use std::{any::TypeId, sync::Arc};

use bevy_app::Plugin;
use bevy_camera::{Camera3d, Projection};
use bevy_core_pipeline::{
    core_3d::{
        AlphaMask3d, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, Transmissive3d, Transparent3d,
    },
    oit::OrderIndependentTransparencySettings,
    prepass::{
        DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
        OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey,
    },
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_ecs::{
    component::Component,
    prelude::*,
    query::Has,
    system::{Query, ResMut, SystemChangeTick},
};
use bevy_light::{EnvironmentMapLight, IrradianceVolume, ShadowFilteringMethod};
use bevy_mesh::{BaseMeshPipelineKey, MeshVertexBufferLayoutRef};
use bevy_render::{
    Render, RenderApp, RenderDebugFlags, RenderSystems, camera::TemporalJitter, render_phase::{
        BinnedRenderPhase, BinnedRenderPhasePlugin, BinnedRenderPhaseType, PhaseItemExtraIndex, SortedRenderPhase, SortedRenderPhasePlugin, ViewBinnedRenderPhases, ViewSortedRenderPhases
    }, render_resource::{
        RenderPipelineDescriptor, SpecializedMeshPipeline, SpecializedMeshPipelineError,
    }, view::{ExtractedView, Msaa}
};
use bevy_shader::ShaderDefVal;

use crate::{
    alpha_mode_pipeline_key, screen_space_specular_transmission_pipeline_key,
    tonemapping_pipeline_key, DistanceFog, DrawMaterial, ErasedMaterialPipelineKey,
    MaterialFragmentShader, MaterialPipeline, MaterialProperties, MaterialVertexShader,
    MeshPipeline, MeshPipelineKey, OpaqueRendererMethod, Pass, PassId, PassPlugin, PhaseItemExt,
    PhaseParams, PipelineSpecializer, PreparedMaterial, RenderLightmap, RenderMeshInstanceFlags,
    RenderViewLightProbes, ScreenSpaceAmbientOcclusion, ViewKeyCache, ViewSpecializationTicks,
    MATERIAL_BIND_GROUP_INDEX,
};

#[derive(Default)]
pub struct MainPassPlugin {
    pub debug_flags: RenderDebugFlags,
}
impl Plugin for MainPassPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.add_plugins(PassPlugin::<MainPass>::new(self.debug_flags));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_systems(
                Render,
                check_views_need_specialization::<MainPass>.in_set(RenderSystems::PrepareAssets),
            );
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Component)]
pub struct MainPass;

impl Pass for MainPass {
    type Specializer = MaterialPipelineSpecializer;

    type PhaseItems = (Opaque3d, AlphaMask3d, Transmissive3d, Transparent3d);

    type RenderCommand = DrawMaterial;
}

// TODO: Redesign
pub fn check_views_need_specialization<P: Pass>(
    mut view_key_cache: ResMut<ViewKeyCache<P>>,
    mut view_specialization_ticks: ResMut<ViewSpecializationTicks<P>>,
    mut views: Query<(
        &ExtractedView,
        &Msaa,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ShadowFilteringMethod>,
        Has<ScreenSpaceAmbientOcclusion>,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        Option<&Camera3d>,
        Has<TemporalJitter>,
        Option<&Projection>,
        Has<DistanceFog>,
        (
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<RenderViewLightProbes<IrradianceVolume>>,
        ),
        Has<OrderIndependentTransparencySettings>,
    )>,
    ticks: SystemChangeTick,
) {
    for (
        view,
        msaa,
        tonemapping,
        dither,
        shadow_filter_method,
        ssao,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        camera_3d,
        temporal_jitter,
        projection,
        distance_fog,
        (has_environment_maps, has_irradiance_volumes),
        has_oit,
    ) in views.iter_mut()
    {
        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }

        if temporal_jitter {
            view_key |= MeshPipelineKey::TEMPORAL_JITTER;
        }

        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        if has_oit {
            view_key |= MeshPipelineKey::OIT_ENABLED;
        }

        if let Some(projection) = projection {
            view_key |= match projection {
                Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
                Projection::Custom(_) => MeshPipelineKey::VIEW_PROJECTION_NONSTANDARD,
            };
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= tonemapping_pipeline_key(*tonemapping);
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }
        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if distance_fog {
            view_key |= MeshPipelineKey::DISTANCE_FOG;
        }
        if let Some(camera_3d) = camera_3d {
            view_key |= screen_space_specular_transmission_pipeline_key(
                camera_3d.screen_space_specular_transmission_quality,
            );
        }
        if !view_key_cache
            .get_mut(&view.retained_view_entity)
            .is_some_and(|current_key| *current_key == view_key)
        {
            view_key_cache.insert(view.retained_view_entity, view_key);
            view_specialization_ticks.insert(view.retained_view_entity, ticks.this_run());
        }
    }
}

pub struct MaterialPipelineSpecializer {
    pub(crate) pipeline: MaterialPipeline,
    pub(crate) properties: Arc<MaterialProperties>,
    pub(crate) pass_id: PassId,
}

impl PipelineSpecializer for MaterialPipelineSpecializer {
    type Pipeline = MaterialPipeline;

    fn new(pipeline: &Self::Pipeline, material: &PreparedMaterial, pass_id: PassId) -> Self {
        MaterialPipelineSpecializer {
            pipeline: pipeline.clone(),
            properties: material.properties.clone(),
            pass_id,
        }
    }

    fn create_key(
        view_key: MeshPipelineKey,
        base_mesh_key: &BaseMeshPipelineKey,
        mesh_instance_flags: &RenderMeshInstanceFlags,
        material: &PreparedMaterial,
        lightmap: Option<&RenderLightmap>,
        has_crossfade: bool,
        type_id: TypeId,
    ) -> Option<Self::Key> {
        let mut mesh_pipeline_key_bits = material.properties.mesh_pipeline_key_bits;
        mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
            material.properties.alpha_mode,
            &Msaa::from_samples(view_key.msaa_samples()),
        ));
        let mut mesh_key = view_key
            | MeshPipelineKey::from_bits_retain(base_mesh_key.bits())
            | mesh_pipeline_key_bits;

        if let Some(lightmap) = lightmap {
            mesh_key |= MeshPipelineKey::LIGHTMAPPED;

            if lightmap.bicubic_sampling {
                mesh_key |= MeshPipelineKey::LIGHTMAP_BICUBIC_SAMPLING;
            }
        }

        if has_crossfade {
            mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
        }

        if view_key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
            // If the previous frame have skins or morph targets, note that.
            if mesh_instance_flags.contains(RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN) {
                mesh_key |= MeshPipelineKey::HAS_PREVIOUS_SKIN;
            }
            if mesh_instance_flags.contains(RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH) {
                mesh_key |= MeshPipelineKey::HAS_PREVIOUS_MORPH;
            }
        }

        let material_key = material.properties.material_key.clone();

        Some(Self::Key {
            mesh_key,
            material_key,
            type_id,
        })
    }

    fn pass_id(&self) -> PassId {
        self.pass_id
    }
}

impl SpecializedMeshPipeline for MaterialPipelineSpecializer {
    type Key = ErasedMaterialPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self
            .pipeline
            .mesh_pipeline
            .specialize(key.mesh_key, layout)?;
        descriptor.vertex.shader_defs.push(ShaderDefVal::UInt(
            "MATERIAL_BIND_GROUP".into(),
            MATERIAL_BIND_GROUP_INDEX as u32,
        ));
        if let Some(ref mut fragment) = descriptor.fragment {
            fragment.shader_defs.push(ShaderDefVal::UInt(
                "MATERIAL_BIND_GROUP".into(),
                MATERIAL_BIND_GROUP_INDEX as u32,
            ));
        };
        if let Some(vertex_shader) = self
            .properties
            .get_shader(MaterialVertexShader(self.pass_id))
        {
            descriptor.vertex.shader = vertex_shader.clone();
        }

        if let Some(fragment_shader) = self
            .properties
            .get_shader(MaterialFragmentShader(self.pass_id))
        {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader.clone();
        }

        descriptor
            .layout
            .insert(3, self.properties.material_layout.as_ref().unwrap().clone());

        if let Some(specialize) = self.properties.specialize {
            specialize(&self.pipeline, &mut descriptor, layout, key, self.pass_id)?;
        }

        // If bindless mode is on, add a `BINDLESS` define.
        if self.properties.bindless {
            descriptor.vertex.shader_defs.push("BINDLESS".into());
            if let Some(ref mut fragment) = descriptor.fragment {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }

        Ok(descriptor)
    }
}

impl PhaseItemExt for Opaque3d {
    type Phase = BinnedRenderPhase<Self>;
    type Phases = ViewBinnedRenderPhases<Self>;
    type Plugin = BinnedRenderPhasePlugin<Self, MeshPipeline>;

    fn queue(render_phase: &mut Self::Phase, params: &PhaseParams) {
        if params.material.properties.render_method == OpaqueRendererMethod::Deferred {
            // Even though we aren't going to insert the entity into
            // a bin, we still want to update its cache entry. That
            // way, we know we don't need to re-examine it in future
            // frames.
            render_phase.update_cache(params.main_entity, None, params.current_change_tick);
            return;
        }
        let (vertex_slab, index_slab) = params
            .mesh_allocator
            .mesh_slabs(&params.mesh_instance.mesh_asset_id);

        render_phase.add(
            Opaque3dBatchSetKey {
                pipeline: params.pipeline,
                draw_function: params.draw_function,
                material_bind_group_index: Some(params.material.binding.group.0),
                vertex_slab: vertex_slab.unwrap_or_default(),
                index_slab,
                lightmap_slab: params
                    .mesh_instance
                    .shared
                    .lightmap_slab_index
                    .map(|index| *index),
            },
            Opaque3dBinKey {
                asset_id: params.mesh_instance.mesh_asset_id.into(),
            },
            (params.entity, params.main_entity),
            params.mesh_instance.current_uniform_index,
            BinnedRenderPhaseType::mesh(
                params.mesh_instance.should_batch(),
                &params.gpu_preprocessing_support,
            ),
            params.current_change_tick,
        );
    }
}

impl PhaseItemExt for AlphaMask3d {
    type Phase = BinnedRenderPhase<Self>;
    type Phases = ViewBinnedRenderPhases<Self>;
    type Plugin = BinnedRenderPhasePlugin<Self, MeshPipeline>;

    fn queue(render_phase: &mut Self::Phase, params: &PhaseParams) {
        let (vertex_slab, index_slab) = params
            .mesh_allocator
            .mesh_slabs(&params.mesh_instance.mesh_asset_id);

        render_phase.add(
            OpaqueNoLightmap3dBatchSetKey {
                pipeline: params.pipeline,
                draw_function: params.draw_function,
                material_bind_group_index: Some(params.material.binding.group.0),
                vertex_slab: vertex_slab.unwrap_or_default(),
                index_slab,
            },
            OpaqueNoLightmap3dBinKey {
                asset_id: params.mesh_instance.mesh_asset_id.into(),
            },
            (params.entity, params.main_entity),
            params.mesh_instance.current_uniform_index,
            BinnedRenderPhaseType::mesh(
                params.mesh_instance.should_batch(),
                &params.gpu_preprocessing_support,
            ),
            params.current_change_tick,
        );
    }
}

impl PhaseItemExt for Transmissive3d {
    type Phase = SortedRenderPhase<Self>;
    type Phases = ViewSortedRenderPhases<Self>;
    type Plugin = SortedRenderPhasePlugin<Self, MeshPipeline>;

    fn queue(render_phase: &mut Self::Phase, params: &PhaseParams) {
        let (_, index_slab) = params
            .mesh_allocator
            .mesh_slabs(&params.mesh_instance.mesh_asset_id);
        let distance = params
            .rangefinder
            .distance_translation(&params.mesh_instance.translation)
            + params.material.properties.depth_bias;

        render_phase.add(Transmissive3d {
            entity: (params.entity, params.main_entity),
            draw_function: params.draw_function,
            pipeline: params.pipeline,
            distance,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: index_slab.is_some(),
        });
    }
}

impl PhaseItemExt for Transparent3d {
    type Phase = SortedRenderPhase<Self>;
    type Phases = ViewSortedRenderPhases<Self>;
    type Plugin = SortedRenderPhasePlugin<Self, MeshPipeline>;

    fn queue(render_phase: &mut Self::Phase, params: &PhaseParams) {
        let (_, index_slab) = params
            .mesh_allocator
            .mesh_slabs(&params.mesh_instance.mesh_asset_id);
        let distance = params
            .rangefinder
            .distance_translation(&params.mesh_instance.translation)
            + params.material.properties.depth_bias;

        render_phase.add(Transparent3d {
            entity: (params.entity, params.main_entity),
            draw_function: params.draw_function,
            pipeline: params.pipeline,
            distance,
            batch_range: 0..1,
            extra_index: PhaseItemExtraIndex::None,
            indexed: index_slab.is_some(),
        });
    }
}
