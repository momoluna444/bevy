use std::{marker::PhantomData, ops::Range};

use bevy_core_pipeline::core_3d::{Opaque3dBatchSetKey, Opaque3dBinKey};
use bevy_ecs::entity::Entity;
use bevy_render::{
    render_phase::{
        BinnedPhaseItem, BinnedRenderPhase, BinnedRenderPhasePlugin, DrawFunctionId, PhaseItem,
        PhaseItemExtraIndex, ViewBinnedRenderPhases,
    },
    sync_world::MainEntity,
};

use crate::{MeshPipeline, Pass, PhaseItemExt, PhaseItems, PhaseParams, RenderPhaseType};

macro_rules! define_dummy_phase {
    ($name:ident) => {
        pub struct $name<P>(PhantomData<P>);

        impl<P: Pass> PhaseItemExt for $name<P> {
            const PHASE_TYPES: RenderPhaseType = RenderPhaseType::empty();
            type Phase = BinnedRenderPhase<Self>;
            type Phases = ViewBinnedRenderPhases<Self>;
            type Plugin = BinnedRenderPhasePlugin<Self, MeshPipeline>;

            fn queue(_render_phase: &mut Self::Phase, _params: &PhaseParams) {
                panic!()
            }
        }

        impl<P: Pass> PhaseItem for $name<P> {
            fn entity(&self) -> Entity {
                panic!()
            }

            fn main_entity(&self) -> MainEntity {
                panic!()
            }

            fn draw_function(&self) -> DrawFunctionId {
                panic!()
            }

            fn batch_range(&self) -> &Range<u32> {
                panic!()
            }

            fn batch_range_mut(&mut self) -> &mut Range<u32> {
                panic!()
            }

            fn extra_index(&self) -> PhaseItemExtraIndex {
                panic!()
            }

            fn batch_range_and_extra_index_mut(
                &mut self,
            ) -> (&mut Range<u32>, &mut PhaseItemExtraIndex) {
                panic!()
            }
        }

        impl<P: Pass> BinnedPhaseItem for $name<P> {
            type BatchSetKey = Opaque3dBatchSetKey;
            type BinKey = Opaque3dBinKey;

            fn new(
                _batch_set_key: Self::BatchSetKey,
                _bin_key: Self::BinKey,
                _representative_entity: (Entity, MainEntity),
                _batch_range: Range<u32>,
                _extra_index: PhaseItemExtraIndex,
            ) -> Self {
                panic!()
            }
        }
    };
}

define_dummy_phase!(DummyPhase2);
define_dummy_phase!(DummyPhase3);
define_dummy_phase!(DummyPhase4);

impl<P, PIE> PhaseItems<P> for PIE
where
    P: Pass,
    PIE: PhaseItemExt,
{
    type Phase1 = PIE;
    type Phase2 = DummyPhase2<P>;
    type Phase3 = DummyPhase3<P>;
    type Phase4 = DummyPhase4<P>;
}

impl<P, PIE1> PhaseItems<P> for (PIE1,)
where
    P: Pass,
    PIE1: PhaseItemExt,
{
    type Phase1 = PIE1;
    type Phase2 = DummyPhase2<P>;
    type Phase3 = DummyPhase3<P>;
    type Phase4 = DummyPhase4<P>;
}

impl<P, PIE1, PIE2> PhaseItems<P> for (PIE1, PIE2)
where
    P: Pass,
    PIE1: PhaseItemExt,
    PIE2: PhaseItemExt,
{
    type Phase1 = PIE1;
    type Phase2 = PIE2;
    type Phase3 = DummyPhase3<P>;
    type Phase4 = DummyPhase4<P>;
}

impl<P, PIE1, PIE2, PIE3> PhaseItems<P> for (PIE1, PIE2, PIE3)
where
    P: Pass,
    PIE1: PhaseItemExt,
    PIE2: PhaseItemExt,
    PIE3: PhaseItemExt,
{
    type Phase1 = PIE1;
    type Phase2 = PIE2;
    type Phase3 = PIE3;
    type Phase4 = DummyPhase4<P>;
}

impl<P, PIE1, PIE2, PIE3, PIE4> PhaseItems<P> for (PIE1, PIE2, PIE3, PIE4)
where
    P: Pass,
    PIE1: PhaseItemExt,
    PIE2: PhaseItemExt,
    PIE3: PhaseItemExt,
    PIE4: PhaseItemExt,
{
    type Phase1 = PIE1;
    type Phase2 = PIE2;
    type Phase3 = PIE3;
    type Phase4 = PIE4;
}
