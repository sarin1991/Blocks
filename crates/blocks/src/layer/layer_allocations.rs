use crate::block::EqualBlock;
pub trait LayerAllocations : EqualBlock + std::marker::Sync
{
    type AllocationConfig : std::marker::Sync;
    fn allocate_parameters(config:&Self::AllocationConfig) -> Self::P;
    fn create_allocations(config:&Self::AllocationConfig) -> Self::A;
    fn allocate_forward_context(config:&Self::AllocationConfig) -> Self::F;
}