use crate::block::BaseBlock;
pub trait LayerAllocations : BaseBlock + std::marker::Sync
{
    type AllocationConfig : std::marker::Sync;
    fn allocate_parameters(config:&Self::AllocationConfig) -> Self::P;
    fn create_allocations(chunk_size:usize,config:&Self::AllocationConfig) -> Self::A;
    fn allocate_forward_context(chunk_size:usize,config:&Self::AllocationConfig) -> Self::F;
}