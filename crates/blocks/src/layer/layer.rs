use crate::block::EqualBlock;
use crate::types::types::{ViewRepr, ViewMutRepr};
use ndarray::{Array2,ArrayView2,ArrayViewMut2};
use crate::utils::copy_array;
use ndarray::Axis;
use rayon::prelude::*;
use core::ops::AddAssign;
use super::layer_allocations::LayerAllocations;

pub struct Layer<T> 
where 
    T:EqualBlock<I = Array2<f32>> + LayerAllocations,
    T::P: AddAssign,
    for <'a> <T::P as ViewMutRepr>::ViewMut<'a> : AddAssign<T::P>,
{
    pub chunk_size: usize,
    pub num_blocks: usize,
    pub num_threads_per_block: usize,
    pub block: T,
    pub allocation_config: T::AllocationConfig,
}

impl<T> Layer<T> 
where
    T:EqualBlock<I = Array2<f32>> + LayerAllocations,
    T::P: AddAssign,
    for <'a> <T::P as ViewMutRepr>::ViewMut<'a> : AddAssign<T::P>,
{
    pub fn new(chunk_size:usize, num_blocks: usize,
        num_threads_per_block: usize, block: T, allocation_config: T::AllocationConfig) -> Self {
        Self { 
            chunk_size,
            num_blocks,
            num_threads_per_block,
            block,
            allocation_config,
        }
    }
    pub fn forward<'p>(&self,
        parameters:&<T::P as ViewRepr>::View<'p>,
        input:&ArrayView2<f32>,
        output:&mut ArrayViewMut2<f32>) {
        let dim_in = input.shape()[0];
        let dim_out = output.shape()[0];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.num_blocks).build().unwrap();
        let input_iter = input.axis_chunks_iter(Axis(1), self.chunk_size);
        let output_iter = output.axis_chunks_iter_mut(Axis(1),self.chunk_size);
        pool.install(||{
            (output_iter,input_iter)
            .into_par_iter()
            .for_each_init(||(rayon::ThreadPoolBuilder::new().num_threads(self.num_threads_per_block).build().unwrap(),
                Array2::<f32>::zeros((dim_in,self.chunk_size)),Array2::<f32>::zeros((dim_out,self.chunk_size)),
                T::create_allocations(&self.allocation_config),T::allocate_forward_context(&self.allocation_config)), 
                |(block_pool,block_input,block_output,
                allocations, forward_context),
                (mut output, input)|{
                let actual_chunk_size = input.shape()[1];
                let allocation_mut_view = &mut allocations.view_mut_repr();
                let fc_mut_view = &mut forward_context.view_mut_repr();
                if actual_chunk_size==self.chunk_size {
                    copy_array(&mut block_input.view_mut(),&input);
                    block_pool.install(||{
                        self.block.forward(parameters, &block_input.view(), 
                        &mut block_output.view_mut(), allocation_mut_view, fc_mut_view);
                    });
                    copy_array(&mut output,&block_output.view());
                }
                else {
                    let mut block_input = Array2::<f32>::zeros((dim_in,actual_chunk_size));
                    let mut block_output = Array2::<f32>::zeros((dim_out,actual_chunk_size));
                    copy_array(&mut block_input.view_mut(),&input);
                    block_pool.install(||{
                        self.block.forward(parameters, &block_input.view(), 
                        &mut block_output.view_mut(), allocation_mut_view, fc_mut_view);
                    });
                    copy_array(&mut output,&block_output.view());
                }
            });
        });
    }
    pub fn backward<'gp,'p>(&self,
        parameter_gradients:&mut <T::P as ViewMutRepr>::ViewMut<'gp>,
        input_gradients:&mut ArrayViewMut2<f32>,
        output_gradients:&ArrayView2<f32>,
        input:&ArrayView2<f32>,
        parameters:&<T::P as ViewRepr>::View<'p>,
    ) {
        let dim_in = input.shape()[0];
        let dim_out = output_gradients.shape()[0];
        let pool = rayon::ThreadPoolBuilder::new().num_threads(self.num_blocks).build().unwrap();
        let in_grad_iter = input_gradients.axis_chunks_iter_mut(Axis(1), self.chunk_size);
        let out_grad_iter = output_gradients.axis_chunks_iter(Axis(1), self.chunk_size);
        let input_iter = input.axis_chunks_iter(Axis(1), self.chunk_size);
        pool.install(||{
            let acc_param_grad : T::P = (in_grad_iter,out_grad_iter,input_iter)
            .into_par_iter()
            .fold(|| (rayon::ThreadPoolBuilder::new().num_threads(self.num_threads_per_block).build().unwrap(),
            T::allocate_parameters(&self.allocation_config),T::allocate_forward_context(&self.allocation_config),
            T::create_allocations(&self.allocation_config),Array2::<f32>::zeros((dim_in,self.chunk_size)),
            Array2::<f32>::zeros((dim_out,self.chunk_size)),Array2::<f32>::zeros((dim_out,self.chunk_size)),
            Array2::<f32>::zeros((dim_in,self.chunk_size))),
            |(block_pool,mut param_grad,mut fc, mut alloc,
                mut block_in_grad, mut block_out_grad, 
                mut block_output, mut block_input),
            (mut in_grad, out_grad,
                input)| {
                let actual_chunk_size = input.shape()[1];
                block_pool.install(||{
                    if actual_chunk_size==self.chunk_size {
                        copy_array(&mut block_out_grad.view_mut(),&out_grad);
                        copy_array(&mut block_input.view_mut(),&input);
                        self.block.forward(parameters, &block_input.view(), &mut block_output.view_mut(), 
                        &mut alloc.view_mut_repr(), &mut fc.view_mut_repr());
                        self.block.backward(&mut param_grad.view_mut_repr(), 
                        &mut block_in_grad.view_mut_repr(), &block_out_grad.view(), 
                        &block_output.view(), &block_input.view(), parameters, 
                        &mut alloc.view_mut_repr(), &fc.view_repr());
                        copy_array(&mut in_grad,&block_in_grad.view());
                    }
                    else {
                        let mut block_in_grad = Array2::<f32>::zeros((dim_in,actual_chunk_size));
                        let mut block_out_grad = Array2::<f32>::zeros((dim_out,actual_chunk_size));
                        let mut block_input = Array2::<f32>::zeros((dim_in,actual_chunk_size));
                        let mut block_output = Array2::<f32>::zeros((dim_out,actual_chunk_size));
                        copy_array(&mut block_out_grad.view_mut(),&out_grad);
                        copy_array(&mut block_input.view_mut(),&input);
                        self.block.forward(parameters, &block_input.view(), &mut block_output.view_mut(), 
                        &mut alloc.view_mut_repr(), &mut fc.view_mut_repr());
                        self.block.backward(&mut param_grad.view_mut_repr(), 
                        &mut block_in_grad.view_mut_repr(), &block_out_grad.view(), 
                        &block_output.view(), &block_input.view(), parameters, 
                        &mut alloc.view_mut_repr(), &fc.view_repr());
                        copy_array(&mut in_grad,&block_in_grad.view());
                    }
                });   
                // return identity
                (block_pool,param_grad,fc, alloc,
                    block_in_grad, block_out_grad, 
                    block_output, block_input)
            }).map(|(_block_pool,param_grad,_fc, _alloc,
                _block_in_grad, _block_out_grad, 
                _block_output, _block_input)|{
                param_grad
            }).reduce(||T::allocate_parameters(&self.allocation_config),
                |mut acc_param_grad, param_grad|{
                acc_param_grad += param_grad;
                acc_param_grad
            });
            *parameter_gradients += acc_param_grad;
        });
    }
}