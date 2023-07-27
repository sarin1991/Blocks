use crate::{block::EqualBlock, layer::layer_allocations::LayerAllocations, types::types::BlockTypeAllocate};
use super::parameters::{DyadBlockParameter,
    DyadBlockParameterMutView,DyadBlockParameterView};
use ndarray::{Array2,ArrayView2,ArrayViewMut2};
use super::parameters::DyadBlockParameterConfig;
use ndarray::{Axis,Zip};
use rayon::prelude::*;
use crate::blas::{sgemm,gemm};
use crate::utils::copy_array;

#[derive(Clone, Copy, Debug)]
pub struct DyadBlock {
    pub parameter_config: DyadBlockParameterConfig,
}

impl DyadBlock {
    pub fn new(parameter_config:&DyadBlockParameterConfig) -> Self {
        Self { parameter_config:*parameter_config }
    }
}

impl EqualBlock for DyadBlock {
    type P = DyadBlockParameter;
    type A = f32;
    type I = Array2<f32>;
    type F = f32;
    fn forward(&self, 
            parameters:&DyadBlockParameterView,
            input:&ArrayView2<f32>,
            output:&mut ArrayViewMut2<f32>,
            _allocations:&mut &mut f32,
            _forward_context:&mut &mut f32) {
        let d_i = self.parameter_config.dim_in;
        let d_o = self.parameter_config.dim_out;
        let dy = self.parameter_config.dyad_dim;
        let chunk_size = input.shape()[1];
        let input_slice = input.as_slice().unwrap();
        let mut output_dyad = output.view_mut().into_shape(
            (dy,d_o,chunk_size)).unwrap();
        output_dyad.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(index,mut out_block)|{
            let out_slice = out_block.as_slice_mut().unwrap();
            let wl_mat = parameters.w_lower.index_axis(Axis(0), index);
            let wl_block = wl_mat.as_slice().unwrap();
            let wu_mat = parameters.w_lower.index_axis(Axis(0), index);
            let wu_block = wu_mat.as_slice().unwrap();
            let transa = cblas::Transpose::None;
            let transb = cblas::Transpose::None;
            // lower
            let start_index = index*d_i*chunk_size;
            let in_l = &input_slice[start_index..];
            sgemm(transa, transb, 1.0f32, wl_block, in_l, 0.0f32, out_slice, 
                d_o, chunk_size, d_i, d_o, chunk_size, chunk_size);
            // upper
            let start_index = index*chunk_size;
            let in_u = &input_slice[start_index..];
            let ldb = dy*chunk_size;
            sgemm(transa, transb, 1.0f32, wu_block, in_u, 1.0f32, out_slice, 
                d_o, chunk_size, d_i, d_o, ldb, chunk_size);
        });
        match &parameters.bias {
            Some(b) =>  {
                Zip::from(output.axis_iter_mut(Axis(0)))
                .and(b)
                .for_each(|mut out_block,b_val| {
                    out_block += *b_val;
                })
            },
            None =>{},
        }
    }
    fn backward<'gp,'gi,'go,'o,'i,'p,'a,'f>(&self,
            parameter_gradients:&mut DyadBlockParameterMutView,
            input_gradients:&mut ArrayViewMut2<f32>,
            output_gradients:&ArrayView2<f32>,
            _output:&ArrayView2<f32>,
            input:&ArrayView2<f32>,
            parameters:&DyadBlockParameterView,
            _allocations:&mut &mut f32,
            _forward_context:&&f32,
        ) {
        let d_i = self.parameter_config.dim_in;
        let d_o = self.parameter_config.dim_out;
        let dy = self.parameter_config.dyad_dim;
        let rayon_min_len = (dy/rayon::current_num_threads()).max(1);
        let chunk_size = input.shape()[1];
        let out_grad_dyad = output_gradients.view().into_shape(
            (dy,d_o,chunk_size)).unwrap();
        // w_u
        let input_dyad = input.view().into_shape(
            (d_i,dy,chunk_size)).unwrap();
        let mut in_grad_dyad = input_gradients.view_mut().into_shape(
            (d_i,dy,chunk_size)).unwrap();
        let in_grad_iter = in_grad_dyad.axis_iter_mut(Axis(1));
        let param_wu_iter = parameter_gradients.w_upper.axis_iter_mut(Axis(0));
        let in_iter = input_dyad.axis_iter(Axis(1));
        (in_grad_iter,param_wu_iter,in_iter)
        .into_par_iter()
        .enumerate()
        .with_min_len(rayon_min_len)
        .for_each_init(|| Array2::<f32>::zeros((d_i,chunk_size)), 
        |in_mat, (index,
            (mut in_grad_block,wu_grad,in_block))|{
            let wu_mat = parameters.w_upper.index_axis(Axis(0), index);
            let out_grad_block = out_grad_dyad.index_axis(Axis(0), index);
            // copy input
            copy_array(&mut in_mat.view_mut(),&in_block);
            // w grad
            let transa = cblas::Transpose::None;
            let transb = cblas::Transpose::Ordinary;
            gemm(transa, transb, 1.0f32, out_grad_block,in_mat.view(), 1.0f32, wu_grad);
            // in grad
            let transa = cblas::Transpose::Ordinary;
            let transb = cblas::Transpose::None;
            gemm(transa, transb, 1.0f32, wu_mat, out_grad_block, 0.0f32, in_mat.view_mut());
            // copy input grad
            copy_array(&mut in_grad_block,&in_mat.view());
        });
        // w_l
        let input_dyad = input.view().into_shape(
            (dy,d_i,chunk_size)).unwrap();
        let mut in_grad_dyad = input_gradients.view_mut().into_shape(
            (dy,d_i,chunk_size)).unwrap();
        let in_grad_iter = in_grad_dyad.axis_iter_mut(Axis(0));
        let param_wl_iter = parameter_gradients.w_lower.axis_iter_mut(Axis(0));
        (param_wl_iter,in_grad_iter)
        .into_par_iter()
        .enumerate()
        .with_min_len(rayon_min_len)
        .for_each(|(index,(wl_grad, in_grad_block))|{
            let wl_mat = parameters.w_lower.index_axis(Axis(0), index);
            let out_grad_block = out_grad_dyad.index_axis(Axis(0), index);
            let in_block = input_dyad.index_axis(Axis(0), index);
            // w grad
            let transa = cblas::Transpose::None;
            let transb = cblas::Transpose::Ordinary;
            gemm(transa, transb, 1.0f32, out_grad_block, in_block, 1.0f32, wl_grad);
            // in grad
            let transa = cblas::Transpose::Ordinary;
            let transb = cblas::Transpose::None;
            gemm(transa, transb, 1.0f32, wl_mat, out_grad_block, 1.0f32, in_grad_block);
        });
        // bias grad
        match &mut parameter_gradients.bias {
            Some(b) => {
                let mut b_mut_view = b.view_mut();
                b_mut_view += &output_gradients.sum_axis(Axis(1));
            },
            None => {},
        }
    }
}

impl LayerAllocations for DyadBlock{
    type AllocationConfig = DyadBlockParameterConfig;
    fn allocate_parameters(config:&Self::AllocationConfig) -> Self::P {
        DyadBlockParameter::allocate(config)
    }
    fn create_allocations(_config:&Self::AllocationConfig) -> Self::A {
        0.0f32
    }
    fn allocate_forward_context(_config:&Self::AllocationConfig) -> Self::F {
        0.0f32
    }
}