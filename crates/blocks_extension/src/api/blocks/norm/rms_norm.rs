use blocks::norm::rms_norm::RMSNormBlock;
use numpy::ndarray::{ArrayViewMut2, ArrayView1, ArrayViewMut1};
use pyo3::prelude::*;
use crate::conversion::{FromRefPy, FromMutRefPy};
use crate::types::array::{ArrayMutRef1D,ArrayRef1D};
use blocks::block::FusedBlock;
use crate::types::array::ArrayMutRef2D;

#[pyclass]
pub struct PyRMSNormBlock {
    rms_norm_block:RMSNormBlock,
}

#[pymethods]
impl PyRMSNormBlock {
    #[new]
    fn new(dim: usize, chunk_size: usize) -> Self {
        Self {
            rms_norm_block: RMSNormBlock::new(dim,chunk_size),
        }
    }
    fn forward(&self, parameters: ArrayRef1D,mut input_output:ArrayMutRef2D, mut forward_context: ArrayMutRef1D) {
        let params = ArrayView1::from_py(&parameters);
        let mut fc = ArrayViewMut1::from_py(&mut forward_context);
        let mut in_out = ArrayViewMut2::from_py(&mut input_output);
        self.rms_norm_block.forward(&params,&mut in_out,&mut &mut 0.0f32,&mut fc);
    }
    fn backward(&self, mut parameter_gradients: ArrayMutRef1D,mut input_output_gradients: ArrayMutRef2D,
        mut input_output: ArrayMutRef2D, forward_context: ArrayRef1D, parameters: ArrayRef1D) {
        let fc = ArrayView1::from_py(&forward_context);
        let params = ArrayView1::from_py(&parameters);
        let mut param_grad = ArrayViewMut1::from_py(&mut parameter_gradients);
        let mut in_out_grad = ArrayViewMut2::from_py(&mut input_output_gradients);
        let mut in_out = ArrayViewMut2::from_py(&mut input_output);
        self.rms_norm_block.backward(&mut param_grad, &mut in_out_grad, 
            &mut in_out, &params, &mut &mut 0.0f32, &fc);
    }
}