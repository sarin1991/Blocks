use blocks::activations::relu::ReLUBlock;
use numpy::ndarray::ArrayViewMut2;
use pyo3::prelude::*;
use crate::conversion::FromMutRefPy;
use blocks::block::FusedBlock;
use crate::types::array::ArrayMutRef2D;

#[pyclass]
pub struct PyReLUBlock {
    relu_block:ReLUBlock,
}

#[pymethods]
impl PyReLUBlock {
    #[new]
    fn new() -> Self {
        Self {
            relu_block: ReLUBlock::new(),
        }
    }
    fn forward(&self, mut input_output:ArrayMutRef2D) {
        let mut relu_in_out = ArrayViewMut2::from_py(&mut input_output);
        self.relu_block.forward(&&0.0f32,&mut relu_in_out,&mut &mut 0.0f32,&mut &mut 0.0f32);
    }
    fn backward(&self, mut input_output_gradients: ArrayMutRef2D,
        mut output: ArrayMutRef2D) {
        let mut relu_in_out_grad = ArrayViewMut2::from_py(&mut input_output_gradients);
        let mut relu_out = ArrayViewMut2::from_py(&mut output);
        self.relu_block.backward(&mut &mut 0.0f32, &mut relu_in_out_grad, 
            &mut relu_out, &&0.0f32, &mut &mut 0.0f32, &&0.0f32);
    }
}