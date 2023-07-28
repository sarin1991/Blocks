use blocks::activations::relu::ReLUBlock;
use numpy::ndarray::{ArrayView2, ArrayViewMut2};
use pyo3::prelude::*;
use crate::conversion::{FromMutRefPy,FromRefPy};
use blocks::block::FusedBlock;
use crate::types::array::{ArrayRef2D,ArrayMutRef2D};

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
        output: ArrayRef2D) {
        let mut relu_in_out_grad = ArrayViewMut2::from_py(&mut input_output_gradients);
        let relu_out = ArrayView2::from_py(&output);
        self.relu_block.backward(&mut &mut 0.0f32, &mut relu_in_out_grad, 
            &relu_out, &&0.0f32, &mut &mut 0.0f32, &&0.0f32);
    }
}