use numpy::ndarray::{ArrayView2, ArrayViewMut2};
use pyo3::prelude::*;
use blocks::linear_blocks::dyad::dyad_block::dyad_block::DyadBlock;
use blocks::linear_blocks::dyad::dyad_block::parameters::DyadBlockParameterConfig;
use blocks::linear_blocks::dyad::dyad_block::parameters::{DyadBlockParameterView,
                                                        DyadBlockParameterMutView};
use blocks::layer::layer_allocations::LayerAllocations;
use super::types::{PyDyadParameter,PyDyadParameterView,PyDyadParameterMutView};
use crate::conversion::{FromMutRefPy,FromRefPy,IntoPy};
use blocks::block::EqualBlock;
use crate::types::array::{ArrayRef2D,ArrayMutRef2D};

#[pyclass]
pub struct PyDyadBlock {
    dyad_block:DyadBlock,
}

#[pymethods]
impl PyDyadBlock {
    #[new]
    fn new(dyad_dim:usize,dim_in:usize,dim_out:usize,has_bias:bool) -> Self {
        let parameter_config = DyadBlockParameterConfig::new(dyad_dim, 
            dim_in, dim_out, has_bias);
        let dyad_block = DyadBlock::new(&parameter_config);
        Self {
            dyad_block
        }
    }
    fn allocate_parameters(&self,py: Python<'_>) -> PyDyadParameter {
        let dyad_parameters = DyadBlock::allocate_parameters(&self.dyad_block.parameter_config);
        dyad_parameters.into_py(py)
    }
    fn forward(&self,parameters: PyDyadParameterView, input:ArrayRef2D, mut output: ArrayMutRef2D) {
        let dyad_params = DyadBlockParameterView::from_py(&parameters);
        let dyad_input = ArrayView2::from_py(&input);
        let mut dyad_output = ArrayViewMut2::from_py(&mut output);
        self.dyad_block.forward(&dyad_params, &dyad_input, &mut dyad_output, 
            &mut &mut 0.0f32, &mut &mut 0.0f32);
    }
    fn backward(&self, mut parameter_gradients: PyDyadParameterMutView, mut input_gradients: ArrayMutRef2D,
        output_gradients: ArrayRef2D, output: ArrayRef2D, input: ArrayRef2D,
        parameters: PyDyadParameterView) {
        let mut dyad_param_grad = DyadBlockParameterMutView::from_py(&mut parameter_gradients);
        let mut dyad_in_grad = ArrayViewMut2::from_py(&mut input_gradients);
        let dyad_out_grad = ArrayView2::from_py(&output_gradients);
        let dyad_out = ArrayView2::from_py(&output);
        let dyad_in = ArrayView2::from_py(&input);
        let dyad_params = DyadBlockParameterView::from_py(&parameters);
        self.dyad_block.backward(&mut dyad_param_grad, &mut dyad_in_grad, 
            &dyad_out_grad, &dyad_out, &dyad_in, 
            &dyad_params, &mut &mut 0.0f32, &&0.0f32);
    }
}