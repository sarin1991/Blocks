use numpy::ndarray::{Array1,ArrayView1,ArrayViewMut1, ArrayViewMut2};
use pyo3::prelude::*;
use blocks::non_linear_blocks::rms_dyad_relu::{RMSDyadReLUBlock,RMSDyadReLUBlockParameterConfig};
use blocks::linear_blocks::dyad::dyad_block::parameters::{DyadBlockParameterView,
                                                        DyadBlockParameterMutView, DyadBlockParameter};
use crate::api::blocks::linear_blocks::dyad_block::types::{PyDyadParameter,PyDyadParameterMutView,PyDyadParameterView};
use crate::types::array::{Array1D,ArrayMutRef1D,ArrayRef1D};
use blocks::layer::layer_allocations::LayerAllocations;
use crate::conversion::{FromMutRefPy,FromRefPy,IntoPy};
use crate::types::array::ArrayMutRef2D;
use blocks::block::OwnedBlock;
pub type PyRMSDyadReLUBlockParamView<'py> = (ArrayRef1D<'py>,PyDyadParameterView<'py>);
pub type PyRMSDyadReLUBlockParamMutView<'py> = (ArrayMutRef1D<'py>,PyDyadParameterMutView<'py>);
pub type PyRMSDyadReLUBlockParam = (Array1D,PyDyadParameter);

impl<'r> FromRefPy<'r> for (ArrayView1<'r,f32>,DyadBlockParameterView<'r>){
    type PyType<'py> = PyRMSDyadReLUBlockParamView<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        (ArrayView1::from_py(&py_type.0),DyadBlockParameterView::from_py(&py_type.1))
    }
}

impl<'r> FromMutRefPy<'r> for (ArrayViewMut1<'r,f32>,DyadBlockParameterMutView<'r>)
where Self:'r,
{
    type PyType<'py> = PyRMSDyadReLUBlockParamMutView<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        (ArrayViewMut1::from_py(&mut py_type.0),
        DyadBlockParameterMutView::from_py(&mut py_type.1))
    }
}

impl IntoPy for (Array1<f32>,DyadBlockParameter) {
    type PyType = PyRMSDyadReLUBlockParam;
    fn into_py(self, py: pyo3::Python<'_>) -> Self::PyType {
        (
            self.0.into_py(py),
            self.1.into_py(py),
        )
    }
}

#[pyclass]
pub struct PyRMSDyadReLUBlock {
    rms_dyad_relu_block:RMSDyadReLUBlock,
}

#[pymethods]
impl PyRMSDyadReLUBlock {
    #[new]
    fn new(dyad_dim:usize,dim_in:usize,dim_out:usize,has_bias:bool,rms_norm_chunk_size:usize) -> Self {
        let parameter_config = 
            RMSDyadReLUBlockParameterConfig::new(dyad_dim, dim_in, dim_out, has_bias, rms_norm_chunk_size);
        let rms_dyad_relu_block = RMSDyadReLUBlock::new(&parameter_config);
        Self {
            rms_dyad_relu_block
        }
    }
    fn allocate_parameters(&self,py: Python<'_>) -> PyRMSDyadReLUBlockParam {
        let rms_dyad_relu_block_parameters = RMSDyadReLUBlock::allocate_parameters(&self.rms_dyad_relu_block.parameter_config);
        rms_dyad_relu_block_parameters.into_py(py)
    }
    fn forward(&self,parameters: PyRMSDyadReLUBlockParamView, mut input:ArrayMutRef2D, 
            mut output: ArrayMutRef2D, mut allocations:ArrayMutRef1D) {
        let rms_dyad_relu_params = 
            <(ArrayView1::<f32>,DyadBlockParameterView)>::from_py(&parameters);
        let mut block_input = ArrayViewMut2::from_py(&mut input);
        let mut block_output = ArrayViewMut2::from_py(&mut output);
        let mut block_alloc = ArrayViewMut1::from_py(&mut allocations);
        self.rms_dyad_relu_block.forward(&rms_dyad_relu_params, &mut block_input, &mut block_output, 
            &mut block_alloc, &mut &mut 0.0f32);
    }
    fn backward(&self, mut parameter_gradients: PyRMSDyadReLUBlockParamMutView, mut input_gradients: ArrayMutRef2D,
        mut output_gradients: ArrayMutRef2D, mut output: ArrayMutRef2D, mut input: ArrayMutRef2D,
        parameters: PyRMSDyadReLUBlockParamView, mut allocations:ArrayMutRef1D) {
        let mut block_param_grad = 
            <(ArrayViewMut1<f32>,DyadBlockParameterMutView)>::from_py(&mut parameter_gradients);
        let mut block_in_grad = ArrayViewMut2::from_py(&mut input_gradients);
        let mut block_out_grad = ArrayViewMut2::from_py(&mut output_gradients);
        let mut block_out = ArrayViewMut2::from_py(&mut output);
        let mut block_in = ArrayViewMut2::from_py(&mut input);
        let block_params = <(ArrayView1::<f32>,DyadBlockParameterView)>::from_py(&parameters);
        let mut block_alloc = ArrayViewMut1::from_py(&mut allocations);
        self.rms_dyad_relu_block.backward(&mut block_param_grad, 
            &mut block_in_grad, &mut block_out_grad, &mut block_out, 
            &mut block_in, &block_params, &mut block_alloc, 
            &&0.0f32);
    }
}