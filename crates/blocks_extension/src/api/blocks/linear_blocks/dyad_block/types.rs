use blocks::linear_blocks::dyad::dyad_block::parameters::{DyadBlockParameter,
    DyadBlockParameterMutView,DyadBlockParameterView};
use numpy::ndarray::{ArrayView3, ArrayView1, ArrayViewMut3, ArrayViewMut1};
use crate::conversion::{FromRefPy,FromMutRefPy,IntoPy};
use crate::types::array::{Array1D,ArrayRef1D,ArrayMutRef1D};
use crate::types::array::{Array3D,ArrayRef3D,ArrayMutRef3D};

pub type PyDyadParameterView<'py> = (ArrayRef3D<'py>,ArrayRef3D<'py>,Option<ArrayRef1D<'py>>);
pub type PyDyadParameterMutView<'py> = (ArrayMutRef3D<'py>,ArrayMutRef3D<'py>,Option<ArrayMutRef1D<'py>>);
pub type PyDyadParameter = (Array3D,Array3D,Option<Array1D>);

impl<'r> FromRefPy<'r> for DyadBlockParameterView<'r>{
    type PyType<'py> = PyDyadParameterView<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        Self { 
            w_upper: ArrayView3::from_py(&py_type.0), 
            w_lower: ArrayView3::from_py(&py_type.1), 
            bias: Option::<ArrayView1<f32>>::from_py(&py_type.2),
        }
    }
}

impl<'r> FromMutRefPy<'r> for DyadBlockParameterMutView<'r> 
where Self:'r,
{
    type PyType<'py> = PyDyadParameterMutView<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        Self{
            w_upper: ArrayViewMut3::from_py(&mut py_type.0),
            w_lower: ArrayViewMut3::from_py(&mut py_type.1),
            bias: Option::<ArrayViewMut1<f32>>::from_py(&mut py_type.2),
        }
    }
}

impl IntoPy for DyadBlockParameter {
    type PyType = PyDyadParameter;
    fn into_py(self, py: pyo3::Python<'_>) -> Self::PyType {
        (
            self.w_upper.into_py(py),
            self.w_lower.into_py(py),
            self.bias.into_py(py)
        )
    }
}