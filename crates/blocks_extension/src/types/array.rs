use pyo3::prelude::*;
use numpy::IntoPyArray;
use numpy::{PyReadonlyArray1,PyArray1,PyReadwriteArray1};
use numpy::ndarray::{Array1,ArrayView1,ArrayViewMut1};
use numpy::{PyReadonlyArray2,PyArray2,PyReadwriteArray2};
use numpy::ndarray::{Array2,ArrayView2,ArrayViewMut2};
use numpy::{PyReadonlyArray3,PyArray3,PyReadwriteArray3};
use numpy::ndarray::{Array3,ArrayView3,ArrayViewMut3};
use crate::conversion::{FromRefPy,FromMutRefPy,IntoPy};

pub type Array1D = Py<PyArray1<f32>>;
pub type ArrayRef1D<'py> = PyReadonlyArray1<'py,f32>;
pub type ArrayMutRef1D<'py> = PyReadwriteArray1<'py,f32>;

impl IntoPy for Array1<f32>{
    type PyType = Array1D;
    fn into_py(self, py: Python<'_>) -> Self::PyType {
        self.into_pyarray(py).into()
    }
}
impl<'r> FromRefPy<'r> for ArrayView1<'r,f32> {
    type PyType<'py> = ArrayRef1D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        py_type.as_array()
    }
}
impl<'r> FromMutRefPy<'r> for ArrayViewMut1<'r,f32> 
where Self:'r,
{
    type PyType<'py> = ArrayMutRef1D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        py_type.as_array_mut()
    }
}

pub type Array2D = Py<PyArray2<f32>>;
pub type ArrayRef2D<'py> = PyReadonlyArray2<'py,f32>;
pub type ArrayMutRef2D<'py> = PyReadwriteArray2<'py,f32>;

impl IntoPy for Array2<f32>{
    type PyType = Array2D;
    fn into_py(self, py: Python<'_>) -> Self::PyType {
        self.into_pyarray(py).into()
    }
}
impl<'r> FromRefPy<'r> for ArrayView2<'r,f32> {
    type PyType<'py> = ArrayRef2D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        py_type.as_array()
    }
}
impl<'r> FromMutRefPy<'r> for ArrayViewMut2<'r,f32> 
where Self:'r,
{
    type PyType<'py> = ArrayMutRef2D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        py_type.as_array_mut()
    }
}

pub type Array3D = Py<PyArray3<f32>>;
pub type ArrayRef3D<'py> = PyReadonlyArray3<'py,f32>;
pub type ArrayMutRef3D<'py> = PyReadwriteArray3<'py,f32>;

impl IntoPy for Array3<f32>{
    type PyType = Array3D;
    fn into_py(self, py: Python<'_>) -> Self::PyType {
        self.into_pyarray(py).into()
    }
}
impl<'r> FromRefPy<'r> for ArrayView3<'r,f32> {
    type PyType<'py> = ArrayRef3D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        py_type.as_array()
    }
}
impl<'r> FromMutRefPy<'r> for ArrayViewMut3<'r,f32> 
where Self:'r,
{
    type PyType<'py> = ArrayMutRef3D<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        py_type.as_array_mut()
    }
}