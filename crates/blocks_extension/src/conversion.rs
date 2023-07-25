use pyo3::Python;

pub trait FromRefPy<'r> 
where Self:'r,
{
    type PyType<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self;
}
impl<'r,T> FromRefPy<'r> for Option<T>
where 
    T: FromRefPy<'r>,
    Self: 'r,
{
    type PyType<'py> = Option<T::PyType<'py>>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r Self::PyType<'py>) -> Self {
        match py_type {
            Some(py_type) => Some(T::from_py(py_type)),
            None => None,
        }
    }
}

pub trait FromMutRefPy<'r> 
where Self:'r,
{
    type PyType<'py>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self;
}
impl<'r,T> FromMutRefPy<'r> for Option<T>
where 
    T: FromMutRefPy<'r>,
    Self: 'r,
{
    type PyType<'py> = Option<T::PyType<'py>>
    where 'py:'r;
    fn from_py<'py>(py_type: &'r mut Self::PyType<'py>) -> Self {
        match py_type {
            Some(py_type) => Some(T::from_py(py_type)),
            None => None,
        }
    }
}

pub trait IntoPy {
    type PyType;
    fn into_py(self, py: Python<'_>) -> Self::PyType;
}
impl<T> IntoPy for Option<T>
where T: IntoPy,
{
    type PyType = Option<T::PyType>;
    fn into_py(self, py: Python<'_>) -> Self::PyType {
        match self {
            Some(s) => Some(s.into_py(py)),
            None => None,
        }
    }
}