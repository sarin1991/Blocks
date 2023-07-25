use pyo3::prelude::*;
pub mod api;
pub mod types;
pub mod conversion;
use crate::api::blocks::blocks::register_blocks_module;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn blocks_extension(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    register_blocks_module(py,m)?;
    Ok(())
}