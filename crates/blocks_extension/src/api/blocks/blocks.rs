use pyo3::prelude::*;

pub(crate) fn register_blocks_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let blocks_module = PyModule::new(py, "blocks")?;
    
    parent_module.add_submodule(blocks_module)?;
    Ok(())
}