use pyo3::prelude::*;
use super::linear_blocks::dyad_block::dyad_block::PyDyadBlock;

pub(crate) fn register_blocks_module(py: Python<'_>, parent_module: &PyModule) -> PyResult<()> {
    let blocks_module = PyModule::new(py, "blocks")?;
    blocks_module.add_class::<PyDyadBlock>()?;
    parent_module.add_submodule(blocks_module)?;
    Ok(())
}