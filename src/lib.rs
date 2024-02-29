use pyo3::prelude::*;
// use std::time::{Duration, SystemTime};

#[pyfunction]
fn test_func() -> String {
    "Hello World!".to_string()
}

#[pymodule]
fn tis_chessin_stats(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_func, m)?)?;
    return Ok(());
}
