use pyo3::prelude::*;

#[pyfunction]
fn hello() -> &'static str {
    "hello from ypir_rs (Rust extension)"
}

#[pymodule]
fn ypir_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    Ok(())
}
