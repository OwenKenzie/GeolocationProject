use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::client::Client as SpiralClient;
use spiral_rs::params::Params as SpiralParams;

use ypir::client::{pack_query, YClient};
use ypir::params::{params_for_scenario, params_for_scenario_simplepir};
use ypir::server::YServer;

// ---------- helpers: bytes <-> u64 words (little-endian) ----------

unsafe fn shrink_client_lifetime<'a>(
    c: &'a mut SpiralClient<'static>,
) -> &'a mut SpiralClient<'a> {
    std::mem::transmute::<&'a mut SpiralClient<'static>, &'a mut SpiralClient<'a>>(c)
}

unsafe fn shrink_params_lifetime<'a>(
    p: &'static SpiralParams,
) -> &'a SpiralParams {
    std::mem::transmute::<&'static SpiralParams, &'a SpiralParams>(p)
}

fn bytes_to_u64_le(b: &[u8]) -> PyResult<Vec<u64>> {
    if b.len() % 8 != 0 {
        return Err(PyValueError::new_err("byte length must be multiple of 8"));
    }
    let mut out = Vec::with_capacity(b.len() / 8);
    for chunk in b.chunks_exact(8) {
        out.push(u64::from_le_bytes(chunk.try_into().unwrap()));
    }
    Ok(out)
}

fn u64_to_bytes_le(words: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 8);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

fn aligned64_to_bytes_le(mem: &AlignedMemory64) -> Vec<u8> {
    u64_to_bytes_le(mem.as_slice())
}

// ---------- Python-exposed wrapper types ----------
// IMPORTANT: mark unsendable so PyO3 does NOT require Send/Sync.

#[pyclass(unsendable)]
struct PyYpirParams {
    params: &'static SpiralParams,
    is_simplepir: bool,
    item_size_bits: usize,
}

#[pymethods]
impl PyYpirParams {
    fn __repr__(&self) -> String {
        format!(
            "PyYpirParams(simplepir={}, item_size_bits={})",
            self.is_simplepir, self.item_size_bits
        )
    }
}

#[pyclass(unsendable)]
struct PyYpirClient {
    params: &'static SpiralParams,
    inner: SpiralClient<'static>,
}

#[pyclass(unsendable)]
struct PyYpirServer {
    params: &'static SpiralParams,
    db_bytes: Vec<u8>,
    inner: YServer<'static, u8>,
}

// ---------- constructors / API ----------

#[pyfunction]
fn params_db_dim_1(params: &PyYpirParams) -> usize {
    params.params.db_dim_1
}

#[pyfunction]
fn required_db_bytes(params: &PyYpirParams) -> usize {
    let p = params.params;
    let db_rows = 1 << (p.db_dim_1 + p.poly_len_log2);
    let db_cols = if params.is_simplepir {
        p.instances * p.poly_len
    } else {
        1 << (p.db_dim_2 + p.poly_len_log2)
    };
    db_rows * db_cols
}


/// Build spiral params from scenario helpers in ypir::params
#[pyfunction]
fn params_for(num_items: usize, item_size_bytes: usize, is_simplepir: bool) -> PyResult<PyYpirParams> {
    if num_items == 0 {
        return Err(PyValueError::new_err("num_items must be > 0"));
    }
    if item_size_bytes == 0 {
        return Err(PyValueError::new_err("item_size_bytes must be > 0"));
    }

    let item_size_bits = item_size_bytes * 8;

    let p: SpiralParams = if is_simplepir {
        params_for_scenario_simplepir(num_items, item_size_bits)
    } else {
        params_for_scenario(num_items, item_size_bits)
    };

    let leaked: &'static SpiralParams = Box::leak(Box::new(p));

    Ok(PyYpirParams {
        params: leaked,
        is_simplepir,
        item_size_bits,
    })
}

#[pyfunction]
fn client_new(params: &PyYpirParams) -> PyResult<PyYpirClient> {
    let mut c = SpiralClient::init(params.params);
    c.generate_secret_keys();
    Ok(PyYpirClient {
        params: params.params,
        inner: c,
    })
}

/// Create server using u8 DB elements (required by ToM512 bounds in server.rs).
#[pyfunction]
fn server_new(
    params: &PyYpirParams,
    db_bytes: Vec<u8>,
    inp_transposed: bool,
    pad_rows: bool,
) -> PyResult<PyYpirServer> {
    let p = params.params;

    let db_rows = 1 << (p.db_dim_1 + p.poly_len_log2);
    let db_cols = if params.is_simplepir {
        p.instances * p.poly_len
    } else {
        1 << (p.db_dim_2 + p.poly_len_log2)
    };
    let needed = db_rows * db_cols;

    if db_bytes.len() < needed {
        return Err(PyValueError::new_err(format!(
            "db_bytes too small: got {} bytes, need at least {} (db_rows={} db_cols={})",
            db_bytes.len(),
            needed,
            db_rows,
            db_cols
        )));
    }

    let db_for_server = db_bytes[..needed].to_vec();
    let iter = db_for_server.iter().copied();

    let s = YServer::<u8>::new(p, iter, params.is_simplepir, inp_transposed, pad_rows);

    Ok(PyYpirServer {
        params: p,
        db_bytes: db_for_server,
        inner: s,
    })
}

/// Generate a query. If `pack=true`, return packed query bytes suitable for server.answer().
#[pyfunction]
fn query(
    client: &mut PyYpirClient,
    public_seed_idx: u8,
    dim_log2: usize,
    packing: bool,
    index_row: usize,
    pack: bool,
) -> PyResult<Vec<u8>> {
    let q_words: Vec<u64> = unsafe {
        let inner = shrink_client_lifetime(&mut client.inner);
        let params = shrink_params_lifetime(client.params);
        let y = YClient::new(inner, params);
        y.generate_query(public_seed_idx, dim_log2, packing, index_row)
    };

    if pack {
        let packed = pack_query(client.params, &q_words);
        Ok(aligned64_to_bytes_le(&packed))
    } else {
        Ok(u64_to_bytes_le(&q_words))
    }
}


#[pyfunction]
fn answer(server: &PyYpirServer, packed_query_bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    let packed_words = bytes_to_u64_le(&packed_query_bytes)?;
    let resp: AlignedMemory64 = server.inner.answer_query(&packed_words);
    Ok(aligned64_to_bytes_le(&resp))
}

#[pyfunction]
fn extract(client: &mut PyYpirClient, response_bytes: Vec<u8>) -> PyResult<Vec<u8>> {
    let resp_words = bytes_to_u64_le(&response_bytes)?;

    let out: Vec<u64> = unsafe {
        let inner = shrink_client_lifetime(&mut client.inner);
        let params = shrink_params_lifetime(client.params);
        let y = YClient::new(inner, params);
        y.decode_response(&resp_words)
    };

    Ok(u64_to_bytes_le(&out))
}

#[pymodule]
fn ypir_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(params_for, m)?)?;
    m.add_function(wrap_pyfunction!(client_new, m)?)?;
    m.add_function(wrap_pyfunction!(server_new, m)?)?;
    m.add_function(wrap_pyfunction!(query, m)?)?;
    m.add_function(wrap_pyfunction!(answer, m)?)?;
    m.add_function(wrap_pyfunction!(extract, m)?)?;

    m.add_function(wrap_pyfunction!(params_db_dim_1, m)?)?;
    m.add_function(wrap_pyfunction!(required_db_bytes, m)?)?;

    m.add_class::<PyYpirParams>()?;
    m.add_class::<PyYpirClient>()?;
    m.add_class::<PyYpirServer>()?;
    Ok(())
}
