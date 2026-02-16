use spiral_rs::{arith::*, params::*};

use super::server::ToM512;

/// Portable implementation (no AVX2/AVX-512).
///
/// Keeps the same signature/name so the rest of the codebase doesn’t change.
/// Despite the name, this version is scalar and safe on CPUs without AVX2/AVX-512.
///
/// Layout assumptions (matching the old code):
/// - `a` contains K batches concatenated, each batch length = `a_elems`.
/// - `b_t` is the database in *transposed* layout: for column j, row k is at `b_t[j*b_rows + k]`.
/// - Each `a` element is a u64 holding two 32-bit CRT limbs (low/high).
pub fn fast_batched_dot_product_avx512<const K: usize, T: Copy>(
    params: &Params,
    c: &mut [u64],
    a: &[u64],
    a_elems: usize,
    b_t: &[T], // transposed
    b_rows: usize,
    b_cols: usize,
) where
    *const T: ToM512,
{
    assert_eq!(a_elems, b_rows);
    assert_eq!(c.len(), K * b_cols);
    assert_eq!(a.len(), K * a_elems);
    assert_eq!(b_t.len(), b_cols * b_rows);

    // Split output into K batches (same as old code).
    let c_batches = c.chunks_exact_mut(b_cols);

    // Split input `a` into K batches.
    let a_batches = a.chunks_exact(a_elems);

    // For each output column j, compute dot-products for all K batches.
    // We keep the same “wrap then Barrett reduce” behavior as the AVX-512 version:
    // - accumulate in u64 with wrapping arithmetic
    // - reduce low/high limbs with barrett_coeff_u64
    // - crt_compose_2 and barrett_u64 for final accumulation into c
    for (batch_idx, (c_batch, a_batch)) in c_batches.zip(a_batches).enumerate() {
        debug_assert!(batch_idx < K);

        for j in 0..b_cols {
            let mut sum_lo: u64 = 0;
            let mut sum_hi: u64 = 0;

            let base = j * b_rows;

            for k in 0..a_elems {
                // Read db value (u8/u16/u32) through ToM512 fallback (scalar on non-avx512f).
                let b_val_u64: u64 = unsafe { b_t.as_ptr().add(base + k).to_m512() };

                let a_val: u64 = a_batch[k];
                let a_lo: u64 = a_val & 0xFFFF_FFFF;
                let a_hi: u64 = a_val >> 32;

                // Match old behavior: multiply 32-bit limbs by db word, accumulate with wrapping.
                sum_lo = sum_lo.wrapping_add(a_lo.wrapping_mul(b_val_u64));
                sum_hi = sum_hi.wrapping_add(a_hi.wrapping_mul(b_val_u64));
            }

            // Reduce both limbs, compose CRT, and accumulate into output (same as old kernel).
            let lo = barrett_coeff_u64(params, sum_lo, 0);
            let hi = barrett_coeff_u64(params, sum_hi, 1);
            let res = params.crt_compose_2(lo, hi);

            c_batch[j] = barrett_u64(params, c_batch[j].wrapping_add(res));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use spiral_rs::aligned_memory::AlignedMemory64;
    use spiral_rs::poly::*;

    // Note: this is a lightweight sanity test ensuring the function runs.
    // It does NOT validate cryptographic correctness (that requires reference outputs).
    #[test]
    fn test_fast_batched_dot_product_portable_runs() {
        // If you have a helper for params in your repo, use it.
        // Here we just construct a minimal Params in the same way your project does.
        // Replace this with your real params constructor if needed.
        let params = Params::default();

        const K: usize = 1;
        let a_elems = 1024;
        let b_rows = a_elems;
        let b_cols = 64;

        let a = vec![1u64; K * a_elems];
        let mut c = vec![0u64; K * b_cols];

        let mut b = AlignedMemory64::new(b_rows * b_cols);
        for i in 0..b.len() {
            b[i] = 1;
        }
        let b_u16 =
            unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() * 4) };

        fast_batched_dot_product_avx512::<K, _>(&params, &mut c, &a, a_elems, b_u16, b_rows, b_cols);
    }
}
