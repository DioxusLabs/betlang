use super::{
    activation::gelu_simd,
    constants::{BINS, EMBED},
};
use fearless_simd::{Level, Simd, SimdBase, dispatch, f32x4};
use std::sync::OnceLock;

/// Detect the best available SIMD level once per process.
pub(crate) fn simd_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

#[inline(always)]
pub(crate) fn as_array_chunks<const N: usize>(slice: &[f32]) -> &[[f32; N]] {
    let (chunks, remainder) = slice.as_chunks::<N>();
    debug_assert!(remainder.is_empty());
    chunks
}

#[inline(always)]
pub(crate) fn as_array_chunks_mut<const N: usize>(slice: &mut [f32]) -> &mut [[f32; N]] {
    let (chunks, remainder) = slice.as_chunks_mut::<N>();
    debug_assert!(remainder.is_empty());
    chunks
}

/// Sum the K=3 hashed embedding rows for one unit-id into `dst` (length EMBED).
#[inline]
pub(crate) fn embed_position(embedding: &[f32], unit: u32, dst: &mut [f32]) {
    let b0 = hash_bin(unit, 0);
    let b1 = hash_bin(unit, 1);
    let b2 = hash_bin(unit, 2);
    let rows = as_array_chunks::<EMBED>(embedding);
    let row0 = &rows[b0];
    let row1 = &rows[b1];
    let row2 = &rows[b2];
    for (((d, &v0), &v1), &v2) in dst.iter_mut().zip(row0).zip(row1).zip(row2) {
        *d = v0 + v1 + v2;
    }
}

/// K=3 prime-mix hash matching `hash_unit_indices` in the Python trainer.
fn hash_bin(unit: u32, head: usize) -> usize {
    const PRIMES: [u32; 4] = [2_654_435_761, 2_246_822_519, 3_266_489_917, 668_265_263];
    let p1 = PRIMES[head % PRIMES.len()];
    let p2 = PRIMES[(head + 1) % PRIMES.len()];
    let mut h = unit.wrapping_mul(p1);
    h ^= h >> 13;
    h = h.wrapping_mul(p2);
    (h as usize) % BINS
}

/// Apply GELU in place over a buffer whose length is a multiple of 4.
pub(crate) fn gelu_in_place(values: &mut [f32]) {
    let level = simd_level();
    dispatch!(level, simd => gelu_in_place_simd(simd, values));
}

#[inline(always)]
fn gelu_in_place_simd<S: Simd>(simd: S, values: &mut [f32]) {
    for chunk in as_array_chunks_mut::<4>(values) {
        let v = f32x4::from_slice(simd, chunk);
        gelu_simd(simd, v).store_slice(chunk);
    }
}

pub(crate) fn dense_forward(input: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
    let in_len = input.len();
    let out_len = out.len();
    debug_assert_eq!(kernel.len(), in_len * out_len);
    out.copy_from_slice(bias);
    if out_len == 0 {
        return;
    }
    for (&x, krow) in input.iter().zip(kernel.chunks_exact(out_len)) {
        for (o, &w) in out.iter_mut().zip(krow) {
            *o = w.mul_add(x, *o);
        }
    }
}
