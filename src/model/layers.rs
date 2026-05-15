use super::{
    activation::gelu_simd,
    constants::{BINS, EMBED},
};
use fearless_simd::{Level, Simd, SimdBase, SimdFloat, dispatch, f32x4};
use std::sync::OnceLock;

/// Detect the best available SIMD level once per process.
fn simd_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

#[inline(always)]
fn as_array_chunks<const N: usize>(slice: &[f32]) -> &[[f32; N]] {
    let (chunks, remainder) = slice.as_chunks::<N>();
    debug_assert!(remainder.is_empty());
    chunks
}

#[inline(always)]
fn as_array_chunks_mut<const N: usize>(slice: &mut [f32]) -> &mut [[f32; N]] {
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
    const PRIMES: [u64; 4] = [2_654_435_761, 2_246_822_519, 3_266_489_917, 668_265_263];
    let mask = 0xFFFF_FFFFu64;
    let p1 = PRIMES[head % PRIMES.len()];
    let p2 = PRIMES[(head + 1) % PRIMES.len()];
    let mut h = ((unit as u64).wrapping_mul(p1)) & mask;
    h ^= h >> 13;
    h = (h.wrapping_mul(p2)) & mask;
    (h as usize) % BINS
}

/// Conv1d (SAME pad) for a block of `BLOCK` consecutive output positions
/// starting at `t_base`. Accumulates into `accs` (`BLOCK * out_channels` long).
///
/// `kernel` is `[k][in_c][out_c]` with the inner row contiguous over
/// out_channels.
///
/// BLOCK=4 + all positions in-bounds → SIMD fast path. When out_channels
/// is a multiple of 16, the group-16 kernel keeps 16 `f32x4` accumulators
/// register-resident across the full (k, in_c) inner loop; otherwise a
/// chunk-inner SIMD path with 4 named acc rows is used.
/// Edges and partial blocks use the plain f32 path in this kernel.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv1d_block<S: Simd, const BLOCK: usize>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    t_base: usize,
    accs: &mut [f32],
) {
    debug_assert_eq!(accs.len(), BLOCK * out_channels);
    debug_assert!(out_channels.is_multiple_of(4));
    let pad = (kernel_size - 1) / 2;
    if BLOCK == 4 {
        let all_in_bounds =
            t_base >= pad && t_base + 4 + (kernel_size - 1).saturating_sub(pad) <= seq_len;
        if all_in_bounds {
            conv1d_block4_simd_inner(
                simd,
                input,
                in_channels,
                kernel,
                kernel_size,
                out_channels,
                bias,
                t_base,
                pad,
                accs,
            );
            return;
        }
    }
    // Edge/partial-block path.
    for acc in accs.chunks_exact_mut(out_channels) {
        acc.copy_from_slice(bias);
    }
    for k in 0..kernel_size {
        let src_t_at_s0 = t_base as isize + k as isize - pad as isize;
        let s_lo = if src_t_at_s0 < 0 {
            ((-src_t_at_s0) as usize).min(BLOCK)
        } else {
            0
        };
        let s_hi_signed = seq_len as isize - src_t_at_s0;
        let s_hi = if s_hi_signed > 0 {
            (s_hi_signed as usize).min(BLOCK)
        } else {
            0
        };
        if s_lo >= s_hi {
            continue;
        }
        let kbase = k * in_channels * out_channels;
        let krows = kernel[kbase..kbase + in_channels * out_channels].chunks_exact(out_channels);
        for (in_c, krow) in krows.enumerate() {
            for (s, acc) in accs
                .chunks_exact_mut(out_channels)
                .enumerate()
                .skip(s_lo)
                .take(s_hi - s_lo)
            {
                let src_t = (src_t_at_s0 + s as isize) as usize;
                let x = input[src_t * in_channels + in_c];
                for (a, &w) in acc.iter_mut().zip(krow) {
                    *a = w.mul_add(x, *a);
                }
            }
        }
    }
}

/// BLOCK=4 SIMD hot path. Out-channels are processed in 16-wide groups so
/// that each group's 4 accumulator rows × 4 chunks (= 16 NEON registers)
/// live in regs across the entire (k, in_c) inner loop. Combined with the
/// chunk-inner inner pattern, kernel and input reads stay cache-sequential
/// while the per-iter accs load/store traffic is eliminated.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv1d_block4_simd_inner<S: Simd>(
    simd: S,
    input: &[f32],
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    t_base: usize,
    pad: usize,
    accs: &mut [f32],
) {
    const GROUP: usize = 16;
    if out_channels.is_multiple_of(GROUP) {
        conv1d_block4_group16::<S>(
            simd,
            input,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            t_base,
            pad,
            accs,
        );
        return;
    }
    // Chunk-inner SIMD path for out_channels not a multiple of 16.
    for acc in accs.chunks_exact_mut(out_channels) {
        acc.copy_from_slice(bias);
    }
    let (a01, a23) = accs.split_at_mut(2 * out_channels);
    let (a0, a1) = a01.split_at_mut(out_channels);
    let (a2, a3) = a23.split_at_mut(out_channels);
    for k in 0..kernel_size {
        let base_t = t_base + k - pad;
        let row0_off = base_t * in_channels;
        let row1_off = (base_t + 1) * in_channels;
        let row2_off = (base_t + 2) * in_channels;
        let row3_off = (base_t + 3) * in_channels;
        let kbase = k * in_channels * out_channels;
        let krows = kernel[kbase..kbase + in_channels * out_channels].chunks_exact(out_channels);
        for (in_c, krow) in krows.enumerate() {
            let xv0 = f32x4::splat(simd, input[row0_off + in_c]);
            let xv1 = f32x4::splat(simd, input[row1_off + in_c]);
            let xv2 = f32x4::splat(simd, input[row2_off + in_c]);
            let xv3 = f32x4::splat(simd, input[row3_off + in_c]);
            for ((((kr_c, a0_c), a1_c), a2_c), a3_c) in as_array_chunks::<4>(krow)
                .iter()
                .zip(as_array_chunks_mut::<4>(a0).iter_mut())
                .zip(as_array_chunks_mut::<4>(a1).iter_mut())
                .zip(as_array_chunks_mut::<4>(a2).iter_mut())
                .zip(as_array_chunks_mut::<4>(a3).iter_mut())
            {
                let kr = f32x4::from_slice(simd, kr_c);
                let av0 = f32x4::from_slice(simd, a0_c);
                let av1 = f32x4::from_slice(simd, a1_c);
                let av2 = f32x4::from_slice(simd, a2_c);
                let av3 = f32x4::from_slice(simd, a3_c);
                kr.mul_add(xv0, av0).store_slice(a0_c);
                kr.mul_add(xv1, av1).store_slice(a1_c);
                kr.mul_add(xv2, av2).store_slice(a2_c);
                kr.mul_add(xv3, av3).store_slice(a3_c);
            }
        }
    }
}

/// Group-16 SIMD kernel. For each 16-wide out-channel group, holds
/// `4 acc rows × 4 chunks = 16 f32x4` accumulators in NEON registers
/// across the entire (k, in_c) inner loop.
///
/// Per (k, in_c) iter: 4 kernel f32x4 loads + 4 broadcast x loads + 16 FMAs.
/// All 16 accumulators stay register-resident — no per-iter accs load/store.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv1d_block4_group16<S: Simd>(
    simd: S,
    input: &[f32],
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    t_base: usize,
    pad: usize,
    accs: &mut [f32],
) {
    let (acc01, acc23) = accs.split_at_mut(2 * out_channels);
    let (acc0, acc1) = acc01.split_at_mut(out_channels);
    let (acc2, acc3) = acc23.split_at_mut(out_channels);
    for (g, ((((bias_group, acc0_group), acc1_group), acc2_group), acc3_group)) in
        as_array_chunks::<16>(bias)
            .iter()
            .zip(as_array_chunks_mut::<16>(acc0).iter_mut())
            .zip(as_array_chunks_mut::<16>(acc1).iter_mut())
            .zip(as_array_chunks_mut::<16>(acc2).iter_mut())
            .zip(as_array_chunks_mut::<16>(acc3).iter_mut())
            .enumerate()
    {
        let g_off = g * 16;
        let bias_chunks = as_array_chunks::<4>(bias_group);
        let b0 = f32x4::from_slice(simd, &bias_chunks[0]);
        let b1 = f32x4::from_slice(simd, &bias_chunks[1]);
        let b2 = f32x4::from_slice(simd, &bias_chunks[2]);
        let b3 = f32x4::from_slice(simd, &bias_chunks[3]);
        let (mut a0_0, mut a0_1, mut a0_2, mut a0_3) = (b0, b1, b2, b3);
        let (mut a1_0, mut a1_1, mut a1_2, mut a1_3) = (b0, b1, b2, b3);
        let (mut a2_0, mut a2_1, mut a2_2, mut a2_3) = (b0, b1, b2, b3);
        let (mut a3_0, mut a3_1, mut a3_2, mut a3_3) = (b0, b1, b2, b3);

        for k in 0..kernel_size {
            let base_t = t_base + k - pad;
            let row0_off = base_t * in_channels;
            let row1_off = (base_t + 1) * in_channels;
            let row2_off = (base_t + 2) * in_channels;
            let row3_off = (base_t + 3) * in_channels;
            let kbase = k * in_channels * out_channels;
            let krows =
                kernel[kbase..kbase + in_channels * out_channels].chunks_exact(out_channels);
            for (in_c, krow) in krows.enumerate() {
                let kernel_chunks = as_array_chunks::<4>(&krow[g_off..g_off + 16]);
                let kr0 = f32x4::from_slice(simd, &kernel_chunks[0]);
                let kr1 = f32x4::from_slice(simd, &kernel_chunks[1]);
                let kr2 = f32x4::from_slice(simd, &kernel_chunks[2]);
                let kr3 = f32x4::from_slice(simd, &kernel_chunks[3]);
                let xv0 = f32x4::splat(simd, input[row0_off + in_c]);
                let xv1 = f32x4::splat(simd, input[row1_off + in_c]);
                let xv2 = f32x4::splat(simd, input[row2_off + in_c]);
                let xv3 = f32x4::splat(simd, input[row3_off + in_c]);
                a0_0 = kr0.mul_add(xv0, a0_0);
                a0_1 = kr1.mul_add(xv0, a0_1);
                a0_2 = kr2.mul_add(xv0, a0_2);
                a0_3 = kr3.mul_add(xv0, a0_3);
                a1_0 = kr0.mul_add(xv1, a1_0);
                a1_1 = kr1.mul_add(xv1, a1_1);
                a1_2 = kr2.mul_add(xv1, a1_2);
                a1_3 = kr3.mul_add(xv1, a1_3);
                a2_0 = kr0.mul_add(xv2, a2_0);
                a2_1 = kr1.mul_add(xv2, a2_1);
                a2_2 = kr2.mul_add(xv2, a2_2);
                a2_3 = kr3.mul_add(xv2, a2_3);
                a3_0 = kr0.mul_add(xv3, a3_0);
                a3_1 = kr1.mul_add(xv3, a3_1);
                a3_2 = kr2.mul_add(xv3, a3_2);
                a3_3 = kr3.mul_add(xv3, a3_3);
            }
        }

        // Store the 4×4 accumulator tile back into accs[s][g_off..g_off+16].
        let acc0_chunks = as_array_chunks_mut::<4>(acc0_group);
        a0_0.store_slice(&mut acc0_chunks[0]);
        a0_1.store_slice(&mut acc0_chunks[1]);
        a0_2.store_slice(&mut acc0_chunks[2]);
        a0_3.store_slice(&mut acc0_chunks[3]);
        let acc1_chunks = as_array_chunks_mut::<4>(acc1_group);
        a1_0.store_slice(&mut acc1_chunks[0]);
        a1_1.store_slice(&mut acc1_chunks[1]);
        a1_2.store_slice(&mut acc1_chunks[2]);
        a1_3.store_slice(&mut acc1_chunks[3]);
        let acc2_chunks = as_array_chunks_mut::<4>(acc2_group);
        a2_0.store_slice(&mut acc2_chunks[0]);
        a2_1.store_slice(&mut acc2_chunks[1]);
        a2_2.store_slice(&mut acc2_chunks[2]);
        a2_3.store_slice(&mut acc2_chunks[3]);
        let acc3_chunks = as_array_chunks_mut::<4>(acc3_group);
        a3_0.store_slice(&mut acc3_chunks[0]);
        a3_1.store_slice(&mut acc3_chunks[1]);
        a3_2.store_slice(&mut acc3_chunks[2]);
        a3_3.store_slice(&mut acc3_chunks[3]);
    }
}

/// Fused conv1d (SAME pad) + GELU + MaxPool(pool).
/// Always accumulates BLOCK=4 consecutive conv positions per outer iteration
/// (max NEON-friendly krow reuse), then applies POOL-wide maxpool over them.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_maxpool(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let pooled_len = seq_len / pool;
    assert_eq!(out.len(), pooled_len * out_channels);
    let level = simd_level();
    dispatch!(level, simd => conv_gelu_maxpool_simd(
        simd, input, seq_len, in_channels, kernel, kernel_size,
        out_channels, bias, pool, pooled_len, out, scratch,
    ));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_maxpool_simd<S: Simd>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    pooled_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    match pool {
        4 => conv_gelu_maxpool_run::<S, 4, 4>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            pooled_len,
            out,
            scratch,
        ),
        2 => conv_gelu_maxpool_run::<S, 4, 2>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            pooled_len,
            out,
            scratch,
        ),
        _ => conv_gelu_maxpool_run::<S, 1, 1>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            pooled_len,
            out,
            scratch,
        ),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_maxpool_run<S: Simd, const BLOCK: usize, const POOL: usize>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pooled_len: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    debug_assert_eq!(BLOCK % POOL, 0);
    assert!(scratch.len() >= BLOCK * out_channels);
    let outs_per_block: usize = BLOCK / POOL;
    let block_count = pooled_len / outs_per_block;
    {
        let accs = &mut scratch[..BLOCK * out_channels];
        for tb in 0..block_count {
            let t_base = tb * BLOCK;
            conv1d_block::<S, BLOCK>(
                simd,
                input,
                seq_len,
                in_channels,
                kernel,
                kernel_size,
                out_channels,
                bias,
                t_base,
                accs,
            );
            let out_start = tb * outs_per_block * out_channels;
            let out_block = &mut out[out_start..out_start + outs_per_block * out_channels];
            for (op, dst) in out_block.chunks_exact_mut(out_channels).enumerate() {
                let s_first = op * POOL;
                let pool_start = s_first * out_channels;
                let pool_accs = &accs[pool_start..pool_start + POOL * out_channels];
                let first = &pool_accs[..out_channels];
                for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
                    .iter_mut()
                    .zip(as_array_chunks::<4>(first))
                {
                    let v = f32x4::from_slice(simd, a_c);
                    gelu_simd(simd, v).store_slice(d_c);
                }
                for acc in pool_accs[out_channels..].chunks_exact(out_channels) {
                    for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
                        .iter_mut()
                        .zip(as_array_chunks::<4>(acc))
                    {
                        let v = f32x4::from_slice(simd, a_c);
                        let g = gelu_simd(simd, v);
                        let dv = f32x4::from_slice(simd, d_c);
                        g.max(dv).store_slice(d_c);
                    }
                }
            }
        }
    }
    let processed = block_count * outs_per_block;
    if processed < pooled_len {
        let tail_accs = &mut scratch[..POOL * out_channels];
        for (tail_offset, dst) in out[processed * out_channels..]
            .chunks_exact_mut(out_channels)
            .enumerate()
        {
            let tp = processed + tail_offset;
            let t_base = tp * POOL;
            conv1d_block::<S, POOL>(
                simd,
                input,
                seq_len,
                in_channels,
                kernel,
                kernel_size,
                out_channels,
                bias,
                t_base,
                &mut *tail_accs,
            );
            let first = &tail_accs[..out_channels];
            for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
                .iter_mut()
                .zip(as_array_chunks::<4>(first))
            {
                let v = f32x4::from_slice(simd, a_c);
                gelu_simd(simd, v).store_slice(d_c);
            }
            for acc in tail_accs[out_channels..].chunks_exact(out_channels) {
                for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
                    .iter_mut()
                    .zip(as_array_chunks::<4>(acc))
                {
                    let v = f32x4::from_slice(simd, a_c);
                    let g = gelu_simd(simd, v);
                    let dv = f32x4::from_slice(simd, d_c);
                    g.max(dv).store_slice(d_c);
                }
            }
        }
    }
}

/// Fused conv1d (SAME pad) + GELU + (GlobalMax || GlobalAvg) pool.
/// Writes max into `out_max` and average into `out_avg` (each `out_channels` long).
/// Processes T_BLOCK=4 positions per outer iteration to reuse kernel rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_global_pool(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    out_max: &mut [f32],
    out_avg: &mut [f32],
    scratch: &mut [f32],
) {
    out_max.fill(f32::NEG_INFINITY);
    out_avg.fill(0.0);
    if seq_len == 0 {
        return;
    }
    let level = simd_level();
    dispatch!(level, simd => conv_gelu_global_pool_simd(
        simd, input, seq_len, in_channels, kernel, kernel_size,
        out_channels, bias, out_max, out_avg, scratch,
    ));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_global_pool_simd<S: Simd>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    out_max: &mut [f32],
    out_avg: &mut [f32],
    scratch: &mut [f32],
) {
    const T_BLOCK: usize = 4;
    assert!(scratch.len() >= T_BLOCK * out_channels);
    let block_count = seq_len / T_BLOCK;
    {
        let accs = &mut scratch[..T_BLOCK * out_channels];
        for tb in 0..block_count {
            let t_base = tb * T_BLOCK;
            conv1d_block::<S, T_BLOCK>(
                simd,
                input,
                seq_len,
                in_channels,
                kernel,
                kernel_size,
                out_channels,
                bias,
                t_base,
                accs,
            );
            for acc in accs.chunks_exact(out_channels) {
                for ((mx_c, av_c), a_c) in as_array_chunks_mut::<4>(out_max)
                    .iter_mut()
                    .zip(as_array_chunks_mut::<4>(out_avg).iter_mut())
                    .zip(as_array_chunks::<4>(acc))
                {
                    let v = f32x4::from_slice(simd, a_c);
                    let g = gelu_simd(simd, v);
                    let mx_v = f32x4::from_slice(simd, mx_c);
                    let av_v = f32x4::from_slice(simd, av_c);
                    g.max(mx_v).store_slice(mx_c);
                    (av_v + g).store_slice(av_c);
                }
            }
        }
    }
    let tail_accs = &mut scratch[..out_channels];
    for t in (block_count * T_BLOCK)..seq_len {
        conv1d_block::<S, 1>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            t,
            &mut *tail_accs,
        );
        for ((mx_c, av_c), a_c) in as_array_chunks_mut::<4>(out_max)
            .iter_mut()
            .zip(as_array_chunks_mut::<4>(out_avg).iter_mut())
            .zip(as_array_chunks::<4>(tail_accs))
        {
            let v = f32x4::from_slice(simd, a_c);
            let g = gelu_simd(simd, v);
            let mx_v = f32x4::from_slice(simd, mx_c);
            let av_v = f32x4::from_slice(simd, av_c);
            g.max(mx_v).store_slice(mx_c);
            (av_v + g).store_slice(av_c);
        }
    }
    let inv = 1.0 / seq_len as f32;
    for av in out_avg.iter_mut() {
        *av *= inv;
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
