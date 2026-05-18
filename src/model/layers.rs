use super::{
    activation::gelu_simd,
    constants::{BINS, EMBED},
};
use fearless_simd::{Level, Simd, SimdBase, SimdFloat, dispatch, f32x4};
use std::{ops::Range, sync::OnceLock};

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
    const PRIMES: [u32; 4] = [2_654_435_761, 2_246_822_519, 3_266_489_917, 668_265_263];
    let p1 = PRIMES[head % PRIMES.len()];
    let p2 = PRIMES[(head + 1) % PRIMES.len()];
    let mut h = unit.wrapping_mul(p1);
    h ^= h >> 13;
    h = h.wrapping_mul(p2);
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

#[derive(Clone, Debug)]
struct RepeatedRows {
    range: Range<usize>,
    extends_right_padding: bool,
}

impl RepeatedRows {
    fn new(start: usize, end: usize, extends_right_padding: bool) -> Self {
        Self {
            range: start..end,
            extends_right_padding,
        }
    }

    fn empty(at: usize) -> Self {
        Self::new(at, at, false)
    }

    fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Tensor<'a> {
    data: &'a [f32],
    rows: usize,
    channels: usize,
    repeated_tail: RepeatedRows,
}

impl<'a> Tensor<'a> {
    pub(crate) fn with_repeated_tail(
        data: &'a [f32],
        rows: usize,
        channels: usize,
        start: usize,
    ) -> Self {
        let start = start.min(rows);
        let materialized_rows = if start < rows { start + 1 } else { rows };
        debug_assert!(data.len() >= materialized_rows * channels);
        debug_assert!(data.len() <= rows * channels);
        Self {
            data,
            rows,
            channels,
            repeated_tail: RepeatedRows::new(start, rows, true),
        }
    }

    #[cfg(test)]
    pub(crate) fn copy_to_dense(&self, out: &mut [f32]) {
        debug_assert_eq!(out.len(), self.rows * self.channels);
        for row_index in 0..self.rows {
            out[row_index * self.channels..(row_index + 1) * self.channels]
                .copy_from_slice(self.row(row_index));
        }
    }

    fn row(&self, index: usize) -> &'a [f32] {
        debug_assert!(index < self.rows);
        let region = &self.repeated_tail;
        let index = if region.range.contains(&index) {
            region.range.start
        } else {
            index
        };
        row(self.data, self.channels, index)
    }
}

struct ConvSpec<'a> {
    seq_len: usize,
    in_channels: usize,
    kernel: &'a [f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &'a [f32],
    pool: usize,
}

impl ConvSpec<'_> {
    fn row_count(&self) -> usize {
        self.seq_len / self.pool
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_maxpool_tensor<'a>(
    input: Tensor<'_>,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    out: &'a mut [f32],
    scratch: &mut [f32],
) -> Tensor<'a> {
    let spec = ConvSpec {
        seq_len: input.rows,
        in_channels: input.channels,
        kernel,
        kernel_size,
        out_channels,
        bias,
        pool,
    };
    let out_region = pooled_repeated_region(&input, &spec);
    let row_count = spec.row_count();
    debug_assert_eq!(out.len(), row_count * out_channels);

    if out_region.is_empty() {
        conv_gelu_range(&input, &spec, 0..row_count, out, scratch);
        return Tensor::with_repeated_tail(out, row_count, out_channels, row_count);
    }

    let out_start = out_region.range.start;
    let out_end = out_region.range.end;
    conv_gelu_range(&input, &spec, 0..out_start, out, scratch);
    let value_row = row_range_mut(out, out_channels, out_start..out_start + 1);
    const_output_value(&input, &spec, value_row);
    conv_gelu_range(&input, &spec, out_end..row_count, out, scratch);

    Tensor {
        data: out,
        rows: row_count,
        channels: out_channels,
        repeated_tail: out_region,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_gelu_global_pool_tensor(
    input: Tensor<'_>,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    out_max: &mut [f32],
    out_avg: &mut [f32],
    tmp: &mut [f32],
    scratch: &mut [f32],
) {
    let rows_len = input.rows * out_channels;
    let conv = conv_gelu_maxpool_tensor(
        input,
        kernel,
        kernel_size,
        out_channels,
        bias,
        1,
        &mut tmp[..rows_len],
        scratch,
    );
    global_max_avg_pool(conv, out_max, out_avg);
}

pub(crate) fn global_max_avg_pool(input: Tensor<'_>, out_max: &mut [f32], out_avg: &mut [f32]) {
    debug_assert_eq!(out_max.len(), input.channels);
    debug_assert_eq!(out_avg.len(), input.channels);
    out_max.fill(f32::NEG_INFINITY);
    out_avg.fill(0.0);
    if input.rows == 0 {
        return;
    }

    let region = &input.repeated_tail;
    if region.is_empty() {
        accumulate_pool_rows(input.data, input.channels, out_max, out_avg);
    } else {
        accumulate_pool_rows(
            row_range(input.data, input.channels, 0..region.range.start),
            input.channels,
            out_max,
            out_avg,
        );
        accumulate_const_rows(
            input.row(region.range.start),
            region.range.len(),
            out_max,
            out_avg,
        );
        accumulate_pool_rows(
            row_range(input.data, input.channels, region.range.end..input.rows),
            input.channels,
            out_max,
            out_avg,
        );
    }

    let inv = 1.0 / input.rows as f32;
    for avg in out_avg {
        *avg *= inv;
    }
}

fn pooled_repeated_region(input: &Tensor<'_>, spec: &ConvSpec<'_>) -> RepeatedRows {
    let input_region = &input.repeated_tail;
    if input_region.is_empty() {
        return RepeatedRows::empty(spec.row_count());
    }

    let pad = (spec.kernel_size - 1) / 2;
    let right = spec.kernel_size - 1 - pad;
    let conv_start = input_region
        .range
        .start
        .saturating_add(pad)
        .min(spec.seq_len);
    let conv_end = if input_region.extends_right_padding {
        spec.seq_len
    } else {
        input_region
            .range
            .end
            .saturating_sub(right)
            .min(spec.seq_len)
    };

    if conv_start >= conv_end {
        return RepeatedRows::empty(spec.row_count());
    }

    let start = conv_start.div_ceil(spec.pool);
    let end = conv_end / spec.pool;
    if start >= end {
        RepeatedRows::empty(spec.row_count())
    } else {
        RepeatedRows::new(start, end, false)
    }
}

fn conv_gelu_range(
    input: &Tensor<'_>,
    spec: &ConvSpec<'_>,
    rows: Range<usize>,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    if rows.is_empty() {
        return;
    }
    let row_start = rows.start;
    let range = row_range_mut(out, spec.out_channels, rows.clone());
    if !range_touches_repeated(input, spec, rows) {
        conv_gelu_maxpool_range(
            input.data,
            spec.seq_len,
            spec.in_channels,
            spec.kernel,
            spec.kernel_size,
            spec.out_channels,
            spec.bias,
            spec.pool,
            row_start,
            range,
            scratch,
        );
    } else {
        conv_gelu_maxpool_tensor_range(input, spec, row_start, range, scratch);
    }
}

fn range_touches_repeated(input: &Tensor<'_>, spec: &ConvSpec<'_>, rows: Range<usize>) -> bool {
    let region = &input.repeated_tail;
    if rows.is_empty() {
        return false;
    }
    if region.is_empty() {
        return false;
    }
    let pad = (spec.kernel_size - 1) / 2;
    let right = spec.kernel_size - 1 - pad;
    let first_output = rows.start * spec.pool;
    let last_output = (rows.end - 1) * spec.pool + (spec.pool - 1);
    let src_start = first_output.saturating_sub(pad);
    let src_end = (last_output + right + 1).min(input.rows);
    src_start < region.range.end && region.range.start < src_end
}

#[allow(clippy::too_many_arguments)]
fn conv_gelu_maxpool_range(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    row_start: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    let row_count = out.len() / out_channels;
    if row_count == 0 {
        return;
    }
    let level = simd_level();
    dispatch!(level, simd => conv_gelu_maxpool_range_simd(
        simd, input, seq_len, in_channels, kernel, kernel_size,
        out_channels, bias, pool, row_start, out, scratch,
    ));
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv_gelu_maxpool_range_simd<S: Simd>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    row_start: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    match pool {
        4 => conv_gelu_maxpool_range_run::<S, 4>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            row_start,
            out,
            scratch,
        ),
        2 => conv_gelu_maxpool_range_run::<S, 2>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            row_start,
            out,
            scratch,
        ),
        _ => conv_gelu_maxpool_range_run::<S, 1>(
            simd,
            input,
            seq_len,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            row_start,
            out,
            scratch,
        ),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn conv_gelu_maxpool_range_run<S: Simd, const POOL: usize>(
    simd: S,
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    row_start: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    debug_assert!(scratch.len() >= POOL * out_channels);
    let accs = &mut scratch[..POOL * out_channels];
    for (row_offset, dst) in out.chunks_exact_mut(out_channels).enumerate() {
        let t_base = (row_start + row_offset) * POOL;
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
            &mut *accs,
        );
        pool_gelu_rows(simd, accs, out_channels, dst);
    }
}

fn conv_gelu_maxpool_tensor_range(
    input: &Tensor<'_>,
    spec: &ConvSpec<'_>,
    row_start: usize,
    out: &mut [f32],
    scratch: &mut [f32],
) {
    debug_assert!(scratch.len() >= spec.pool * spec.out_channels);
    let accs = &mut scratch[..spec.pool * spec.out_channels];
    for (row_offset, dst) in out.chunks_exact_mut(spec.out_channels).enumerate() {
        let t_base = (row_start + row_offset) * spec.pool;
        conv1d_tensor_block(input, spec, t_base, spec.pool, accs);
        for value in accs.iter_mut() {
            *value = super::activation::gelu(*value);
        }
        pool_rows_scalar(accs, spec.out_channels, dst);
    }
}

fn conv1d_tensor_block(
    input: &Tensor<'_>,
    spec: &ConvSpec<'_>,
    t_base: usize,
    block: usize,
    accs: &mut [f32],
) {
    debug_assert_eq!(accs.len(), block * spec.out_channels);
    for acc in accs.chunks_exact_mut(spec.out_channels) {
        acc.copy_from_slice(spec.bias);
    }
    let pad = (spec.kernel_size - 1) / 2;
    for k in 0..spec.kernel_size {
        let src_t_at_s0 = t_base as isize + k as isize - pad as isize;
        let kbase = k * spec.in_channels * spec.out_channels;
        let krows = spec.kernel[kbase..kbase + spec.in_channels * spec.out_channels]
            .chunks_exact(spec.out_channels);
        for (in_c, krow) in krows.enumerate() {
            for (s, acc) in accs.chunks_exact_mut(spec.out_channels).enumerate() {
                let src_t = src_t_at_s0 + s as isize;
                if src_t < 0 || src_t >= input.rows as isize {
                    continue;
                }
                let x = input.row(src_t as usize)[in_c];
                for (a, &w) in acc.iter_mut().zip(krow) {
                    *a = w.mul_add(x, *a);
                }
            }
        }
    }
}

fn pool_rows_scalar(rows: &[f32], channels: usize, dst: &mut [f32]) {
    dst.copy_from_slice(&rows[..channels]);
    for row in rows[channels..].chunks_exact(channels) {
        for (dst, &value) in dst.iter_mut().zip(row) {
            *dst = dst.max(value);
        }
    }
}

fn const_output_value(input: &Tensor<'_>, spec: &ConvSpec<'_>, out: &mut [f32]) {
    let input_region = &input.repeated_tail;
    debug_assert!(!input_region.is_empty());
    let input_row = input.row(input_region.range.start);
    out.copy_from_slice(spec.bias);
    for krow in spec
        .kernel
        .chunks_exact(spec.in_channels * spec.out_channels)
    {
        for (&x, weights) in input_row.iter().zip(krow.chunks_exact(spec.out_channels)) {
            for (o, &w) in out.iter_mut().zip(weights) {
                *o = w.mul_add(x, *o);
            }
        }
    }
    for v in out {
        *v = super::activation::gelu(*v);
    }
}

fn row(data: &[f32], channels: usize, index: usize) -> &[f32] {
    &data[index * channels..(index + 1) * channels]
}

fn row_range(data: &[f32], channels: usize, rows: Range<usize>) -> &[f32] {
    &data[rows.start * channels..rows.end * channels]
}

fn row_range_mut(data: &mut [f32], channels: usize, rows: Range<usize>) -> &mut [f32] {
    &mut data[rows.start * channels..rows.end * channels]
}

fn accumulate_pool_rows(rows: &[f32], channels: usize, out_max: &mut [f32], out_avg: &mut [f32]) {
    for row in rows.chunks_exact(channels) {
        for ((mx, avg), &v) in out_max.iter_mut().zip(out_avg.iter_mut()).zip(row) {
            *mx = mx.max(v);
            *avg += v;
        }
    }
}

fn accumulate_const_rows(value: &[f32], count: usize, out_max: &mut [f32], out_avg: &mut [f32]) {
    for ((mx, avg), &v) in out_max.iter_mut().zip(out_avg.iter_mut()).zip(value) {
        *mx = mx.max(v);
        for _ in 0..count {
            *avg += v;
        }
    }
}

#[inline(always)]
fn pool_gelu_rows<S: Simd>(simd: S, rows: &[f32], channels: usize, dst: &mut [f32]) {
    debug_assert_eq!(dst.len(), channels);
    let mut rows = rows.chunks_exact(channels);
    let first = rows.next().expect("maxpool requires at least one row");
    store_gelu_row(simd, first, dst);
    for row in rows {
        max_gelu_row(simd, row, dst);
    }
}

#[inline(always)]
fn store_gelu_row<S: Simd>(simd: S, row: &[f32], dst: &mut [f32]) {
    for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
        .iter_mut()
        .zip(as_array_chunks::<4>(row))
    {
        let v = f32x4::from_slice(simd, a_c);
        gelu_simd(simd, v).store_slice(d_c);
    }
}

#[inline(always)]
fn max_gelu_row<S: Simd>(simd: S, row: &[f32], dst: &mut [f32]) {
    for (d_c, a_c) in as_array_chunks_mut::<4>(dst)
        .iter_mut()
        .zip(as_array_chunks::<4>(row))
    {
        let v = f32x4::from_slice(simd, a_c);
        let g = gelu_simd(simd, v);
        let dv = f32x4::from_slice(simd, d_c);
        g.max(dv).store_slice(d_c);
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
