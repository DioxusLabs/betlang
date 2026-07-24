//! Length-aware Winograd forward pass.
//!
//! The model was trained over a fixed padded 2048-unit shape; computing it
//! naively costs the same for every input. This engine restructures the
//! arithmetic around a few ideas, all in portable `fearless_simd` kernels:
//!
//! - **Winograd convolutions**: each conv stage runs as F(4, k) minimal
//!   filtering (`wino` holds the generated Cook-Toom transforms), cutting
//!   the multiply count per output tile by 1.9-2.8x. Weights are
//!   transformed once at load; per tile, inputs are transformed, multiplied
//!   pointwise in the interpolation domain, and transformed back.
//! - **Padded input buffers**: each stage writes into a buffer with the conv
//!   padding materialized, so the tile kernels are branch-free — no edge
//!   cases and no bounds checks in the hot loop.
//! - **Packed weights, group-outer loops**: transformed weights are packed
//!   into contiguous per-16-output-channel groups and the group loop sits
//!   outside the channel loop, so the inner loops stream a small weight
//!   slice from cache instead of re-streaming the whole kernel.
//! - **Precomputed tails** ([`ForwardPlan`]): every row at or beyond the
//!   data-dependent range of a stage equals the same row of a forward pass
//!   over an all-padding sequence, which is computed once per process,
//!   including suffix max/sum tables for the global pool. Per call, only the
//!   rows that depend on the input are computed.
//! - **Endpoint max-pooling, fused in registers**: GELU is unimodal, so
//!   `max(gelu(x_i))` over a pool group is attained at an endpoint of
//!   `[min x_i, max x_i]` — two GELU evaluations per pooled output, applied
//!   directly to the output-transform registers without a memory round-trip.

use super::{
    activation::gelu_simd,
    constants::*,
    layers::{as_array_chunks, as_array_chunks_mut, gelu_in_place, simd_level},
    runtime::Model,
    wino,
};
use fearless_simd::{Simd, SimdBase, SimdFloat, dispatch, f32x4};
use std::cell::RefCell;

/// A generated Winograd transform over `f32x4` lanes.
type Transform<S, const IN: usize, const OUT: usize> = fn(S, &[f32x4<S>; IN]) -> [f32x4<S>; OUT];

const T1: usize = MAX_UNITS / CONV0_POOL;
const T2: usize = T1 / CONV1_POOL;
const PAD0: usize = (CONV0_KERNEL - 1) / 2;
const PAD1: usize = (CONV1_KERNEL - 1) / 2;
const PAD2: usize = (CONV2_KERNEL - 1) / 2;
/// Output positions per Winograd tile (the `m` in F(m, k)).
const TILE: usize = 4;
/// Tiles processed together so the pointwise stage can share weight loads.
const TILE_BLOCK: usize = 4;
/// Output channels are processed in groups of this many.
const GROUP: usize = 16;

/// Transform a `[k][in_c][out_c]` conv kernel into the Winograd domain and
/// pack it as contiguous `[group][j][in_c][GROUP]` slices: `u[j] = G g` per
/// `(in, out)` pair, laid out so the pointwise loop reads one output-channel
/// group's weights for interpolation point `j` as a single sequential
/// stream.
pub(crate) fn pack_wino_kernel(
    kernel: &[f32],
    kernel_size: usize,
    in_channels: usize,
    out_channels: usize,
    g_matrix: &[f32],
    points: usize,
) -> Box<[f32]> {
    debug_assert!(out_channels.is_multiple_of(GROUP));
    debug_assert_eq!(g_matrix.len(), points * kernel_size);
    debug_assert_eq!(kernel.len(), kernel_size * in_channels * out_channels);
    let mut packed = vec![0.0f32; points * in_channels * out_channels].into_boxed_slice();
    for group in 0..out_channels / GROUP {
        for j in 0..points {
            for in_c in 0..in_channels {
                for lane in 0..GROUP {
                    let out_c = group * GROUP + lane;
                    let mut value = 0.0f32;
                    for k in 0..kernel_size {
                        value += g_matrix[j * kernel_size + k]
                            * kernel[(k * in_channels + in_c) * out_channels + out_c];
                    }
                    let dst = ((group * points + j) * in_channels + in_c) * GROUP + lane;
                    packed[dst] = value;
                }
            }
        }
    }
    packed
}

/// Input-independent rows of the forward pass: everything at or beyond the
/// data-dependent row range of each stage matches the same row of a forward
/// pass over an all-padding (zero-unit) sequence.
#[derive(Debug)]
pub(crate) struct ForwardPlan {
    /// Pool0 rows of the zero-input pass (`T1 x CONV0`).
    tail0: Vec<f32>,
    /// Pool1 rows of the zero-input pass (`T2 x CONV1`).
    tail1: Vec<f32>,
    /// Suffix max over the zero-input conv2 GELU rows (`(T2 + 1) x CONV2`).
    suffix_max2: Vec<f32>,
    /// Suffix sum over the zero-input conv2 GELU rows (`(T2 + 1) x CONV2`).
    suffix_sum2: Vec<f32>,
}

impl ForwardPlan {
    /// Build the plan by running the stage kernels densely over an
    /// all-padding (zero-unit) sequence, once per process.
    pub(crate) fn new(model: &Model) -> Self {
        let mut tail0 = vec![0.0f32; T1 * CONV0];
        let mut tail1 = vec![0.0f32; T2 * CONV1];
        let mut conv2 = vec![0.0f32; T2 * CONV2];
        let level = simd_level();
        dispatch!(level, simd => plan_stages(simd, model, &mut tail0, &mut tail1, &mut conv2));

        let mut suffix_max2 = vec![f32::NEG_INFINITY; (T2 + 1) * CONV2];
        let mut suffix_sum2 = vec![0.0f32; (T2 + 1) * CONV2];
        for row in (0..T2).rev() {
            for channel in 0..CONV2 {
                let value = conv2[row * CONV2 + channel];
                suffix_max2[row * CONV2 + channel] =
                    value.max(suffix_max2[(row + 1) * CONV2 + channel]);
                suffix_sum2[row * CONV2 + channel] =
                    value + suffix_sum2[(row + 1) * CONV2 + channel];
            }
        }

        Self {
            tail0,
            tail1,
            suffix_max2,
            suffix_sum2,
        }
    }
}

/// Dense zero-input pass: the same conv stages over the full padded shape,
/// materializing every stage's rows for [`ForwardPlan`]. With `pool == 1`
/// the endpoint pooling reduces to plain GELU rows.
#[inline(always)]
fn plan_stages<S: Simd>(
    simd: S,
    model: &Model,
    tail0: &mut [f32],
    tail1: &mut [f32],
    conv2: &mut [f32],
) {
    let mut v_buf = vec![0.0f32; wino::W0_POINTS * (MAX_UNITS / TILE) * EMBED];
    let mut m_buf = vec![0.0f32; wino::W0_POINTS * TILE_BLOCK * GROUP];
    let buf0 = vec![0.0f32; (MAX_UNITS + 2 * PAD0) * EMBED];
    stage_w0(
        simd,
        &buf0,
        MAX_UNITS,
        EMBED,
        &model.conv0_wino,
        CONV0,
        &model.conv0_bias,
        CONV0_POOL,
        tail0,
        &mut v_buf,
        &mut m_buf,
    );

    let mut buf1 = vec![0.0f32; (T1 + 2 * PAD1) * CONV0];
    buf1[PAD1 * CONV0..(PAD1 + T1) * CONV0].copy_from_slice(tail0);
    stage_w1(
        simd,
        &buf1,
        T1,
        CONV0,
        &model.conv1_wino,
        CONV1,
        &model.conv1_bias,
        CONV1_POOL,
        tail1,
        &mut v_buf,
        &mut m_buf,
    );

    let mut buf2 = vec![0.0f32; (T2 + 2 * PAD2) * CONV1];
    buf2[PAD2 * CONV1..(PAD2 + T2) * CONV1].copy_from_slice(tail1);
    stage_w2(
        simd,
        &buf2,
        T2,
        CONV1,
        &model.conv2_wino,
        CONV2,
        &model.conv2_bias,
        1,
        conv2,
        &mut v_buf,
        &mut m_buf,
    );
}

/// Stamp out one Winograd conv stage for `$n` interpolation points with the
/// generated `$input`/`$output` transforms.
///
/// Per block of up to [`TILE_BLOCK`] tiles:
/// 1. transform the input rows of each tile into `v[j][tile][in_c]`,
/// 2. for every output-channel group and point `j`, multiply pointwise
///    against the packed transformed weights, accumulating one `f32x4`
///    quartet per tile entirely in registers (weight loads are shared
///    across the tiles in the block),
/// 3. transform back per tile, add bias, and max-pool GELU into `dst` via
///    the endpoint identity (`pool == 1` stores plain GELU rows).
macro_rules! wino_stage {
    ($name:ident, $global_name:ident, $n:expr, $input:ident, $output:ident) => {
        #[allow(clippy::too_many_arguments)]
        #[inline(always)]
        fn $name<S: Simd>(
            simd: S,
            buffer: &[f32],
            positions: usize,
            in_channels: usize,
            packed: &[f32],
            out_channels: usize,
            bias: &[f32],
            pool: usize,
            dst: &mut [f32],
            v_buf: &mut [f32],
            m_buf: &mut [f32],
        ) {
            debug_assert!(positions.is_multiple_of(TILE));
            let tiles = positions / TILE;
            let group_len = $n * in_channels * GROUP;
            input_transform::<S, $n>(simd, buffer, in_channels, tiles, $input, v_buf);
            for (group, group_weights) in packed.chunks_exact(group_len).enumerate() {
                let bias = as_array_chunks::<4>(&bias[group * GROUP..(group + 1) * GROUP]);
                for block_start in (0..tiles).step_by(TILE_BLOCK) {
                    let block_tiles = (tiles - block_start).min(TILE_BLOCK);
                    pointwise::<S, $n>(
                        simd,
                        v_buf,
                        tiles,
                        block_start,
                        in_channels,
                        group_weights,
                        block_tiles,
                        m_buf,
                    );
                    for tile in 0..block_tiles {
                        let y = untransform::<S, $n>(simd, m_buf, tile, bias, $output);
                        let dst_row = (block_start + tile) * TILE / pool;
                        store_pooled(
                            simd,
                            &y,
                            pool,
                            &mut dst[dst_row * out_channels + group * GROUP..],
                            out_channels,
                        );
                    }
                }
            }
        }

        /// The same stage fused into the global max/sum pool over the first
        /// `valid_rows` outputs (used by the final conv).
        #[allow(clippy::too_many_arguments, dead_code)]
        #[inline(always)]
        fn $global_name<S: Simd>(
            simd: S,
            buffer: &[f32],
            valid_rows: usize,
            in_channels: usize,
            packed: &[f32],
            bias: &[f32],
            out_max: &mut [f32],
            out_sum: &mut [f32],
            v_buf: &mut [f32],
            m_buf: &mut [f32],
        ) {
            let tiles = valid_rows.div_ceil(TILE);
            let group_len = $n * in_channels * GROUP;
            input_transform::<S, $n>(simd, buffer, in_channels, tiles, $input, v_buf);
            for (group, group_weights) in packed.chunks_exact(group_len).enumerate() {
                let bias = as_array_chunks::<4>(&bias[group * GROUP..(group + 1) * GROUP]);
                let max_chunks =
                    as_array_chunks_mut::<4>(&mut out_max[group * GROUP..(group + 1) * GROUP]);
                let sum_chunks =
                    as_array_chunks_mut::<4>(&mut out_sum[group * GROUP..(group + 1) * GROUP]);
                for block_start in (0..tiles).step_by(TILE_BLOCK) {
                    let block_tiles = (tiles - block_start).min(TILE_BLOCK);
                    pointwise::<S, $n>(
                        simd,
                        v_buf,
                        tiles,
                        block_start,
                        in_channels,
                        group_weights,
                        block_tiles,
                        m_buf,
                    );
                    for tile in 0..block_tiles {
                        let y = untransform::<S, $n>(simd, m_buf, tile, bias, $output);
                        let take = (valid_rows - (block_start + tile) * TILE).min(TILE);
                        for chunk in 0..4 {
                            let mut mx = f32x4::from_slice(simd, &max_chunks[chunk]);
                            let mut sm = f32x4::from_slice(simd, &sum_chunks[chunk]);
                            for row in &y[..take] {
                                let v = gelu_simd(simd, row[chunk]);
                                mx = mx.max(v);
                                sm += v;
                            }
                            mx.store_slice(&mut max_chunks[chunk]);
                            sm.store_slice(&mut sum_chunks[chunk]);
                        }
                    }
                }
            }
        }
    };
}

wino_stage!(stage_w0, stage_w0_global, 10, input_w0_fn, output_w0_fn);
wino_stage!(stage_w1, stage_w1_global, 8, input_w1_fn, output_w1_fn);
wino_stage!(stage_w2, stage_w2_global, 6, input_w2_fn, output_w2_fn);

// Monomorphizable wrappers so the macro can pass the transforms as values.
#[inline(always)]
fn input_w0_fn<S: Simd>(simd: S, d: &[f32x4<S>; 10]) -> [f32x4<S>; 10] {
    wino::input_w0(simd, d)
}
#[inline(always)]
fn output_w0_fn<S: Simd>(simd: S, m: &[f32x4<S>; 10]) -> [f32x4<S>; 4] {
    wino::output_w0(simd, m)
}
#[inline(always)]
fn input_w1_fn<S: Simd>(simd: S, d: &[f32x4<S>; 8]) -> [f32x4<S>; 8] {
    wino::input_w1(simd, d)
}
#[inline(always)]
fn output_w1_fn<S: Simd>(simd: S, m: &[f32x4<S>; 8]) -> [f32x4<S>; 4] {
    wino::output_w1(simd, m)
}
#[inline(always)]
fn input_w2_fn<S: Simd>(simd: S, d: &[f32x4<S>; 6]) -> [f32x4<S>; 6] {
    wino::input_w2(simd, d)
}
#[inline(always)]
fn output_w2_fn<S: Simd>(simd: S, m: &[f32x4<S>; 6]) -> [f32x4<S>; 4] {
    wino::output_w2(simd, m)
}

/// Transform the `N` input rows of every tile in the stage, vectorized
/// across input channels: `v_buf[(j * tiles + tile) * in_channels + c]`
/// holds interpolation point `j` of channel `c`.
#[inline(always)]
fn input_transform<S: Simd, const N: usize>(
    simd: S,
    buffer: &[f32],
    in_channels: usize,
    tiles: usize,
    transform: Transform<S, N, N>,
    v_buf: &mut [f32],
) {
    for tile in 0..tiles {
        let row0 = tile * TILE;
        for chunk_start in (0..in_channels).step_by(4) {
            let mut d = [f32x4::splat(simd, 0.0); N];
            for (j, value) in d.iter_mut().enumerate() {
                let offset = (row0 + j) * in_channels + chunk_start;
                *value =
                    f32x4::from_slice(simd, &as_array_chunks::<4>(&buffer[offset..offset + 4])[0]);
            }
            let v = transform(simd, &d);
            for (j, value) in v.iter().enumerate() {
                let offset = (j * tiles + tile) * in_channels + chunk_start;
                value.store_slice(&mut as_array_chunks_mut::<4>(&mut v_buf[offset..offset + 4])[0]);
            }
        }
    }
}

/// Pointwise multiply in the interpolation domain: for every point `j`,
/// `m[tile][out] = sum_c u[j][c][out] * v[j][tile][c]`, with the weight
/// loads shared across the tiles of the block and the accumulators kept in
/// registers. Results land in `m_buf[(j * TILE_BLOCK + tile) * GROUP..]`.
/// Dispatches on the tile count so the accumulator array has a constant
/// bound and stays in registers.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pointwise<S: Simd, const N: usize>(
    simd: S,
    v_buf: &[f32],
    tiles: usize,
    block_start: usize,
    in_channels: usize,
    group_weights: &[f32],
    block_tiles: usize,
    m_buf: &mut [f32],
) {
    match block_tiles {
        4 => pointwise_tb::<S, N, 4>(
            simd,
            v_buf,
            tiles,
            block_start,
            in_channels,
            group_weights,
            m_buf,
        ),
        3 => pointwise_tb::<S, N, 3>(
            simd,
            v_buf,
            tiles,
            block_start,
            in_channels,
            group_weights,
            m_buf,
        ),
        2 => pointwise_tb::<S, N, 2>(
            simd,
            v_buf,
            tiles,
            block_start,
            in_channels,
            group_weights,
            m_buf,
        ),
        _ => pointwise_tb::<S, N, 1>(
            simd,
            v_buf,
            tiles,
            block_start,
            in_channels,
            group_weights,
            m_buf,
        ),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn pointwise_tb<S: Simd, const N: usize, const TB: usize>(
    simd: S,
    v_buf: &[f32],
    tiles: usize,
    block_start: usize,
    in_channels: usize,
    group_weights: &[f32],
    m_buf: &mut [f32],
) {
    for (j, u) in group_weights.chunks_exact(in_channels * GROUP).enumerate() {
        let v_rows = &v_buf[(j * tiles + block_start) * in_channels..][..TB * in_channels];
        let zero = f32x4::splat(simd, 0.0);
        let mut acc = [[zero; 4]; TB];
        for (c, w) in u.chunks_exact(GROUP).enumerate() {
            let w = as_array_chunks::<4>(w);
            let w0 = f32x4::from_slice(simd, &w[0]);
            let w1 = f32x4::from_slice(simd, &w[1]);
            let w2 = f32x4::from_slice(simd, &w[2]);
            let w3 = f32x4::from_slice(simd, &w[3]);
            for (tile, acc) in acc.iter_mut().enumerate() {
                let x = f32x4::splat(simd, v_rows[tile * in_channels + c]);
                acc[0] = w0.mul_add(x, acc[0]);
                acc[1] = w1.mul_add(x, acc[1]);
                acc[2] = w2.mul_add(x, acc[2]);
                acc[3] = w3.mul_add(x, acc[3]);
            }
        }
        for (tile, acc) in acc.iter().enumerate() {
            let dst =
                &mut m_buf[(j * TILE_BLOCK + tile) * GROUP..(j * TILE_BLOCK + tile + 1) * GROUP];
            let dst = as_array_chunks_mut::<4>(dst);
            for chunk in 0..4 {
                acc[chunk].store_slice(&mut dst[chunk]);
            }
        }
    }
}

/// Apply the output transform for one tile and channel group, returning the
/// `TILE` position rows (each 4 chunks of 4 channels) with bias added.
#[inline(always)]
fn untransform<S: Simd, const N: usize>(
    simd: S,
    m_buf: &[f32],
    tile: usize,
    bias: &[[f32; 4]],
    transform: Transform<S, N, 4>,
) -> [[f32x4<S>; 4]; TILE] {
    let zero = f32x4::splat(simd, 0.0);
    let mut y = [[zero; 4]; TILE];
    for chunk in 0..4 {
        let mut m = [zero; N];
        for (j, value) in m.iter_mut().enumerate() {
            let offset = (j * TILE_BLOCK + tile) * GROUP + chunk * 4;
            *value = f32x4::from_slice(simd, &as_array_chunks::<4>(&m_buf[offset..offset + 4])[0]);
        }
        let out = transform(simd, &m);
        let b = f32x4::from_slice(simd, &bias[chunk]);
        for (position, value) in out.iter().enumerate() {
            y[position][chunk] = *value + b;
        }
    }
    y
}

/// Max-pool GELU over the tile's `TILE` position rows via the endpoint
/// identity and store into `dst` (`pool == 1` stores plain GELU rows).
#[inline(always)]
fn store_pooled<S: Simd>(
    simd: S,
    y: &[[f32x4<S>; 4]; TILE],
    pool: usize,
    dst: &mut [f32],
    out_channels: usize,
) {
    match pool {
        1 => {
            for (position, row) in y.iter().enumerate() {
                let dst = as_array_chunks_mut::<4>(&mut dst[position * out_channels..][..GROUP]);
                for chunk in 0..4 {
                    gelu_simd(simd, row[chunk]).store_slice(&mut dst[chunk]);
                }
            }
        }
        2 => {
            for (pair, rows) in y.chunks_exact(2).enumerate() {
                let dst = as_array_chunks_mut::<4>(&mut dst[pair * out_channels..][..GROUP]);
                for chunk in 0..4 {
                    let hi = rows[0][chunk].max(rows[1][chunk]);
                    let lo = rows[0][chunk].min(rows[1][chunk]);
                    let best = gelu_simd(simd, hi).max(gelu_simd(simd, lo));
                    best.store_slice(&mut dst[chunk]);
                }
            }
        }
        _ => {
            let dst = as_array_chunks_mut::<4>(&mut dst[..GROUP]);
            for chunk in 0..4 {
                let hi = y[0][chunk]
                    .max(y[1][chunk])
                    .max(y[2][chunk].max(y[3][chunk]));
                let lo = y[0][chunk]
                    .min(y[1][chunk])
                    .min(y[2][chunk].min(y[3][chunk]));
                let best = gelu_simd(simd, hi).max(gelu_simd(simd, lo));
                best.store_slice(&mut dst[chunk]);
            }
        }
    }
}

/// Fill `dst` (a padded stage buffer) so that conceptual input rows
/// `[data_rows, limit)` hold `tail` rows `[data_rows, tail_rows)` followed by
/// zeros (beyond-sequence padding), and the `pad` left-padding rows are zero.
/// Rows `[0, data_rows)` were already written by the previous stage.
fn fill_padding(
    dst: &mut [f32],
    pad: usize,
    data_rows: usize,
    tail: &[f32],
    tail_rows: usize,
    limit: usize,
    channels: usize,
) {
    dst[..pad * channels].fill(0.0);
    let dst = &mut dst[pad * channels..];
    let data_end = data_rows.min(limit);
    let tail_end = tail_rows.min(limit);
    if tail_end > data_end {
        dst[data_end * channels..tail_end * channels]
            .copy_from_slice(&tail[data_end * channels..tail_end * channels]);
    }
    if limit > tail_end {
        dst[tail_end * channels..limit * channels].fill(0.0);
    }
}

impl Model {
    /// Length-aware forward pass.
    pub(crate) fn logits(&self, units: &[i32]) -> [f32; CLASSES] {
        let n = units.len().min(MAX_UNITS);
        let plan = self.forward_plan();
        if n == 0 {
            let mut pooled = [0.0f32; POOLED];
            let (max_slice, avg_slice) = pooled.split_at_mut(CONV2);
            max_slice.copy_from_slice(&plan.suffix_max2[..CONV2]);
            avg_slice.copy_from_slice(&plan.suffix_sum2[..CONV2]);
            let inv = 1.0 / T2 as f32;
            for avg in avg_slice.iter_mut() {
                *avg *= inv;
            }
            return self.dense_head(&pooled);
        }

        // Data-dependent pooled row counts per stage, and the conv position
        // counts rounded up to whole tiles. Positions past the data-dependent
        // range are cheap to compute and discarded or overwritten.
        let d0 = (n + PAD0).div_ceil(CONV0_POOL).min(T1);
        let t0 = d0 * CONV0_POOL;
        let d1 = (d0 + PAD1).div_ceil(CONV1_POOL).min(T2);
        let t1 = (d1 * CONV1_POOL).next_multiple_of(TILE).min(T1);
        let d1_up = t1 / CONV1_POOL;
        let d2 = (d1 + PAD2).min(T2);
        let d2_padded = d2.next_multiple_of(TILE);

        // Reused per-thread scratch with fixed worst-case stage offsets.
        const BUF0_LEN: usize = (MAX_UNITS + 2 * PAD0) * EMBED;
        const BUF1_LEN: usize = (T1 + 2 * PAD1) * CONV0;
        const BUF2_LEN: usize = (T2 + TILE + 2 * PAD2) * CONV1;
        const V_LEN: usize = wino::W0_POINTS * (MAX_UNITS / TILE) * EMBED;
        const M_LEN: usize = wino::W0_POINTS * TILE_BLOCK * GROUP;
        const SCRATCH_LEN: usize = BUF0_LEN + BUF1_LEN + BUF2_LEN + V_LEN + M_LEN;
        thread_local! {
            static SCRATCH: RefCell<Vec<f32>> = RefCell::new(vec![0.0; SCRATCH_LEN]);
        }

        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (buf0, rest) = scratch.split_at_mut(BUF0_LEN);
            let (buf1, rest) = rest.split_at_mut(BUF1_LEN);
            let (buf2, rest) = rest.split_at_mut(BUF2_LEN);
            let (v_buf, m_buf) = rest.split_at_mut(V_LEN);

            // Stage 0: embed into the zero-padded buffer. Level-0 padding is
            // the zero row, so both the left padding and everything past the
            // embedded rows is zero-filled.
            buf0[..PAD0 * EMBED].fill(0.0);
            buf0[(PAD0 + n) * EMBED..(t0 + 2 * PAD0) * EMBED].fill(0.0);
            let embed_rows = &mut buf0[PAD0 * EMBED..(PAD0 + n) * EMBED];
            self.embed_units(&units[..n], embed_rows);
            gelu_in_place(embed_rows);

            let mut pooled = [0.0f32; POOLED];
            let level = simd_level();
            dispatch!(level, simd => forward_stages(
                simd,
                self,
                plan,
                buf0,
                buf1,
                buf2,
                v_buf,
                m_buf,
                &mut pooled,
                t0,
                d0,
                t1,
                d1,
                d1_up,
                d2,
                d2_padded,
            ));

            let (_, avg_slice) = pooled.split_at_mut(CONV2);
            let inv = 1.0 / T2 as f32;
            for avg in avg_slice.iter_mut() {
                *avg *= inv;
            }

            self.dense_head(&pooled)
        })
    }
}

/// All three conv stages under a single SIMD dispatch. `pooled` receives the
/// global max in its first half and the (unnormalized) global sum in its
/// second half.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_stages<S: Simd>(
    simd: S,
    model: &Model,
    plan: &ForwardPlan,
    buf0: &[f32],
    buf1: &mut [f32],
    buf2: &mut [f32],
    v_buf: &mut [f32],
    m_buf: &mut [f32],
    pooled: &mut [f32; POOLED],
    t0: usize,
    d0: usize,
    t1: usize,
    d1: usize,
    d1_up: usize,
    d2: usize,
    d2_padded: usize,
) {
    // Stage 0 -> pooled rows [0, d0) of the stage-1 buffer.
    stage_w0(
        simd,
        &buf0[..(t0 + 2 * PAD0) * EMBED],
        t0,
        EMBED,
        &model.conv0_wino,
        CONV0,
        &model.conv0_bias,
        CONV0_POOL,
        &mut buf1[PAD1 * CONV0..(PAD1 + d0) * CONV0],
        v_buf,
        m_buf,
    );

    // Stage 1: halo from the precomputed tail, pooled rows -> stage-2 buffer.
    // The conv may produce up to `d1_up` pooled rows; rows at or beyond `d1`
    // are recomputable constants and are overwritten by the next fill.
    fill_padding(buf1, PAD1, d0, &plan.tail0, T1, t1 + PAD1, CONV0);
    stage_w1(
        simd,
        &buf1[..(t1 + 2 * PAD1) * CONV0],
        t1,
        CONV0,
        &model.conv1_wino,
        CONV1,
        &model.conv1_bias,
        CONV1_POOL,
        &mut buf2[PAD2 * CONV1..(PAD2 + d1_up) * CONV1],
        v_buf,
        m_buf,
    );

    // Stage 2: conv2 + fused global max/sum over the first d2 rows, seeded
    // with the precomputed suffix contributions of the remaining rows.
    fill_padding(buf2, PAD2, d1, &plan.tail1, T2, d2_padded + PAD2, CONV1);
    let (out_max, out_sum) = pooled.split_at_mut(CONV2);
    out_max.copy_from_slice(&plan.suffix_max2[d2 * CONV2..(d2 + 1) * CONV2]);
    out_sum.copy_from_slice(&plan.suffix_sum2[d2 * CONV2..(d2 + 1) * CONV2]);
    stage_w2_global(
        simd,
        &buf2[..(d2_padded + 2 * PAD2) * CONV1],
        d2,
        CONV1,
        &model.conv2_wino,
        &model.conv2_bias,
        out_max,
        out_sum,
        v_buf,
        m_buf,
    );
}
