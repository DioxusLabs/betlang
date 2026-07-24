//! Length-aware streaming forward pass.
//!
//! The reference pipeline in `runtime` computes every stage over the fixed
//! padded 2048-unit shape with per-call handling for the constant padded
//! tail. This engine restructures the same arithmetic around three ideas,
//! all in portable `fearless_simd` kernels:
//!
//! - **Padded input buffers**: each stage writes into a buffer with the conv
//!   padding materialized, so every convolution block takes the branch-free
//!   SIMD fast path — no edge cases inside the hot loop.
//! - **Precomputed tails** ([`ForwardPlan`]): every row at or beyond the
//!   data-dependent range of a stage equals the same row of a forward pass
//!   over an all-padding sequence, which is computed once per process,
//!   including suffix max/sum tables for the global pool. Per call, only the
//!   rows that depend on the input are computed.
//! - **Endpoint max-pooling**: GELU is unimodal, so `max(gelu(x_i))` over a
//!   pool group is attained at an endpoint of `[min x_i, max x_i]` — two
//!   GELU evaluations per pooled output instead of one per position.

use super::{
    constants::*,
    layers::{conv1d_block, gelu_in_place, gelu_max_sum_rows, pool_endpoint_gelu_rows, simd_level},
    runtime::Model,
};
use fearless_simd::{Simd, dispatch};
use std::cell::RefCell;

const T1: usize = MAX_UNITS / CONV0_POOL;
const T2: usize = T1 / CONV1_POOL;
const PAD0: usize = (CONV0_KERNEL - 1) / 2;
const PAD1: usize = (CONV1_KERNEL - 1) / 2;
const PAD2: usize = (CONV2_KERNEL - 1) / 2;
/// Conv positions are processed in blocks of this many outputs.
const BLOCK: usize = 4;

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
    /// Build the plan from the reference pipeline so the tail rows match it
    /// bit-for-bit.
    pub(crate) fn new(model: &Model) -> Self {
        let (tail0, tail1, conv2) = model.reference_stage_outputs(&[]);

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

/// One conv stage over a padded buffer: for every block of `BLOCK` output
/// positions, run the branch-free conv kernel and max-pool `gelu` into `dst`
/// via the endpoint identity. The buffer holds `pad` left-padding rows, so
/// conceptual input row `j` lives at buffer row `j + pad` and every block is
/// fully in bounds.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn conv_stage<S: Simd>(
    simd: S,
    buffer: &[f32],
    positions: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
    accs: &mut [f32],
    dst: &mut [f32],
) {
    debug_assert!(positions.is_multiple_of(BLOCK));
    let pad = (kernel_size - 1) / 2;
    let rows = buffer.len() / in_channels;
    let accs = &mut accs[..BLOCK * out_channels];
    for block_start in (0..positions).step_by(BLOCK) {
        conv1d_block::<S, BLOCK>(
            simd,
            buffer,
            rows,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            block_start + pad,
            accs,
        );
        let dst_start = block_start / pool * out_channels;
        pool_endpoint_gelu_rows(
            simd,
            accs,
            out_channels,
            pool,
            &mut dst[dst_start..dst_start + (BLOCK / pool) * out_channels],
        );
    }
}

/// The conv2 stage feeds the global pool directly: conv each block, then
/// fold GELU into the running max/sum over the first `valid_rows` outputs.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn conv_global_stage<S: Simd>(
    simd: S,
    buffer: &[f32],
    valid_rows: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    accs: &mut [f32],
    out_max: &mut [f32],
    out_sum: &mut [f32],
) {
    let pad = (kernel_size - 1) / 2;
    let rows = buffer.len() / in_channels;
    let accs = &mut accs[..BLOCK * out_channels];
    for block_start in (0..valid_rows).step_by(BLOCK) {
        conv1d_block::<S, BLOCK>(
            simd,
            buffer,
            rows,
            in_channels,
            kernel,
            kernel_size,
            out_channels,
            bias,
            block_start + pad,
            accs,
        );
        let take = (valid_rows - block_start).min(BLOCK);
        gelu_max_sum_rows(
            simd,
            &accs[..take * out_channels],
            out_channels,
            out_max,
            out_sum,
        );
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
    /// Length-aware forward pass. Matches [`Model::logits_reference`] up to
    /// floating-point reassociation in the pooling reductions.
    pub(crate) fn logits_fast(&self, units: &[i32]) -> [f32; CLASSES] {
        let n = units.len().min(MAX_UNITS);
        if n == 0 {
            return self.logits_reference(units);
        }
        let plan = self.forward_plan();

        // Data-dependent pooled row counts per stage, and the conv position
        // counts rounded up to whole blocks. Positions past the data-dependent
        // range are cheap to compute and discarded or overwritten.
        let d0 = (n + PAD0).div_ceil(CONV0_POOL).min(T1);
        let t0 = d0 * CONV0_POOL;
        let d1 = (d0 + PAD1).div_ceil(CONV1_POOL).min(T2);
        let t1 = (d1 * CONV1_POOL).next_multiple_of(BLOCK).min(T1);
        let d1_up = t1 / CONV1_POOL;
        let d2 = (d1 + PAD2).min(T2);

        // Reused per-thread scratch with fixed worst-case stage offsets.
        const BUF0_LEN: usize = (MAX_UNITS + 2 * PAD0) * EMBED;
        const BUF1_LEN: usize = (T1 + 2 * PAD1) * CONV0;
        const BUF2_LEN: usize = (T2 + BLOCK + 2 * PAD2) * CONV1;
        const ACCS_LEN: usize = 2 * CONV2 + BLOCK * CONV1;
        const SCRATCH_LEN: usize = BUF0_LEN + BUF1_LEN + BUF2_LEN + ACCS_LEN;
        thread_local! {
            static SCRATCH: RefCell<Vec<f32>> = RefCell::new(vec![0.0; SCRATCH_LEN]);
        }

        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let (buf0, rest) = scratch.split_at_mut(BUF0_LEN);
            let (buf1, rest) = rest.split_at_mut(BUF1_LEN);
            let (buf2, accs) = rest.split_at_mut(BUF2_LEN);

            // Stage 0: embed into the zero-padded buffer, then conv0 + pool4,
            // writing the pooled rows straight into the stage-1 buffer.
            // Level-0 padding is the zero row, so both the left padding and
            // everything past the embedded rows is zero-filled.
            buf0[..PAD0 * EMBED].fill(0.0);
            buf0[(PAD0 + n) * EMBED..(t0 + 2 * PAD0) * EMBED].fill(0.0);
            let embed_rows = &mut buf0[PAD0 * EMBED..(PAD0 + n) * EMBED];
            self.embed_units(&units[..n], embed_rows);
            gelu_in_place(embed_rows);

            let level = simd_level();
            dispatch!(level, simd => forward_stages(
                simd,
                self,
                plan,
                buf0,
                buf1,
                buf2,
                accs,
                t0,
                d0,
                t1,
                d1,
                d1_up,
                d2,
            ));

            // Global max/avg: precomputed suffix over the input-independent
            // rows, already seeded by `forward_stages`.
            let mut pooled = [0.0f32; POOLED];
            let (max_slice, avg_slice) = pooled.split_at_mut(CONV2);
            max_slice.copy_from_slice(&accs[..CONV2]);
            avg_slice.copy_from_slice(&accs[CONV2..2 * CONV2]);
            let inv = 1.0 / T2 as f32;
            for avg in avg_slice.iter_mut() {
                *avg *= inv;
            }

            self.dense_head(&pooled)
        })
    }
}

/// All three conv stages under a single SIMD dispatch. The final stage's
/// max/sum accumulators are returned through the head of `accs`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn forward_stages<S: Simd>(
    simd: S,
    model: &Model,
    plan: &ForwardPlan,
    buf0: &[f32],
    buf1: &mut [f32],
    buf2: &mut [f32],
    accs: &mut [f32],
    t0: usize,
    d0: usize,
    t1: usize,
    d1: usize,
    d1_up: usize,
    d2: usize,
) {
    // Stage 0 -> pooled rows [0, d0) of the stage-1 buffer.
    conv_stage(
        simd,
        &buf0[..(t0 + 2 * PAD0) * EMBED],
        t0,
        EMBED,
        &model.conv0_kernel,
        CONV0_KERNEL,
        CONV0,
        &model.conv0_bias,
        CONV0_POOL,
        accs,
        &mut buf1[PAD1 * CONV0..(PAD1 + d0) * CONV0],
    );

    // Stage 1: halo from the precomputed tail, pooled rows -> stage-2 buffer.
    // The conv may produce up to `d1_up` pooled rows; rows at or beyond `d1`
    // are recomputable constants and are overwritten by the next fill.
    fill_padding(buf1, PAD1, d0, &plan.tail0, T1, t1 + PAD1, CONV0);
    conv_stage(
        simd,
        &buf1[..(t1 + 2 * PAD1) * CONV0],
        t1,
        CONV0,
        &model.conv1_kernel,
        CONV1_KERNEL,
        CONV1,
        &model.conv1_bias,
        CONV1_POOL,
        accs,
        &mut buf2[PAD2 * CONV1..(PAD2 + d1_up) * CONV1],
    );

    // Stage 2: conv2 + fused global max/sum over the first d2 rows, seeded
    // with the precomputed suffix contributions of the remaining rows.
    let d2_padded = d2.next_multiple_of(BLOCK).min(T2 + BLOCK);
    fill_padding(buf2, PAD2, d1, &plan.tail1, T2, d2_padded + PAD2, CONV1);
    let (out_max, rest) = accs.split_at_mut(CONV2);
    let (out_sum, accs2) = rest.split_at_mut(CONV2);
    out_max.copy_from_slice(&plan.suffix_max2[d2 * CONV2..(d2 + 1) * CONV2]);
    out_sum.copy_from_slice(&plan.suffix_sum2[d2 * CONV2..(d2 + 1) * CONV2]);
    conv_global_stage(
        simd,
        &buf2[..(d2_padded + 2 * PAD2) * CONV1],
        d2,
        CONV1,
        &model.conv2_kernel,
        CONV2_KERNEL,
        CONV2,
        &model.conv2_bias,
        accs2,
        out_max,
        out_sum,
    );
}
