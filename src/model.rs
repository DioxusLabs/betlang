//! Inference for the wordseq student.
//!
//! Loads `assets/magika/source-student-q4.bin` (~50 KB MSQ1 export) and
//! runs a forward pass: byte-window tokenization → word-unit tokenization
//! → HashEmbedding lookup (K=3) → 3 conv stages with max-pool → global
//! max+avg pool → 2 dense layers → 48-class softmax logits.
//!
//! Model architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
//! - 1024-bin × 24-dim shared HashEmbedding table (4-bit, ~12 KB)
//! - QConv1D k=7 24→64ch (2-bit ternary)
//! - MaxPool(4)
//! - QConv1D k=5 64→128ch (2-bit)
//! - MaxPool(2)
//! - QConv1D k=3 128→128ch (2-bit)
//! - GlobalMax ⊕ GlobalAvg → 256-dim
//! - QDense 256→96 (2-bit) + GELU
//! - QDense 96→48 (4-bit)
//!
//! Convolution hot paths use `fearless_simd` with scalar fallbacks for
//! edges and out-channel counts not divisible by 16.

use crate::language::{CLASS_LANGUAGES, Language};
use fearless_simd::{Level, Simd, SimdBase, SimdFloat, dispatch, f32x4};
use std::sync::OnceLock;

/// Detect the best available SIMD level once per process.
fn simd_level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

static MODEL_BYTES: &[u8] = include_bytes!("../assets/magika/source-student-q4.bin");

const MODEL_MAGIC: [u8; 8] = [0x4d, 0x53, 0x51, 0x31, 0x01, 0x00, 0x00, 0x00];

const MAGIKA_BEG_SIZE: usize = 1_024;
const MAGIKA_END_SIZE: usize = 1_024;
const MAGIKA_BLOCK_SIZE: usize = 4_096;

// Wordseq architecture constants. Must match what the model was trained with.
const BINS: usize = 1_024;
const MAX_UNITS: usize = 2_048;
const EMBED: usize = 24;
const CONV0_KERNEL: usize = 7;
const CONV0: usize = 64;
const CONV0_POOL: usize = 4;
const CONV1_KERNEL: usize = 5;
const CONV1: usize = 128;
const CONV1_POOL: usize = 2;
const CONV2_KERNEL: usize = 3;
const CONV2: usize = 128;
const POOLED: usize = CONV2 * 2; // GlobalMax + GlobalAvg
const DENSE: usize = 96;
pub(crate) const CLASSES: usize = 48;
const RELIABLE_LOGIT_MARGIN: f32 = 3.0;

// v2 tokenizer flag bits. Must match `_PUNCT_FLAG`/etc. in the Python trainer.
const WORD_MASK: u32 = 0x00FF_FFFF;
const STYLE_BIT: u32 = 0x0100_0000;
const PUNCT_FLAG: u32 = 0x1000_0000;
const INDENT_FLAG: u32 = 0x2000_0000;
const NUM_FLAG: u32 = 0x4000_0000;
const BRACKET_FLAG: u32 = 0x5000_0000;
const STRING_FLAG: u32 = 0x7000_0000;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TokenizerVersion {
    V2,
    V3,
    V4,
}

#[derive(Debug)]
struct Model {
    tokenizer_version: TokenizerVersion,
    /// 1024 × 24 dequantized embedding rows.
    embedding: Vec<f32>,
    /// `[k][in_c][out_c]` — inner kernel row is contiguous over out_channels.
    conv0_kernel: Vec<f32>,
    conv0_bias: [f32; CONV0],
    conv1_kernel: Vec<f32>,
    conv1_bias: [f32; CONV1],
    conv2_kernel: Vec<f32>,
    conv2_bias: [f32; CONV2],
    /// (POOLED, DENSE) flattened.
    dense0_kernel: Vec<f32>,
    dense0_bias: [f32; DENSE],
    /// (DENSE, CLASSES) flattened.
    output_kernel: Vec<f32>,
    output_bias: [f32; CLASSES],
}

impl Model {
    fn load() -> Self {
        assert!(MODEL_BYTES.starts_with(&MODEL_MAGIC), "bad MSQ1 magic");
        let metadata_start = rfind_bytes(MODEL_BYTES, br#"{"bits""#).expect("no metadata");
        let metadata = std::str::from_utf8(&MODEL_BYTES[metadata_start..]).expect("utf-8 meta");
        assert!(
            metadata.contains(r#""architecture":"wordseq-b1024-k3-m2048-tiny-3conv-hidden""#),
            "shipped model is not the expected wordseq architecture",
        );
        let tokenizer_version = parse_tokenizer_version(metadata);
        let scales = parse_scales(metadata);
        // 6 layers, each with 1 weight tensor → 6 scales total.
        assert_eq!(scales.len(), 6, "expected 6 weight scales");

        let mut cur = MODEL_MAGIC.len();

        // q_hash_embedding: weights [(1024, 24)] int4
        let embedding = read_int4_dequant(&mut cur, BINS * EMBED, scales[0]);

        // q_conv_0: weights [(7, 24, 64)] ternary, bias [(64,)] f32
        let conv0_kernel = read_ternary_dequant(&mut cur, CONV0_KERNEL * EMBED * CONV0, scales[1]);
        let conv0_bias = read_f32_array::<CONV0>(&mut cur);

        // q_conv_1: weights [(5, 64, 128)] ternary, bias [(128,)] f32
        let conv1_kernel = read_ternary_dequant(&mut cur, CONV1_KERNEL * CONV0 * CONV1, scales[2]);
        let conv1_bias = read_f32_array::<CONV1>(&mut cur);

        // q_conv_2: weights [(3, 128, 128)] ternary, bias [(128,)] f32
        let conv2_kernel = read_ternary_dequant(&mut cur, CONV2_KERNEL * CONV1 * CONV2, scales[3]);
        let conv2_bias = read_f32_array::<CONV2>(&mut cur);

        // q_dense_0: weights [(256, 96)] ternary, bias [(96,)] f32
        let dense0_kernel = read_ternary_dequant(&mut cur, POOLED * DENSE, scales[4]);
        let dense0_bias = read_f32_array::<DENSE>(&mut cur);

        // q_output: weights [(96, 48)] int4, bias [(48,)] f32
        let output_kernel = read_int4_dequant(&mut cur, DENSE * CLASSES, scales[5]);
        let output_bias = read_f32_array::<CLASSES>(&mut cur);

        // Trailing 4-byte LE metadata length (not used for indexing here, just sanity).
        assert_eq!(cur + 4, metadata_start, "unexpected payload length");
        let metadata_len = u32::from_le_bytes(MODEL_BYTES[cur..cur + 4].try_into().unwrap());
        assert_eq!(metadata_len as usize, MODEL_BYTES.len() - metadata_start);

        Self {
            tokenizer_version,
            embedding,
            conv0_kernel,
            conv0_bias,
            conv1_kernel,
            conv1_bias,
            conv2_kernel,
            conv2_bias,
            dense0_kernel,
            dense0_bias,
            output_kernel,
            output_bias,
        }
    }

    fn get() -> &'static Self {
        static MODEL: OnceLock<Model> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    /// Run the full forward pass on a unit-id sequence (length = `len`).
    fn logits(&self, units: &[i32], len: usize) -> [f32; CLASSES] {
        // 1) HashEmbedding (K=3 hashed rows summed) + GELU, fused per position.
        //    Padding positions (unit_id < 0) → gelu(0) = 0.
        let t = len.min(MAX_UNITS);
        let mut embed = vec![0.0f32; t * EMBED];
        for pos in 0..t.min(units.len()) {
            let id = units[pos];
            if id < 0 {
                continue;
            }
            let dst = &mut embed[pos * EMBED..(pos + 1) * EMBED];
            embed_position(&self.embedding, id as u32, dst);
            for v in dst.iter_mut() {
                *v = gelu(*v);
            }
        }

        // 2) Conv0 (k=7, 24→64) + GELU + MaxPool(4), fused to avoid the
        //    ~524 KB intermediate buffer and the second read pass.
        let (pool0, t1) = conv_gelu_maxpool(
            &embed,
            t,
            EMBED,
            &self.conv0_kernel,
            CONV0_KERNEL,
            CONV0,
            &self.conv0_bias,
            CONV0_POOL,
        );

        // 3) Conv1 (k=5, 64→128) + GELU + MaxPool(2), fused.
        let (pool1, t2) = conv_gelu_maxpool(
            &pool0,
            t1,
            CONV0,
            &self.conv1_kernel,
            CONV1_KERNEL,
            CONV1,
            &self.conv1_bias,
            CONV1_POOL,
        );

        // 4) Conv2 (k=3, 128→128) + GELU + GlobalMax/AvgPool → 256-dim,
        //    fused so the conv2 output is never materialized.
        let mut pooled = [0.0f32; POOLED];
        let (max_slice, avg_slice) = pooled.split_at_mut(CONV2);
        conv_gelu_global_pool(
            &pool1,
            t2,
            CONV1,
            &self.conv2_kernel,
            CONV2_KERNEL,
            CONV2,
            &self.conv2_bias,
            max_slice,
            avg_slice,
        );

        // 5) Dense 256→96 + GELU
        let mut dense0_out = [0.0f32; DENSE];
        dense_forward(
            &pooled,
            &self.dense0_kernel,
            &self.dense0_bias,
            &mut dense0_out,
        );
        for v in &mut dense0_out {
            *v = gelu(*v);
        }

        // 6) Dense 96→48 → logits
        let mut logits = [0.0f32; CLASSES];
        dense_forward(
            &dense0_out,
            &self.output_kernel,
            &self.output_bias,
            &mut logits,
        );
        logits
    }

    /// Run inference using the padded 2048-position shape used by the shipped
    /// Python evaluator. Shorter runtime sequences must not shrink the CNN,
    /// because pooling/global-average behavior changes materially.
    fn logits_for_runtime_units(&self, units: &[i32]) -> [f32; CLASSES] {
        self.logits(units, MAX_UNITS)
    }

    fn tokenize_units(&self, bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
        match self.tokenizer_version {
            TokenizerVersion::V2 => tokenize_v2(bytes, padding_mask),
            TokenizerVersion::V3 => tokenize_v3(bytes, padding_mask),
            TokenizerVersion::V4 => tokenize_v4(bytes, padding_mask),
        }
    }
}

/// Sum the K=3 hashed embedding rows for one unit-id into `dst` (length EMBED).
#[inline]
fn embed_position(embedding: &[f32], unit: u32, dst: &mut [f32]) {
    let b0 = hash_bin(unit, 0);
    let b1 = hash_bin(unit, 1);
    let b2 = hash_bin(unit, 2);
    let row0 = &embedding[b0 * EMBED..b0 * EMBED + EMBED];
    let row1 = &embedding[b1 * EMBED..b1 * EMBED + EMBED];
    let row2 = &embedding[b2 * EMBED..b2 * EMBED + EMBED];
    for i in 0..EMBED {
        dst[i] = row0[i] + row1[i] + row2[i];
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
/// out_channels; both scalar and SIMD paths read it directly.
///
/// BLOCK=4 + all positions in-bounds → SIMD fast path. When out_channels
/// is a multiple of 16, the group-16 kernel keeps 16 `f32x4` accumulators
/// register-resident across the full (k, in_c) inner loop; otherwise a
/// chunk-inner SIMD path with 4 named acc rows is used.
/// Edges / BLOCK != 4 → scalar fallback.
#[inline(always)]
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
    debug_assert_eq!(out_channels % 4, 0);
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
    // Scalar fallback (BLOCK != 4 OR sequence edge). Uses the original layout
    // where the inner kernel row is contiguous.
    for s in 0..BLOCK {
        accs[s * out_channels..(s + 1) * out_channels].copy_from_slice(bias);
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
        for in_c in 0..in_channels {
            let krow_off = kbase + in_c * out_channels;
            let krow = &kernel[krow_off..krow_off + out_channels];
            for s in s_lo..s_hi {
                let src_t = (src_t_at_s0 + s as isize) as usize;
                let x = input[src_t * in_channels + in_c];
                let acc = &mut accs[s * out_channels..(s + 1) * out_channels];
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
    if out_channels % GROUP == 0 {
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
    // Fallback for out_channels not a multiple of 16: chunk-inner SIMD.
    for s in 0..4 {
        accs[s * out_channels..(s + 1) * out_channels].copy_from_slice(bias);
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
        for in_c in 0..in_channels {
            let krow = &kernel[kbase + in_c * out_channels..kbase + (in_c + 1) * out_channels];
            let xv0 = f32x4::splat(simd, input[row0_off + in_c]);
            let xv1 = f32x4::splat(simd, input[row1_off + in_c]);
            let xv2 = f32x4::splat(simd, input[row2_off + in_c]);
            let xv3 = f32x4::splat(simd, input[row3_off + in_c]);
            for ((((kr_c, a0_c), a1_c), a2_c), a3_c) in krow
                .chunks_exact(4)
                .zip(a0.chunks_exact_mut(4))
                .zip(a1.chunks_exact_mut(4))
                .zip(a2.chunks_exact_mut(4))
                .zip(a3.chunks_exact_mut(4))
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
    let groups = out_channels / 16;
    for g in 0..groups {
        let g_off = g * 16;
        let b0 = f32x4::from_slice(simd, &bias[g_off..g_off + 4]);
        let b1 = f32x4::from_slice(simd, &bias[g_off + 4..g_off + 8]);
        let b2 = f32x4::from_slice(simd, &bias[g_off + 8..g_off + 12]);
        let b3 = f32x4::from_slice(simd, &bias[g_off + 12..g_off + 16]);
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
            for in_c in 0..in_channels {
                let krow_off = kbase + in_c * out_channels + g_off;
                let kr0 = f32x4::from_slice(simd, &kernel[krow_off..krow_off + 4]);
                let kr1 = f32x4::from_slice(simd, &kernel[krow_off + 4..krow_off + 8]);
                let kr2 = f32x4::from_slice(simd, &kernel[krow_off + 8..krow_off + 12]);
                let kr3 = f32x4::from_slice(simd, &kernel[krow_off + 12..krow_off + 16]);
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
        a0_0.store_slice(&mut accs[g_off..g_off + 4]);
        a0_1.store_slice(&mut accs[g_off + 4..g_off + 8]);
        a0_2.store_slice(&mut accs[g_off + 8..g_off + 12]);
        a0_3.store_slice(&mut accs[g_off + 12..g_off + 16]);
        a1_0.store_slice(&mut accs[out_channels + g_off..out_channels + g_off + 4]);
        a1_1.store_slice(&mut accs[out_channels + g_off + 4..out_channels + g_off + 8]);
        a1_2.store_slice(&mut accs[out_channels + g_off + 8..out_channels + g_off + 12]);
        a1_3.store_slice(&mut accs[out_channels + g_off + 12..out_channels + g_off + 16]);
        a2_0.store_slice(&mut accs[2 * out_channels + g_off..2 * out_channels + g_off + 4]);
        a2_1.store_slice(&mut accs[2 * out_channels + g_off + 4..2 * out_channels + g_off + 8]);
        a2_2.store_slice(&mut accs[2 * out_channels + g_off + 8..2 * out_channels + g_off + 12]);
        a2_3.store_slice(&mut accs[2 * out_channels + g_off + 12..2 * out_channels + g_off + 16]);
        a3_0.store_slice(&mut accs[3 * out_channels + g_off..3 * out_channels + g_off + 4]);
        a3_1.store_slice(&mut accs[3 * out_channels + g_off + 4..3 * out_channels + g_off + 8]);
        a3_2.store_slice(&mut accs[3 * out_channels + g_off + 8..3 * out_channels + g_off + 12]);
        a3_3.store_slice(&mut accs[3 * out_channels + g_off + 12..3 * out_channels + g_off + 16]);
    }
}

/// Fused conv1d (SAME pad) + GELU + MaxPool(pool).
/// Always accumulates BLOCK=4 consecutive conv positions per outer iteration
/// (max NEON-friendly krow reuse), then applies POOL-wide maxpool over them.
fn conv_gelu_maxpool(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    pool: usize,
) -> (Vec<f32>, usize) {
    let pooled_len = seq_len / pool;
    let mut out = vec![0.0f32; pooled_len * out_channels];
    let level = simd_level();
    dispatch!(level, simd => conv_gelu_maxpool_simd(
        simd, input, seq_len, in_channels, kernel, kernel_size,
        out_channels, bias, pool, pooled_len, &mut out,
    ));
    (out, pooled_len)
}

#[inline(always)]
fn conv_gelu_maxpool_simd<S: Simd>(
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
        ),
    }
}

#[inline(always)]
fn conv_gelu_maxpool_run<S: Simd, const BLOCK: usize, const POOL: usize>(
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
) {
    debug_assert_eq!(BLOCK % POOL, 0);
    let outs_per_block: usize = BLOCK / POOL;
    let mut accs = vec![0.0f32; BLOCK * out_channels];
    let block_count = pooled_len / outs_per_block;
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
            &mut accs,
        );
        for op in 0..outs_per_block {
            let pooled_idx = tb * outs_per_block + op;
            let dst = &mut out[pooled_idx * out_channels..(pooled_idx + 1) * out_channels];
            let s_first = op * POOL;
            let first = &accs[s_first * out_channels..(s_first + 1) * out_channels];
            for (d_c, a_c) in dst.chunks_exact_mut(4).zip(first.chunks_exact(4)) {
                let v = f32x4::from_slice(simd, a_c);
                gelu_simd(simd, v).store_slice(d_c);
            }
            for s in 1..POOL {
                let acc_idx = s_first + s;
                let acc = &accs[acc_idx * out_channels..(acc_idx + 1) * out_channels];
                for (d_c, a_c) in dst.chunks_exact_mut(4).zip(acc.chunks_exact(4)) {
                    let v = f32x4::from_slice(simd, a_c);
                    let g = gelu_simd(simd, v);
                    let dv = f32x4::from_slice(simd, d_c);
                    g.max(dv).store_slice(d_c);
                }
            }
        }
    }
    let processed = block_count * outs_per_block;
    if processed < pooled_len {
        let mut tail_accs = vec![0.0f32; POOL * out_channels];
        for tp in processed..pooled_len {
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
                &mut tail_accs,
            );
            let dst = &mut out[tp * out_channels..(tp + 1) * out_channels];
            let first = &tail_accs[..out_channels];
            for (d_c, a_c) in dst.chunks_exact_mut(4).zip(first.chunks_exact(4)) {
                let v = f32x4::from_slice(simd, a_c);
                gelu_simd(simd, v).store_slice(d_c);
            }
            for s in 1..POOL {
                let acc = &tail_accs[s * out_channels..(s + 1) * out_channels];
                for (d_c, a_c) in dst.chunks_exact_mut(4).zip(acc.chunks_exact(4)) {
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
fn conv_gelu_global_pool(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
    out_max: &mut [f32],
    out_avg: &mut [f32],
) {
    out_max.fill(f32::NEG_INFINITY);
    out_avg.fill(0.0);
    if seq_len == 0 {
        return;
    }
    let level = simd_level();
    dispatch!(level, simd => conv_gelu_global_pool_simd(
        simd, input, seq_len, in_channels, kernel, kernel_size,
        out_channels, bias, out_max, out_avg,
    ));
}

#[inline(always)]
fn conv_gelu_global_pool_simd<S: Simd>(
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
) {
    const T_BLOCK: usize = 4;
    let mut accs = vec![0.0f32; T_BLOCK * out_channels];
    let block_count = seq_len / T_BLOCK;
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
            &mut accs,
        );
        for s in 0..T_BLOCK {
            let acc = &accs[s * out_channels..(s + 1) * out_channels];
            for ((mx_c, av_c), a_c) in out_max
                .chunks_exact_mut(4)
                .zip(out_avg.chunks_exact_mut(4))
                .zip(acc.chunks_exact(4))
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
    let mut tail_accs = vec![0.0f32; out_channels];
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
            &mut tail_accs,
        );
        for ((mx_c, av_c), a_c) in out_max
            .chunks_exact_mut(4)
            .zip(out_avg.chunks_exact_mut(4))
            .zip(tail_accs.chunks_exact(4))
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

fn dense_forward(input: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
    let in_len = input.len();
    let out_len = out.len();
    debug_assert_eq!(kernel.len(), in_len * out_len);
    out.copy_from_slice(bias);
    for (i, &x) in input.iter().enumerate() {
        let krow = &kernel[i * out_len..(i + 1) * out_len];
        for (o, &w) in out.iter_mut().zip(krow) {
            *o = w.mul_add(x, *o);
        }
    }
}

#[inline(always)]
fn gelu(x: f32) -> f32 {
    // Tanh-form GELU (Hendrycks & Gimpel), matching how the student was trained.
    // 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    0.5 * x * (1.0 + tanh_approx(0.797_884_56 * (x + 0.044_715 * x * x * x)))
}

/// [7/6] Padé approximation of tanh, accurate to ~1e-6 over the clamped range.
/// Replaces the libm `tanh` call so the GELU step auto-vectorizes (no function call).
#[inline(always)]
fn tanh_approx(x: f32) -> f32 {
    let x = x.clamp(-5.0, 5.0);
    let x2 = x * x;
    let num = x * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)));
    let den = 135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0));
    num / den
}

#[inline(always)]
fn gelu_simd<S: Simd>(simd: S, x: f32x4<S>) -> f32x4<S> {
    let half = f32x4::splat(simd, 0.5);
    let one = f32x4::splat(simd, 1.0);
    let c1 = f32x4::splat(simd, 0.797_884_56);
    let c2 = f32x4::splat(simd, 0.044_715);
    let x2 = x * x;
    let x3 = x * x2;
    let inner = c1 * (x + c2 * x3);
    let t = tanh_approx_simd(simd, inner);
    half * x * (one + t)
}

/// SIMD [7/6] Padé tanh approximation matching `tanh_approx`.
#[inline(always)]
fn tanh_approx_simd<S: Simd>(simd: S, x: f32x4<S>) -> f32x4<S> {
    let neg5 = f32x4::splat(simd, -5.0);
    let pos5 = f32x4::splat(simd, 5.0);
    let x = x.max(neg5).min(pos5);
    let x2 = x * x;
    let c28 = f32x4::splat(simd, 28.0);
    let c378 = f32x4::splat(simd, 378.0);
    let c3150 = f32x4::splat(simd, 3150.0);
    let c17325 = f32x4::splat(simd, 17325.0);
    let c62370 = f32x4::splat(simd, 62370.0);
    let c135135 = f32x4::splat(simd, 135135.0);
    let num = x * (c135135 + x2 * (c17325 + x2 * (c378 + x2)));
    let den = c135135 + x2 * (c62370 + x2 * (c3150 + c28 * x2));
    num / den
}

// ============================================================================
// MSQ1 file format readers
// ============================================================================

fn read_int4_dequant(cursor: &mut usize, count: usize, scale: f32) -> Vec<f32> {
    let bytes = count.div_ceil(2);
    let payload = &MODEL_BYTES[*cursor..*cursor + bytes];
    *cursor += bytes;
    let mut out = Vec::with_capacity(count);
    for &packed in payload {
        if out.len() < count {
            let lo = (packed & 0x0f) as i8 - 8;
            out.push(lo as f32 * scale);
        }
        if out.len() < count {
            let hi = (packed >> 4) as i8 - 8;
            out.push(hi as f32 * scale);
        }
    }
    out
}

fn read_ternary_dequant(cursor: &mut usize, count: usize, scale: f32) -> Vec<f32> {
    let bytes = count.div_ceil(4);
    let payload = &MODEL_BYTES[*cursor..*cursor + bytes];
    *cursor += bytes;
    let mut out = Vec::with_capacity(count);
    for &packed in payload {
        for shift in [0u32, 2, 4, 6] {
            if out.len() >= count {
                break;
            }
            let code = (packed >> shift) & 0x03;
            let v = match code {
                0 => -scale,
                2 => scale,
                _ => 0.0, // 1 → 0; codes outside {0,1,2} treated as 0 too
            };
            out.push(v);
        }
    }
    out
}

fn read_f32_array<const N: usize>(cursor: &mut usize) -> [f32; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        let off = *cursor + i * 4;
        out[i] = f32::from_le_bytes(MODEL_BYTES[off..off + 4].try_into().unwrap());
    }
    *cursor += N * 4;
    out
}

fn rfind_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .rposition(|w| w == needle)
        .map(|i| i)
}

fn parse_scales(metadata: &str) -> Vec<f32> {
    let mut scales = Vec::new();
    let mut rest = metadata;
    while let Some(idx) = rest.find(r#""scale":"#) {
        let value_start = idx + r#""scale":"#.len();
        let value_end = rest[value_start..]
            .find([',', '}'])
            .map(|e| value_start + e)
            .expect("scale terminator");
        scales.push(
            rest[value_start..value_end]
                .parse::<f32>()
                .expect("scale parse"),
        );
        rest = &rest[value_end..];
    }
    scales
}

fn parse_tokenizer_version(metadata: &str) -> TokenizerVersion {
    let Some(version) = parse_usize_field(metadata, "tokenizer_version") else {
        // Legacy exported checkpoints predate tokenizer metadata. The shipped
        // wordseq asset was trained with v2.
        return TokenizerVersion::V2;
    };
    match version {
        2 => TokenizerVersion::V2,
        3 => TokenizerVersion::V3,
        4 => TokenizerVersion::V4,
        other => {
            panic!("unsupported wordseq tokenizer_version {other}; runtime supports v2, v3, and v4")
        }
    }
}

fn parse_usize_field(metadata: &str, field: &str) -> Option<usize> {
    let key = format!(r#""{field}""#);
    let after_key = &metadata[metadata.find(&key)? + key.len()..];
    let rest = after_key[after_key.find(':')? + 1..].trim_start();
    let end = rest
        .find(|ch: char| !ch.is_ascii_digit())
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    rest[..end].parse().ok()
}

// ============================================================================
// Byte windowing + v2 word-unit tokenization
// ============================================================================

/// v2 tokenizer ported from `numpy_word_units_apply_v2` in the Python trainer.
///
/// Walks the input bytes, treating any value >= 0x80 with the high bit set
/// as "padding" and stopping at the first such byte (matching the Python
/// `if value >= PADDING_TOKEN: break`). Returns a vector of unit IDs (i32)
/// up to MAX_UNITS.
fn tokenize_v2(bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
    const PRIME: u64 = 2_654_435_761;
    let mut out: Vec<i32> = Vec::with_capacity(MAX_UNITS);
    let mut word: Vec<u8> = Vec::new();
    let mut number: Vec<u8> = Vec::new();
    let mut punct: Vec<u8> = Vec::new();
    let mut at_line_start = true;
    let mut indent_units: u32 = 0;

    let flush_word = |word: &mut Vec<u8>, out: &mut Vec<i32>| {
        if word.is_empty() || out.len() >= MAX_UNITS {
            return;
        }
        let mut h: u64 = 0;
        for &b in word.iter() {
            h = h.wrapping_mul(PRIME).wrapping_add(b as u64) & 0xFFFF_FFFF;
        }
        out.push(((h as u32) & WORD_MASK) as i32);
        word.clear();
    };
    let flush_number = |number: &mut Vec<u8>, out: &mut Vec<i32>| {
        if number.is_empty() || out.len() >= MAX_UNITS {
            return;
        }
        let mut h: u64 = 0;
        for &b in number.iter() {
            h = h.wrapping_mul(PRIME).wrapping_add(b as u64) & 0xFFFF_FFFF;
        }
        out.push((((h as u32) & WORD_MASK) | NUM_FLAG) as i32);
        number.clear();
    };
    let flush_punct = |punct: &mut Vec<u8>, out: &mut Vec<i32>| {
        if punct.is_empty() || out.len() >= MAX_UNITS {
            return;
        }
        let mut h: u64 = 0;
        for &b in punct.iter() {
            h = h.wrapping_mul(PRIME).wrapping_add(b as u64) & 0xFFFF_FFFF;
        }
        out.push((((h as u32) & WORD_MASK) | PUNCT_FLAG) as i32);
        punct.clear();
    };
    let push_indent = |out: &mut Vec<i32>, indent: u32| {
        if indent > 0 && out.len() < MAX_UNITS {
            out.push((indent.min(63) | INDENT_FLAG) as i32);
        }
    };

    for (col, &value) in bytes.iter().enumerate() {
        if padding_mask[col] {
            // Stop at first padding byte.
            break;
        }

        let is_letter =
            (b'a'..=b'z').contains(&value) || (b'A'..=b'Z').contains(&value) || value == b'_';
        let is_digit = (b'0'..=b'9').contains(&value);
        let is_newline = value == b'\n';
        let is_cr = value == b'\r';
        let is_space = value == b' ' || value == b'\t';

        if !is_letter {
            flush_word(&mut word, &mut out);
        }
        if !(is_digit || value == b'.') {
            flush_number(&mut number, &mut out);
        }
        let need_flush_punct =
            is_letter || is_digit || is_space || is_newline || is_cr || value == b'.';
        if need_flush_punct {
            flush_punct(&mut punct, &mut out);
        }

        if out.len() >= MAX_UNITS {
            break;
        }

        if is_letter {
            if at_line_start {
                push_indent(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            word.push(value);
            continue;
        }
        if is_digit || value == b'.' {
            // Lone `.` (not after a digit) is punctuation, not a number start.
            if value == b'.' && number.is_empty() {
                if at_line_start {
                    push_indent(&mut out, indent_units);
                }
                at_line_start = false;
                indent_units = 0;
                punct.push(value);
                continue;
            }
            if at_line_start {
                push_indent(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            number.push(value);
            continue;
        }
        if is_newline {
            if at_line_start {
                push_indent(&mut out, indent_units);
            }
            if out.len() < MAX_UNITS {
                out.push(((b'\n' as u32) | PUNCT_FLAG) as i32);
            }
            at_line_start = true;
            indent_units = 0;
            continue;
        }
        if is_cr {
            continue;
        }
        if at_line_start && is_space {
            indent_units += if value == b' ' { 1 } else { 4 };
            continue;
        }
        if at_line_start {
            push_indent(&mut out, indent_units);
        }
        at_line_start = false;
        indent_units = 0;
        if is_space {
            // Collapse multiple spaces to one PUNCT_FLAG | b' '.
            let space_token = ((b' ' as u32) | PUNCT_FLAG) as i32;
            if out.last() != Some(&space_token) && out.len() < MAX_UNITS {
                out.push(space_token);
            }
            continue;
        }
        // Otherwise: punctuation char, accumulate in punct run.
        punct.push(value);
    }
    // End-of-input: flush any pending word/number/punct.
    flush_word(&mut word, &mut out);
    flush_number(&mut number, &mut out);
    flush_punct(&mut punct, &mut out);

    out
}

fn hash_unit_bytes(bytes: &[u8]) -> u32 {
    const PRIME: u64 = 2_654_435_761;
    let mut h: u64 = 0;
    for &b in bytes {
        h = h.wrapping_mul(PRIME).wrapping_add(b as u64) & 0xFFFF_FFFF;
    }
    h as u32
}

fn flush_hashed(buffer: &mut Vec<u8>, out: &mut Vec<i32>, flag: u32, extra_bits: u32) {
    if !buffer.is_empty() && out.len() < MAX_UNITS {
        out.push(((hash_unit_bytes(buffer) & WORD_MASK) | flag | extra_bits) as i32);
    }
    buffer.clear();
}

fn push_indent_unit(out: &mut Vec<i32>, indent: u32) {
    if indent > 0 && out.len() < MAX_UNITS {
        out.push((indent.min(63) | INDENT_FLAG) as i32);
    }
}

/// v3 tokenizer ported from `numpy_word_units_apply_v3` in the Python trainer.
///
/// This keeps v2's digit/punctuation compression, then case-folds word hashes
/// and isolates unambiguous brackets as stable BRACKET_FLAG tokens.
fn tokenize_v3(bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(MAX_UNITS);
    let mut word: Vec<u8> = Vec::new();
    let mut number: Vec<u8> = Vec::new();
    let mut punct: Vec<u8> = Vec::new();
    let mut at_line_start = true;
    let mut indent_units: u32 = 0;

    for (col, &raw_value) in bytes.iter().enumerate() {
        if padding_mask[col] {
            break;
        }

        let value = raw_value.to_ascii_lowercase();
        let is_letter = value.is_ascii_lowercase() || value == b'_';
        let is_digit = value.is_ascii_digit();
        let is_newline = value == b'\n';
        let is_cr = value == b'\r';
        let is_space = value == b' ' || value == b'\t';
        let is_bracket = matches!(value, b'(' | b')' | b'[' | b']' | b'{' | b'}');

        if !is_letter {
            flush_hashed(&mut word, &mut out, 0, 0);
        }
        if !(is_digit || value == b'.') {
            flush_hashed(&mut number, &mut out, NUM_FLAG, 0);
        }
        let need_flush_punct =
            is_letter || is_digit || is_space || is_newline || is_cr || is_bracket || value == b'.';
        if need_flush_punct {
            flush_hashed(&mut punct, &mut out, PUNCT_FLAG, 0);
        }

        if out.len() >= MAX_UNITS {
            break;
        }

        if is_letter {
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            word.push(value);
            continue;
        }
        if is_digit || value == b'.' {
            if value == b'.' && number.is_empty() {
                if at_line_start {
                    push_indent_unit(&mut out, indent_units);
                }
                at_line_start = false;
                indent_units = 0;
                punct.push(value);
                continue;
            }
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            number.push(value);
            continue;
        }
        if is_newline {
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            if out.len() < MAX_UNITS {
                out.push(((b'\n' as u32) | PUNCT_FLAG) as i32);
            }
            at_line_start = true;
            indent_units = 0;
            continue;
        }
        if is_cr {
            continue;
        }
        if at_line_start && is_space {
            indent_units += if value == b' ' { 1 } else { 4 };
            continue;
        }
        if at_line_start {
            push_indent_unit(&mut out, indent_units);
        }
        at_line_start = false;
        indent_units = 0;
        if is_space {
            let space_token = ((b' ' as u32) | PUNCT_FLAG) as i32;
            if out.last() != Some(&space_token) && out.len() < MAX_UNITS {
                out.push(space_token);
            }
            continue;
        }
        if is_bracket {
            if out.len() < MAX_UNITS {
                out.push(((value as u32) | BRACKET_FLAG) as i32);
            }
            continue;
        }
        punct.push(value);
    }

    flush_hashed(&mut word, &mut out, 0, 0);
    flush_hashed(&mut number, &mut out, NUM_FLAG, 0);
    flush_hashed(&mut punct, &mut out, PUNCT_FLAG, 0);

    out
}

/// v4 tokenizer ported from `numpy_word_units_apply_v4` in the Python trainer.
///
/// This keeps v2's digit/punctuation compression, then case-folds word hashes,
/// isolates brackets, marks CamelCase-ish identifiers with STYLE_BIT, and
/// collapses double-quoted string literals to one STRING_FLAG token.
fn tokenize_v4(bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(MAX_UNITS);
    let mut word: Vec<u8> = Vec::new();
    let mut word_had_upper = false;
    let mut number: Vec<u8> = Vec::new();
    let mut punct: Vec<u8> = Vec::new();
    let mut at_line_start = true;
    let mut indent_units: u32 = 0;
    let mut in_string = false;
    let mut string_escape = false;

    for (col, &raw_value) in bytes.iter().enumerate() {
        if padding_mask[col] {
            break;
        }

        if in_string {
            if string_escape {
                string_escape = false;
            } else if raw_value == b'\\' {
                string_escape = true;
            } else if raw_value == b'"' {
                if out.len() < MAX_UNITS {
                    out.push((STRING_FLAG | b'"' as u32) as i32);
                }
                in_string = false;
            }
            continue;
        }

        let saw_upper_now = raw_value.is_ascii_uppercase();
        let value = if saw_upper_now {
            raw_value.to_ascii_lowercase()
        } else {
            raw_value
        };
        let is_letter = value.is_ascii_lowercase() || value == b'_';
        let is_digit = value.is_ascii_digit();
        let is_newline = value == b'\n';
        let is_cr = value == b'\r';
        let is_space = value == b' ' || value == b'\t';
        let is_bracket = matches!(value, b'(' | b')' | b'[' | b']' | b'{' | b'}');
        let is_dquote = value == b'"';

        if !is_letter && !word.is_empty() {
            let style = if word_had_upper { STYLE_BIT } else { 0 };
            flush_hashed(&mut word, &mut out, 0, style);
            word_had_upper = false;
        }
        if !(is_digit || value == b'.') && !number.is_empty() {
            flush_hashed(&mut number, &mut out, NUM_FLAG, 0);
        }
        let need_flush_punct = is_letter
            || is_digit
            || is_space
            || is_newline
            || is_cr
            || is_bracket
            || is_dquote
            || value == b'.';
        if need_flush_punct && !punct.is_empty() {
            flush_hashed(&mut punct, &mut out, PUNCT_FLAG, 0);
        }

        if out.len() >= MAX_UNITS {
            break;
        }

        if is_letter {
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            if saw_upper_now {
                word_had_upper = true;
            }
            word.push(value);
            continue;
        }
        if is_digit || value == b'.' {
            if value == b'.' && number.is_empty() {
                if at_line_start {
                    push_indent_unit(&mut out, indent_units);
                }
                at_line_start = false;
                indent_units = 0;
                punct.push(value);
                continue;
            }
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            number.push(value);
            continue;
        }
        if is_newline {
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            if out.len() < MAX_UNITS {
                out.push(((b'\n' as u32) | PUNCT_FLAG) as i32);
            }
            at_line_start = true;
            indent_units = 0;
            continue;
        }
        if is_cr {
            continue;
        }
        if at_line_start && is_space {
            indent_units += if value == b' ' { 1 } else { 4 };
            continue;
        }
        if at_line_start {
            push_indent_unit(&mut out, indent_units);
        }
        at_line_start = false;
        indent_units = 0;
        if is_space {
            let space_token = ((b' ' as u32) | PUNCT_FLAG) as i32;
            if out.last() != Some(&space_token) && out.len() < MAX_UNITS {
                out.push(space_token);
            }
            continue;
        }
        if is_bracket {
            if out.len() < MAX_UNITS {
                out.push((value as u32 | BRACKET_FLAG) as i32);
            }
            continue;
        }
        if is_dquote {
            in_string = true;
            string_escape = false;
            continue;
        }
        punct.push(value);
    }

    if in_string && out.len() < MAX_UNITS {
        out.push((STRING_FLAG | b'"' as u32) as i32);
    }
    if !word.is_empty() {
        let style = if word_had_upper { STYLE_BIT } else { 0 };
        flush_hashed(&mut word, &mut out, 0, style);
    }
    if !number.is_empty() {
        flush_hashed(&mut number, &mut out, NUM_FLAG, 0);
    }
    if !punct.is_empty() {
        flush_hashed(&mut punct, &mut out, PUNCT_FLAG, 0);
    }

    out
}

fn trim_start_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(bytes.len());
    &bytes[start..]
}

fn trim_end_ascii(bytes: &[u8]) -> &[u8] {
    let end = bytes
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map(|i| i + 1)
        .unwrap_or(0);
    &bytes[..end]
}

/// Build the (begin + end) byte window with a parallel padding mask.
/// Mirrors `magika_features` in the trainer.
fn build_window(source: &[u8]) -> Option<(Vec<u8>, Vec<bool>)> {
    if source.is_empty() {
        return None;
    }
    let block = source.len().min(MAGIKA_BLOCK_SIZE);
    let stripped_beg_full = trim_start_ascii(&source[..block]);
    if stripped_beg_full.len() < 8 {
        return None;
    }
    let stripped_end_full = trim_end_ascii(&source[source.len() - block..]);

    let beg_len = stripped_beg_full.len().min(MAGIKA_BEG_SIZE);
    let end_len = stripped_end_full.len().min(MAGIKA_END_SIZE);
    let total = MAGIKA_BEG_SIZE + MAGIKA_END_SIZE;
    let mut buf = vec![0u8; total];
    let mut pad = vec![false; total];

    buf[..beg_len].copy_from_slice(&stripped_beg_full[..beg_len]);
    for slot in pad.iter_mut().take(MAGIKA_BEG_SIZE).skip(beg_len) {
        *slot = true;
    }
    let end_start = MAGIKA_BEG_SIZE + (MAGIKA_END_SIZE - end_len);
    for slot in pad.iter_mut().take(end_start).skip(MAGIKA_BEG_SIZE) {
        *slot = true;
    }
    let end_src = &stripped_end_full[stripped_end_full.len() - end_len..];
    buf[end_start..end_start + end_len].copy_from_slice(end_src);
    Some((buf, pad))
}

// ============================================================================
// Public API
// ============================================================================

/// Detect the source language for a source string.
///
/// Returns `Some(Language)` when the model's top class clearly outranks the
/// runner-up, otherwise `None`.
pub fn detect(source: &str) -> Option<Language> {
    let (bytes, pad) = build_window(source.as_bytes())?;
    let model = Model::get();
    let units = model.tokenize_units(&bytes, &pad);
    let logits = model.logits_for_runtime_units(&units);
    let class_index = reliable_class_from_logits(&logits)?;
    Some(CLASS_LANGUAGES[class_index])
}

fn reliable_class_from_logits(logits: &[f32; CLASSES]) -> Option<usize> {
    let (class_index, top, runner_up) = top_two_logits(logits)?;
    if top - runner_up >= RELIABLE_LOGIT_MARGIN {
        return Some(class_index);
    }
    let mut sum = 0.0;
    let mut sum_squares = 0.0;
    for &logit in logits {
        let value = (logit - top).exp();
        sum += value;
        sum_squares += value * value;
    }
    let probability = 1.0 / sum;
    let mean = 1.0 / CLASSES as f32;
    let variance = ((sum_squares / (sum * sum)) - mean).max(0.0) / (CLASSES - 1) as f32;
    (probability > mean + 2.0 * variance.sqrt()).then_some(class_index)
}

fn top_two_logits(logits: &[f32; CLASSES]) -> Option<(usize, f32, f32)> {
    let mut top = f32::NEG_INFINITY;
    let mut runner_up = f32::NEG_INFINITY;
    let mut class_index = 0;
    for (index, &logit) in logits.iter().enumerate() {
        if logit >= top {
            runner_up = top;
            top = logit;
            class_index = index;
        } else if logit > runner_up {
            runner_up = logit;
        }
    }
    top.is_finite().then_some((class_index, top, runner_up))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_embedded_model() {
        let model = Model::get();
        assert_eq!(model.tokenizer_version, TokenizerVersion::V3);
        assert_eq!(model.embedding.len(), BINS * EMBED);
        assert_eq!(model.output_kernel.len(), DENSE * CLASSES);
    }

    #[test]
    fn tokenizer_version_defaults_legacy_checkpoints_to_v2() {
        assert_eq!(
            parse_tokenizer_version(r#"{"bits":4}"#),
            TokenizerVersion::V2
        );
        assert_eq!(
            parse_tokenizer_version(r#"{"bits":4,"tokenizer_version":3}"#),
            TokenizerVersion::V3
        );
        assert_eq!(
            parse_tokenizer_version(r#"{"bits":4,"tokenizer_version":4}"#),
            TokenizerVersion::V4
        );
    }

    #[test]
    fn tokenizer_v3_casefolds_and_isolates_brackets() {
        let source = b"Foo(foo)\n";
        let pad = vec![false; source.len()];
        let units = tokenize_v3(source, &pad);

        assert_eq!(units[0] as u32, hash_unit_bytes(b"foo") & WORD_MASK);
        assert!(units.contains(&((BRACKET_FLAG | b'(' as u32) as i32)));
        assert!(units.contains(&((BRACKET_FLAG | b')' as u32) as i32)));
    }

    #[test]
    fn tokenizer_v4_casefolds_brackets_and_strings() {
        let source = b"Foo { \"Bar\\\"Baz\" }\n";
        let pad = vec![false; source.len()];
        let units = tokenize_v4(source, &pad);
        let expected = [
            16_986_836,
            268_435_488,
            1_342_177_403,
            268_435_488,
            1_879_048_226,
            268_435_488,
            1_342_177_405,
            268_435_466,
        ];

        assert_eq!(
            units[0] as u32,
            (hash_unit_bytes(b"foo") & WORD_MASK) | STYLE_BIT
        );
        assert_eq!(&units[..expected.len()], &expected);
        assert!(units.contains(&((BRACKET_FLAG | b'{' as u32) as i32)));
        assert!(units.contains(&((STRING_FLAG | b'"' as u32) as i32)));
        assert!(units.contains(&((BRACKET_FLAG | b'}' as u32) as i32)));
    }

    #[test]
    fn detects_rust_from_source() {
        let lang = detect("use std::fmt;\nfn main() { println!(\"hi\"); }");
        assert_eq!(lang, Some(Language::Rust));
    }

    #[test]
    fn detects_python_from_source() {
        let lang = detect(
            "import os\n\ndef main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n",
        );
        assert_eq!(lang, Some(Language::Python));
    }

    #[test]
    fn detects_javascript_from_source() {
        let lang = detect(
            "const greet = (name) => { console.log(`Hello, ${name}!`); };\ngreet('world');\n",
        );
        assert_eq!(lang, Some(Language::JavaScript));
    }

    #[test]
    fn runtime_inference_pads_short_sources_to_eval_shape() {
        let source = "use std::fmt;\nfn main() { println!(\"hi\"); }\n";
        let (bytes, pad) = build_window(source.as_bytes()).unwrap();
        let model = Model::get();
        let units = model.tokenize_units(&bytes, &pad);
        assert!(units.len() < MAX_UNITS);

        let mut padded = units.clone();
        padded.resize(MAX_UNITS, -1);

        let runtime_logits = model.logits_for_runtime_units(&units);
        let eval_shape_logits = model.logits(&padded, MAX_UNITS);

        for (runtime, eval_shape) in runtime_logits.iter().zip(eval_shape_logits) {
            assert_eq!(*runtime, eval_shape);
        }
    }

    #[test]
    fn empty_input_returns_none() {
        assert_eq!(detect(""), None);
    }

    #[test]
    fn very_short_input_returns_none() {
        // < 8 non-whitespace bytes
        assert_eq!(detect("hi"), None);
    }
}
