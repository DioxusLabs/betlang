//! Inference for the wordseq Magika source-language student.
//!
//! Loads `assets/magika/source-student-q4.bin` (49.92 KB MSQ1 export) and
//! runs a forward pass: byte-window tokenization → v2 word-unit tokenization
//! → HashEmbedding lookup (K=3) → 3 conv stages with max-pool → global
//! max+avg pool → 2 dense layers → 67-class softmax logits.
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
//! - QDense 96→67 (4-bit)
//!
//! Implementation is scalar Rust. SIMD specializations have been removed
//! relative to the prior conv-xwide-hash-hidden model; can be re-added.

use crate::language::{CLASS_LANGUAGES, Language};
use std::sync::OnceLock;

static MODEL_BYTES: &[u8] = include_bytes!("../assets/magika/source-student-q4.bin");

const MODEL_MAGIC: [u8; 8] = [0x4d, 0x53, 0x51, 0x31, 0x01, 0x00, 0x00, 0x00];

const MAGIKA_BEG_SIZE: usize = 1_024;
const MAGIKA_END_SIZE: usize = 1_024;
const MAGIKA_BLOCK_SIZE: usize = 4_096;

// Wordseq architecture constants. Must match what the model was trained with.
const BINS: usize = 1_024;
const HASH_COUNT: usize = 3;
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
pub(crate) const CLASSES: usize = 67;
const RELIABLE_LOGIT_MARGIN: f32 = 3.0;

// v2 tokenizer flag bits. Must match `_PUNCT_FLAG`/etc. in the Python trainer.
const WORD_MASK: u32 = 0x00FF_FFFF;
const PUNCT_FLAG: u32 = 0x1000_0000;
const INDENT_FLAG: u32 = 0x2000_0000;
const NUM_FLAG: u32 = 0x4000_0000;

#[derive(Debug)]
struct Model {
    /// 1024 × 24 dequantized embedding rows.
    embedding: Vec<f32>,
    /// (CONV0_KERNEL, EMBED, CONV0) flattened.
    conv0_kernel: Vec<f32>,
    conv0_bias: [f32; CONV0],
    /// (CONV1_KERNEL, CONV0, CONV1) flattened.
    conv1_kernel: Vec<f32>,
    conv1_bias: [f32; CONV1],
    /// (CONV2_KERNEL, CONV1, CONV2) flattened.
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

        // q_output: weights [(96, 67)] int4, bias [(67,)] f32
        let output_kernel = read_int4_dequant(&mut cur, DENSE * CLASSES, scales[5]);
        let output_bias = read_f32_array::<CLASSES>(&mut cur);

        // Trailing 4-byte LE metadata length (not used for indexing here, just sanity).
        assert_eq!(cur + 4, metadata_start, "unexpected payload length");
        let metadata_len = u32::from_le_bytes(MODEL_BYTES[cur..cur + 4].try_into().unwrap());
        assert_eq!(metadata_len as usize, MODEL_BYTES.len() - metadata_start);

        Self {
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
        // 1) HashEmbedding: for each position, hash unit_id into K=3 bins,
        //    sum the K embedding rows. Pad positions (unit_id < 0) → zero vector.
        let t = len.min(MAX_UNITS);
        let mut embed = vec![0.0f32; t * EMBED];
        for pos in 0..t {
            let id = units[pos];
            if id < 0 {
                continue;
            }
            let unsigned = id as u32;
            let dst = &mut embed[pos * EMBED..(pos + 1) * EMBED];
            for h in 0..HASH_COUNT {
                let bin = hash_bin(unsigned, h);
                let row = &self.embedding[bin * EMBED..(bin + 1) * EMBED];
                for (d, w) in dst.iter_mut().zip(row) {
                    *d += *w;
                }
            }
        }
        // 2) GELU
        for v in &mut embed {
            *v = gelu(*v);
        }

        // 3) QConv1D k=7 24→64 with SAME padding; activation GELU
        let conv0 = conv1d_same(&embed, t, EMBED, &self.conv0_kernel, CONV0_KERNEL, CONV0, &self.conv0_bias);
        let conv0_act = gelu_vec(conv0);

        // 4) MaxPool(4) along sequence dim
        let (pool0, t1) = max_pool_1d(&conv0_act, t, CONV0, CONV0_POOL);

        // 5) QConv1D k=5 64→128 with SAME padding; activation GELU
        let conv1 = conv1d_same(&pool0, t1, CONV0, &self.conv1_kernel, CONV1_KERNEL, CONV1, &self.conv1_bias);
        let conv1_act = gelu_vec(conv1);

        // 6) MaxPool(2)
        let (pool1, t2) = max_pool_1d(&conv1_act, t1, CONV1, CONV1_POOL);

        // 7) QConv1D k=3 128→128 with SAME padding; activation GELU
        let conv2 = conv1d_same(&pool1, t2, CONV1, &self.conv2_kernel, CONV2_KERNEL, CONV2, &self.conv2_bias);
        let conv2_act = gelu_vec(conv2);

        // 8) GlobalMaxPool ⊕ GlobalAvgPool → 256-dim
        let mut pooled = [0.0f32; POOLED];
        global_max_pool(&conv2_act, t2, CONV2, &mut pooled[..CONV2]);
        global_avg_pool(&conv2_act, t2, CONV2, &mut pooled[CONV2..]);

        // 9) QDense 256→96 + GELU
        let mut dense0_out = [0.0f32; DENSE];
        dense_forward(&pooled, &self.dense0_kernel, &self.dense0_bias, &mut dense0_out);
        for v in &mut dense0_out {
            *v = gelu(*v);
        }

        // 10) QDense 96→67 → logits
        let mut logits = [0.0f32; CLASSES];
        dense_forward(&dense0_out, &self.output_kernel, &self.output_bias, &mut logits);
        logits
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

/// SAME-padded 1D conv: out[t,c_out] = bias[c_out] + sum_k sum_c kernel[k,c,c_out] * x[t+k-pad, c]
/// where pad = (kernel_size - 1) / 2.
fn conv1d_same(
    input: &[f32],
    seq_len: usize,
    in_channels: usize,
    kernel: &[f32],
    kernel_size: usize,
    out_channels: usize,
    bias: &[f32],
) -> Vec<f32> {
    let pad = (kernel_size - 1) / 2;
    let mut out = vec![0.0f32; seq_len * out_channels];
    for t in 0..seq_len {
        let out_row = &mut out[t * out_channels..(t + 1) * out_channels];
        out_row.copy_from_slice(bias);
        for k in 0..kernel_size {
            let src_t = (t + k).wrapping_sub(pad);
            // SAME padding: zero out-of-bounds positions.
            if src_t >= seq_len {
                continue;
            }
            let src = &input[src_t * in_channels..(src_t + 1) * in_channels];
            // Kernel layout: [k, in_c, out_c] → row offset (k * in_channels + in_c) * out_channels
            for in_c in 0..in_channels {
                let x = src[in_c];
                if x == 0.0 {
                    continue;
                }
                let krow_off = (k * in_channels + in_c) * out_channels;
                let krow = &kernel[krow_off..krow_off + out_channels];
                for (o, &w) in out_row.iter_mut().zip(krow) {
                    *o += x * w;
                }
            }
        }
    }
    out
}

fn max_pool_1d(input: &[f32], seq_len: usize, channels: usize, pool: usize) -> (Vec<f32>, usize) {
    let new_len = seq_len / pool;
    let mut out = vec![f32::NEG_INFINITY; new_len * channels];
    for t in 0..new_len {
        let dst = &mut out[t * channels..(t + 1) * channels];
        for k in 0..pool {
            let src_t = t * pool + k;
            let src = &input[src_t * channels..(src_t + 1) * channels];
            for (d, &s) in dst.iter_mut().zip(src) {
                if s > *d {
                    *d = s;
                }
            }
        }
    }
    (out, new_len)
}

fn global_max_pool(input: &[f32], seq_len: usize, channels: usize, out: &mut [f32]) {
    out.fill(f32::NEG_INFINITY);
    for t in 0..seq_len {
        let src = &input[t * channels..(t + 1) * channels];
        for (d, &s) in out.iter_mut().zip(src) {
            if s > *d {
                *d = s;
            }
        }
    }
}

fn global_avg_pool(input: &[f32], seq_len: usize, channels: usize, out: &mut [f32]) {
    out.fill(0.0);
    if seq_len == 0 {
        return;
    }
    for t in 0..seq_len {
        let src = &input[t * channels..(t + 1) * channels];
        for (d, &s) in out.iter_mut().zip(src) {
            *d += s;
        }
    }
    let inv = 1.0 / seq_len as f32;
    for d in out.iter_mut() {
        *d *= inv;
    }
}

fn dense_forward(input: &[f32], kernel: &[f32], bias: &[f32], out: &mut [f32]) {
    let in_len = input.len();
    let out_len = out.len();
    debug_assert_eq!(kernel.len(), in_len * out_len);
    out.copy_from_slice(bias);
    for i in 0..in_len {
        let x = input[i];
        if x == 0.0 {
            continue;
        }
        let krow = &kernel[i * out_len..(i + 1) * out_len];
        for (o, &w) in out.iter_mut().zip(krow) {
            *o += x * w;
        }
    }
}

fn gelu(x: f32) -> f32 {
    // Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))).
    // libm not available without dep; use the standard approximation from Hendrycks & Gimpel.
    0.5 * x * (1.0 + tanh_approx(0.797_884_56 * (x + 0.044_715 * x * x * x)))
}

fn gelu_vec(mut v: Vec<f32>) -> Vec<f32> {
    for x in v.iter_mut() {
        *x = gelu(*x);
    }
    v
}

#[inline]
fn tanh_approx(x: f32) -> f32 {
    // Use std's tanh — it's exact-ish via libm.
    x.tanh()
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
    haystack.windows(needle.len()).rposition(|w| w == needle).map(|i| i)
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
        scales.push(rest[value_start..value_end].parse::<f32>().expect("scale parse"));
        rest = &rest[value_end..];
    }
    scales
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

        let is_letter = (b'a'..=b'z').contains(&value)
            || (b'A'..=b'Z').contains(&value)
            || value == b'_';
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
        let need_flush_punct = is_letter
            || is_digit
            || is_space
            || is_newline
            || is_cr
            || value == b'.';
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

fn trim_start_ascii(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace()).unwrap_or(bytes.len());
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
    let units = tokenize_v2(&bytes, &pad);
    let logits = Model::get().logits(&units, units.len());
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
        assert_eq!(model.embedding.len(), BINS * EMBED);
        assert_eq!(model.output_kernel.len(), DENSE * CLASSES);
    }

    #[test]
    fn detects_rust_from_source() {
        let lang = detect("use std::fmt;\nfn main() { println!(\"hi\"); }");
        assert_eq!(lang, Some(Language::Rust));
    }

    #[test]
    fn detects_python_from_source() {
        let lang = detect("import os\n\ndef main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n");
        assert_eq!(lang, Some(Language::Python));
    }

    #[test]
    fn detects_javascript_from_source() {
        let lang = detect("const greet = (name) => { console.log(`Hello, ${name}!`); };\ngreet('world');\n");
        assert_eq!(lang, Some(Language::JavaScript));
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
