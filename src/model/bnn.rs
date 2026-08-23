//! Binary CNN inference over the Shannon-coded byte window.
//!
//! The raw (begin + end) byte window is entropy-coded with a canonical
//! Shannon code learned from the training byte distribution, producing two
//! fixed 16384-bit planes (code bits + codeword-boundary markers). Inference
//! is entirely bitwise/integer: XNOR + popcount convolutions (optionally a
//! sum of binary bases combined with fixed-point integer scales), folded
//! integer thresholds, OR-pooling, segmented popcount head counts, and an
//! integer classifier (per-class f32 scale). Multiple models may be stored
//! in one artifact; their logits are summed (ensemble).

use super::constants::CLASSES;
use std::sync::OnceLock;

pub(crate) static BNN_BYTES: &[u8] = include_bytes!("../../assets/bnn/source-bnn2.bin");

const MAGIC: u32 = 0x324E_4242; // "BBN2"
const VERSION: u32 = 2;
pub(crate) const BITS: usize = 16_384;
const BIT_WORDS: usize = BITS / 64;

struct Layer {
    m: usize,
    cin: usize,
    k: usize,
    stride: usize,
    pad: usize,
    cout: usize,
    /// Per basis: `cout` rows of `(cin * k).div_ceil(64)` words
    /// (bit index within a row = tap * cin + ci).
    weights: Vec<Vec<u64>>,
    /// Per basis: fixed-point (x4096) per-out-channel scales.
    aq: Vec<Vec<i32>>,
    /// 1 => fires when z_q >= thr, 0 => fires when z_q <= thr.
    sign: Vec<u8>,
    thr: Vec<i32>,
}

struct Model {
    c: usize,
    segs: usize,
    stem: Layer,
    conv1: Layer,
    conv2: Layer,
    head_weights: Vec<i16>,
    head_scale: Vec<f32>,
    head_bias: Vec<f32>,
}

pub(crate) struct Bnn {
    /// Canonical Shannon codes: (code value, bit length) per byte.
    codes: [(u32, u32); 256],
    models: Vec<Model>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    cur: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.bytes[self.cur..self.cur + 4].try_into().unwrap());
        self.cur += 4;
        v
    }

    fn u8s(&mut self, n: usize) -> &'a [u8] {
        let s = &self.bytes[self.cur..self.cur + n];
        self.cur += n;
        s
    }

    fn i32s(&mut self, n: usize) -> Vec<i32> {
        (0..n).map(|_| self.u32() as i32).collect()
    }

    fn i16s(&mut self, n: usize) -> Vec<i16> {
        (0..n)
            .map(|_| {
                let v = i16::from_le_bytes(self.bytes[self.cur..self.cur + 2].try_into().unwrap());
                self.cur += 2;
                v
            })
            .collect()
    }

    fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| f32::from_bits(self.u32())).collect()
    }

    /// Reads `rows` bit-rows of `nbits` (u32-packed, LSB-first) into u64 rows.
    fn bit_rows(&mut self, rows: usize, nbits: usize) -> Vec<u64> {
        let words32 = nbits.div_ceil(32);
        let words64 = nbits.div_ceil(64);
        let mut out = vec![0u64; rows * words64];
        for r in 0..rows {
            for w in 0..words32 {
                let v = u64::from(self.u32());
                out[r * words64 + w / 2] |= v << ((w % 2) * 32);
            }
        }
        out
    }
}

/// Rebuild canonical codes from code lengths: codes are assigned in
/// (length, byte) order, left-justified within the max length.
fn canonical_codes(lengths: &[u8]) -> [(u32, u32); 256] {
    let mut order: Vec<u8> = (0..=255u8).collect();
    order.sort_by_key(|&b| (lengths[b as usize], b));
    let mut codes = [(0u32, 0u32); 256];
    let mut code = 0u32;
    let mut prev_len = 0u32;
    for &b in &order {
        let len = u32::from(lengths[b as usize]);
        code <<= len - prev_len;
        codes[b as usize] = (code, len);
        code += 1;
        prev_len = len;
    }
    codes
}

fn read_layer(r: &mut Reader) -> Layer {
    let m = r.u32() as usize;
    let cin = r.u32() as usize;
    let k = r.u32() as usize;
    // cout is implied by the enclosing model's C; stored thresholds follow.
    // Weight rows and aq come per basis.
    // The caller patches stride/pad/cout.
    let mut layer = Layer {
        m,
        cin,
        k,
        stride: 1,
        pad: k / 2,
        cout: 0,
        weights: Vec::with_capacity(m),
        aq: Vec::with_capacity(m),
        sign: Vec::new(),
        thr: Vec::new(),
    };
    layer
        .weights
        .reserve_exact(m.saturating_sub(layer.weights.capacity()));
    layer
}

impl Bnn {
    fn load() -> Self {
        let mut r = Reader {
            bytes: BNN_BYTES,
            cur: 0,
        };
        assert_eq!(r.u32(), MAGIC, "bad BBN2 magic");
        assert_eq!(r.u32(), VERSION, "bad BBN2 version");
        assert_eq!(r.u32() as usize, BITS);
        assert_eq!(r.u32() as usize, CLASSES);
        let n_models = r.u32() as usize;

        let lengths: [u8; 256] = r.u8s(256).try_into().unwrap();
        let codes = canonical_codes(&lengths);

        let mut models = Vec::with_capacity(n_models);
        for _ in 0..n_models {
            let c = r.u32() as usize;
            let stem_k = r.u32() as usize;
            let stem_stride = r.u32() as usize;
            let segs = r.u32() as usize;
            let mut layers = Vec::with_capacity(3);
            for li in 0..3 {
                let mut layer = read_layer(&mut r);
                layer.cout = c;
                if li == 0 {
                    assert_eq!(layer.k, stem_k);
                    layer.stride = stem_stride;
                    layer.pad = stem_k / 2;
                }
                for _ in 0..layer.m {
                    layer.weights.push(r.bit_rows(c, layer.cin * layer.k));
                    layer.aq.push(r.i32s(c));
                }
                layer.sign = r.u8s(c).to_vec();
                layer.thr = r.i32s(c);
                layers.push(layer);
            }
            let conv2 = layers.pop().unwrap();
            let conv1 = layers.pop().unwrap();
            let stem = layers.pop().unwrap();
            let head_weights = r.i16s(CLASSES * segs * c);
            let head_scale = r.f32s(CLASSES);
            let head_bias = r.f32s(CLASSES);
            models.push(Model {
                c,
                segs,
                stem,
                conv1,
                conv2,
                head_weights,
                head_scale,
                head_bias,
            });
        }
        assert_eq!(r.cur, BNN_BYTES.len(), "unexpected BBN2 payload length");
        Self { codes, models }
    }

    pub(crate) fn get() -> &'static Self {
        static MODEL: OnceLock<Bnn> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    /// Shannon-encode window bytes into two fixed-length bitplanes
    /// (code bits, codeword-boundary markers), truncated at the last
    /// codeword that fits and zero padded.
    pub(crate) fn encode_planes(&self, window: &[u8]) -> ([u64; BIT_WORDS], [u64; BIT_WORDS]) {
        let mut bits = [0u64; BIT_WORDS];
        let mut bounds = [0u64; BIT_WORDS];
        let mut pos = 0usize;
        for &b in window {
            let (code, len) = self.codes[b as usize];
            if pos + len as usize > BITS {
                break;
            }
            bounds[pos / 64] |= 1u64 << (pos % 64);
            // MSB of the codeword first, matching the trainer.
            for i in (0..len).rev() {
                if (code >> i) & 1 == 1 {
                    bits[pos / 64] |= 1u64 << (pos % 64);
                }
                pos += 1;
            }
        }
        (bits, bounds)
    }

    pub(crate) fn logits(&self, planes: &([u64; BIT_WORDS], [u64; BIT_WORDS])) -> [f32; CLASSES] {
        let mut logits = [0.0f32; CLASSES];
        for model in &self.models {
            let ml = model.logits(planes);
            for (acc, l) in logits.iter_mut().zip(ml.iter()) {
                *acc += l;
            }
        }
        let n = self.models.len().max(1) as f32;
        for logit in &mut logits {
            *logit /= n;
        }
        logits
    }
}

impl Model {
    fn logits(&self, planes: &([u64; BIT_WORDS], [u64; BIT_WORDS])) -> [f32; CLASSES] {
        let c = self.c;
        let cw = c.div_ceil(64);
        // Stem over the two input planes -> position-major rows of C bits,
        // OR-pooled by 4.
        let l0 = (BITS + 2 * self.stem.pad - self.stem.k) / self.stem.stride + 1;
        let l1 = (l0 / 4).max(1);
        let mut a1 = vec![0u64; l1 * cw];
        for t in 0..l0 {
            let start = (t * self.stem.stride) as isize - self.stem.pad as isize;
            let (x, mask, n_valid) = stem_window(planes, start, self.stem.k);
            let slot = t / 4;
            if slot >= l1 {
                break;
            }
            for ch in 0..c {
                let mut zq = 0i64;
                for j in 0..self.stem.m {
                    let w = self.stem.weights[j][ch];
                    let m = (!(x ^ w) & mask).count_ones() as i64;
                    let z = 2 * m - n_valid as i64;
                    zq += i64::from(self.stem.aq[j][ch]) * z;
                }
                let fire = if self.stem.sign[ch] == 1 {
                    zq >= i64::from(self.stem.thr[ch])
                } else {
                    zq <= i64::from(self.stem.thr[ch])
                };
                if fire {
                    a1[slot * cw + ch / 64] |= 1u64 << (ch % 64);
                }
            }
        }

        let l2 = l1 / 4;
        let a2 = conv3(&a1, l1, &self.conv1, 4, c);
        let a3 = conv3(&a2, l2, &self.conv2, 1, c);

        // Segmented popcount head.
        let segs = self.segs;
        let seg = l2 / segs;
        let mut counts = vec![0i32; segs * c];
        for t in 0..seg * segs {
            let row = &a3[t * cw..(t + 1) * cw];
            let base = (t / seg) * c;
            for (wi, &word) in row.iter().enumerate() {
                let mut bitsw = word;
                while bitsw != 0 {
                    let b = bitsw.trailing_zeros() as usize;
                    counts[base + wi * 64 + b] += 1;
                    bitsw &= bitsw - 1;
                }
            }
        }

        let mut logits = [0.0f32; CLASSES];
        let width = segs * c;
        for (k, logit) in logits.iter_mut().enumerate() {
            let row = &self.head_weights[k * width..(k + 1) * width];
            let mut acc = 0i64;
            for (w, &f) in row.iter().zip(counts.iter()) {
                acc += i64::from(*w) * i64::from(f);
            }
            *logit = acc as f32 * self.head_scale[k] + self.head_bias[k];
        }
        logits
    }
}

/// Gather a stem window over the two input planes: bit index tap*2 + plane.
/// Returns (bits, valid mask, n_valid).
fn stem_window(
    planes: &([u64; BIT_WORDS], [u64; BIT_WORDS]),
    start: isize,
    k: usize,
) -> (u64, u64, usize) {
    let mut x = 0u64;
    let mut mask = 0u64;
    let mut n_valid = 0usize;
    for tap in 0..k {
        let p = start + tap as isize;
        if p >= 0 && (p as usize) < BITS {
            let p = p as usize;
            n_valid += 2;
            mask |= 0b11 << (tap * 2);
            if planes.0[p / 64] >> (p % 64) & 1 == 1 {
                x |= 1u64 << (tap * 2);
            }
            if planes.1[p / 64] >> (p % 64) & 1 == 1 {
                x |= 1u64 << (tap * 2 + 1);
            }
        }
    }
    (x, mask, n_valid)
}

/// k=3 (pad=1) multi-basis binary conv + threshold + OR-pool over
/// position-major rows of `c` bits. Output: len/pool rows.
fn conv3(rows: &[u64], len: usize, layer: &Layer, pool: usize, c: usize) -> Vec<u64> {
    let cw = c.div_ceil(64);
    let kw = 3 * cw;
    let out_len = len / pool;
    let mut out = vec![0u64; out_len * cw];
    let zero = vec![0u64; cw];
    for t in 0..len {
        let taps: [&[u64]; 3] = [
            if t == 0 {
                &zero
            } else {
                &rows[(t - 1) * cw..t * cw]
            },
            &rows[t * cw..(t + 1) * cw],
            if t + 1 >= len {
                &zero
            } else {
                &rows[(t + 1) * cw..(t + 2) * cw]
            },
        ];
        let edge = [t == 0, false, t + 1 >= len];
        let n_valid = (3 - edge.iter().filter(|&&e| e).count()) * c;
        let slot = t / pool;
        if slot >= out_len {
            break;
        }
        for ch in 0..c {
            let mut zq = 0i64;
            for j in 0..layer.m {
                let w = &layer.weights[j][ch * kw..(ch + 1) * kw];
                let mut m = 0i64;
                for tap in 0..3 {
                    if edge[tap] {
                        continue;
                    }
                    for wi in 0..cw {
                        m += i64::from((!(taps[tap][wi] ^ w[tap * cw + wi])).count_ones());
                    }
                }
                let z = 2 * m - n_valid as i64;
                zq += i64::from(layer.aq[j][ch]) * z;
            }
            let fire = if layer.sign[ch] == 1 {
                zq >= i64::from(layer.thr[ch])
            } else {
                zq <= i64::from(layer.thr[ch])
            };
            if fire {
                out[slot * cw + ch / 64] |= 1u64 << (ch % 64);
            }
        }
    }
    out
}

#[cfg(test)]
mod parity_tests {
    use super::*;

    #[test]
    fn python_parity() {
        let Ok(fixtures) = std::env::var("BNN_FIXTURES") else {
            return;
        };
        let raw = std::fs::read_to_string(fixtures).unwrap();
        let model = Bnn::get();
        // fixtures: [{"window_hex": "...", "logits": [f, ...]}, ...]
        let mut cases = Vec::new();
        for chunk in raw.split("\"window_hex\": \"").skip(1) {
            let hex_end = chunk.find('"').unwrap();
            let hex = &chunk[..hex_end];
            let rest = &chunk[hex_end..];
            let ls = rest.find('[').unwrap() + 1;
            let le = rest.find(']').unwrap();
            let logits: Vec<f32> = rest[ls..le]
                .split(',')
                .map(|s| s.trim().parse().unwrap())
                .collect();
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
                .collect();
            cases.push((bytes, logits));
        }
        assert!(!cases.is_empty());
        for (window, expected) in cases {
            let planes = model.encode_planes(&window);
            let got = model.logits(&planes);
            for (g, e) in got.iter().zip(expected.iter()) {
                let tol = 1e-3 * g.abs().max(e.abs()).max(1.0);
                assert!((g - e).abs() < tol, "logit mismatch: {g} vs {e}");
            }
        }
    }
}
