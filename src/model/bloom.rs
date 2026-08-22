//! Shannon n-gram counting-Bloom binary model (MBL3) runtime.
//!
//! The encoder turns the 2048-token Magika window into a fixed-width binary
//! signature using only table lookups, XOR, rotates, and integer compares:
//! every byte n-gram (orders 1-8, computed separately for the begin/end half)
//! is given a 64-bit Zobrist code — the XOR of per-offset random codes, i.e. a
//! Shannon random block code over (symbol, offset) pairs — and folded into a
//! per-(half, order) counting-Bloom block. Bucket counts pass through a fixed
//! thermometer (1/2/4/8 for orders <= 3, 1/2 for 4-5, presence for 6-8), a
//! quantized log-frequency — Shannon surprisal — encoding of each n-gram.
//!
//! The classifier is binary as well: an optional block-diagonal binary hidden
//! layer (XOR + popcount + threshold per plane) followed by a binary dense
//! layer, so inference is XOR/popcount end to end. The dense layer combines
//! per-plane bipolar dot products with a small per-(class, plane) float scale
//! (XNOR-net style calibration; CLASSES * PLANES multiply-adds per file).

use std::sync::OnceLock;

pub(crate) static BLOOM_BYTES: &[u8] = include_bytes!("../../assets/magika/source-bloom.bin");

pub(crate) const BLOOM_MAGIC: [u8; 4] = *b"MBL3";

pub(crate) const TOKENS: usize = 2_048;
pub(crate) const PAD_TOKEN: u16 = 256;
pub(crate) const SYMBOLS: usize = 257;
pub(crate) const HALF: usize = TOKENS / 2;
pub(crate) const CLASSES: usize = 48;

pub(crate) const MAX_ORDER: usize = 8;
pub(crate) const BLOCK_BITS: usize = 4_096;
pub(crate) const BLOCK_WORDS: usize = BLOCK_BITS / 64;
pub(crate) const NGRAM_GROUPS: usize = 2 * MAX_ORDER;
/// Word-unit counting groups per half: unigrams (two independent hash
/// folds), bigrams, trigrams, and line-start unigrams.
pub(crate) const WORD_GROUPS: usize = 10;
/// Wordseq-unit n-gram groups over the tokenizer-v3 unit stream.
pub(crate) const UNIT_MAX_ORDER: usize = 4;
pub(crate) const UNIT_GROUPS: usize = UNIT_MAX_ORDER;
pub(crate) const GROUPS: usize = NGRAM_GROUPS + WORD_GROUPS + UNIT_GROUPS;
pub(crate) const COUNT_BUCKETS: usize = GROUPS * BLOCK_BITS;

/// Count thermometer levels per n-gram order (1-based order = index + 1).
pub(crate) const ORDER_LEVELS: [&[u16]; MAX_ORDER] = [
    &[1, 2, 4, 8],
    &[1, 2, 4, 8],
    &[1, 2, 4, 8],
    &[1, 2],
    &[1, 2],
    &[1],
    &[1],
    &[1],
];
/// Count thermometer levels per word feature kind (uni fold 1, bigram,
/// uni fold 2, trigram, line-start unigram).
pub(crate) const WORD_LEVELS: [&[u16]; 5] = [
    &[1, 2, 4, 8],
    &[1, 2],
    &[1, 2, 4, 8],
    &[1, 2],
    &[1, 2, 4, 8],
];
/// Count thermometer levels per wordseq-unit n-gram order.
pub(crate) const UNIT_LEVELS: [&[u16]; UNIT_MAX_ORDER] = [&[1, 2, 4, 8], &[1, 2], &[1], &[1]];

/// Total signature planes: one BLOCK_BITS-wide bit plane per (order, half,
/// level) nested in that order, followed by word (kind, half, level) planes
/// and wordseq-unit (order, level) planes.
pub(crate) const PLANES: usize = 78;
pub(crate) const BITS: usize = PLANES * BLOCK_BITS;
pub(crate) const WORDS: usize = BITS / 64;

const ZOBRIST_SEED: u64 = 0xBE7A_1AB5_5EED_0001;
const UNIT_SEED: u64 = 0xBE7A_1AB5_5EED_0002;

/// Deterministic xorshift64* stream; must match the Python trainer exactly.
fn xorshift64_star(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545_F491_4F6C_DD1D)
}

/// SplitMix64 finalizer; must match the Python trainer exactly.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn unit_salts() -> [u64; UNIT_MAX_ORDER] {
    let mut state = UNIT_SEED;
    let mut salts = [0u64; UNIT_MAX_ORDER];
    for slot in salts.iter_mut() {
        *slot = xorshift64_star(&mut state);
    }
    salts
}

/// Per-offset n-gram code tables plus one trailing word-symbol table.
const TABLES: usize = MAX_ORDER + 1;

fn zobrist_tables() -> Box<[[u64; SYMBOLS]; TABLES]> {
    let mut state = ZOBRIST_SEED;
    let mut tables = Box::new([[0u64; SYMBOLS]; TABLES]);
    for table in tables.iter_mut() {
        for slot in table.iter_mut() {
            *slot = xorshift64_star(&mut state);
        }
    }
    tables
}

fn is_word_symbol(symbol: u16) -> bool {
    symbol < 256 && ((symbol as u8).is_ascii_alphanumeric() || symbol == b'_' as u16)
}

fn casefold(symbol: u16) -> u16 {
    if (b'A' as u16..=b'Z' as u16).contains(&symbol) {
        symbol + 32
    } else {
        symbol
    }
}

/// Encode a token window plus its tokenizer-v3 unit stream into the packed
/// binary signature.
pub(crate) fn encode_signature(tokens: &[u16; TOKENS], units: &[i32]) -> Box<[u64; WORDS]> {
    static ZOBRIST: OnceLock<Box<[[u64; SYMBOLS]; TABLES]>> = OnceLock::new();
    let tables = ZOBRIST.get_or_init(zobrist_tables);

    let mut counts = vec![0u16; COUNT_BUCKETS];
    for half in 0..2 {
        let htok = &tokens[half * HALF..(half + 1) * HALF];
        for position in 0..HALF {
            let mut acc = 0u64;
            let mut valid = true;
            for (order_index, table) in tables.iter().take(MAX_ORDER).enumerate() {
                let symbol = if position >= order_index {
                    htok[position - order_index]
                } else {
                    PAD_TOKEN
                };
                acc ^= table[symbol as usize];
                valid &= symbol != PAD_TOKEN;
                if valid {
                    let rotated = acc.rotate_left((7 * order_index + half) as u32 & 63);
                    let bucket = (rotated as usize) & (BLOCK_BITS - 1);
                    let group = 2 * order_index + half;
                    counts[group * BLOCK_BITS + bucket] =
                        counts[group * BLOCK_BITS + bucket].saturating_add(1);
                }
            }
        }

        // Word-unit scan: rotate-XOR chain over casefolded identifier bytes,
        // emitting unigrams (two folds), a bigram/trigram with the previous
        // words of the same half, and a line-start unigram when the word
        // begins a line.
        let word_table = &tables[MAX_ORDER];
        let uni_group = NGRAM_GROUPS + half;
        let bi_group = NGRAM_GROUPS + 2 + half;
        let uni2_group = NGRAM_GROUPS + 4 + half;
        let tri_group = NGRAM_GROUPS + 6 + half;
        let ls_group = NGRAM_GROUPS + 8 + half;
        let mut bump = |group: usize, hash: u64, rot: usize| {
            let bucket = (hash.rotate_left(rot as u32) as usize) & (BLOCK_BITS - 1);
            counts[group * BLOCK_BITS + bucket] =
                counts[group * BLOCK_BITS + bucket].saturating_add(1);
        };
        let mut acc_w = 0u64;
        let mut prev_hash = 0u64;
        let mut prev2_hash = 0u64;
        let mut have_prev = false;
        let mut have_prev2 = false;
        let mut at_linestart = false;
        for position in 0..HALF {
            let symbol = htok[position];
            if !is_word_symbol(symbol) {
                acc_w = 0;
                continue;
            }
            if position == 0 || !is_word_symbol(htok[position - 1]) {
                at_linestart = position == 0
                    || htok[position - 1] == u16::from(b'\n')
                    || htok[position - 1] == u16::from(b'\r');
            }
            acc_w = acc_w.rotate_left(1) ^ word_table[casefold(symbol) as usize];
            let at_end = position + 1 >= HALF || !is_word_symbol(htok[position + 1]);
            if !at_end {
                continue;
            }
            bump(uni_group, acc_w, 23 + half);
            bump(uni2_group, acc_w, 41 + half);
            let bg = prev_hash.rotate_left(17) ^ acc_w;
            if have_prev {
                bump(bi_group, bg, 29 + half);
            }
            if have_prev2 {
                let tg = prev2_hash.rotate_left(34) ^ bg;
                bump(tri_group, tg, 47 + half);
            }
            if at_linestart {
                bump(ls_group, acc_w, 53 + half);
            }
            prev2_hash = prev_hash;
            prev_hash = acc_w;
            have_prev2 = have_prev;
            have_prev = true;
        }
    }

    // Wordseq-unit n-grams: SplitMix64 codes salted per offset, XOR-combined.
    let salts = unit_salts();
    for position in 0..units.len() {
        let mut acc = 0u64;
        for (order_index, salt) in salts.iter().enumerate() {
            if position < order_index {
                break;
            }
            acc ^= splitmix64(units[position - order_index] as u64 ^ salt);
            let group = NGRAM_GROUPS + WORD_GROUPS + order_index;
            let bucket =
                (acc.rotate_left((11 * order_index + 5) as u32) as usize) & (BLOCK_BITS - 1);
            counts[group * BLOCK_BITS + bucket] =
                counts[group * BLOCK_BITS + bucket].saturating_add(1);
        }
    }

    let mut signature = Box::new([0u64; WORDS]);
    let mut plane = 0;
    let mut emit = |group: usize, level: u16, plane: &mut usize| {
        let block = &counts[group * BLOCK_BITS..(group + 1) * BLOCK_BITS];
        let base = *plane * BLOCK_WORDS;
        for (bucket, &count) in block.iter().enumerate() {
            let bit = (count >= level) as u64;
            signature[base + bucket / 64] |= bit << (bucket % 64);
        }
        *plane += 1;
    };
    for (order_index, levels) in ORDER_LEVELS.iter().enumerate() {
        for half in 0..2 {
            for &level in levels.iter() {
                emit(2 * order_index + half, level, &mut plane);
            }
        }
    }
    for (kind, levels) in WORD_LEVELS.iter().enumerate() {
        for half in 0..2 {
            for &level in levels.iter() {
                emit(NGRAM_GROUPS + 2 * kind + half, level, &mut plane);
            }
        }
    }
    for (order_index, levels) in UNIT_LEVELS.iter().enumerate() {
        for &level in levels.iter() {
            emit(NGRAM_GROUPS + WORD_GROUPS + order_index, level, &mut plane);
        }
    }
    debug_assert_eq!(plane, PLANES);
    signature
}

pub(crate) struct BloomModel {
    hidden_per_block: usize,
    /// `PLANES * hidden` rows of `BLOCK_WORDS` packed hidden weights.
    hidden_weights: Box<[u64]>,
    /// Per-unit mismatch thresholds: `bit = (mismatches <= thr) ^ flip`.
    hidden_thresholds: Box<[i16]>,
    /// Packed per-unit output flips.
    hidden_flips: Box<[u64]>,
    /// `CLASSES` rows of packed head weights over the head input bits.
    head_weights: Box<[u64]>,
    head_words: usize,
    /// Bits per head-input block (`BLOCK_BITS`, or `hidden_per_block`).
    head_block_bits: usize,
    /// `CLASSES * PLANES` per-(class, plane) scales.
    scale: Box<[f32]>,
    bias: [f32; CLASSES],
}

fn read_u32(bytes: &[u8], cur: &mut usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*cur..*cur + 4]);
    *cur += 4;
    u32::from_le_bytes(b)
}

fn read_f32(bytes: &[u8], cur: &mut usize) -> f32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&bytes[*cur..*cur + 4]);
    *cur += 4;
    f32::from_le_bytes(b)
}

fn read_words(bytes: &[u8], cur: &mut usize, count: usize) -> Box<[u64]> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = *cur + i * 8;
        let mut b = [0u8; 8];
        b.copy_from_slice(&bytes[at..at + 8]);
        out.push(u64::from_le_bytes(b));
    }
    *cur += count * 8;
    out.into_boxed_slice()
}

fn read_i16s(bytes: &[u8], cur: &mut usize, count: usize) -> Box<[i16]> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let at = *cur + i * 2;
        out.push(i16::from_le_bytes([bytes[at], bytes[at + 1]]));
    }
    *cur += count * 2;
    out.into_boxed_slice()
}

fn words_for_bits(bits: usize) -> usize {
    bits.div_ceil(64)
}

impl BloomModel {
    fn load() -> Self {
        let bytes = BLOOM_BYTES;
        debug_assert!(bytes.starts_with(&BLOOM_MAGIC), "bad MBL3 magic");
        let mut cur = BLOOM_MAGIC.len();
        let block_bits = read_u32(bytes, &mut cur) as usize;
        let planes = read_u32(bytes, &mut cur) as usize;
        let hidden_per_block = read_u32(bytes, &mut cur) as usize;
        let classes = read_u32(bytes, &mut cur) as usize;
        debug_assert_eq!(block_bits, BLOCK_BITS, "unexpected MBL3 block bits");
        debug_assert_eq!(planes, PLANES, "unexpected MBL3 plane count");
        debug_assert_eq!(classes, CLASSES, "unexpected MBL3 class count");
        // Per-plane popcount subtotals require word-aligned head-input blocks.
        debug_assert!(hidden_per_block.is_multiple_of(64) || hidden_per_block == 0);

        let mut scale = Vec::with_capacity(CLASSES * PLANES);
        for _ in 0..CLASSES * PLANES {
            scale.push(read_f32(bytes, &mut cur));
        }
        let mut bias = [0.0f32; CLASSES];
        for slot in bias.iter_mut() {
            *slot = read_f32(bytes, &mut cur);
        }

        let (hidden_weights, hidden_thresholds, hidden_flips, head_bits);
        if hidden_per_block > 0 {
            let units = planes * hidden_per_block;
            hidden_weights = read_words(bytes, &mut cur, units * BLOCK_WORDS);
            hidden_thresholds = read_i16s(bytes, &mut cur, units);
            hidden_flips = read_words(bytes, &mut cur, words_for_bits(units));
            head_bits = units;
        } else {
            hidden_weights = Box::default();
            hidden_thresholds = Box::default();
            hidden_flips = Box::default();
            head_bits = BITS;
        }
        let head_words = words_for_bits(head_bits);
        let head_weights = read_words(bytes, &mut cur, CLASSES * head_words);
        debug_assert_eq!(cur, bytes.len(), "unexpected MBL3 payload length");

        Self {
            hidden_per_block,
            hidden_weights,
            hidden_thresholds,
            hidden_flips,
            head_weights,
            head_words,
            head_block_bits: if hidden_per_block > 0 {
                hidden_per_block
            } else {
                BLOCK_BITS
            },
            scale: scale.into_boxed_slice(),
            bias,
        }
    }

    pub(crate) fn get() -> &'static Self {
        static MODEL: OnceLock<BloomModel> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    /// Forward pass: encode, optional binary hidden layer, binary dense head.
    pub(crate) fn logits(&self, tokens: &[u16; TOKENS], units: &[i32]) -> [f32; CLASSES] {
        let signature = encode_signature(tokens, units);
        let head_in: Vec<u64> = if self.hidden_per_block > 0 {
            let units = PLANES * self.hidden_per_block;
            let mut hidden = vec![0u64; words_for_bits(units)];
            for unit in 0..units {
                let plane = unit / self.hidden_per_block;
                let block = &signature[plane * BLOCK_WORDS..(plane + 1) * BLOCK_WORDS];
                let row = &self.hidden_weights[unit * BLOCK_WORDS..(unit + 1) * BLOCK_WORDS];
                let mut mismatches = 0u32;
                for (word, weight) in block.iter().zip(row.iter()) {
                    mismatches += (word ^ weight).count_ones();
                }
                let flip = (self.hidden_flips[unit / 64] >> (unit % 64)) & 1;
                let bit =
                    ((mismatches as i32 <= self.hidden_thresholds[unit] as i32) as u64) ^ flip;
                hidden[unit / 64] |= bit << (unit % 64);
            }
            hidden
        } else {
            signature.to_vec()
        };

        let block_words = self.head_block_bits / 64;
        let mut logits = [0.0f32; CLASSES];
        for (class, logit) in logits.iter_mut().enumerate() {
            let row = &self.head_weights[class * self.head_words..(class + 1) * self.head_words];
            let scales = &self.scale[class * PLANES..(class + 1) * PLANES];
            let mut acc = self.bias[class];
            for (block, &scale) in scales.iter().enumerate() {
                let at = block * block_words;
                let mut mismatches = 0u32;
                for (word, weight) in head_in[at..at + block_words]
                    .iter()
                    .zip(row[at..at + block_words].iter())
                {
                    mismatches += (word ^ weight).count_ones();
                }
                let z = self.head_block_bits as i32 - 2 * mismatches as i32;
                acc += scale * z as f32;
            }
            *logit = acc;
        }
        logits
    }
}

/// Build the 2048-token Magika window exactly as the trainer's
/// `magika_features`: strip leading whitespace from the first block and
/// trailing whitespace from the last, keep the first 1024 bytes (right-padded)
/// and the last 1024 bytes (left-padded) with the padding token.
pub(crate) fn build_token_window(source: &[u8]) -> Option<[u16; TOKENS]> {
    if source.is_empty() {
        return None;
    }
    let block = source.len().min(super::constants::MAGIKA_BLOCK_SIZE);
    let stripped_beg = source[..block].trim_ascii_start();
    if stripped_beg.len() < 8 {
        return None;
    }
    let stripped_end = source[source.len() - block..].trim_ascii_end();

    let mut tokens = [PAD_TOKEN; TOKENS];
    let beg_len = stripped_beg.len().min(TOKENS / 2);
    for (slot, &byte) in tokens[..beg_len].iter_mut().zip(stripped_beg.iter()) {
        *slot = byte as u16;
    }
    let end_len = stripped_end.len().min(TOKENS / 2);
    let end_src = &stripped_end[stripped_end.len() - end_len..];
    for (slot, &byte) in tokens[TOKENS - end_len..].iter_mut().zip(end_src.iter()) {
        *slot = byte as u16;
    }
    Some(tokens)
}
