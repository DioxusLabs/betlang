//! Shannon n-gram counting-Bloom binary model (MBL5) runtime.
//!
//! The encoder turns the 2048-token Magika window into a binary signature
//! using only table lookups, XOR, rotates, and integer compares: every byte
//! n-gram (orders 1-8, computed separately for the begin/end half) is given a
//! 64-bit Zobrist code — the XOR of per-offset random codes, i.e. a Shannon
//! random block code over (symbol, offset) pairs — and counted in 4,096
//! counting-Bloom buckets per (half, order) group; word-unit and tokenizer-v3
//! unit streams get their own groups. A candidate signature bit is a bucket
//! count passed through a thermometer threshold (quantized log-frequency —
//! Shannon surprisal — of each n-gram), giving 319,488 candidate bits that
//! are nearly free to compute.
//!
//! Storing head weights for every candidate is not free, so the artifact
//! keeps only the trainer-selected columns: per plane (a (group, threshold)
//! pair), a sorted list of selected bucket ids, u64-aligned per plane. The
//! classifier is a binary {-1,+1} linear head over the selected columns,
//! evaluated with XOR + popcount per (class, plane), combined with a
//! per-(class, plane) int8 scale and a per-class f32 step (XNOR-net style
//! calibration), so inference stays binary end to end with one float multiply
//! per class.
//!
//! The plane table and column selection are read from the artifact, not
//! hard-coded: the trainer's selection drives both feature encoding and the
//! head.

use std::sync::OnceLock;

pub(crate) static BLOOM_BYTES: &[u8] = include_bytes!("../../assets/magika/source-bloom.bin");

pub(crate) const BLOOM_MAGIC: [u8; 4] = *b"MBL5";

pub(crate) const TOKENS: usize = 2_048;
pub(crate) const PAD_TOKEN: u16 = 256;
pub(crate) const SYMBOLS: usize = 257;
pub(crate) const HALF: usize = TOKENS / 2;
pub(crate) const CLASSES: usize = 48;

pub(crate) const MAX_ORDER: usize = 8;
pub(crate) const NGRAM_GROUPS: usize = 2 * MAX_ORDER;
/// Word-unit counting groups per half: unigrams (two independent hash
/// folds), bigrams, trigrams, and line-start unigrams.
pub(crate) const WORD_GROUPS: usize = 10;
/// Wordseq-unit n-gram groups over the tokenizer-v3 unit stream.
pub(crate) const UNIT_MAX_ORDER: usize = 4;
pub(crate) const UNIT_GROUPS: usize = UNIT_MAX_ORDER;
pub(crate) const GROUPS: usize = NGRAM_GROUPS + WORD_GROUPS + UNIT_GROUPS;
/// Counting-Bloom buckets per group (fixed full resolution).
pub(crate) const BLOCK: usize = 4_096;

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

/// One signature plane: the selected columns of one (group, threshold)
/// candidate plane. Bit i of the segment is `counts[group][ids[i]] >= level`.
struct Plane {
    group: usize,
    level: u8,
    /// Selected columns in this plane (a multiple of 64).
    width: usize,
    /// Start of this plane's bits in the packed signature.
    word_offset: usize,
    /// Start of this plane's bucket ids in `bucket_ids`.
    id_offset: usize,
}

pub(crate) struct BloomModel {
    planes: Box<[Plane]>,
    /// Selected bucket ids, ascending within each plane's segment.
    bucket_ids: Box<[u16]>,
    /// Total signature bits (a multiple of 64).
    bits: usize,
    /// `CLASSES` rows of packed head weights over the signature bits.
    head_weights: Box<[u64]>,
    /// Per-(class, plane) int8 scale, applied as `step[class] * q`.
    scale_q: Box<[i8]>,
    scale_step: [f32; CLASSES],
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

impl BloomModel {
    fn load() -> Self {
        let bytes = BLOOM_BYTES;
        debug_assert!(bytes.starts_with(&BLOOM_MAGIC), "bad MBL5 magic");
        let mut cur = BLOOM_MAGIC.len();
        let bits = read_u32(bytes, &mut cur) as usize;
        let plane_count = read_u32(bytes, &mut cur) as usize;
        let classes = read_u32(bytes, &mut cur) as usize;
        debug_assert_eq!(classes, CLASSES, "unexpected MBL5 class count");
        debug_assert!(bits.is_multiple_of(64), "unaligned MBL5 signature");

        // Plane table: (group, level, selected width) per plane.
        let mut planes = Vec::with_capacity(plane_count);
        let mut bit_at = 0usize;
        for _ in 0..plane_count {
            let group = bytes[cur] as usize;
            let level = bytes[cur + 1];
            let width = u16::from_le_bytes([bytes[cur + 2], bytes[cur + 3]]) as usize;
            cur += 4;
            debug_assert!(group < GROUPS, "bad MBL5 group id");
            debug_assert!(level >= 1, "bad MBL5 threshold");
            debug_assert!(width.is_multiple_of(64), "unaligned MBL5 plane");
            debug_assert!(width <= BLOCK, "plane wider than its group");
            planes.push(Plane {
                group,
                level,
                width,
                word_offset: bit_at / 64,
                id_offset: bit_at,
            });
            bit_at += width;
        }
        debug_assert_eq!(bit_at, bits, "MBL5 plane widths disagree with header");

        // Selected bucket ids, one u16 per signature bit, per-plane ascending.
        let mut bucket_ids = Vec::with_capacity(bits);
        for i in 0..bits {
            let at = cur + 2 * i;
            let id = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
            debug_assert!((id as usize) < BLOCK, "bad MBL5 bucket id");
            bucket_ids.push(id);
        }
        cur += 2 * bits;

        let mut scale_q = Vec::with_capacity(CLASSES * plane_count);
        for i in 0..CLASSES * plane_count {
            scale_q.push(bytes[cur + i] as i8);
        }
        cur += CLASSES * plane_count;
        let mut scale_step = [0.0f32; CLASSES];
        for slot in scale_step.iter_mut() {
            *slot = read_f32(bytes, &mut cur);
        }
        let mut bias = [0.0f32; CLASSES];
        for slot in bias.iter_mut() {
            *slot = read_f32(bytes, &mut cur);
        }
        let head_weights = read_words(bytes, &mut cur, CLASSES * (bits / 64));
        debug_assert_eq!(cur, bytes.len(), "unexpected MBL5 payload length");

        Self {
            planes: planes.into_boxed_slice(),
            bucket_ids: bucket_ids.into_boxed_slice(),
            bits,
            head_weights,
            scale_q: scale_q.into_boxed_slice(),
            scale_step,
            bias,
        }
    }

    pub(crate) fn get() -> &'static Self {
        static MODEL: OnceLock<BloomModel> = OnceLock::new();
        MODEL.get_or_init(Self::load)
    }

    #[inline]
    fn words(&self) -> usize {
        self.bits / 64
    }

    #[inline]
    fn bump(&self, counts: &mut [u8], group: usize, hash: u64) {
        let slot = group * BLOCK + (hash & (BLOCK as u64 - 1)) as usize;
        counts[slot] = counts[slot].saturating_add(1);
    }

    /// Encode a token window plus its tokenizer-v3 unit stream into the packed
    /// binary signature.
    fn encode_signature(&self, tokens: &[u16; TOKENS], units: &[i32]) -> Vec<u64> {
        static ZOBRIST: OnceLock<Box<[[u64; SYMBOLS]; TABLES]>> = OnceLock::new();
        let tables = ZOBRIST.get_or_init(zobrist_tables);

        let mut counts = vec![0u8; GROUPS * BLOCK];
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
                        self.bump(&mut counts, 2 * order_index + half, rotated);
                    }
                }
            }

            // Word-unit scan: rotate-XOR chain over casefolded identifier
            // bytes, emitting unigrams (two folds), a bigram/trigram with the
            // previous words of the same half, and a line-start unigram when
            // the word begins a line.
            let word_table = &tables[MAX_ORDER];
            let uni_group = NGRAM_GROUPS + half;
            let bi_group = NGRAM_GROUPS + 2 + half;
            let uni2_group = NGRAM_GROUPS + 4 + half;
            let tri_group = NGRAM_GROUPS + 6 + half;
            let ls_group = NGRAM_GROUPS + 8 + half;
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
                self.bump(
                    &mut counts,
                    uni_group,
                    acc_w.rotate_left((23 + half) as u32),
                );
                self.bump(
                    &mut counts,
                    uni2_group,
                    acc_w.rotate_left((41 + half) as u32),
                );
                let bg = prev_hash.rotate_left(17) ^ acc_w;
                if have_prev {
                    self.bump(&mut counts, bi_group, bg.rotate_left((29 + half) as u32));
                }
                if have_prev2 {
                    let tg = prev2_hash.rotate_left(34) ^ bg;
                    self.bump(&mut counts, tri_group, tg.rotate_left((47 + half) as u32));
                }
                if at_linestart {
                    self.bump(&mut counts, ls_group, acc_w.rotate_left((53 + half) as u32));
                }
                prev2_hash = prev_hash;
                prev_hash = acc_w;
                have_prev2 = have_prev;
                have_prev = true;
            }
        }

        // Wordseq-unit n-grams: SplitMix64 codes salted per offset,
        // XOR-combined.
        let salts = unit_salts();
        for position in 0..units.len() {
            let mut acc = 0u64;
            for (order_index, salt) in salts.iter().enumerate() {
                if position < order_index {
                    break;
                }
                acc ^= splitmix64(units[position - order_index] as u64 ^ salt);
                let group = NGRAM_GROUPS + WORD_GROUPS + order_index;
                self.bump(
                    &mut counts,
                    group,
                    acc.rotate_left((11 * order_index + 5) as u32),
                );
            }
        }

        // Gather the selected columns: counts[group][id] >= level, packed
        // little-endian per plane segment.
        let mut signature = vec![0u64; self.words()];
        for plane in self.planes.iter() {
            let block = &counts[plane.group * BLOCK..(plane.group + 1) * BLOCK];
            let ids = &self.bucket_ids[plane.id_offset..plane.id_offset + plane.width];
            let base = plane.word_offset * 64;
            for (index, &id) in ids.iter().enumerate() {
                let bit = (block[id as usize] >= plane.level) as u64;
                let at = base + index;
                signature[at / 64] |= bit << (at % 64);
            }
        }
        signature
    }

    /// Forward pass: encode, then the binary dense head. All per-plane work is
    /// XOR + popcount with an integer accumulator; each class ends with one
    /// float multiply-add.
    pub(crate) fn logits(&self, tokens: &[u16; TOKENS], units: &[i32]) -> [f32; CLASSES] {
        let signature = self.encode_signature(tokens, units);

        let words = self.words();
        let mut logits = [0.0f32; CLASSES];
        for (class, logit) in logits.iter_mut().enumerate() {
            let row = &self.head_weights[class * words..(class + 1) * words];
            let scales = &self.scale_q[class * self.planes.len()..(class + 1) * self.planes.len()];
            let mut acc = 0i32;
            for (plane, &q) in self.planes.iter().zip(scales.iter()) {
                let at = plane.word_offset;
                let plane_words = plane.width / 64;
                let mut mismatches = 0u32;
                for (word, weight) in signature[at..at + plane_words]
                    .iter()
                    .zip(row[at..at + plane_words].iter())
                {
                    mismatches += (word ^ weight).count_ones();
                }
                let z = plane.width as i32 - 2 * mismatches as i32;
                acc += i32::from(q) * z;
            }
            *logit = self.scale_step[class] * acc as f32 + self.bias[class];
        }
        logits
    }

    /// Total signature bits (used by tests).
    #[cfg(test)]
    pub(crate) fn signature_bits(&self) -> usize {
        self.bits
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
