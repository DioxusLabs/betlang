pub(crate) const MODEL_MAGIC: [u8; 8] = [0x4d, 0x53, 0x51, 0x31, 0x01, 0x00, 0x00, 0x00];
pub(crate) const SCALE_COUNT: usize = 6;

pub(crate) const MAGIKA_BEG_SIZE: usize = 1_024;
pub(crate) const MAGIKA_END_SIZE: usize = 1_024;
pub(crate) const MAGIKA_BLOCK_SIZE: usize = 4_096;

// Wordseq architecture constants. Must match what the model was trained with.
pub(crate) const BINS: usize = 1_024;
pub(crate) const MAX_UNITS: usize = 2_048;
pub(crate) const EMBED: usize = 24;
pub(crate) const CONV0_KERNEL: usize = 7;
pub(crate) const CONV0: usize = 64;
pub(crate) const CONV0_POOL: usize = 4;
pub(crate) const CONV1_KERNEL: usize = 5;
pub(crate) const CONV1: usize = 128;
pub(crate) const CONV1_POOL: usize = 2;
pub(crate) const CONV2_KERNEL: usize = 3;
pub(crate) const CONV2: usize = 128;
pub(crate) const POOLED: usize = CONV2 * 2; // GlobalMax + GlobalAvg
pub(crate) const DENSE: usize = 96;
pub(crate) const CLASSES: usize = 48;

const EMBED_SCRATCH: usize = MAX_UNITS * EMBED;
const POOL0_SCRATCH: usize = (MAX_UNITS / CONV0_POOL) * CONV0;
const POOL1_SCRATCH: usize = (MAX_UNITS / CONV0_POOL / CONV1_POOL) * CONV1;

const fn max_usize(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}

pub(crate) const ACTIVATION_SCRATCH: usize =
    max_usize(EMBED_SCRATCH + POOL0_SCRATCH, POOL0_SCRATCH + POOL1_SCRATCH);
pub(crate) const CONV_SCRATCH: usize = 4 * CONV2;
pub(crate) const INFERENCE_SCRATCH: usize = ACTIVATION_SCRATCH + CONV_SCRATCH;

// Tokenizer flag bits. Must match `_PUNCT_FLAG`/etc. in the Python trainer.
pub(crate) const WORD_MASK: u32 = 0x00FF_FFFF;
pub(crate) const PUNCT_FLAG: u32 = 0x1000_0000;
pub(crate) const INDENT_FLAG: u32 = 0x2000_0000;
pub(crate) const NUM_FLAG: u32 = 0x4000_0000;
pub(crate) const BRACKET_FLAG: u32 = 0x5000_0000;
