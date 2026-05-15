pub(crate) const MODEL_PAYLOAD_LEN: usize = 100_444;

pub(crate) const MAGIKA_BEG_SIZE: usize = 1_024;
pub(crate) const MAGIKA_END_SIZE: usize = 1_024;
pub(crate) const MAGIKA_BLOCK_SIZE: usize = 4_096;

// Wordseq architecture constants. Must match what the model was trained with.
pub(crate) const BINS: usize = 1_536;
pub(crate) const MAX_UNITS: usize = 2_048;
pub(crate) const EMBED: usize = 28;
pub(crate) const CONV0_KERNEL: usize = 7;
pub(crate) const CONV0: usize = 96;
pub(crate) const CONV0_POOL: usize = 4;
pub(crate) const CONV1_KERNEL: usize = 5;
pub(crate) const CONV1: usize = 192;
pub(crate) const CONV1_POOL: usize = 2;
pub(crate) const CONV2_KERNEL: usize = 3;
pub(crate) const CONV2: usize = 192;
pub(crate) const POOLED: usize = CONV2 * 2; // GlobalMax + GlobalAvg
pub(crate) const DENSE: usize = 160;
pub(crate) const CLASSES: usize = 67;

#[allow(clippy::excessive_precision)]
pub(crate) const MODEL_WEIGHT_SCALES: [f32; 6] = [
    0.162_534_645_625_523_17,
    0.180_766_612_291_336_06,
    0.134_378_731_250_762_94,
    0.128_200_232_982_635_5,
    0.155_264_601_111_412_05,
    0.294_296_571_186_610_6,
];

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
