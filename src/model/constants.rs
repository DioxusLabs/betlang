pub(crate) const MAGIKA_BEG_SIZE: usize = 1_024;
pub(crate) const MAGIKA_END_SIZE: usize = 1_024;
pub(crate) const MAGIKA_WINDOW_SIZE: usize = MAGIKA_BEG_SIZE + MAGIKA_END_SIZE;
pub(crate) const MAGIKA_BLOCK_SIZE: usize = 4_096;

pub(crate) const MAX_UNITS: usize = 2_048;
pub(crate) const CLASSES: usize = 48;

// Tokenizer flag bits. Must match `_PUNCT_FLAG`/etc. in the Python trainer.
pub(crate) const WORD_MASK: u32 = 0x00FF_FFFF;
pub(crate) const PUNCT_FLAG: u32 = 0x1000_0000;
pub(crate) const INDENT_FLAG: u32 = 0x2000_0000;
pub(crate) const NUM_FLAG: u32 = 0x4000_0000;
pub(crate) const BRACKET_FLAG: u32 = 0x5000_0000;
