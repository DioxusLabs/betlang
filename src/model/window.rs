use super::constants::{MAGIKA_BEG_SIZE, MAGIKA_BLOCK_SIZE, MAGIKA_END_SIZE, MAGIKA_WINDOW_SIZE};

pub(crate) struct TokenWindow {
    bytes: [u8; MAGIKA_WINDOW_SIZE],
    valid_len: usize,
}

impl TokenWindow {
    pub(crate) fn bytes(&self) -> &[u8] {
        &self.bytes[..self.valid_len]
    }
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

/// Build the (begin + end) byte window and its first padding boundary.
/// Mirrors `magika_features` in the trainer.
pub(crate) fn build_window(source: &[u8]) -> Option<TokenWindow> {
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
    let mut buf = [0u8; MAGIKA_WINDOW_SIZE];

    buf[..beg_len].copy_from_slice(&stripped_beg_full[..beg_len]);
    let end_start = MAGIKA_BEG_SIZE + (MAGIKA_END_SIZE - end_len);
    let end_src = &stripped_end_full[stripped_end_full.len() - end_len..];
    buf[end_start..end_start + end_len].copy_from_slice(end_src);
    let valid_len = if beg_len < MAGIKA_BEG_SIZE {
        beg_len
    } else if end_start > MAGIKA_BEG_SIZE {
        MAGIKA_BEG_SIZE
    } else {
        MAGIKA_WINDOW_SIZE
    };
    Some(TokenWindow {
        bytes: buf,
        valid_len,
    })
}
