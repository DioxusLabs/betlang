//! Byte-level input canonicalization shared with the training corpus.
//!
//! The corpus builder (`scripts/build_prompt_corpus.py`) applies the same
//! transforms, so the model sees one canonical text form at training and
//! inference time:
//!
//! - `\r\n` and `\r` become `\n`
//! - tabs and non-breaking spaces (U+00A0) become spaces
//! - UTF-8 BOM and zero-width characters (U+200B..U+200D, U+FEFF) are removed
//! - other C0 control bytes (except `\n`) are removed
//! - runs of spaces collapse to a single space
//! - leading/trailing ASCII whitespace is trimmed

/// Canonicalize raw input bytes before windowing/tokenization.
pub(crate) fn normalize(source: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(source.len());
    let mut index = 0;
    while index < source.len() {
        let byte = source[index];
        match byte {
            b'\r' => {
                out.push(b'\n');
                if source.get(index + 1) == Some(&b'\n') {
                    index += 1;
                }
            }
            b'\t' => push_space(&mut out),
            b' ' => push_space(&mut out),
            0xc2 if source.get(index + 1) == Some(&0xa0) => {
                push_space(&mut out);
                index += 1;
            }
            0xe2 if source.get(index + 1) == Some(&0x80)
                && matches!(source.get(index + 2), Some(0x8b..=0x8d)) =>
            {
                index += 2;
            }
            0xef if source.get(index + 1) == Some(&0xbb)
                && source.get(index + 2) == Some(&0xbf) =>
            {
                index += 2;
            }
            0x00..=0x1f if byte != b'\n' => {}
            _ => out.push(byte),
        }
        index += 1;
    }
    let start = out
        .iter()
        .position(|b| !b.is_ascii_whitespace())
        .unwrap_or(out.len());
    let end = out
        .iter()
        .rposition(|b| !b.is_ascii_whitespace())
        .map_or(start, |p| p + 1);
    out.drain(end..);
    out.drain(..start);
    out
}

fn push_space(out: &mut Vec<u8>) {
    if out.last() != Some(&b' ') {
        out.push(b' ');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_canonicalizes_whitespace_and_invisibles() {
        assert_eq!(normalize(b"  ls   -la\t\t/tmp  "), b"ls -la /tmp".to_vec());
        assert_eq!(normalize(b"a\r\nb\rc"), b"a\nb\nc".to_vec());
        assert_eq!(
            normalize("\u{feff}write\u{200b} a poem\u{a0}now".as_bytes()),
            b"write a poem now".to_vec()
        );
        assert_eq!(normalize(b"a\x00\x08b"), b"ab".to_vec());
        assert_eq!(normalize(b"   \t  "), Vec::<u8>::new());
    }
}
