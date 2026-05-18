use super::constants::{BRACKET_FLAG, INDENT_FLAG, MAX_UNITS, NUM_FLAG, PUNCT_FLAG, WORD_MASK};

pub(crate) fn hash_unit_bytes(bytes: &[u8]) -> u32 {
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

/// Production word-unit tokenizer version 3.
///
/// Case-folds word hashes and emits unambiguous brackets as BRACKET_FLAG tokens.
pub(crate) fn tokenize(bytes: &[u8], padding_mask: &[bool]) -> Vec<i32> {
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
