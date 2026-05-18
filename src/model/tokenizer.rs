use super::{
    constants::{BRACKET_FLAG, INDENT_FLAG, MAX_UNITS, NUM_FLAG, PUNCT_FLAG, WORD_MASK},
    window::TokenWindow,
};

pub(crate) fn hash_unit_bytes(bytes: &[u8]) -> u32 {
    const PRIME: u64 = 2_654_435_761;
    let mut h: u64 = 0;
    for &b in bytes {
        h = h.wrapping_mul(PRIME).wrapping_add(b as u64) & 0xFFFF_FFFF;
    }
    h as u32
}

fn push_indent_unit(out: &mut Vec<i32>, indent: u32) {
    if indent > 0 && out.len() < MAX_UNITS {
        out.push((indent.min(63) | INDENT_FLAG) as i32);
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokenKind {
    Empty,
    Word,
    Number,
    Punct,
}

struct TokenBuffer {
    kind: TokenKind,
    bytes: Vec<u8>,
}

impl TokenBuffer {
    fn new() -> Self {
        Self {
            kind: TokenKind::Empty,
            bytes: Vec::new(),
        }
    }

    fn is_number(&self) -> bool {
        self.kind == TokenKind::Number
    }

    fn flush(&mut self, out: &mut Vec<i32>) {
        let flag = match self.kind {
            TokenKind::Empty => return,
            TokenKind::Word => 0,
            TokenKind::Number => NUM_FLAG,
            TokenKind::Punct => PUNCT_FLAG,
        };
        if out.len() < MAX_UNITS {
            out.push(((hash_unit_bytes(&self.bytes) & WORD_MASK) | flag) as i32);
        }
        self.bytes.clear();
        self.kind = TokenKind::Empty;
    }

    fn push(&mut self, kind: TokenKind, value: u8, out: &mut Vec<i32>) {
        if self.kind != kind {
            self.flush(out);
            self.kind = kind;
        }
        self.bytes.push(value);
    }
}

/// Production word-unit tokenizer version 3.
///
/// Case-folds word hashes and emits unambiguous brackets as BRACKET_FLAG tokens.
pub(crate) fn tokenize(window: &TokenWindow) -> Vec<i32> {
    let bytes = window.bytes();
    let mut out: Vec<i32> = Vec::with_capacity(MAX_UNITS);
    let mut current = TokenBuffer::new();
    let mut at_line_start = true;
    let mut indent_units: u32 = 0;

    for &raw_value in bytes {
        let value = raw_value.to_ascii_lowercase();
        let is_letter = value.is_ascii_lowercase() || value == b'_';
        let is_digit = value.is_ascii_digit();
        let is_newline = value == b'\n';
        let is_cr = value == b'\r';
        let is_space = value == b' ' || value == b'\t';
        let is_bracket = matches!(value, b'(' | b')' | b'[' | b']' | b'{' | b'}');

        if out.len() >= MAX_UNITS {
            break;
        }

        if is_letter {
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            current.push(TokenKind::Word, value, &mut out);
            continue;
        }
        if is_digit || value == b'.' {
            if value == b'.' && !current.is_number() {
                if at_line_start {
                    push_indent_unit(&mut out, indent_units);
                }
                at_line_start = false;
                indent_units = 0;
                current.flush(&mut out);
                current.push(TokenKind::Punct, value, &mut out);
                continue;
            }
            if at_line_start {
                push_indent_unit(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            current.push(TokenKind::Number, value, &mut out);
            continue;
        }
        if is_newline {
            current.flush(&mut out);
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
            current.flush(&mut out);
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
            current.flush(&mut out);
            let space_token = ((b' ' as u32) | PUNCT_FLAG) as i32;
            if out.last() != Some(&space_token) && out.len() < MAX_UNITS {
                out.push(space_token);
            }
            continue;
        }
        if is_bracket {
            current.flush(&mut out);
            if out.len() < MAX_UNITS {
                out.push(((value as u32) | BRACKET_FLAG) as i32);
            }
            continue;
        }
        current.push(TokenKind::Punct, value, &mut out);
    }

    current.flush(&mut out);

    out
}
