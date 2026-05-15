#![warn(missing_docs)]

//! CPU inference for the compact Magika wordseq student.
//!
//! Betlang classifies source code into an Arborium/tree-sitter-style language
//! slug using a small embedded neural model.
//!
//! ```
//! let detection = betlang::detect("fn main() { println!(\"hi\"); }");
//!
//! assert_eq!(detection.language(), Some(betlang::Language::Rust));
//! ```

mod language;
mod model;

pub use language::{Language, ParseLanguageError};

/// Source-language detection result.
///
/// Internally, this stores language probabilities sorted from most likely to
/// least likely. Use [`Detection::language`] to read the top language. It
/// returns [`None`] when the input is empty, effectively whitespace only, or too
/// short to build the model window.
///
/// ```
/// let detection = betlang::detect("fn main() { println!(\"hi\"); }");
///
/// assert_eq!(detection.language(), Some(betlang::Language::Rust));
/// ```
#[derive(Debug)]
pub struct Detection {
    predictions: Vec<(f32, Language)>,
}

impl Detection {
    /// Return the most likely detected language.
    ///
    /// Returns [`None`] when the input is empty, effectively whitespace only, or
    /// too short to build the model window.
    ///
    /// ```
    /// let detection = betlang::detect("fn main() { println!(\"hi\"); }");
    ///
    /// assert_eq!(detection.language(), Some(betlang::Language::Rust));
    /// ```
    pub fn language(&self) -> Option<Language> {
        self.predictions.first().map(|(_, language)| *language)
    }

    pub(crate) fn from_predictions(predictions: Vec<(f32, Language)>) -> Self {
        Self { predictions }
    }
}

/// Detect the source language for a source string.
///
/// Use [`Language::slug`] to map predicted languages to Arborium/tree-sitter
/// identifiers. [`Detection::language`] returns [`None`] when the input is
/// empty, effectively whitespace only, or too short to build the model window.
///
/// ```
/// let detection = betlang::detect("fn main() { println!(\"hi\"); }");
///
/// assert_eq!(detection.language(), Some(betlang::Language::Rust));
/// ```
pub fn detect(source: &str) -> Detection {
    model::detect_bytes(source.as_bytes())
}

/// Detect the source language for raw bytes.
///
/// This uses the same byte-window tokenizer as [`detect`], but does not require
/// callers to validate UTF-8 before classification.
///
/// ```
/// let detection = betlang::detect_bytes(b"fn main() { println!(\"hi\"); }\n\xff");
///
/// assert_eq!(detection.language(), Some(betlang::Language::Rust));
/// ```
pub fn detect_bytes(source: &[u8]) -> Detection {
    model::detect_bytes(source)
}
