#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod language;
mod model;

pub use language::{Language, ParseLanguageError};

/// Source-language detection result.
///
/// Use [`Detection::language`] to read the top language and
/// [`Detection::top_languages`] to iterate over ranked probability/language
/// pairs. [`Detection::language`] returns [`None`] when the input is empty,
/// effectively whitespace only, or too short to build the model window.
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

    /// Return detected languages sorted from most likely to least likely.
    ///
    /// The iterator yields `(probability, language)` pairs. Probabilities are
    /// aggregated across embedded model classes that map to the same public
    /// [`Language`], so each public language appears at most once.
    ///
    /// ```
    /// let detection = betlang::detect("fn main() { println!(\"hi\"); }");
    /// let (probability, language) = detection.top_languages().next().unwrap();
    ///
    /// assert_eq!(language, betlang::Language::Rust);
    /// assert!(probability > 0.0);
    /// ```
    pub fn top_languages(&self) -> impl Iterator<Item = (f32, Language)> + '_ {
        self.predictions.iter().copied()
    }

    pub(crate) fn from_predictions(predictions: Vec<(f32, Language)>) -> Self {
        Self { predictions }
    }
}

/// Detect the source language for bytes-like input.
///
/// Use [`Language::slug`] to map predicted languages to Arborium/tree-sitter
/// identifiers. [`Detection::language`] returns [`None`] when the input is
/// empty, effectively whitespace only, or too short to build the model window.
/// The input may be a UTF-8 string, raw byte slice, or another type that can be
/// borrowed as bytes.
///
/// ```
/// let detection = betlang::detect("fn main() { println!(\"hi\"); }");
///
/// assert_eq!(detection.language(), Some(betlang::Language::Rust));
/// ```
pub fn detect(source: impl AsRef<[u8]>) -> Detection {
    model::detect(source.as_ref())
}
