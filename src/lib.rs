//! CPU inference for the compact wordseq v3 student.

mod language;
mod model;

pub use language::Language;

/// Detect the source language for a source string.
///
/// Use [`Language::label`] for the exact embedded model class and
/// [`Language::slug`] for an Arborium/tree-sitter language identifier.
pub fn detect(source: &str) -> Option<Language> {
    model::detect(source)
}
