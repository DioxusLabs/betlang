//! CPU inference for the compact Magika source-language student.

mod language;
mod model;

pub use language::Language;

/// Detect the source language for a source string.
///
/// Use [`Language::slug`] to map the result to an Arborium/tree-sitter language
/// identifier.
pub fn detect(source: &str) -> Option<Language> {
    model::detect(source)
}
