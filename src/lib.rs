#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

mod kind;
mod model;

pub use kind::{Kind, ParseKindError};

/// Prompt-vs-shell-command detection result.
///
/// Use [`Detection::kind`] to read the top kind and [`Detection::top_kinds`]
/// to iterate over ranked probability/kind pairs. [`Detection::kind`] returns
/// [`None`] when the input is empty, effectively whitespace only, or too
/// short to build the model window.
///
/// ```
/// let detection = betlang::detect("Write a short poem about the ocean.");
///
/// assert_eq!(detection.kind(), Some(betlang::Kind::Prompt));
/// ```
#[derive(Debug)]
pub struct Detection {
    predictions: Vec<(f32, Kind)>,
}

impl Detection {
    /// Return the most likely detected kind.
    ///
    /// Returns [`None`] when the input is empty, effectively whitespace only, or
    /// too short to build the model window.
    ///
    /// ```
    /// let detection = betlang::detect("Write a short poem about the ocean.");
    ///
    /// assert_eq!(detection.kind(), Some(betlang::Kind::Prompt));
    /// ```
    pub fn kind(&self) -> Option<Kind> {
        self.predictions.first().map(|(_, kind)| *kind)
    }

    /// Return detected kinds sorted from most likely to least likely.
    ///
    /// The iterator yields `(probability, kind)` pairs, one per model output
    /// kind.
    ///
    /// ```
    /// let detection = betlang::detect("Write a short poem about the ocean.");
    /// let Some((probability, kind)) = detection.top_kinds().next() else {
    ///     panic!("expected a kind prediction");
    /// };
    ///
    /// assert_eq!(kind, betlang::Kind::Prompt);
    /// assert!(probability > 0.0);
    /// ```
    pub fn top_kinds(&self) -> impl Iterator<Item = (f32, Kind)> + '_ {
        self.predictions.iter().copied()
    }

    pub(crate) fn from_predictions(predictions: Vec<(f32, Kind)>) -> Self {
        Self { predictions }
    }
}

/// Detect whether bytes-like input is an LLM prompt or a shell command.
///
/// Use [`Kind::slug`] to read the model label slug. [`Detection::kind`]
/// returns [`None`] when the input is empty, effectively whitespace only, or too
/// short to build the model window. The input may be a UTF-8 string, raw byte
/// slice, or another type that can be borrowed as bytes.
///
/// ```
/// let detection = betlang::detect("Write a short poem about the ocean.");
///
/// assert_eq!(detection.kind(), Some(betlang::Kind::Prompt));
/// ```
pub fn detect(source: impl AsRef<[u8]>) -> Detection {
    model::detect(source.as_ref())
}
