use std::{error::Error, fmt, str::FromStr};

/// Error returned when parsing a [`Kind`] from an unknown slug.
///
/// ```
/// let error = "not-a-kind".parse::<betlang::Kind>().unwrap_err();
///
/// assert_eq!(error.to_string(), "unknown betlang kind slug");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseKindError;

impl fmt::Display for ParseKindError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown betlang kind slug")
    }
}

impl Error for ParseKindError {}

/// Kind of text predicted by the embedded student model.
///
/// Kinds parse from their model label slugs with [`str::parse`].
///
/// ```
/// let kind = "prompt".parse::<betlang::Kind>()?;
///
/// assert_eq!(kind, betlang::Kind::Prompt);
/// assert_eq!(kind.slug(), "prompt");
/// # Ok::<(), betlang::ParseKindError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum Kind {
    /// Model label `"natural_language"`: prose that is not addressed to a
    /// model — narrative text, articles, reviews, conversation.
    NaturalLanguage = 0,
    /// Model label `"prompt"`: text written to instruct a language model —
    /// task requests, questions for an assistant, role-play setups.
    Prompt = 1,
}

impl Kind {
    pub(crate) const MODEL_LABEL_COUNT: usize = 2;

    /// Model label slug for this detected kind.
    ///
    /// ```
    /// assert_eq!(betlang::Kind::Prompt.slug(), "prompt");
    /// ```
    pub const fn slug(self) -> &'static str {
        match self {
            Self::NaturalLanguage => "natural_language",
            Self::Prompt => "prompt",
        }
    }

    #[cfg(test)]
    const fn model_index(self) -> usize {
        self as u8 as usize
    }

    pub(crate) fn from_model_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::NaturalLanguage),
            1 => Some(Self::Prompt),
            _ => None,
        }
    }
}

/// Parses a [`Kind`] from its model label slug.
///
/// ```
/// assert_eq!(
///     "natural_language".parse::<betlang::Kind>()?,
///     betlang::Kind::NaturalLanguage
/// );
/// # Ok::<(), betlang::ParseKindError>(())
/// ```
impl FromStr for Kind {
    type Err = ParseKindError;

    fn from_str(slug: &str) -> Result<Self, Self::Err> {
        Ok(match slug {
            "natural_language" => Self::NaturalLanguage,
            "prompt" => Self::Prompt,
            _ => return Err(ParseKindError),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn model_output_kinds_have_unique_slugs_and_roundtrip() {
        let mut slugs = HashSet::new();
        for index in 0..Kind::MODEL_LABEL_COUNT {
            let kind = Kind::from_model_index(index).expect("model label");
            assert_eq!(kind.model_index(), index);
            assert!(slugs.insert(kind.slug()), "duplicate slug {}", kind.slug());
            assert_eq!(kind.slug().parse::<Kind>(), Ok(kind));
        }
        assert_eq!(slugs.len(), Kind::MODEL_LABEL_COUNT);
        assert!(Kind::from_model_index(Kind::MODEL_LABEL_COUNT).is_none());
        assert_eq!("unknown".parse::<Kind>(), Err(ParseKindError));
    }
}
