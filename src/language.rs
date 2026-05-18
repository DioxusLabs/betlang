use std::{error::Error, fmt, str::FromStr};

/// Error returned when parsing a [`Language`] from an unknown slug.
///
/// ```
/// let error = "not-a-language"
///     .parse::<betlang::Language>()
///     .unwrap_err();
///
/// assert_eq!(error.to_string(), "unknown betlang language slug");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParseLanguageError;

impl fmt::Display for ParseLanguageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown betlang language slug")
    }
}

impl Error for ParseLanguageError {}

/// Source language predicted by the embedded Magika student model.
///
/// Languages parse from their model label slugs with [`str::parse`].
///
/// ```
/// let language = "rust".parse::<betlang::Language>()?;
///
/// assert_eq!(language, betlang::Language::Rust);
/// assert_eq!(language.slug(), "rust");
/// # Ok::<(), betlang::ParseLanguageError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[non_exhaustive]
pub enum Language {
    /// Model label `"asm"`.
    Asm = 0,
    /// Model label `"batch"`.
    Batch = 1,
    /// Model label `"c"`.
    C = 2,
    /// Model label `"clojure"`.
    Clojure = 3,
    /// Model label `"cmake"`.
    CMake = 4,
    /// Model label `"cobol"`.
    Cobol = 5,
    /// Model label `"cpp"`.
    Cpp = 6,
    /// Model label `"cs"`.
    Cs = 7,
    /// Model label `"css"`.
    Css = 8,
    /// Model label `"dart"`.
    Dart = 9,
    /// Model label `"dockerfile"`.
    Dockerfile = 10,
    /// Model label `"elixir"`.
    Elixir = 11,
    /// Model label `"erlang"`.
    Erlang = 12,
    /// Model label `"gemfile"`.
    Gemfile = 13,
    /// Model label `"gemspec"`.
    Gemspec = 14,
    /// Model label `"go"`.
    Go = 15,
    /// Model label `"gradle"`.
    Gradle = 16,
    /// Model label `"groovy"`.
    Groovy = 17,
    /// Model label `"haskell"`.
    Haskell = 18,
    /// Model label `"html"`.
    Html = 19,
    /// Model label `"ini"`.
    Ini = 20,
    /// Model label `"java"`.
    Java = 21,
    /// Model label `"javascript"`.
    JavaScript = 22,
    /// Model label `"json"`.
    Json = 23,
    /// Model label `"julia"`.
    Julia = 24,
    /// Model label `"kotlin"`.
    Kotlin = 25,
    /// Model label `"lisp"`.
    Lisp = 26,
    /// Model label `"lua"`.
    Lua = 27,
    /// Model label `"markdown"`.
    Markdown = 28,
    /// Model label `"objectivec"`.
    ObjectiveC = 29,
    /// Model label `"ocaml"`.
    Ocaml = 30,
    /// Model label `"perl"`.
    Perl = 31,
    /// Model label `"php"`.
    Php = 32,
    /// Model label `"powershell"`.
    Powershell = 33,
    /// Model label `"python"`.
    Python = 34,
    /// Model label `"r"`.
    R = 35,
    /// Model label `"ruby"`.
    Ruby = 36,
    /// Model label `"rust"`.
    Rust = 37,
    /// Model label `"scala"`.
    Scala = 38,
    /// Model label `"shell"`.
    Shell = 39,
    /// Model label `"sql"`.
    Sql = 40,
    /// Model label `"swift"`.
    Swift = 41,
    /// Model label `"toml"`.
    Toml = 42,
    /// Model label `"typescript"`.
    TypeScript = 43,
    /// Model label `"vba"`.
    Vba = 44,
    /// Model label `"verilog"`.
    Verilog = 45,
    /// Model label `"xml"`.
    Xml = 46,
    /// Model label `"yaml"`.
    Yaml = 47,
}

impl Language {
    pub(crate) const MODEL_LABEL_COUNT: usize = 48;

    /// Model label slug for this detected language.
    ///
    /// ```
    /// assert_eq!(betlang::Language::Rust.slug(), "rust");
    /// ```
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Asm => "asm",
            Self::Batch => "batch",
            Self::C => "c",
            Self::Clojure => "clojure",
            Self::CMake => "cmake",
            Self::Cobol => "cobol",
            Self::Cpp => "cpp",
            Self::Cs => "cs",
            Self::Css => "css",
            Self::Dart => "dart",
            Self::Dockerfile => "dockerfile",
            Self::Elixir => "elixir",
            Self::Erlang => "erlang",
            Self::Gemfile => "gemfile",
            Self::Gemspec => "gemspec",
            Self::Go => "go",
            Self::Gradle => "gradle",
            Self::Groovy => "groovy",
            Self::Haskell => "haskell",
            Self::Html => "html",
            Self::Ini => "ini",
            Self::Java => "java",
            Self::JavaScript => "javascript",
            Self::Json => "json",
            Self::Julia => "julia",
            Self::Kotlin => "kotlin",
            Self::Lisp => "lisp",
            Self::Lua => "lua",
            Self::Markdown => "markdown",
            Self::ObjectiveC => "objectivec",
            Self::Ocaml => "ocaml",
            Self::Perl => "perl",
            Self::Php => "php",
            Self::Powershell => "powershell",
            Self::Python => "python",
            Self::R => "r",
            Self::Ruby => "ruby",
            Self::Rust => "rust",
            Self::Scala => "scala",
            Self::Shell => "shell",
            Self::Sql => "sql",
            Self::Swift => "swift",
            Self::Toml => "toml",
            Self::TypeScript => "typescript",
            Self::Vba => "vba",
            Self::Verilog => "verilog",
            Self::Xml => "xml",
            Self::Yaml => "yaml",
        }
    }

    #[cfg(test)]
    const fn model_index(self) -> usize {
        self as u8 as usize
    }

    pub(crate) fn from_model_index(index: usize) -> Option<Self> {
        match index {
            0 => Some(Self::Asm),
            1 => Some(Self::Batch),
            2 => Some(Self::C),
            3 => Some(Self::Clojure),
            4 => Some(Self::CMake),
            5 => Some(Self::Cobol),
            6 => Some(Self::Cpp),
            7 => Some(Self::Cs),
            8 => Some(Self::Css),
            9 => Some(Self::Dart),
            10 => Some(Self::Dockerfile),
            11 => Some(Self::Elixir),
            12 => Some(Self::Erlang),
            13 => Some(Self::Gemfile),
            14 => Some(Self::Gemspec),
            15 => Some(Self::Go),
            16 => Some(Self::Gradle),
            17 => Some(Self::Groovy),
            18 => Some(Self::Haskell),
            19 => Some(Self::Html),
            20 => Some(Self::Ini),
            21 => Some(Self::Java),
            22 => Some(Self::JavaScript),
            23 => Some(Self::Json),
            24 => Some(Self::Julia),
            25 => Some(Self::Kotlin),
            26 => Some(Self::Lisp),
            27 => Some(Self::Lua),
            28 => Some(Self::Markdown),
            29 => Some(Self::ObjectiveC),
            30 => Some(Self::Ocaml),
            31 => Some(Self::Perl),
            32 => Some(Self::Php),
            33 => Some(Self::Powershell),
            34 => Some(Self::Python),
            35 => Some(Self::R),
            36 => Some(Self::Ruby),
            37 => Some(Self::Rust),
            38 => Some(Self::Scala),
            39 => Some(Self::Shell),
            40 => Some(Self::Sql),
            41 => Some(Self::Swift),
            42 => Some(Self::Toml),
            43 => Some(Self::TypeScript),
            44 => Some(Self::Vba),
            45 => Some(Self::Verilog),
            46 => Some(Self::Xml),
            47 => Some(Self::Yaml),
            _ => None,
        }
    }
}

/// Parses a [`Language`] from its model label slug.
///
/// ```
/// assert_eq!("rust".parse::<betlang::Language>()?, betlang::Language::Rust);
/// # Ok::<(), betlang::ParseLanguageError>(())
/// ```
impl FromStr for Language {
    type Err = ParseLanguageError;

    fn from_str(slug: &str) -> Result<Self, Self::Err> {
        Ok(match slug {
            "asm" => Self::Asm,
            "batch" => Self::Batch,
            "c" => Self::C,
            "clojure" => Self::Clojure,
            "cmake" => Self::CMake,
            "cobol" => Self::Cobol,
            "cpp" => Self::Cpp,
            "cs" => Self::Cs,
            "css" => Self::Css,
            "dart" => Self::Dart,
            "dockerfile" => Self::Dockerfile,
            "elixir" => Self::Elixir,
            "erlang" => Self::Erlang,
            "gemfile" => Self::Gemfile,
            "gemspec" => Self::Gemspec,
            "go" => Self::Go,
            "gradle" => Self::Gradle,
            "groovy" => Self::Groovy,
            "haskell" => Self::Haskell,
            "html" => Self::Html,
            "ini" => Self::Ini,
            "java" => Self::Java,
            "javascript" => Self::JavaScript,
            "json" => Self::Json,
            "julia" => Self::Julia,
            "kotlin" => Self::Kotlin,
            "lisp" => Self::Lisp,
            "lua" => Self::Lua,
            "markdown" => Self::Markdown,
            "objectivec" => Self::ObjectiveC,
            "ocaml" => Self::Ocaml,
            "perl" => Self::Perl,
            "php" => Self::Php,
            "powershell" => Self::Powershell,
            "python" => Self::Python,
            "r" => Self::R,
            "ruby" => Self::Ruby,
            "rust" => Self::Rust,
            "scala" => Self::Scala,
            "shell" => Self::Shell,
            "sql" => Self::Sql,
            "swift" => Self::Swift,
            "toml" => Self::Toml,
            "typescript" => Self::TypeScript,
            "vba" => Self::Vba,
            "verilog" => Self::Verilog,
            "xml" => Self::Xml,
            "yaml" => Self::Yaml,
            _ => return Err(ParseLanguageError),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn model_output_languages_have_unique_slugs_and_roundtrip() {
        let mut slugs = HashSet::new();
        for index in 0..Language::MODEL_LABEL_COUNT {
            let language = Language::from_model_index(index).expect("model label");
            assert_eq!(language.model_index(), index);
            assert!(
                slugs.insert(language.slug()),
                "duplicate slug {}",
                language.slug()
            );
            assert_eq!(language.slug().parse::<Language>(), Ok(language));
        }
        assert_eq!(slugs.len(), Language::MODEL_LABEL_COUNT);
        assert!(Language::from_model_index(Language::MODEL_LABEL_COUNT).is_none());
        assert_eq!("unknown".parse::<Language>(), Err(ParseLanguageError));
    }
}
