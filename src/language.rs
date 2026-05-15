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
/// Languages parse from their public slugs with [`str::parse`].
///
/// ```
/// let language = "rust".parse::<betlang::Language>()?;
///
/// assert_eq!(language, betlang::Language::Rust);
/// assert_eq!(language.slug(), "rust");
/// # Ok::<(), betlang::ParseLanguageError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    /// Arborium slug `"asm"`.
    Asm,
    /// Arborium slug `"awk"`.
    Awk,
    /// Arborium slug `"batch"`.
    Batch,
    /// Arborium slug `"bash"`.
    Bash,
    /// Arborium slug `"c"`.
    C,
    /// Arborium slug `"c-sharp"`.
    CSharp,
    /// Arborium slug `"clojure"`.
    Clojure,
    /// Arborium slug `"cmake"`.
    CMake,
    /// Arborium slug `"cobol"`.
    Cobol,
    /// Arborium slug `"commonlisp"`.
    CommonLisp,
    /// Arborium slug `"cpp"`.
    Cpp,
    /// Arborium slug `"css"`.
    Css,
    /// Arborium slug `"dart"`.
    Dart,
    /// Arborium slug `"diff"`.
    Diff,
    /// Arborium slug `"dockerfile"`.
    Dockerfile,
    /// Arborium slug `"elixir"`.
    Elixir,
    /// Arborium slug `"erlang"`.
    Erlang,
    /// Arborium slug `"go"`.
    Go,
    /// Arborium slug `"groovy"`.
    Groovy,
    /// Arborium slug `"haskell"`.
    Haskell,
    /// Arborium slug `"hcl"`.
    Hcl,
    /// Arborium slug `"html"`.
    Html,
    /// Arborium slug `"ini"`.
    Ini,
    /// Arborium slug `"java"`.
    Java,
    /// Arborium slug `"javascript"`.
    JavaScript,
    /// Arborium slug `"jinja2"`.
    Jinja2,
    /// Arborium slug `"json"`.
    Json,
    /// Arborium slug `"julia"`.
    Julia,
    /// Arborium slug `"kotlin"`.
    Kotlin,
    /// Arborium slug `"lua"`.
    Lua,
    /// Arborium slug `"markdown"`.
    Markdown,
    /// Arborium slug `"matlab"`.
    Matlab,
    /// Arborium slug `"objc"`.
    ObjectiveC,
    /// Arborium slug `"ocaml"`.
    Ocaml,
    /// Arborium slug `"perl"`.
    Perl,
    /// Arborium slug `"php"`.
    Php,
    /// Arborium slug `"postscript"`.
    Postscript,
    /// Arborium slug `"powershell"`.
    Powershell,
    /// Arborium slug `"prolog"`.
    Prolog,
    /// Arborium slug `"python"`.
    Python,
    /// Arborium slug `"r"`.
    R,
    /// Arborium slug `"ruby"`.
    Ruby,
    /// Arborium slug `"rust"`.
    Rust,
    /// Arborium slug `"scala"`.
    Scala,
    /// Arborium slug `"scss"`.
    Scss,
    /// Arborium slug `"solidity"`.
    Solidity,
    /// Arborium slug `"sql"`.
    Sql,
    /// Arborium slug `"starlark"`.
    Starlark,
    /// Arborium slug `"swift"`.
    Swift,
    /// Arborium slug `"textproto"`.
    TextProto,
    /// Arborium slug `"toml"`.
    Toml,
    /// Arborium slug `"typescript"`.
    TypeScript,
    /// Arborium slug `"vb"`.
    Vb,
    /// Arborium slug `"verilog"`.
    Verilog,
    /// Arborium slug `"vhdl"`.
    Vhdl,
    /// Arborium slug `"vue"`.
    Vue,
    /// Arborium slug `"xml"`.
    Xml,
    /// Arborium slug `"yaml"`.
    Yaml,
    /// Arborium slug `"zig"`.
    Zig,
}

impl Language {
    /// Arborium/tree-sitter language slug for this detected language.
    ///
    /// ```
    /// assert_eq!(betlang::Language::Rust.slug(), "rust");
    /// ```
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Asm => "asm",
            Self::Awk => "awk",
            Self::Batch => "batch",
            Self::Bash => "bash",
            Self::C => "c",
            Self::CSharp => "c-sharp",
            Self::Clojure => "clojure",
            Self::CMake => "cmake",
            Self::Cobol => "cobol",
            Self::CommonLisp => "commonlisp",
            Self::Cpp => "cpp",
            Self::Css => "css",
            Self::Dart => "dart",
            Self::Diff => "diff",
            Self::Dockerfile => "dockerfile",
            Self::Elixir => "elixir",
            Self::Erlang => "erlang",
            Self::Go => "go",
            Self::Groovy => "groovy",
            Self::Haskell => "haskell",
            Self::Hcl => "hcl",
            Self::Html => "html",
            Self::Ini => "ini",
            Self::Java => "java",
            Self::JavaScript => "javascript",
            Self::Jinja2 => "jinja2",
            Self::Json => "json",
            Self::Julia => "julia",
            Self::Kotlin => "kotlin",
            Self::Lua => "lua",
            Self::Markdown => "markdown",
            Self::Matlab => "matlab",
            Self::ObjectiveC => "objc",
            Self::Ocaml => "ocaml",
            Self::Perl => "perl",
            Self::Php => "php",
            Self::Postscript => "postscript",
            Self::Powershell => "powershell",
            Self::Prolog => "prolog",
            Self::Python => "python",
            Self::R => "r",
            Self::Ruby => "ruby",
            Self::Rust => "rust",
            Self::Scala => "scala",
            Self::Scss => "scss",
            Self::Solidity => "solidity",
            Self::Sql => "sql",
            Self::Starlark => "starlark",
            Self::Swift => "swift",
            Self::TextProto => "textproto",
            Self::Toml => "toml",
            Self::TypeScript => "typescript",
            Self::Vb => "vb",
            Self::Verilog => "verilog",
            Self::Vhdl => "vhdl",
            Self::Vue => "vue",
            Self::Xml => "xml",
            Self::Yaml => "yaml",
            Self::Zig => "zig",
        }
    }
}

/// Parses a [`Language`] from its public slug.
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
            "awk" => Self::Awk,
            "batch" => Self::Batch,
            "bash" => Self::Bash,
            "c" => Self::C,
            "c-sharp" => Self::CSharp,
            "clojure" => Self::Clojure,
            "cmake" => Self::CMake,
            "cobol" => Self::Cobol,
            "commonlisp" => Self::CommonLisp,
            "cpp" => Self::Cpp,
            "css" => Self::Css,
            "dart" => Self::Dart,
            "diff" => Self::Diff,
            "dockerfile" => Self::Dockerfile,
            "elixir" => Self::Elixir,
            "erlang" => Self::Erlang,
            "go" => Self::Go,
            "groovy" => Self::Groovy,
            "haskell" => Self::Haskell,
            "hcl" => Self::Hcl,
            "html" => Self::Html,
            "ini" => Self::Ini,
            "java" => Self::Java,
            "javascript" => Self::JavaScript,
            "jinja2" => Self::Jinja2,
            "json" => Self::Json,
            "julia" => Self::Julia,
            "kotlin" => Self::Kotlin,
            "lua" => Self::Lua,
            "markdown" => Self::Markdown,
            "matlab" => Self::Matlab,
            "objc" => Self::ObjectiveC,
            "ocaml" => Self::Ocaml,
            "perl" => Self::Perl,
            "php" => Self::Php,
            "postscript" => Self::Postscript,
            "powershell" => Self::Powershell,
            "prolog" => Self::Prolog,
            "python" => Self::Python,
            "r" => Self::R,
            "ruby" => Self::Ruby,
            "rust" => Self::Rust,
            "scala" => Self::Scala,
            "scss" => Self::Scss,
            "solidity" => Self::Solidity,
            "sql" => Self::Sql,
            "starlark" => Self::Starlark,
            "swift" => Self::Swift,
            "textproto" => Self::TextProto,
            "toml" => Self::Toml,
            "typescript" => Self::TypeScript,
            "vb" => Self::Vb,
            "verilog" => Self::Verilog,
            "vhdl" => Self::Vhdl,
            "vue" => Self::Vue,
            "xml" => Self::Xml,
            "yaml" => Self::Yaml,
            "zig" => Self::Zig,
            _ => return Err(ParseLanguageError),
        })
    }
}

pub(crate) const CLASS_LANGUAGES: [Language; 67] = [
    Language::Asm,
    Language::Awk,
    Language::Batch,
    Language::Starlark,
    Language::C,
    Language::Clojure,
    Language::CMake,
    Language::Cobol,
    Language::Cpp,
    Language::CSharp,
    Language::Xml,
    Language::Css,
    Language::Dart,
    Language::Diff,
    Language::Dockerfile,
    Language::Elixir,
    Language::Ruby,
    Language::Erlang,
    Language::Ruby,
    Language::Ruby,
    Language::Go,
    Language::Groovy,
    Language::Groovy,
    Language::Haskell,
    Language::Hcl,
    Language::Html,
    Language::Ini,
    Language::Json,
    Language::Java,
    Language::JavaScript,
    Language::Jinja2,
    Language::Json,
    Language::Json,
    Language::Julia,
    Language::Kotlin,
    Language::CommonLisp,
    Language::Lua,
    Language::Markdown,
    Language::Matlab,
    Language::ObjectiveC,
    Language::Ocaml,
    Language::Perl,
    Language::Php,
    Language::Postscript,
    Language::Powershell,
    Language::Prolog,
    Language::Python,
    Language::R,
    Language::Ruby,
    Language::Rust,
    Language::Scala,
    Language::Scss,
    Language::Bash,
    Language::Solidity,
    Language::Sql,
    Language::Swift,
    Language::TextProto,
    Language::Toml,
    Language::TypeScript,
    Language::Vb,
    Language::Xml,
    Language::Verilog,
    Language::Vhdl,
    Language::Vue,
    Language::Xml,
    Language::Yaml,
    Language::Zig,
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    const PUBLIC_LANGUAGES: [Language; 59] = [
        Language::Asm,
        Language::Awk,
        Language::Batch,
        Language::Bash,
        Language::C,
        Language::CSharp,
        Language::Clojure,
        Language::CMake,
        Language::Cobol,
        Language::CommonLisp,
        Language::Cpp,
        Language::Css,
        Language::Dart,
        Language::Diff,
        Language::Dockerfile,
        Language::Elixir,
        Language::Erlang,
        Language::Go,
        Language::Groovy,
        Language::Haskell,
        Language::Hcl,
        Language::Html,
        Language::Ini,
        Language::Java,
        Language::JavaScript,
        Language::Jinja2,
        Language::Json,
        Language::Julia,
        Language::Kotlin,
        Language::Lua,
        Language::Markdown,
        Language::Matlab,
        Language::ObjectiveC,
        Language::Ocaml,
        Language::Perl,
        Language::Php,
        Language::Postscript,
        Language::Powershell,
        Language::Prolog,
        Language::Python,
        Language::R,
        Language::Ruby,
        Language::Rust,
        Language::Scala,
        Language::Scss,
        Language::Solidity,
        Language::Sql,
        Language::Starlark,
        Language::Swift,
        Language::TextProto,
        Language::Toml,
        Language::TypeScript,
        Language::Vb,
        Language::Verilog,
        Language::Vhdl,
        Language::Vue,
        Language::Xml,
        Language::Yaml,
        Language::Zig,
    ];

    #[test]
    fn public_languages_have_unique_slugs_and_roundtrip() {
        let mut slugs = HashSet::new();
        for language in PUBLIC_LANGUAGES {
            assert!(
                slugs.insert(language.slug()),
                "duplicate slug {}",
                language.slug()
            );
            assert_eq!(language.slug().parse::<Language>(), Ok(language));
        }
        assert_eq!("unknown".parse::<Language>(), Err(ParseLanguageError));
    }
}
