/// Source language predicted by the embedded Magika student model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    /// Arborium slug `"asm"`.
    Asm,
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
    /// Arborium slug `"html"`.
    Html,
    /// Arborium slug `"ini"`.
    Ini,
    /// Arborium slug `"java"`.
    Java,
    /// Arborium slug `"javascript"`.
    JavaScript,
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
    /// Arborium slug `"objc"`.
    ObjectiveC,
    /// Arborium slug `"ocaml"`.
    Ocaml,
    /// Arborium slug `"perl"`.
    Perl,
    /// Arborium slug `"php"`.
    Php,
    /// Arborium slug `"powershell"`.
    Powershell,
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
    /// Arborium slug `"sql"`.
    Sql,
    /// Arborium slug `"swift"`.
    Swift,
    /// Arborium slug `"toml"`.
    Toml,
    /// Arborium slug `"typescript"`.
    TypeScript,
    /// Arborium slug `"vb"`.
    Vb,
    /// Arborium slug `"verilog"`.
    Verilog,
    /// Arborium slug `"xml"`.
    Xml,
    /// Arborium slug `"yaml"`.
    Yaml,
}

impl Language {
    /// Arborium/tree-sitter language slug for this detected language.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Asm => "asm",
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
            Self::Dockerfile => "dockerfile",
            Self::Elixir => "elixir",
            Self::Erlang => "erlang",
            Self::Go => "go",
            Self::Groovy => "groovy",
            Self::Haskell => "haskell",
            Self::Html => "html",
            Self::Ini => "ini",
            Self::Java => "java",
            Self::JavaScript => "javascript",
            Self::Json => "json",
            Self::Julia => "julia",
            Self::Kotlin => "kotlin",
            Self::Lua => "lua",
            Self::Markdown => "markdown",
            Self::ObjectiveC => "objc",
            Self::Ocaml => "ocaml",
            Self::Perl => "perl",
            Self::Php => "php",
            Self::Powershell => "powershell",
            Self::Python => "python",
            Self::R => "r",
            Self::Ruby => "ruby",
            Self::Rust => "rust",
            Self::Scala => "scala",
            Self::Sql => "sql",
            Self::Swift => "swift",
            Self::Toml => "toml",
            Self::TypeScript => "typescript",
            Self::Vb => "vb",
            Self::Verilog => "verilog",
            Self::Xml => "xml",
            Self::Yaml => "yaml",
        }
    }
}

pub(crate) const CLASS_LANGUAGES: [Language; 48] = [
    Language::Asm,
    Language::Batch,
    Language::C,
    Language::Clojure,
    Language::CMake,
    Language::Cobol,
    Language::Cpp,
    Language::CSharp,
    Language::Css,
    Language::Dart,
    Language::Dockerfile,
    Language::Elixir,
    Language::Erlang,
    Language::Ruby,
    Language::Ruby,
    Language::Go,
    Language::Groovy,
    Language::Groovy,
    Language::Haskell,
    Language::Html,
    Language::Ini,
    Language::Java,
    Language::JavaScript,
    Language::Json,
    Language::Julia,
    Language::Kotlin,
    Language::CommonLisp,
    Language::Lua,
    Language::Markdown,
    Language::ObjectiveC,
    Language::Ocaml,
    Language::Perl,
    Language::Php,
    Language::Powershell,
    Language::Python,
    Language::R,
    Language::Ruby,
    Language::Rust,
    Language::Scala,
    Language::Bash,
    Language::Sql,
    Language::Swift,
    Language::Toml,
    Language::TypeScript,
    Language::Vb,
    Language::Verilog,
    Language::Xml,
    Language::Yaml,
];
