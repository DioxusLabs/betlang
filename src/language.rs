/// Source language predicted by the embedded Magika student model.
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
