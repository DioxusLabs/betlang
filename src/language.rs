/// Source class predicted by the embedded Magika student model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Language {
    /// Model label `"asm"`.
    Asm,
    /// Model label `"batch"`.
    Batch,
    /// Model label `"c"`.
    C,
    /// Model label `"cs"`.
    CSharp,
    /// Model label `"clojure"`.
    Clojure,
    /// Model label `"cmake"`.
    CMake,
    /// Model label `"cobol"`.
    Cobol,
    /// Model label `"cpp"`.
    Cpp,
    /// Model label `"css"`.
    Css,
    /// Model label `"dart"`.
    Dart,
    /// Model label `"dockerfile"`.
    Dockerfile,
    /// Model label `"elixir"`.
    Elixir,
    /// Model label `"erlang"`.
    Erlang,
    /// Model label `"gemfile"`.
    Gemfile,
    /// Model label `"gemspec"`.
    Gemspec,
    /// Model label `"go"`.
    Go,
    /// Model label `"gradle"`.
    Gradle,
    /// Model label `"groovy"`.
    Groovy,
    /// Model label `"haskell"`.
    Haskell,
    /// Model label `"html"`.
    Html,
    /// Model label `"ini"`.
    Ini,
    /// Model label `"java"`.
    Java,
    /// Model label `"javascript"`.
    JavaScript,
    /// Model label `"json"`.
    Json,
    /// Model label `"julia"`.
    Julia,
    /// Model label `"kotlin"`.
    Kotlin,
    /// Model label `"lisp"`.
    Lisp,
    /// Model label `"lua"`.
    Lua,
    /// Model label `"markdown"`.
    Markdown,
    /// Model label `"objectivec"`.
    ObjectiveC,
    /// Model label `"ocaml"`.
    Ocaml,
    /// Model label `"perl"`.
    Perl,
    /// Model label `"php"`.
    Php,
    /// Model label `"powershell"`.
    Powershell,
    /// Model label `"python"`.
    Python,
    /// Model label `"r"`.
    R,
    /// Model label `"ruby"`.
    Ruby,
    /// Model label `"rust"`.
    Rust,
    /// Model label `"scala"`.
    Scala,
    /// Model label `"shell"`.
    Shell,
    /// Model label `"sql"`.
    Sql,
    /// Model label `"swift"`.
    Swift,
    /// Model label `"toml"`.
    Toml,
    /// Model label `"typescript"`.
    TypeScript,
    /// Model label `"vba"`.
    Vba,
    /// Model label `"verilog"`.
    Verilog,
    /// Model label `"xml"`.
    Xml,
    /// Model label `"yaml"`.
    Yaml,
}

impl Language {
    /// Filesystem/model label for this detected class.
    pub const fn label(self) -> &'static str {
        match self {
            Self::Asm => "asm",
            Self::Batch => "batch",
            Self::C => "c",
            Self::CSharp => "cs",
            Self::Clojure => "clojure",
            Self::CMake => "cmake",
            Self::Cobol => "cobol",
            Self::Cpp => "cpp",
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

    /// Arborium/tree-sitter language slug for this detected class.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Asm => "asm",
            Self::Batch => "batch",
            Self::C => "c",
            Self::CSharp => "c-sharp",
            Self::Clojure => "clojure",
            Self::CMake => "cmake",
            Self::Cobol => "cobol",
            Self::Cpp => "cpp",
            Self::Css => "css",
            Self::Dart => "dart",
            Self::Dockerfile => "dockerfile",
            Self::Elixir => "elixir",
            Self::Erlang => "erlang",
            Self::Gemfile => "ruby",
            Self::Gemspec => "ruby",
            Self::Go => "go",
            Self::Gradle => "groovy",
            Self::Groovy => "groovy",
            Self::Haskell => "haskell",
            Self::Html => "html",
            Self::Ini => "ini",
            Self::Java => "java",
            Self::JavaScript => "javascript",
            Self::Json => "json",
            Self::Julia => "julia",
            Self::Kotlin => "kotlin",
            Self::Lisp => "commonlisp",
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
            Self::Shell => "bash",
            Self::Sql => "sql",
            Self::Swift => "swift",
            Self::Toml => "toml",
            Self::TypeScript => "typescript",
            Self::Vba => "vb",
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
    Language::Gemfile,
    Language::Gemspec,
    Language::Go,
    Language::Gradle,
    Language::Groovy,
    Language::Haskell,
    Language::Html,
    Language::Ini,
    Language::Java,
    Language::JavaScript,
    Language::Json,
    Language::Julia,
    Language::Kotlin,
    Language::Lisp,
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
    Language::Shell,
    Language::Sql,
    Language::Swift,
    Language::Toml,
    Language::TypeScript,
    Language::Vba,
    Language::Verilog,
    Language::Xml,
    Language::Yaml,
];

pub(crate) const CLASS_LABELS: [&str; 48] = [
    "asm",
    "batch",
    "c",
    "clojure",
    "cmake",
    "cobol",
    "cpp",
    "cs",
    "css",
    "dart",
    "dockerfile",
    "elixir",
    "erlang",
    "gemfile",
    "gemspec",
    "go",
    "gradle",
    "groovy",
    "haskell",
    "html",
    "ini",
    "java",
    "javascript",
    "json",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "markdown",
    "objectivec",
    "ocaml",
    "perl",
    "php",
    "powershell",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "shell",
    "sql",
    "swift",
    "toml",
    "typescript",
    "vba",
    "verilog",
    "xml",
    "yaml",
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn class_mapping_is_one_to_one_with_exported_labels() {
        assert_eq!(CLASS_LANGUAGES.len(), CLASS_LABELS.len());

        let mut variants = HashSet::new();
        for (language, label) in CLASS_LANGUAGES.iter().zip(CLASS_LABELS) {
            assert_eq!(language.label(), label);
            assert!(
                variants.insert(*language),
                "duplicate class mapping: {language:?}"
            );
        }
    }
}
