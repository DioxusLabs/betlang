use super::{
    constants::*,
    runtime::Model,
    tokenizer::{hash_unit_bytes, tokenize},
    window::build_window,
};
use crate::Language;
use std::collections::HashSet;
use std::{fs, path::Path};

#[test]
fn loads_embedded_model() {
    let model = Model::get();
    assert_eq!(model.embedding.len(), BINS * EMBED);
    assert_eq!(model.output_kernel.len(), DENSE * CLASSES);
}

#[test]
fn tokenizer_casefolds_and_isolates_brackets() {
    let source = b"Foo(foo)\n";
    let pad = vec![false; source.len()];
    let units = tokenize(source, &pad);

    assert_eq!(units[0] as u32, hash_unit_bytes(b"foo") & WORD_MASK);
    assert!(units.contains(&((BRACKET_FLAG | b'(' as u32) as i32)));
    assert!(units.contains(&((BRACKET_FLAG | b')' as u32) as i32)));
}

#[test]
fn detects_rust_from_source() {
    let detection = crate::detect("use std::fmt;\nfn main() { println!(\"hi\"); }");
    assert_eq!(top_language(&detection), Some(Language::Rust));
}

#[test]
fn detects_python_from_source() {
    let detection = crate::detect(
        "import os\n\ndef main():\n    print('hello world')\n\nif __name__ == '__main__':\n    main()\n",
    );
    assert_eq!(top_language(&detection), Some(Language::Python));
}

#[test]
fn detects_javascript_from_source() {
    let detection = crate::detect(
        "const greet = (name) => { console.log(`Hello, ${name}!`); };\ngreet('world');\n",
    );
    assert_eq!(top_language(&detection), Some(Language::JavaScript));
}

#[test]
fn golden_predictions_cover_representative_sources() {
    let fixtures = [
        (
            Language::Rust,
            "use std::fmt;\nfn main() { println!(\"hi\"); }\n",
        ),
        (
            Language::Python,
            "import pathlib\n\ndef main():\n    print(pathlib.Path.cwd())\n\nif __name__ == '__main__':\n    main()\n",
        ),
        (
            Language::JavaScript,
            "const greet = (name) => {\n  console.log(`hello ${name}`);\n};\ngreet('world');\n",
        ),
        (
            Language::Json,
            r#"{"name":"betlang","version":"0.0.1","keywords":["language","detection"]}"#,
        ),
        (
            Language::Toml,
            "[package]\nname = \"betlang\"\nversion = \"0.0.1\"\nedition = \"2024\"\n",
        ),
        (
            Language::Yaml,
            "name: ci\non:\n  pull_request:\njobs:\n  test:\n    runs-on: ubuntu-latest\n",
        ),
        (
            Language::Html,
            "<!doctype html><html><head><title>Betlang</title></head><body><main>Hello</main></body></html>\n",
        ),
        (
            Language::Css,
            "body {\n  display: grid;\n  grid-template-columns: 1fr;\n  color: #222;\n}\n",
        ),
        (
            Language::Sql,
            "select users.id, users.email from users where users.active = true order by users.id;\n",
        ),
    ];

    for (expected, source) in fixtures {
        let detection = crate::detect(source);
        let Some((probability, language)) = detection.top_languages().next() else {
            panic!("expected a language prediction for {source}");
        };
        assert_eq!(language, expected, "{source}");
        assert_eq!(language.slug(), expected.slug());
        assert!(probability > 0.0, "{source}");
    }
}

#[test]
fn detects_each_language_fixture_file() {
    let mut failures = Vec::new();

    for (expected, path) in LANGUAGE_FIXTURES {
        let source = fs::read(fixture_path(path)).unwrap_or_else(|err| {
            panic!("failed to read fixture {path}: {err}");
        });
        let detection = crate::detect(source);
        let actual = detection.language();

        if actual != Some(expected) {
            let top = detection
                .top_languages()
                .take(3)
                .map(|(probability, language)| format!("{}:{probability:.3}", language.slug()))
                .collect::<Vec<_>>()
                .join(", ");
            failures.push(format!(
                "{path}: expected {}, got {:?}; top [{}]",
                expected.slug(),
                actual.map(Language::slug),
                top
            ));
        }
    }

    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn language_fixtures_have_unique_expected_languages() {
    let mut languages = HashSet::new();

    for (language, _) in LANGUAGE_FIXTURES {
        assert!(
            languages.insert(language),
            "duplicate fixture for {}",
            language.slug()
        );
    }
}

#[test]
fn detect_accepts_non_utf8_inputs() {
    let mut bytes = b"fn main() {\n    println!(\"hello\");\n}\n".to_vec();
    bytes.extend([0xff, 0xfe]);
    let detection = crate::detect(&bytes);
    assert_eq!(top_language(&detection), Some(Language::Rust));
}

#[test]
fn probabilities_sum_to_one_across_public_languages() {
    let detection = crate::detect("use std::fmt;\nfn main() { println!(\"hi\"); }\n");
    let sum: f32 = detection
        .top_languages()
        .map(|(probability, _)| probability)
        .sum();

    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
}

#[test]
fn runtime_inference_pads_short_sources_to_eval_shape() {
    let source = "use std::fmt;\nfn main() { println!(\"hi\"); }\n";
    let Some((bytes, pad)) = build_window(source.as_bytes()) else {
        panic!("expected source to build a model window");
    };
    let model = Model::get();
    let units = model.tokenize_units(&bytes, &pad);
    assert!(units.len() < MAX_UNITS);

    let mut padded = units.clone();
    padded.resize(MAX_UNITS, -1);

    let runtime_logits = model.logits_for_runtime_units(&units);
    let eval_shape_logits = model.logits(&padded, MAX_UNITS);

    for (runtime, eval_shape) in runtime_logits.iter().zip(eval_shape_logits) {
        assert_eq!(*runtime, eval_shape);
    }
}

#[test]
fn empty_input_returns_empty_detection() {
    assert!(crate::detect("").top_languages().next().is_none());
}

#[test]
fn very_short_input_returns_empty_detection() {
    // < 8 non-whitespace bytes
    assert!(crate::detect("hi").top_languages().next().is_none());
}

fn top_language(detection: &crate::Detection) -> Option<Language> {
    detection.language()
}

fn fixture_path(path: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(path)
}

const LANGUAGE_FIXTURES: [(Language, &str); 49] = [
    (Language::Asm, "tests/fixtures/languages/asm.s"),
    (Language::Batch, "tests/fixtures/languages/batch.bat"),
    (Language::Bash, "tests/fixtures/languages/bash.sh"),
    (Language::C, "tests/fixtures/languages/c.c"),
    (Language::CSharp, "tests/fixtures/languages/c-sharp.cs"),
    (Language::Clojure, "tests/fixtures/languages/clojure.clj"),
    (Language::CMake, "tests/fixtures/languages/cmake.cmake"),
    (Language::Cobol, "tests/fixtures/languages/cobol.cob"),
    (
        Language::CommonLisp,
        "tests/fixtures/languages/commonlisp.lisp",
    ),
    (Language::Cpp, "tests/fixtures/languages/cpp.cpp"),
    (Language::Css, "tests/fixtures/languages/css.css"),
    (Language::Diff, "tests/fixtures/languages/diff.diff"),
    (
        Language::Dockerfile,
        "tests/fixtures/languages/dockerfile.Dockerfile",
    ),
    (Language::Elixir, "tests/fixtures/languages/elixir.ex"),
    (Language::Erlang, "tests/fixtures/languages/erlang.erl"),
    (Language::Go, "tests/fixtures/languages/go.go"),
    (Language::Groovy, "tests/fixtures/languages/groovy.gradle"),
    (Language::Haskell, "tests/fixtures/languages/haskell.hs"),
    (Language::Html, "tests/fixtures/languages/html.html"),
    (Language::Ini, "tests/fixtures/languages/ini.ini"),
    (Language::Java, "tests/fixtures/languages/java.java"),
    (
        Language::JavaScript,
        "tests/fixtures/languages/javascript.js",
    ),
    (Language::Json, "tests/fixtures/languages/json.json"),
    (Language::Julia, "tests/fixtures/languages/julia.jl"),
    (Language::Kotlin, "tests/fixtures/languages/kotlin.kt"),
    (Language::Lua, "tests/fixtures/languages/lua.lua"),
    (Language::Markdown, "tests/fixtures/languages/markdown.md"),
    (Language::ObjectiveC, "tests/fixtures/languages/objc.m"),
    (Language::Ocaml, "tests/fixtures/languages/ocaml.ml"),
    (Language::Perl, "tests/fixtures/languages/perl.pl"),
    (Language::Php, "tests/fixtures/languages/php.php"),
    (
        Language::Postscript,
        "tests/fixtures/languages/postscript.ps",
    ),
    (
        Language::Powershell,
        "tests/fixtures/languages/powershell.ps1",
    ),
    (Language::Python, "tests/fixtures/languages/python.py"),
    (Language::R, "tests/fixtures/languages/r.R"),
    (Language::Ruby, "tests/fixtures/languages/ruby.rb"),
    (Language::Rust, "tests/fixtures/languages/rust.rs"),
    (Language::Scala, "tests/fixtures/languages/scala.scala"),
    (Language::Scss, "tests/fixtures/languages/scss.scss"),
    (Language::Sql, "tests/fixtures/languages/sql.sql"),
    (Language::Swift, "tests/fixtures/languages/swift.swift"),
    (Language::Toml, "tests/fixtures/languages/toml.toml"),
    (
        Language::TypeScript,
        "tests/fixtures/languages/typescript.ts",
    ),
    (Language::Vb, "tests/fixtures/languages/vb.vb"),
    (Language::Verilog, "tests/fixtures/languages/verilog.v"),
    (Language::Vhdl, "tests/fixtures/languages/vhdl.vhd"),
    (Language::Vue, "tests/fixtures/languages/vue.vue"),
    (Language::Xml, "tests/fixtures/languages/xml.xml"),
    (Language::Yaml, "tests/fixtures/languages/yaml.yaml"),
];
