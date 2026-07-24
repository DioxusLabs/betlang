use super::{
    constants::*,
    layers::{Tensor, conv_gelu_global_pool_tensor, conv_gelu_maxpool_tensor},
    runtime::Model,
    tokenizer::{hash_unit_bytes, tokenize},
    window::build_window,
};
use crate::Language;
use rand::{Rng, SeedableRng, rngs::StdRng};
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
    let window = build_window(source).expect("source should build a model window");
    let units = tokenize(&window);

    assert_eq!(units[0] as u32, hash_unit_bytes(b"foo") & WORD_MASK);
    assert!(units.contains(&((BRACKET_FLAG | b'(' as u32) as i32)));
    assert!(units.contains(&((BRACKET_FLAG | b')' as u32) as i32)));
}

#[test]
fn tokenizer_matches_legacy_buffer_model_on_fuzzed_windows() {
    let mut rng = StdRng::seed_from_u64(0x544f_4b45_4e49_5a45);

    for case in [
        &b"abc123!x"[..],
        b"1.2 .. ...\r\n\tfoo(bar)[baz]{qux}",
        b"        indented\n    next\n",
        b"UPPER_lower 000.111 <=> -> ::",
    ] {
        let window = build_window(case).expect("case should build a model window");
        assert_eq!(tokenize(&window), legacy_tokenize_bytes(window.bytes()));
    }

    for _ in 0..1024 {
        let len = rng.gen_range(8..6000);
        let mut source = Vec::with_capacity(len);
        source.extend(std::iter::repeat_n(b' ', rng.gen_range(0..16)));
        while source.len() < len {
            source.push(rng.gen_range(0..=255));
        }

        if let Some(window) = build_window(&source) {
            assert_eq!(tokenize(&window), legacy_tokenize_bytes(window.bytes()));
        }
    }
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
fn language_fixtures_cover_model_languages() {
    let fixture_languages = LANGUAGE_FIXTURES
        .into_iter()
        .map(|(language, _)| language)
        .collect::<HashSet<_>>();
    let model_languages = (0..Language::MODEL_LABEL_COUNT)
        .map(|index| Language::from_model_index(index).expect("model label"))
        .collect::<HashSet<_>>();

    assert_eq!(fixture_languages, model_languages);
}

/// Issue #5: a heading plus a bare single-word bullet list is valid Markdown
/// and valid YAML. The Magika teacher labels it markdown with high confidence;
/// the student should agree instead of assigning YAML >0.9.
#[test]
fn ambiguous_markdown_list_prefers_markdown() {
    let detection = crate::detect("# Heading\n\n- first\n- second\n- third\n- fourth\n- fifth");
    assert_eq!(top_language(&detection), Some(Language::Markdown));
}

#[test]
fn markdown_list_with_capitalized_items_prefers_markdown() {
    let detection = crate::detect("# Names\n\n- Alice\n- Bob\n- Carol\n- Dave");
    assert_eq!(top_language(&detection), Some(Language::Markdown));
}

#[test]
fn yaml_sequence_of_mappings_stays_yaml() {
    let detection = crate::detect("- name: build\n  run: make\n- name: test\n  run: make test");
    assert_eq!(top_language(&detection), Some(Language::Yaml));
}

/// The Magika teacher is nearly split (yaml 0.54 / markdown 0.44) on a
/// comment-or-heading followed by a keyed sequence, so only require that the
/// model ranks the two plausible readings first and second.
#[test]
fn commented_yaml_sequence_ranks_yaml_and_markdown_first() {
    let detection = crate::detect("# comment\nitems:\n- first\n- second\n- third");
    let top: Vec<(f32, Language)> = detection.top_languages().take(2).collect();
    let languages = [top[0].1, top[1].1];

    assert!(languages.contains(&Language::Yaml), "{top:?}");
    assert!(languages.contains(&Language::Markdown), "{top:?}");
}

/// A bare `- item` list with no heading is valid YAML and valid Markdown, and
/// the teacher is split between the two. The model should rank them first and
/// second without near-certain confidence in either.
#[test]
fn bare_dash_list_stays_uncertain_between_yaml_and_markdown() {
    let detection = crate::detect("- first\n- second\n- third\n- fourth\n- fifth");
    let top: Vec<(f32, Language)> = detection.top_languages().take(2).collect();
    let languages = [top[0].1, top[1].1];

    assert!(languages.contains(&Language::Yaml), "{top:?}");
    assert!(languages.contains(&Language::Markdown), "{top:?}");
    assert!(
        top[0].0 < 0.9,
        "top prediction should stay uncertain: {top:?}"
    );
}

#[test]
fn detect_accepts_non_utf8_inputs() {
    let mut bytes = b"fn main() {\n    println!(\"hello\");\n}\n".to_vec();
    bytes.extend([0xff, 0xfe]);
    let detection = crate::detect(&bytes);
    assert_eq!(top_language(&detection), Some(Language::Rust));
}

#[test]
fn probabilities_sum_to_one_across_model_languages() {
    let detection = crate::detect("use std::fmt;\nfn main() { println!(\"hi\"); }\n");
    let sum: f32 = detection
        .top_languages()
        .map(|(probability, _)| probability)
        .sum();

    assert!((sum - 1.0).abs() < 1e-5, "{sum}");
}

#[test]
fn runtime_inference_accepts_short_sources() {
    let source = "use std::fmt;\nfn main() { println!(\"hi\"); }\n";
    let Some(window) = build_window(source.as_bytes()) else {
        panic!("expected source to build a model window");
    };
    let model = Model::get();
    let units = model.tokenize_units(&window);
    assert!(units.len() < MAX_UNITS);

    let logits = model.logits(&units);
    assert!(logits.iter().all(|logit| logit.is_finite()));
}

#[test]
fn repeated_tensor_layers_match_full_convolution() {
    for seed in 0..32 {
        check_repeated_tensor_layers(seed, None);
    }
    check_repeated_tensor_layers(32, Some(128));
}

fn check_repeated_tensor_layers(seed: u64, tail_start: Option<usize>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let seq_len = 128;
    let in_channels = 8;
    let mid_channels = 16;
    let out_channels = 12;
    let tail_start = tail_start.unwrap_or_else(|| rng.gen_range(8..seq_len - 8));

    let mut input = random_f32s(&mut rng, seq_len * in_channels);
    for value in &mut input[tail_start * in_channels..] {
        *value = 0.0;
    }
    let input_tensor = Tensor::with_repeated_tail(&input, seq_len, in_channels, tail_start);

    let kernel0 = random_f32s(&mut rng, 5 * in_channels * mid_channels);
    let bias0 = random_f32s(&mut rng, mid_channels);
    let mut full_pool = vec![0.0; (seq_len / 4) * mid_channels];
    let mut const_pool = vec![0.0; full_pool.len()];
    let mut scratch = vec![0.0; 4 * mid_channels.max(out_channels)];
    let full_input_tensor = Tensor::with_repeated_tail(&input, seq_len, in_channels, seq_len);
    conv_gelu_maxpool_tensor(
        full_input_tensor,
        &kernel0,
        5,
        mid_channels,
        &bias0,
        4,
        &mut full_pool,
        &mut scratch,
    );
    let pool_tensor = conv_gelu_maxpool_tensor(
        input_tensor,
        &kernel0,
        5,
        mid_channels,
        &bias0,
        4,
        &mut const_pool,
        &mut scratch,
    );
    let mut materialized_pool = vec![0.0; full_pool.len()];
    pool_tensor.copy_to_dense(&mut materialized_pool);
    assert_f32s_eq(&materialized_pool, &full_pool);

    let kernel1 = random_f32s(&mut rng, 3 * mid_channels * out_channels);
    let bias1 = random_f32s(&mut rng, out_channels);
    let full_pool_tensor =
        Tensor::with_repeated_tail(&full_pool, seq_len / 4, mid_channels, seq_len / 4);
    let mut full_max = vec![0.0; out_channels];
    let mut full_avg = vec![0.0; out_channels];
    let mut full_tmp = vec![0.0; (seq_len / 4) * out_channels];
    conv_gelu_global_pool_tensor(
        full_pool_tensor,
        &kernel1,
        3,
        out_channels,
        &bias1,
        &mut full_max,
        &mut full_avg,
        &mut full_tmp,
        &mut scratch,
    );

    let mut const_max = vec![0.0; out_channels];
    let mut const_avg = vec![0.0; out_channels];
    let mut tmp = vec![0.0; (seq_len / 4) * out_channels];
    conv_gelu_global_pool_tensor(
        pool_tensor,
        &kernel1,
        3,
        out_channels,
        &bias1,
        &mut const_max,
        &mut const_avg,
        &mut tmp,
        &mut scratch,
    );
    assert_f32s_eq(&const_max, &full_max);
    assert_f32s_eq(&const_avg, &full_avg);
}

fn random_f32s(rng: &mut StdRng, len: usize) -> Vec<f32> {
    (0..len).map(|_| rng.gen_range(-0.25..0.25)).collect()
}

fn legacy_tokenize_bytes(bytes: &[u8]) -> Vec<i32> {
    let mut out: Vec<i32> = Vec::with_capacity(MAX_UNITS);
    let mut word: Vec<u8> = Vec::new();
    let mut number: Vec<u8> = Vec::new();
    let mut punct: Vec<u8> = Vec::new();
    let mut at_line_start = true;
    let mut indent_units: u32 = 0;

    for &raw_value in bytes {
        let value = raw_value.to_ascii_lowercase();
        let is_letter = value.is_ascii_lowercase() || value == b'_';
        let is_digit = value.is_ascii_digit();
        let is_newline = value == b'\n';
        let is_cr = value == b'\r';
        let is_space = value == b' ' || value == b'\t';
        let is_bracket = matches!(value, b'(' | b')' | b'[' | b']' | b'{' | b'}');

        if !is_letter {
            legacy_flush(&mut word, &mut out, 0);
        }
        if !(is_digit || value == b'.') {
            legacy_flush(&mut number, &mut out, NUM_FLAG);
        }
        let need_flush_punct =
            is_letter || is_digit || is_space || is_newline || is_cr || is_bracket || value == b'.';
        if need_flush_punct {
            legacy_flush(&mut punct, &mut out, PUNCT_FLAG);
        }

        if out.len() >= MAX_UNITS {
            break;
        }

        if is_letter {
            if at_line_start {
                push_legacy_indent(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            word.push(value);
            continue;
        }
        if is_digit || value == b'.' {
            if value == b'.' && number.is_empty() {
                if at_line_start {
                    push_legacy_indent(&mut out, indent_units);
                }
                at_line_start = false;
                indent_units = 0;
                punct.push(value);
                continue;
            }
            if at_line_start {
                push_legacy_indent(&mut out, indent_units);
            }
            at_line_start = false;
            indent_units = 0;
            number.push(value);
            continue;
        }
        if is_newline {
            if at_line_start {
                push_legacy_indent(&mut out, indent_units);
            }
            if out.len() < MAX_UNITS {
                out.push(((b'\n' as u32) | PUNCT_FLAG) as i32);
            }
            at_line_start = true;
            indent_units = 0;
            continue;
        }
        if is_cr {
            continue;
        }
        if at_line_start && is_space {
            indent_units += if value == b' ' { 1 } else { 4 };
            continue;
        }
        if at_line_start {
            push_legacy_indent(&mut out, indent_units);
        }
        at_line_start = false;
        indent_units = 0;
        if is_space {
            let space_token = ((b' ' as u32) | PUNCT_FLAG) as i32;
            if out.last() != Some(&space_token) && out.len() < MAX_UNITS {
                out.push(space_token);
            }
            continue;
        }
        if is_bracket {
            if out.len() < MAX_UNITS {
                out.push(((value as u32) | BRACKET_FLAG) as i32);
            }
            continue;
        }
        punct.push(value);
    }

    legacy_flush(&mut word, &mut out, 0);
    legacy_flush(&mut number, &mut out, NUM_FLAG);
    legacy_flush(&mut punct, &mut out, PUNCT_FLAG);
    out
}

fn legacy_flush(buffer: &mut Vec<u8>, out: &mut Vec<i32>, flag: u32) {
    if !buffer.is_empty() && out.len() < MAX_UNITS {
        out.push(((hash_unit_bytes(buffer) & WORD_MASK) | flag) as i32);
    }
    buffer.clear();
}

fn push_legacy_indent(out: &mut Vec<i32>, indent: u32) {
    if indent > 0 && out.len() < MAX_UNITS {
        out.push((indent.min(63) | INDENT_FLAG) as i32);
    }
}

fn assert_f32s_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "index {index}");
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

const LANGUAGE_FIXTURES: [(Language, &str); 48] = [
    (Language::Asm, "tests/fixtures/languages/asm.s"),
    (Language::Batch, "tests/fixtures/languages/batch.bat"),
    (Language::C, "tests/fixtures/languages/c.c"),
    (Language::Clojure, "tests/fixtures/languages/clojure.clj"),
    (Language::CMake, "tests/fixtures/languages/cmake.cmake"),
    (Language::Cobol, "tests/fixtures/languages/cobol.cob"),
    (Language::Cpp, "tests/fixtures/languages/cpp.cpp"),
    (Language::Cs, "tests/fixtures/languages/c-sharp.cs"),
    (Language::Css, "tests/fixtures/languages/css.css"),
    (Language::Dart, "tests/fixtures/languages/dart.dart"),
    (
        Language::Dockerfile,
        "tests/fixtures/languages/dockerfile.Dockerfile",
    ),
    (Language::Elixir, "tests/fixtures/languages/elixir.ex"),
    (Language::Erlang, "tests/fixtures/languages/erlang.erl"),
    (Language::Gemfile, "tests/fixtures/languages/Gemfile"),
    (
        Language::Gemspec,
        "tests/fixtures/languages/gemspec.gemspec",
    ),
    (Language::Go, "tests/fixtures/languages/go.go"),
    (Language::Gradle, "tests/fixtures/languages/gradle.gradle"),
    (Language::Groovy, "tests/fixtures/languages/groovy.groovy"),
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
    (Language::Lisp, "tests/fixtures/languages/commonlisp.lisp"),
    (Language::Lua, "tests/fixtures/languages/lua.lua"),
    (Language::Markdown, "tests/fixtures/languages/markdown.md"),
    (Language::ObjectiveC, "tests/fixtures/languages/objc.m"),
    (Language::Ocaml, "tests/fixtures/languages/ocaml.ml"),
    (Language::Perl, "tests/fixtures/languages/perl.pl"),
    (Language::Php, "tests/fixtures/languages/php.php"),
    (
        Language::Powershell,
        "tests/fixtures/languages/powershell.ps1",
    ),
    (Language::Python, "tests/fixtures/languages/python.py"),
    (Language::R, "tests/fixtures/languages/r.R"),
    (Language::Ruby, "tests/fixtures/languages/ruby.rb"),
    (Language::Rust, "tests/fixtures/languages/rust.rs"),
    (Language::Scala, "tests/fixtures/languages/scala.scala"),
    (Language::Shell, "tests/fixtures/languages/bash.sh"),
    (Language::Sql, "tests/fixtures/languages/sql.sql"),
    (Language::Swift, "tests/fixtures/languages/swift.swift"),
    (Language::Toml, "tests/fixtures/languages/toml.toml"),
    (
        Language::TypeScript,
        "tests/fixtures/languages/typescript.ts",
    ),
    (Language::Vba, "tests/fixtures/languages/vb.vb"),
    (Language::Verilog, "tests/fixtures/languages/verilog.v"),
    (Language::Xml, "tests/fixtures/languages/xml.xml"),
    (Language::Yaml, "tests/fixtures/languages/yaml.yaml"),
];
