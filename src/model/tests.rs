use super::{
    constants::*,
    embedded::MODEL_BYTES,
    metadata::{assert_tokenizer_version, rfind_bytes},
    runtime::Model,
    tokenizer::{hash_unit_bytes, tokenize},
    window::build_window,
};
use crate::{Language, language::CLASS_LANGUAGES};
use serde_json::Value;
use sha2::{Digest, Sha256};

const EXPECTED_MODEL_SHA256: &str =
    "52be89bef15515aa93ae924e76d17d72b3943f50ceda8aa9e1c3834f27f8e883";
const EXPECTED_MODEL_LEN: usize = 102_793;
const EXPECTED_METADATA_START: usize = 100_456;
const EXPECTED_METADATA_LEN: usize = 2_337;
const EXPECTED_LABELS: [&str; CLASSES] = [
    "asm",
    "awk",
    "batch",
    "bazel",
    "c",
    "clojure",
    "cmake",
    "cobol",
    "cpp",
    "cs",
    "csproj",
    "css",
    "dart",
    "diff",
    "dockerfile",
    "elixir",
    "erb",
    "erlang",
    "gemfile",
    "gemspec",
    "go",
    "gradle",
    "groovy",
    "haskell",
    "hcl",
    "html",
    "ini",
    "ipynb",
    "java",
    "javascript",
    "jinja",
    "json",
    "jsonl",
    "julia",
    "kotlin",
    "lisp",
    "lua",
    "markdown",
    "matlab",
    "objectivec",
    "ocaml",
    "perl",
    "php",
    "postscript",
    "powershell",
    "prolog",
    "python",
    "r",
    "ruby",
    "rust",
    "scala",
    "scss",
    "shell",
    "solidity",
    "sql",
    "swift",
    "textproto",
    "toml",
    "typescript",
    "vba",
    "vcxproj",
    "verilog",
    "vhdl",
    "vue",
    "xml",
    "yaml",
    "zig",
];

#[test]
fn loads_embedded_model() {
    let model = Model::get();
    assert_eq!(model.embedding.len(), BINS * EMBED);
    assert_eq!(model.output_kernel.len(), DENSE * CLASSES);
}

#[test]
fn embedded_model_asset_matches_expected_contract() {
    assert!(MODEL_BYTES.starts_with(&MODEL_MAGIC));
    assert_eq!(MODEL_BYTES.len(), EXPECTED_MODEL_LEN);

    let metadata_start = rfind_bytes(MODEL_BYTES, br#"{"bits""#).unwrap();
    assert_eq!(metadata_start, EXPECTED_METADATA_START);
    let metadata_len = u32::from_le_bytes(
        MODEL_BYTES[metadata_start - 4..metadata_start]
            .try_into()
            .unwrap(),
    ) as usize;
    assert_eq!(metadata_len, EXPECTED_METADATA_LEN);
    assert_eq!(metadata_len, MODEL_BYTES.len() - metadata_start);

    let digest = Sha256::digest(MODEL_BYTES);
    assert_eq!(format!("{digest:x}"), EXPECTED_MODEL_SHA256);
}

#[test]
fn embedded_model_metadata_matches_runtime_mapping() {
    let metadata = model_metadata_json();
    assert_eq!(metadata["bits"], 4);
    assert_eq!(metadata["token_length"], MAX_UNITS);
    assert_eq!(
        metadata["architecture"],
        "wordseq-b1536-k3-m2048-med-3conv-hidden"
    );
    assert_eq!(metadata["tokenizer_version"], 3);

    let labels = string_array(&metadata["labels"]);
    assert_eq!(labels, EXPECTED_LABELS);

    let slugs = string_array(&metadata["slugs"]);
    assert_eq!(slugs.len(), CLASS_LANGUAGES.len());
    for (slug, language) in slugs.iter().zip(CLASS_LANGUAGES) {
        assert_eq!(*slug, language.slug());
    }

    assert_eq!(CLASS_LANGUAGES[3], Language::Starlark); // bazel
    assert_eq!(CLASS_LANGUAGES[10], Language::Xml); // csproj
    assert_eq!(CLASS_LANGUAGES[16], Language::Ruby); // erb
    assert_eq!(CLASS_LANGUAGES[18], Language::Ruby); // gemfile
    assert_eq!(CLASS_LANGUAGES[19], Language::Ruby); // gemspec
    assert_eq!(CLASS_LANGUAGES[32], Language::Json); // jsonl
    assert_eq!(CLASS_LANGUAGES[52], Language::Bash); // shell
    assert_eq!(CLASS_LANGUAGES[59], Language::Vb); // vba
    assert_eq!(CLASS_LANGUAGES[60], Language::Xml); // vcxproj
}

#[test]
fn embedded_model_tensor_shapes_match_runtime_constants() {
    let metadata = model_metadata_json();
    let layers = metadata["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 6);

    assert_layer(&layers[0], "q_hash_embedding", &[BINS, EMBED], 21_504, None);
    assert_layer(
        &layers[1],
        "q_conv_0",
        &[CONV0_KERNEL, EMBED, CONV0],
        4_704,
        Some((&[CONV0][..], CONV0 * 4)),
    );
    assert_layer(
        &layers[2],
        "q_conv_1",
        &[CONV1_KERNEL, CONV0, CONV1],
        23_040,
        Some((&[CONV1][..], CONV1 * 4)),
    );
    assert_layer(
        &layers[3],
        "q_conv_2",
        &[CONV2_KERNEL, CONV1, CONV2],
        27_648,
        Some((&[CONV2][..], CONV2 * 4)),
    );
    assert_layer(
        &layers[4],
        "q_dense_0",
        &[POOLED, DENSE],
        15_360,
        Some((&[DENSE][..], DENSE * 4)),
    );
    assert_layer(
        &layers[5],
        "q_output",
        &[DENSE, CLASSES],
        5_360,
        Some((&[CLASSES][..], CLASSES * 4)),
    );
}

#[test]
fn tokenizer_version_accepts_only_v3() {
    assert_tokenizer_version(r#"{"bits":4,"tokenizer_version":3}"#);
    assert!(std::panic::catch_unwind(|| assert_tokenizer_version(r#"{"bits":4}"#)).is_err());
    assert!(
        std::panic::catch_unwind(|| {
            assert_tokenizer_version(r#"{"bits":4,"tokenizer_version":99}"#)
        })
        .is_err()
    );
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
        let (probability, language) = detection.top_languages().next().unwrap();
        assert_eq!(language, expected, "{source}");
        assert_eq!(language.slug(), expected.slug());
        assert!(probability > 0.0, "{source}");
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
    let (bytes, pad) = build_window(source.as_bytes()).unwrap();
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

fn model_metadata_json() -> Value {
    let metadata_start = rfind_bytes(MODEL_BYTES, br#"{"bits""#).unwrap();
    serde_json::from_slice(&MODEL_BYTES[metadata_start..]).unwrap()
}

fn string_array(value: &Value) -> Vec<&str> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect()
}

fn assert_layer(
    layer: &Value,
    name: &str,
    weight_shape: &[usize],
    weight_bytes: usize,
    bias: Option<(&[usize], usize)>,
) {
    assert_eq!(layer["name"], name);
    assert_eq!(usize_array(&layer["weights"][0]["shape"]), weight_shape);
    assert_eq!(layer["weights"][0]["bytes"], weight_bytes);

    match bias {
        Some((bias_shape, bias_bytes)) => {
            assert_eq!(usize_array(&layer["biases"][0]["shape"]), bias_shape);
            assert_eq!(layer["biases"][0]["bytes"], bias_bytes);
        }
        None => assert!(layer["biases"].as_array().unwrap().is_empty()),
    }
}

fn usize_array(value: &Value) -> Vec<usize> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}
