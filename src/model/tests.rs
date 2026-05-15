use super::{
    constants::*,
    runtime::Model,
    tokenizer::{hash_unit_bytes, tokenize},
    window::build_window,
};
use crate::Language;

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
