# Betlang

[![Crates.io](https://img.shields.io/crates/v/betlang.svg)](https://crates.io/crates/betlang)
[![Docs.rs](https://docs.rs/betlang/badge.svg)](https://docs.rs/betlang)

CPU source-language detection for code with a binary XOR/popcount model. Try it in browser [here](https://dioxuslabs.github.io/dioxus-code/#playground)

```toml
[dependencies]
betlang = "0.1.1"
```

```rust
let detection = betlang::detect("fn main() {\n    println!(\"hello, world!\");\n}\n");

assert_eq!(detection.language(), Some(betlang::Language::Rust));
```

Use `betlang::detect(source)` for UTF-8 source strings or byte slices. It
returns a `Detection`; call `Detection::language()` to read the top language.
Call `Detection::top_languages()` when you need ranked probabilities.

## Supported Languages

Slugs parse through the standard `FromStr` implementation:

```rust
assert_eq!("rust".parse::<betlang::Language>(), Ok(betlang::Language::Rust));
```

`asm`, `batch`, `c`, `clojure`, `cmake`, `cobol`, `cpp`, `cs`, `css`, `dart`,
`dockerfile`, `elixir`, `erlang`, `gemfile`, `gemspec`, `go`, `gradle`,
`groovy`, `haskell`, `html`, `ini`, `java`, `javascript`, `json`, `julia`,
`kotlin`, `lisp`, `lua`, `markdown`, `objectivec`, `ocaml`, `perl`, `php`,
`powershell`, `python`, `r`, `ruby`, `rust`, `scala`, `shell`, `sql`, `swift`,
`toml`, `typescript`, `vba`, `verilog`, `xml`, `yaml`.

These are the model's 48 output labels. Runtime detections expose them
one-to-one with no label aggregation.

The confusion matrix uses the same labels:

![Betlang bloom confusion](assets/confusion-overall.png)

## Model

The embedded model is `assets/magika/source-bloom.bin`, a 26,808-byte
weights-only MBL4 payload (1.8x smaller than the 47,840-byte convolutional
student it replaces) with SHA-256:

```text
7d78f5778248096b450155c9e56ae5d0281b6932dbf1d6b98142b518a5aca1f0
```

Architecture: a deterministic counting-Bloom n-gram signature (byte n-grams,
case-folded words, and tokenizer-v3 unit n-grams hashed into 50 compact
binary planes totalling 3,968 bits) followed by a binary {-1,+1} linear head
evaluated entirely with XOR + popcount plus one int8 multiply per (class,
plane) and one float multiply-add per class. On the rebuilt held-out
filesystem-label test split it reaches `test_fs_accuracy=0.899` versus
`0.945` for the previous 47,840-byte convolutional student on the same
split — the accuracy cost of shipping a 1.8x smaller artifact that runs
entirely on binary CPU ops.

See [MODEL_CARD.md](MODEL_CARD.md) for the training and evaluation summary.

## Performance

Betlang uses a fixed 4096-byte Magika window. The byte window is hashed into a
3,968-bit binary signature and the 48 logits are computed with XOR +
popcount over packed u64 words. The model is loaded once per process and then
reused through a `OnceLock`.

Benchmark entry points are available through `cargo bench`. Current baseline
numbers are tracked in [BENCHMARKS.md](BENCHMARKS.md); the compact binary
model is ~310x faster than the previous convolutional student on short inputs
and ~740x faster on full 4 KiB windows on the same host.

## License And Attribution

Betlang is licensed under MIT. The embedded model was trained with soft
targets from Google's Magika teacher model; Magika is published by Google
under Apache-2.0. Keep this attribution with redistributed model artifacts.

## Confusion By File Size

The shipped model is evaluated below on the held-out test split. Each panel is
a row-normalized confusion matrix for one file-size bucket: actual labels are
rows, predicted labels are columns, and the diagonal is correct
classification.

![Betlang bloom confusion by file size](assets/confusion-by-size.png)
