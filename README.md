# Betlang

[![Crates.io](https://img.shields.io/crates/v/betlang.svg)](https://crates.io/crates/betlang)
[![Docs.rs](https://docs.rs/betlang/badge.svg)](https://docs.rs/betlang)

CPU source-language detection for code with a fully-binary CNN model. Try it in browser [here](https://dioxuslabs.github.io/dioxus-code/#playground)

```toml
[dependencies]
betlang = "0.1.1"
```

```rust
let detection = betlang::detect("fn main() { println!(\"hi\"); }");

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

## Model

The embedded model is `assets/bnn/source-bnn2.bin`, a 1,165,464-byte packed
"BBN2" payload with SHA-256:

```text
fe36df96d9b6d268118b21b764c3be6dda33596e0fafeacd2bc0b26e3fc143ca
```

Architecture: an ensemble of five binary CNNs over Shannon-coded raw byte
windows. The 2048-byte Magika begin/end window is entropy-coded with a
canonical Shannon code learned from training byte frequencies, and the
resulting bitstream (plus a codeword-boundary bitplane) feeds convolutions
with binary weights and activations. All heavy compute is bitwise:
XNOR + popcount convolutions, integer thresholds, OR-pooling, segmented
popcounts, and an integer classifier head.

On the held-out filesystem-label test split it reaches
`test_accuracy=0.941635` with `macro_recall=0.943226`, versus
`test_accuracy=0.934972` / `macro_recall=0.934863` for the previous wordseq
model on the same split. Ambiguous inputs report split scores instead of a
confident label.

See [MODEL_CARD.md](MODEL_CARD.md) for the training and evaluation summary.

## Performance

Betlang uses a fixed 4096-byte Magika window and Shannon-codes the extracted
2048-byte begin/end window into a fixed 16,384-bit stream. The model is loaded
once per process and then reused through a `OnceLock`.

Inference is pure integer/bitwise CPU work (XNOR, popcount, integer
accumulation, thresholds, bitwise OR pooling). Benchmark entry points are
available through `cargo bench`. Current baseline numbers are tracked in
[BENCHMARKS.md](BENCHMARKS.md).

## License And Attribution

Betlang is licensed under MIT. The embedded models were trained from
outputs of Google's Magika teacher model; Magika is published by Google under
Apache-2.0. Keep this attribution with redistributed model artifacts.
