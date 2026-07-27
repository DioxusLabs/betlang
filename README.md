# Betlang

[![Crates.io](https://img.shields.io/crates/v/betlang.svg)](https://crates.io/crates/betlang)
[![Docs.rs](https://docs.rs/betlang/badge.svg)](https://docs.rs/betlang)

CPU detection of natural language vs LLM prompts with a tiny ~45kb model.
Given a piece of text, betlang predicts whether it is ordinary prose
(`natural_language`) or text written to instruct a language model (`prompt`) —
the same kind of routing decision Warp-style tools make when deciding whether
input should go to an AI.

```toml
[dependencies]
betlang = "0.1.1"
```

```rust
let detection = betlang::detect("Write a short poem about the ocean.");

assert_eq!(detection.kind(), Some(betlang::Kind::Prompt));
```

Use `betlang::detect(source)` for UTF-8 strings or byte slices. It returns a
`Detection`; call `Detection::kind()` to read the top kind. Call
`Detection::top_kinds()` when you need ranked probabilities.

## Kinds

Slugs parse through the standard `FromStr` implementation:

```rust
assert_eq!("prompt".parse::<betlang::Kind>(), Ok(betlang::Kind::Prompt));
```

- `natural_language` — prose that is not addressed to a model: articles, news,
  reviews, narrative and conversational text.
- `prompt` — text written to instruct a language model: task requests,
  questions for an assistant, role-play setups, instructions with pasted
  context.

These are the model's 2 output labels. Runtime detections expose them
one-to-one with no label aggregation.

## Model

The embedded model is `assets/magika/source-student-q4.bin`, a 45,448-byte
weights-only MSQ1 payload.

Architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`, tokenizer version 3.
See [MODEL_CARD.md](MODEL_CARD.md) for the exact hash, training recipe, and
evaluation summary, and `scripts/TRAINING.md` for how to reproduce it.

## Performance

Betlang uses a fixed 4096-byte Magika window and pads runtime inference to the
same 2048-token shape used by evaluation. The model is loaded once per process
and then reused through a `OnceLock`.

Native CPU inference dispatches through `fearless_simd`. Benchmark entry points
are available through `cargo bench`. Current baseline numbers are tracked in
[BENCHMARKS.md](BENCHMARKS.md).

## License And Attribution

Betlang is licensed under MIT. The model architecture and quantized runtime
descend from a student of Google's Magika teacher model; Magika is published
by Google under Apache-2.0. Keep this attribution with redistributed model
artifacts.
