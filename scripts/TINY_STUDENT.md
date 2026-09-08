# 5 KB lexical student experiment

This applies the small hashed lexical representation idea from
[gpu-lexer](https://gpu-lexer.vercel.app) to file-level language detection.
It is a separate experiment; `betlang::detect` retains its production model.

The published gpu-lexer 0.0.1 runtime uses token kinds, lengths, first/last
characters, two word hashes, shape flags and neighboring punctuation features
as additive embeddings. It combines local depthwise filtering with recurrent
and hierarchical file context. Its advertised 27.5 KB is a **minified,
Brotli-compressed browser bundle**, not an uncompressed weights budget.
This experiment borrows compact hashing and local/global lexical context,
not the WebGPU implementation or its trained weights.

## Architecture and byte budget

The detector uses the same beginning/end byte-window policy as betlang.
It splits that window into identifiers, digit runs, punctuation and newlines.
Identifiers are case-folded; digit runs become `0`. Token and adjacent-token
hashes accumulate in 512 shared buckets. `log1p` counts followed by L2
normalization provide file context. A 16-unit ReLU projection maps these
features to the existing 48 output labels.

| BTQ1 component | Raw bytes |
|---|---:|
| Magic/version | 8 |
| Two f32 quantization scales | 8 |
| 512 × 16 int4 projection | 4,096 |
| 16 f32 hidden biases | 64 |
| 16 × 48 int4 output | 384 |
| 48 f32 output biases | 192 |
| **Total** | **4,752** |

The exporter asserts the exact layout and the 5,000-byte budget.
This is model size only: it excludes inference code, runtime allocations
and executable overhead. Quantization-aware training and export use the
existing trainer's symmetric int4 quantizer.

This has much less context capacity than gpu-lexer or betlang's convolutional
model. A size result is not evidence that it can replace the production model.
In particular, confidence calibration and short ambiguous inputs need their
own validation before exposing this as a production detector.

## Run

Python 3.10 or newer:

```sh
python3 -m venv .venv
.venv/bin/pip install tensorflow-cpu==2.20.0 onnxruntime==1.22.1 numpy==2.2.6
export TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=2

# Dataset layout: files/{train,valid,test}/{production-label}/*
.venv/bin/python scripts/train_tiny_student.py \
  --prepare --dataset /path/to/files --cache /path/to/cache \
  --output /path/to/tiny-q4.bin
.venv/bin/python scripts/train_tiny_student.py \
  --cache /path/to/cache --output /path/to/tiny-q4.bin

cargo run --release --example tiny -- /path/to/tiny-q4.bin source.rs
cargo run --release --example tiny -- /path/to/tiny-q4.bin \
  --compare /path/to/cache/test.paths.txt > comparison.tsv
```

Prepare the corpus with repository-disjoint train/validation/test splits
(see `build_finetune_corpus.py`). The feature preparation additionally removes
identical model windows across and within splits, keeping the first occurrence
in train, validation, test order. Re-run preparation whenever the corpus or
tokenizer changes. The paths manifest is the exact evaluation population.
Checkpoint selection uses validation accuracy; test metrics come from the
reloaded packed file. The adjacent JSON report includes all 48 class recalls,
missing test labels, epoch history and the artifact hash. Never compare its
accuracy directly with the model card's score on a different corpus.

The Rust example loads weights once, provides independent native inference,
and can compare both models on the same manifest. The Python inference module
`tiny_student.py` needs only NumPy.

## Checks

```sh
cargo build --release --example tiny
BETLANG_TINY_EXAMPLE="$PWD/target/release/examples/tiny" \
  .venv/bin/python scripts/test_tiny_student.py
cargo test --all-targets
cargo fmt --check
cargo clippy --all-targets -- -D warnings
python3 -m flake8 scripts/{tiny_student,train_tiny_student,test_tiny_student}.py \
  --select E9,F63,F7,F82
```

The Python tests compare QAT against the reloaded int4 model and compare both
features and logits with Rust on UTF-8, arbitrary bytes, whitespace and window
boundaries. They also exercise malformed files and non-finite parameters.
