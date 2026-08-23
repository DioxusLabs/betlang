# Betlang Model Card

## Artifact

- File: `assets/bnn/source-bnn2.bin`
- Format: packed "BBN2" binary-CNN ensemble payload
- Size: 1,165,464 bytes
- SHA-256: `fe36df96d9b6d268118b21b764c3be6dda33596e0fafeacd2bc0b26e3fc143ca`
- Architecture: ensemble of five binary CNNs over Shannon-coded byte windows
  (binary weights and activations, XNOR + popcount convolutions, integer
  thresholds, OR-pooling, segmented popcount heads, i16 quantized classifier)
- Input: canonical Shannon coding of the raw 2048-byte Magika begin/end
  window into a fixed 16,384-bit stream plus a codeword-boundary bitplane
- Output head: 48 model labels exposed one-to-one as public `Language` variants

## Intended Use

Betlang is intended for fast source-language detection on code snippets or
source files. It is suitable for routing files to syntax-aware tooling when a
best-effort content classifier is acceptable. It is not intended for security
decisions, malware classification, or legal identification of file provenance.

## Training Source

The models are trained on a rebuilt public corpus (see
`scripts/bnn/build_corpus.py`): per-language samples from
`bigcode/the-stack-smol-xl` and `bigcode/the-stack`, GitHub repo files for
labels absent from The Stack, and synthetic sets targeting the Markdown/YAML
bare-list ambiguity and other confusable pairs.

Inputs are the raw Magika begin/end byte windows encoded with a canonical
Shannon code learned from training byte frequencies (add-one smoothing,
12-bit maximum code length, codes ordered by `(length, byte)`). No tokenizer
is used. Each window becomes two bitplanes: the Shannon code bits and the
codeword-boundary markers.

Training distills Google's Magika v3.3 teacher predictions alongside
filesystem-extension labels. The binary networks are trained binary from
scratch (annealing a float network collapses); sparse learnable thresholds,
OR-pooling, and segmented popcount heads make the fully-binarized networks
trainable. Ensemble members are fine-tuned on short inputs and synthetic hard
pairs, and ensemble weights are grid-searched under fixture and edge-case
constraints (`scripts/bnn/weight_search.py`).

The model does not contain original source files, but its labels and soft
targets are derived from the training corpus and Magika teacher.

## Evaluation

Held-out filesystem-label test split of the rebuilt corpus (27,465 files,
train/valid/test repositories are disjoint), evaluated with exact integer
export semantics (`scripts/bnn/eval_export2.py`):

- `test_accuracy=0.941635`
- `test_macro_recall=0.943226`

The previous shipped wordseq artifact (`source-student-q4.bin`) scores
`test_accuracy=0.934972` and `test_macro_recall=0.934863` on the same split.

On ambiguous bare `- item` lists (valid YAML and valid Markdown), the model
reports a split YAML/Markdown distribution with top-1 probability below 0.9
rather than picking one with certainty.

Most remaining confusion sits on genuinely ambiguous pairs: `c`/`cpp`,
`javascript`/`typescript`, `markdown`/`yaml`, `ini`/`toml`, `batch`/`shell`,
and `php`/`html`.

## Inference Semantics

All heavy compute is binary CPU work: activations and weights are single bits
packed into 64-bit words, convolutions are XNOR + popcount
(`z' = 2 * matches - fan_in`), basis outputs combine with fixed-point integer
coefficients (`FIX=65536`), activations are integer threshold comparisons,
pooling is bitwise OR, and the classifier is an i16 integer dot product over
segmented popcounts with per-class float scale/bias. Rust inference is
bit-exact against the Python integer reference
(`cargo test --release python_parity`).

## Known Weaknesses

- Very short inputs are intentionally rejected when fewer than eight
  non-whitespace bytes are available.
- Short files (< 512 bytes) are harder than full windows; accuracy on a
  short-truncated validation set is ~0.78 versus ~0.94 on full windows.
- Ambiguous snippets can put several languages close together even when a
  human can infer the language from file naming context.
- The classifier uses content only. It does not inspect file names,
  extensions, shebangs outside the model window, repository metadata, or
  build-system context.
- Non-source formats are out of scope unless represented by a public
  source-language variant.

## Reproducibility

Training and evaluation scripts live under `scripts/bnn/`. The pipeline is
documented in `scripts/bnn/TRAINING.md`, including corpus construction,
Shannon codebook learning, binary training, fine-tuning, ensemble weight
search, export, and the Rust/Python parity check.

The published crate package intentionally includes only the runtime model
artifact and user-facing docs. Training scripts and generated analysis files
remain repository artifacts.

## Attribution

The embedded models were trained from outputs of Google's Magika teacher
model. Magika is published by Google under Apache-2.0. Betlang's source code is
MIT licensed; keep Magika attribution with redistributed model artifacts.
