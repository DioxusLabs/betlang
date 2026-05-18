# Betlang Model Card

## Artifact

- File: `assets/magika/source-student-q4.bin`
- Format: weights-only MSQ1 quantized tensor payload
- Size: 47,840 bytes
- SHA-256: `59ef24167bddd1364eb9c1650add8a67e1a542b5155fac67f5e1cda07df0c0f0`
- Architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
- Tokenizer: word-unit tokenizer version 3
- Output head: 48 model labels exposed one-to-one as public `Language` variants

## Intended Use

Betlang is intended for fast source-language detection on code snippets or
source files. It is suitable for routing files to syntax-aware tooling when a
best-effort content classifier is acceptable. It is not intended for security
decisions, malware classification, or legal identification of file provenance.

## Training Source

The student was trained from Google's Magika v3.3 teacher predictions over a
source-language corpus assembled from extension-suffixed files. The corpus used
for the shipped model was the `bigorig` split, extracted from a GitHub
partial-clone blob index of roughly 6,000 popular code repositories.

The model distills teacher probabilities and filesystem-extension labels. It
does not contain original source files, but its labels and soft targets are
derived from the training corpus and Magika teacher.

## Evaluation

Manifest-aligned held-out filesystem-label test split:

- `test_fs_accuracy=0.965238`
- `macro_recall=0.965411`

The README confusion matrix groups the same held-out split by file-size bucket.

## Known Weaknesses

- Very short inputs are intentionally rejected when fewer than eight
  non-whitespace bytes are available.
- Ambiguous snippets can put several languages close together even when a human
  can infer the language from file naming context.
- The classifier uses content only. It does not inspect file names, extensions,
  shebangs outside the model window, repository metadata, or build-system
  context.
- Non-source formats are out of scope unless represented by a public
  source-language variant.

## Reproducibility

Training and evaluation scripts live under `scripts/`. The frozen recipe is
documented in `scripts/TRAINING.md`, including the expected external Magika
teacher assets, cache layout, training command, evaluation command, and
expected metrics.

The published crate package intentionally includes only the runtime model
artifact and user-facing docs. Training scripts and generated analysis files
remain repository artifacts.

## Attribution

The embedded student model was trained from outputs of Google's Magika teacher
model. Magika is published by Google under Apache-2.0. Betlang's source code is
MIT licensed; keep Magika attribution with redistributed model artifacts.
