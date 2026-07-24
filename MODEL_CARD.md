# Betlang Model Card

## Artifact

- File: `assets/magika/source-student-q4.bin`
- Format: weights-only MSQ1 quantized tensor payload
- Size: 47,840 bytes
- SHA-256: `8493d2d3757572c8661141e414b1c0755aa08d4c4e5382dfbbc6b73b02d89083`
- Architecture: `wordseq-b1024-k3-m2048-tiny-3conv-hidden`
- Tokenizer: word-unit tokenizer version 3
- Output head: 48 model labels exposed one-to-one as public `Language` variants

## Intended Use

Betlang is intended for fast source-language detection on code snippets or
source files. It is suitable for routing files to syntax-aware tooling when a
best-effort content classifier is acceptable. It is not intended for security
decisions, malware classification, or legal identification of file provenance.

## Training Source

The student was originally trained from Google's Magika v3.3 teacher
predictions over a source-language corpus assembled from extension-suffixed
files (the `bigorig` split, extracted from a GitHub partial-clone blob index of
roughly 6,000 popular code repositories).

The shipped artifact is that model fine-tuned from its exported checkpoint on a
rebuilt public corpus (see `scripts/build_finetune_corpus.py`): per-language
samples from `bigcode/the-stack-smol-xl` and `bigcode/the-stack`, GitHub repo
files for labels absent from The Stack, and a small synthetic set targeting the
Markdown/YAML bare-list ambiguity reported in issue #5, including train-only
bare `- item` lists.

The fine-tune distills the same Magika v3.3 teacher one-vs-all: per-class
binary cross-entropy against the teacher's raw head-label marginals, plus a
hard term against the teacher argmax discounted by the teacher's in-head
probability mass. Unlike the original softmax distillation, this never
renormalizes away the probability mass the teacher assigns to labels outside
the 48-label head (such as `txt`), so inputs the teacher considers ambiguous
or out-of-scope train toward uniformly low logits and keep low softmax
confidence at inference.

The shipped run additionally self-distills from a ~2x larger intermediate
parent (`wordseq-b1536-k3-m2048-med-3conv-hidden`, 0.9492 test teacher parity)
trained on the same corpus with the same one-vs-all scheme; the parent's
per-class sigmoid marginals are cached with `scripts/cache_self_distill.py`
and added as a second BCE term.

The model distills teacher probabilities and filesystem-extension labels. It
does not contain original source files, but its labels and soft targets are
derived from the training corpus and Magika teacher.

## Evaluation

Held-out filesystem-label test split of the rebuilt corpus (34,087 files,
train/valid/test repositories are disjoint, rows where the teacher keeps at
most 10% of its probability mass on the head labels are excluded):

- `test_fs_accuracy=0.942353`
- `macro_recall=0.939690`
- `test_teacher_parity=0.944055`

For comparison, the pre-fine-tune artifact scores `test_fs_accuracy=0.926160`,
`macro_recall=0.929904`, and `test_teacher_parity=0.922111` on the same split.
The rebuilt split is balanced across all 48 labels (including rare classes),
so these numbers are not comparable to the `bigorig` metrics reported for
earlier artifacts.

On ambiguous bare `- item` lists (valid YAML and valid Markdown), the median
top-1 probability drops from 0.92 (pre-fine-tune) to 0.51, while YAML or
Markdown remains the top prediction.

Most remaining confusion sits on genuinely ambiguous pairs where the teacher
also splits its probability on the confused files: `c`/`cpp`,
`javascript`/`typescript`, `markdown`/`yaml`, `ini`/`toml`, `batch`/`shell`,
and `php`/`html`. Several other cells are corpus extension-label noise (for
example `.vb` files containing SQL dumps) where the teacher agrees with the
model on 80%+ of the confused files.

The README confusion matrix groups the same held-out split by file-size bucket.

## Known Weaknesses

- Very short inputs are intentionally rejected when fewer than eight
  non-whitespace bytes are available.
- Ambiguous snippets can put several languages close together even when a human
  can infer the language from file naming context. A bare `- item` list with no
  heading is valid YAML and valid Markdown; the model reports a split
  YAML/Markdown distribution for such inputs rather than picking one with
  certainty.
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
