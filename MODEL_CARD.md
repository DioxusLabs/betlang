# Betlang Model Card

## Artifact

- File: `assets/magika/source-bloom.bin`
- Format: weights-only MBL4 binary payload
- Size: 26,808 bytes (1.8x smaller than the 47,840-byte convolutional student it replaces)
- SHA-256: `7d78f5778248096b450155c9e56ae5d0281b6932dbf1d6b98142b518a5aca1f0`
- Architecture: compact counting-Bloom n-gram signature + binary linear head
- Tokenizer: raw byte n-grams plus word-unit tokenizer version 3
- Output head: 48 model labels exposed one-to-one as public `Language` variants

## Architecture

The model replaces the previous quantized convolutional student with a
deterministic binary feature encoder and a binary linear head. Inference is
dominated by integer XOR + popcount:

1. **Byte window** — the first and last 1,024 bytes of the (whitespace-stripped
   within a 4,096-byte block) input, identical to the Magika feature window.
2. **Shannon/Zobrist encoding** — every byte n-gram (orders 1–8, begin and end
   halves separately), case-folded identifier word (two independent hash
   folds), word bigram/trigram, line-start word, and tokenizer-v3 unit n-gram
   (orders 1–4) is hashed with a fixed SplitMix64-seeded Zobrist table into a
   4,096-bucket counting Bloom block.
3. **Compact folding + thermometer binarization** — each 4,096-bucket block is
   folded into one or more small power-of-two views (64–128 buckets, taken
   from different bit fields of the hash), and each folded bucket count passes
   through fixed thresholds (1/2/4/8 depending on the feature group). This
   yields 50 compact binary planes totalling 3,968 bits packed into 62 u64
   words.
4. **Binary head** — each of the 48 classes holds a packed {-1,+1} weight row.
   The logit is `bias + step[class] * Σ_plane q[class][plane] * (width - 2*popcount(x XOR w))`,
   i.e. an XOR/popcount dot product with one int8 multiply per (class, plane)
   and a single float multiply-add per class. A softmax over the 48 logits
   produces probabilities.

The encoder is exact integer arithmetic, so the Python trainer/simulator and
the Rust runtime produce bit-identical signatures and matching logits
(verified by golden vectors in `tests/fixtures/bloom-golden.bin`).

## Intended Use

Betlang is intended for fast source-language detection on code snippets or
source files. It is suitable for routing files to syntax-aware tooling when a
best-effort content classifier is acceptable. It is not intended for security
decisions, malware classification, or legal identification of file provenance.

## Training Source

The head is trained with straight-through-estimator binarization against two
targets on a rebuilt public corpus (see `scripts/build_finetune_corpus.py`):

- **Soft targets** — Google's Magika v3.3 teacher raw per-class head marginals,
  distilled one-vs-all with per-class sigmoid cross-entropy (weight 0.5).
- **Hard targets** — filesystem-extension labels with cross-entropy and 0.05
  label smoothing (weight 0.5).

Training additionally mixes in short prefix crops (12–1,024 bytes, 35% of the
train count, hard labels only) so tiny standalone snippets see n-gram
statistics that match real short files.

The corpus is built entirely from ungated sources: per-language samples from
`bigcode/the-stack-smol-xl`, files harvested from public GitHub repository
tarballs for labels with no per-language subset there (yaml, json, toml, ini,
xml, swift, cobol, objectivec, gradle, gemfile/gemspec), and a small synthetic
set targeting the Markdown/YAML bare-list ambiguity from issue #5. Files from
one repository always land in the same split, and per-repository caps prevent a
single repository from dominating a label.

The model distills teacher probabilities and filesystem-extension labels. It
does not contain original source files, but its labels and soft targets are
derived from the training corpus and Magika teacher.

## Evaluation

Held-out filesystem-label test split of the rebuilt corpus (31,917 files,
train/valid/test repositories are disjoint, rows where the teacher keeps at
most 10% of its probability mass on the head labels are excluded). The rebuilt
corpus avoids the gated `bigcode/the-stack` dataset, so these numbers are not
comparable to metrics previously reported for the retired MSQ1 artifact on its
own rebuilt split.

| Model | size (bytes) | fs_accuracy | macro_recall | teacher_parity |
|---|---:|---:|---:|---:|
| Compact Bloom binary head (shipped) | **26,808** | 0.899176 | 0.879002 | 0.885892 |
| Previous wordseq MSQ1 student | 47,840 | 0.944888 | 0.937713 | 0.952408 |
| Large Bloom binary head (not shipped) | 1,932,116 | 0.947959 | 0.923254 | 0.929505 |

The shipped compact model trades accuracy for size: it is 1.8x smaller than
the previous convolutional student and runs on XOR/popcount instead of
floating-point convolutions, at a ~4.6 point filesystem-accuracy cost. The
same architecture recovers the accuracy when given more bits (the large
319,488-bit variant reaches 0.948), so the signature width is a direct
size/accuracy dial: on the same split a ~19 KB variant (2,752 bits) measured
0.83 and a ~42 KB variant (6,528 bits) measured 0.927.

Most remaining confusion sits on genuinely ambiguous pairs: `c`/`cpp`,
`javascript`/`typescript`, `markdown`/`yaml`, `ini`/`toml`, `batch`/`shell`,
and `php`/`html`.

## Known Weaknesses

- Very short inputs are intentionally rejected when fewer than eight
  non-whitespace bytes are available.
- Very short snippets (under ~100 bytes) remain weaker than long files even
  with short-crop augmentation: the Bloom signature is sparse for tiny inputs
  and n-gram evidence is thin.
- Ambiguous snippets can put several languages close together even when a
  human can infer the language from file naming context.
- The classifier uses content only. It does not inspect file names,
  extensions, shebangs outside the model window, repository metadata, or
  build-system context.
- Non-source formats are out of scope unless represented by a public
  source-language variant.

## Reproducibility

Training and evaluation scripts live under `scripts/`. The recipe is
documented in `scripts/TRAINING.md`, including corpus construction, the
expected external Magika teacher assets, cache layout, training command,
golden-vector export, and expected metrics.

The published crate package intentionally includes only the runtime model
artifact and user-facing docs. Training scripts and generated analysis files
remain repository artifacts.

## Attribution

The embedded model was trained from outputs of Google's Magika teacher model.
Magika is published by Google under Apache-2.0. Betlang's source code is MIT
licensed; keep Magika attribution with redistributed model artifacts.
