# Betlang Model Card

## Artifact

- File: `assets/magika/source-bloom.bin`
- Format: weights-only MBL3 binary payload
- Size: 1,932,116 bytes
- SHA-256: `4d01e7996aee5a9cd5c3582fa97ca025390276cdea022b7594f46ec69da57616`
- Architecture: counting-Bloom n-gram signature + binary linear head
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
3. **Thermometer binarization** — each bucket count passes through fixed
   thresholds (1/2/4/8 depending on the feature group), producing 78 binary
   planes of 4,096 bits: a 319,488-bit signature packed into 4,992 u64 words.
4. **Binary head** — each of the 48 classes holds a packed {-1,+1} weight row.
   The logit is `bias + Σ_plane scale[class][plane] * (4096 - 2*popcount(x XOR w))`,
   i.e. an XOR/popcount dot product with one float multiply-add per
   (class, plane). A softmax over the 48 logits produces probabilities.

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

Training additionally mixes in short prefix crops (12–1,024 bytes, 15% of the
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

| Model | fs_accuracy | macro_recall | teacher_parity |
|---|---:|---:|---:|
| Bloom binary head (shipped) | **0.947959** | 0.923254 | 0.929505 |
| Previous wordseq MSQ1 student | 0.944888 | 0.937713 | 0.952408 |

The binary model beats the previous student on filesystem-label accuracy while
running on XOR/popcount instead of floating-point convolutions. Its macro
recall and teacher parity are lower: the binary head optimizes filesystem
truth directly rather than mimicking the teacher, so it disagrees with the
teacher more often, mostly on ambiguous rows.

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
