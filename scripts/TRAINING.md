# Training the betlang wordseq student model

The model shipped in `assets/magika/source-student-q4.bin` is a 102,793-byte
quantized wordseq student trained on the Magika v3.3 teacher's predictions
over a ~440k-file source-language corpus. It uses the v3 word-unit tokenizer
and hits **0.962517 fs_accuracy** on an 81k held-out test set.

## Files

| File | Lines | Purpose |
|---|---:|---|
| `train_v2_student.py` | 175 | **Standalone driver.** Historical filename; wraps the trainer with the frozen recipe that produced the shipped model. Run this. |
| `train_magika_qat_student.py` | 1372 | Focused QAT trainer for only the shipped `wordseq-b1536-k3-m2048-med-3conv-hidden` model: v3 tokenizer, length buckets, self-distillation, CutMix, training loop, and MSQ1 export. |
| `train_magika_source_student.py` | 673 | Magika teacher loader, byte-window feature extraction, cache iteration helpers. Imported by the QAT trainer. |
| `eval_50kb_model.py` | 167 | Loads an exported MSQ1 `.bin`, runs forward pass, reports `_teacher_parity` and `_fs_accuracy` on a chosen split. |
| `confusion_by_size.py` | 471 | Evaluates an exported wordseq model, aligns cached rows to raw file sizes, and renders the README confusion-matrix image. |

## Recipe (frozen in `train_v2_student.py`)

```
arch:           wordseq-b1536-k3-m2048-med-3conv-hidden
unit_tokenizer: 3  (v2 punct/digit compression + case-folded words + isolated brackets)
length_buckets: yes
hard_loss_weight: 0.5    (cache labels.mmap = teacher argmax)
self_loss_weight: 0.5    (cache self_probabilities.mmap = teacher full softmax)
distill_temperature: 3
label_smoothing: 0.05
cutmix_prob:     0.5
LR:              cosine 8e-4 → 5%, AdamW grad-clip 1.0
epochs:          60, qat_start_epoch=45, early_stop_patience=6
mixed_precision: yes
seed:            2
```

## Reproducing the shipped model

You need:

1. **Magika v3.3 teacher** — `model.onnx` + `config.min.json` from
   `https://github.com/google/magika/tree/main/python/magika/models/standard_v3_3`.
2. **A source-language corpus** — extension-suffixed files split into
   `files/{train,valid,test}/`. The shipped model was trained on a
   605k-file corpus (`bigorig`) extracted from the GitHub partial-clone
   blob index of ~6,000 popular code repositories.
3. **A pre-built training cache** — see "Cache" below.

Then run:

```bash
python3 scripts/train_v2_student.py \
  --dataset /path/to/corpus/files \
  --cache-dir /path/to/cache \
  --magika-model /path/to/magika/standard_v3_3/model.onnx \
  --magika-config /path/to/magika/standard_v3_3/config.min.json \
  --output assets/magika/source-student-q4.bin
```

Expected runtime on a single 12 GB GPU: ~50 minutes for 60 epochs at
~12 k examples/sec.

## Cache

The trainer expects a cache directory containing per-split mmap files:
`tokens.mmap`, `units_v3.mmap`, `labels.mmap`, `probabilities.mmap`, and
`self_probabilities.mmap` for the frozen self-distillation recipe.

`train_magika_qat_student.py` will lazily build `tokens.mmap`,
`probabilities.mmap`, and `labels.mmap` from the corpus by running the Magika
teacher on every file (slow; ~30 min for 500 k files).

`units_v3.mmap` is built lazily on first training run from `tokens.mmap`
via the v3 tokenizer (~5 min for 500 k files).

`self_probabilities.mmap` (the `--self-loss-weight 0.5` term) is the soft
teacher distribution from a larger capacity student trained as an
intermediate step. Skip it (drop `--self-loss-weight`) for a slightly
weaker but simpler training; expect ~1 pp accuracy hit.

## Verifying the trained model

```bash
python3 scripts/eval_50kb_model.py \
  --checkpoint assets/magika/source-student-q4.bin \
  --cache-dir /path/to/cache \
  --architecture wordseq-b1536-k3-m2048-med-3conv-hidden \
  --split test
```

Expected output:
```
test_teacher_parity=0.967618
test_fs_accuracy=0.962517
```

`test_teacher_parity` is fraction matching the cache's `labels.mmap`
(teacher argmax). `test_fs_accuracy` is fraction matching `fs_labels.mmap`
(filesystem-extension labels with teacher fallback for unmapped
extensions). The two should be close (the corpus has ~99.5%
teacher-vs-filesystem agreement).
