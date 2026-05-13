# Training the betlang v2 student model

The model shipped in `assets/magika/source-student-q4.bin` is a 49.92 KB
quantized wordseq student trained on the Magika v3.3 teacher's predictions
over a ~440k-file source-language corpus. It hits **~0.95 fs_accuracy** on
an 81k held-out test set.

## Files

| File | Lines | Purpose |
|---|---:|---|
| `train_v2_student.py` | 175 | **Standalone driver.** Wraps the trainer with the frozen recipe that produced the shipped model. Run this. |
| `train_magika_qat_student.py` | 5411 | The QAT trainer. Architecture builders (wordseq + others), v1/v2 tokenizers, distillation loss, training loop, MSQ1 export. |
| `train_magika_source_student.py` | 673 | Magika teacher loader, byte-window feature extraction, cache iteration helpers. Imported by the QAT trainer. |
| `eval_50kb_model.py` | 142 | Loads an exported MSQ1 `.bin`, runs forward pass, reports `_teacher_parity` and `_fs_accuracy` on a chosen split. |

## Recipe (frozen in `train_v2_student.py`)

```
arch:           wordseq-b1024-k3-m2048-tiny-3conv-hidden
unit_tokenizer: 2  (collapses punct/digit runs into single hashed units)
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

The trainer expects a pre-built cache directory containing per-split mmap
files: `tokens.mmap`, `units_v2.mmap`, `labels.mmap`, `probabilities.mmap`,
`self_probabilities.mmap` (optional), `hidden.mmap` (optional).

`train_magika_qat_student.py` will lazily build `tokens.mmap`,
`probabilities.mmap`, `labels.mmap`, and `hidden.mmap` from the corpus by
running the Magika teacher on every file (slow; ~30 min for 500 k files).

`units_v2.mmap` is built lazily on first training run from `tokens.mmap`
via the v2 tokenizer (~5 min for 500 k files).

`self_probabilities.mmap` (the `--self-loss-weight 0.5` term) is the soft
teacher distribution from a larger capacity student trained as an
intermediate step. Skip it (drop `--self-loss-weight`) for a slightly
weaker but simpler training; expect ~1 pp accuracy hit.

## Verifying the trained model

```bash
python3 scripts/eval_50kb_model.py \
  --checkpoint assets/magika/source-student-q4.bin \
  --cache-dir /path/to/cache \
  --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
  --split test \
  --unit-tokenizer 2
```

Expected output:
```
test_teacher_parity=0.954
test_fs_accuracy=0.950
```

`test_teacher_parity` is fraction matching the cache's `labels.mmap`
(teacher argmax). `test_fs_accuracy` is fraction matching `fs_labels.mmap`
(filesystem-extension labels with teacher fallback for unmapped
extensions). The two should be close (the corpus has ~99.5%
teacher-vs-filesystem agreement).
