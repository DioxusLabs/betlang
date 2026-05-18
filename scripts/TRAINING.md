# Training the betlang wordseq student model

The model shipped in `assets/magika/source-student-q4.bin` is a 47,840-byte
weights-only MSQ1 wordseq payload trained for the fixed 48-label production
head. It uses the v3 word-unit tokenizer and hits **0.965238 fs_accuracy** with
**0.965411 macro_recall** on the manifest-aligned held-out filesystem-label test
split.

## Files

| File | Purpose |
|---|---|
| `train_v2_student.py` | Standalone wrapper for the current fixed production recipe. |
| `train_magika_qat_student.py` | QAT trainer and metadata-free MSQ1 exporter for the shipped `wordseq-b1024-k3-m2048-tiny-3conv-hidden` model. |
| `train_magika_source_student.py` | Magika teacher loader, byte-window feature extraction, cache iteration helpers. Imported by the QAT trainer. |
| `eval_50kb_model.py` | Loads an exported MSQ1 `.bin`, runs a forward pass, and reports accuracy on a chosen split. |
| `confusion_by_size.py` | Evaluates an exported wordseq model, aligns cached rows to raw file sizes, and renders the README confusion-matrix images. |

## Recipe (frozen in `train_v2_student.py`)

```
arch:           wordseq-b1024-k3-m2048-tiny-3conv-hidden
unit_tokenizer: 3  (punct/digit compression + case-folded words + isolated brackets)
length_buckets: yes
classes:        fixed 48-label production head
hard_loss_weight: 0.5
self_loss_weight: 0.5
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

1. **Magika v3.3 teacher** — `model.onnx` from
   `https://github.com/google/magika/tree/main/python/magika/models/standard_v3_3`
   plus a config whose `target_labels_space` is pruned to the fixed 48 exported
   labels.
2. **A source-language corpus** — extension-suffixed files split into
   `files/{train,valid,test}/`. The shipped model was evaluated on a
   manifest-aligned held-out filesystem-label split.
3. **A pre-built training cache** — see "Cache" below.
4. **The fixed 48-label production head** — the exporter intentionally rejects
   checkpoints whose labels or slug mapping differ from the embedded runtime
   head.

Then run:

```bash
python3 scripts/train_v2_student.py \
  --dataset /path/to/corpus/files \
  --cache-dir /path/to/cache \
  --magika-model /path/to/magika/standard_v3_3/model.onnx \
  --magika-config /path/to/magika/standard_v3_3/config.pruned48.min.json \
  --output assets/magika/source-student-q4.bin
```

Expected runtime on a single 12 GB GPU: ~50 minutes for 60 epochs at
~12 k examples/sec.

## Cache

The trainer expects a cache directory containing per-split mmap files:
`tokens.mmap`, `units_v3.mmap`, `labels.mmap`, `probabilities.mmap`, and
`self_probabilities.mmap` for the frozen self-distillation recipe. The cache
metadata must list the fixed 48 exported labels in the same order as the
runtime model head.

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
  --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
  --split test
```

Expected output:
```
test_fs_accuracy=0.965238
```

`test_fs_accuracy` is fraction matching `fs_labels.mmap`
(filesystem-extension labels with teacher fallback for unmapped extensions).
The by-size report in `reports/actual_dataset_confusion_by_size.md` records the
same evaluation and notes cache labels that are intentionally absent from the
48-label model head.
