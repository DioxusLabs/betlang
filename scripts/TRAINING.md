# Training the betlang wordseq student model

The model shipped in `assets/magika/source-student-q4.bin` is a 47,840-byte
weights-only MSQ1 wordseq payload trained for the fixed 48-label production
head. It uses the v3 word-unit tokenizer.

The original artifact was trained from scratch with the frozen recipe below on
the `bigorig` corpus and hit **0.965238 fs_accuracy** with **0.965411
macro_recall** on the manifest-aligned held-out filesystem-label test split.
The shipped artifact is that model fine-tuned on a rebuilt public corpus with
one-vs-all distillation (plus self-distillation from a larger one-vs-all
parent) to fix the Markdown/YAML bare-list confusion and the manufactured
confidence on ambiguous inputs from issue #5 (see "Fine-tuning the shipped
model" below); on the rebuilt held-out split it scores **0.942353
fs_accuracy**, **0.939690 macro_recall**, and **0.944055 teacher parity**
(versus 0.926160 / 0.929904 / 0.922111 for the pre-fine-tune artifact on the
same split).

## Files

| File | Purpose |
|---|---|
| `train_v2_student.py` | Standalone wrapper for the current fixed production recipe. |
| `train_magika_qat_student.py` | QAT trainer and metadata-free MSQ1 exporter for the shipped `wordseq-b1024-k3-m2048-tiny-3conv-hidden` model. |
| `train_magika_source_student.py` | Magika teacher loader, byte-window feature extraction, cache iteration helpers. Imported by the QAT trainer. |
| `build_finetune_corpus.py` | Rebuilds the fine-tuning corpus from The Stack, GitHub, and synthetic ambiguous Markdown/YAML samples. |
| `make_pruned48_config.py` | Generates the pruned 48-label teacher config from the `magika` pip package config. |
| `build_fs_labels.py` | Builds `{split}.fs_labels.mmap` (filesystem truth with teacher fallback) aligned to the cache. |
| `cache_self_distill.py` | Caches a parent student's per-class predictions as `{split}.self_probabilities.mmap` for self-distillation. |
| `hard_gen_*.py` | Teacher-vetted synthetic hard-boundary generators (kept for experiments; not sampled by the shipped recipe). |
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

## Fine-tuning the shipped model

The shipped artifact was produced by fine-tuning the original `bigorig`
checkpoint on a rebuilt public corpus. This recipe is fully reproducible on a
laptop CPU (the fine-tune itself takes ~20 minutes at ~2,000 examples/sec):

1. **Teacher assets** — copy `model.onnx` from the `magika` pip package
   (`magika/models/standard_v3_3/`) and generate the pruned config:

   ```bash
   python3 scripts/make_pruned48_config.py \
     --output /tmp/magika-teacher/config.pruned48.min.json
   ```

2. **Corpus** — `scripts/build_finetune_corpus.py` downloads per-language
   samples from `bigcode/the-stack-smol-xl` (ungated) and `bigcode/the-stack`
   (gated; your HF token must have accepted the dataset terms), harvests
   Objective-C/Gradle/Gemfile files from GitHub repos, and generates a small
   synthetic set of heading + single-word dash-list Markdown files plus YAML
   keyed-sequence counterexamples for the issue #5 ambiguity. It also adds
   train-only bare `- item` lists (no heading): the teacher's renormalized
   target for those is genuinely split between YAML and Markdown, which
   teaches the student to report uncertainty instead of a confident YAML
   label. Files from one repository always land in the same split.

   ```bash
   python3 scripts/build_finetune_corpus.py --output /tmp/betlang-finetune-corpus
   ```

3. **Cache + labels** — build the teacher cache, unit ids, and fs labels:

   ```bash
   python3 scripts/train_magika_qat_student.py \
     --dataset /tmp/betlang-finetune-corpus/files \
     --cache-dir /tmp/betlang-finetune-cache \
     --magika-model /tmp/magika-teacher/model.onnx \
     --magika-config /tmp/magika-teacher/config.pruned48.min.json \
     --output /tmp/out.bin \
     --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
     --unit-tokenizer 3 --prepare-cache-only
   python3 scripts/build_fs_labels.py \
     --dataset /tmp/betlang-finetune-corpus/files \
     --cache-dir /tmp/betlang-finetune-cache
   ```

4. **Parent for self-distillation** — train a ~2x larger student on the same
   cache with the same one-vs-all targets, full-precision (QAT never enabled;
   the metadata-free exporter only supports the production architecture, so
   the best checkpoint is saved as raw weights instead), then cache its
   per-class sigmoid marginals:

   ```bash
   python3 scripts/train_magika_qat_student.py \
     --dataset /tmp/betlang-finetune-corpus/files \
     --cache-dir /tmp/betlang-finetune-cache \
     --magika-model /tmp/magika-teacher/model.onnx \
     --magika-config /tmp/magika-teacher/config.pruned48.min.json \
     --output /tmp/parent_unused.bin \
     --checkpoint-weights /tmp/parent_med.npz \
     --architecture wordseq-b1536-k3-m2048-med-3conv-hidden \
     --unit-tokenizer 3 --length-buckets \
     --min-teacher-head-mass 0.1 \
     --head-marginal-targets --soft-loss-mode bce \
     --mass-discounted-hard-labels \
     --init-from-checkpoint assets/magika/source-student-q4.bin \
     --epochs 18 --batch-size 128 \
     --learning-rate 4e-4 --cosine-decay --min-learning-rate-ratio 0.05 \
     --weight-bits 4 --qat-start-epoch 99 \
     --distill-temperature 1 --hard-loss-weight 0.5 \
     --label-smoothing 0.05 --cutmix-prob 0.5 \
     --early-stop-patience 18 --eval-every 1 --seed 2
   python3 scripts/cache_self_distill.py \
     --checkpoint /tmp/parent_med.npz \
     --architecture wordseq-b1536-k3-m2048-med-3conv-hidden \
     --cache-dir /tmp/betlang-finetune-cache --output-mode sigmoid
   ```

   The parent initializes from the exported tiny checkpoint (smaller tensors
   are tiled up to the wider shapes) and reached 0.9492 test teacher parity.

5. **Fine-tune** — start from the exported checkpoint with QAT active from
   epoch 0 (the ternary/int4 fake-quant is idempotent on exported weights, so
   training starts numerically identical to the shipped model). `--eval-initial`
   makes the initial checkpoint the export bar, so the output only changes if
   validation teacher parity improves:

   ```bash
   python3 scripts/train_magika_qat_student.py \
     --dataset /tmp/betlang-finetune-corpus/files \
     --cache-dir /tmp/betlang-finetune-cache \
     --magika-model /tmp/magika-teacher/model.onnx \
     --magika-config /tmp/magika-teacher/config.pruned48.min.json \
     --output assets/magika/source-student-q4.bin \
     --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
     --unit-tokenizer 3 --length-buckets \
     --min-teacher-head-mass 0.1 \
     --head-marginal-targets --soft-loss-mode bce \
     --mass-discounted-hard-labels \
     --self-probabilities /tmp/betlang-finetune-cache --self-loss-weight 0.5 \
     --init-from-checkpoint assets/magika/source-student-q4.bin \
     --eval-initial \
     --epochs 24 --batch-size 128 \
     --learning-rate 2e-4 --cosine-decay --min-learning-rate-ratio 0.05 \
     --weight-bits 4 --qat-start-epoch 0 \
     --distill-temperature 1 --hard-loss-weight 0.3 \
     --label-smoothing 0.05 --cutmix-prob 0.5 \
     --early-stop-patience 24 --eval-every 1 --seed 2
   ```

   The hard-loss weight drops from the frozen recipe's 0.5 to 0.3: with the
   longer schedule and the parent's second soft term, a 0.5 hard term
   re-sharpens exactly the ambiguous rows whose uncertainty the one-vs-all
   targets are meant to preserve (and drags nearby boundaries like ini/toml).
   A/B runs: hard 0.5 at 24 epochs reached the same parity but pushed the
   bare-list median top-1 back up to 0.71; hard 0.3 keeps it at 0.51.

   Schedule notes from A/B runs on this fine-tune: FP-then-QAT (the frozen
   recipe's 45/60 split) does not survive a short schedule — an FP phase
   reaches ~0.952 valid parity but re-quantization at epoch 12/20 drops it to
   ~0.930 and the cosine tail cannot recover, so short fine-tunes keep QAT
   active from epoch 0.

   The target scheme differs from the frozen from-scratch recipe. The Magika
   teacher predicts over 214 labels, and the original cache renormalized its
   probabilities over the 48 exported labels — which manufactures confident
   targets for inputs the teacher mostly places outside the head (a bare
   `- item` list is mostly `txt` to the teacher, and renormalization turned
   that into `yaml` at 0.9+). Instead:

   - `--head-marginal-targets` caches the teacher's raw per-class marginals
     (rows sum to the in-head mass, <= 1).
   - `--soft-loss-mode bce` distills them one-vs-all with per-class sigmoid
     cross-entropy, which does not require normalized targets, so
     out-of-scope mass simply lowers every class target (cf. "Revisiting
     One-vs-All Classifiers for Predictive Uncertainty and OOD Detection",
     Padhy et al. 2020). Temperature is unused in this mode.
   - `--mass-discounted-hard-labels` scales the hard argmax target by the
     in-head mass so it cannot re-sharpen rows the teacher is unsure about.
   - `--min-teacher-head-mass 0.1` drops rows that are almost entirely
     out-of-scope, whose in-head argmax is noise.

   The runtime is unchanged: its softmax over the 48 logits maps the
   BCE-trained per-class log-odds to a near-uniform distribution when every
   class is unlikely, and to a confident one when a single class dominates.

   The original self-distillation probabilities from the `bigorig` recipe are
   tied to that corpus's cache rows, so the fine-tune trains its own parent
   (step 4) instead of reusing them.

## Verifying the trained model

```bash
python3 scripts/eval_50kb_model.py \
  --checkpoint assets/magika/source-student-q4.bin \
  --cache-dir /path/to/cache \
  --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
  --split test
```

Expected output for the shipped artifact on the rebuilt fine-tune cache:
```
test_teacher_parity=0.944055
test_fs_accuracy=0.942353
```

`test_fs_accuracy` is fraction matching `fs_labels.mmap`
(filesystem-extension labels with teacher fallback for unmapped extensions).
The by-size report in `reports/actual_dataset_confusion_by_size.md` records the
same evaluation and notes cache labels that are intentionally absent from the
48-label model head.
