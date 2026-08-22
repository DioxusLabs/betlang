# Training the betlang Bloom binary model

The model shipped in `assets/magika/source-bloom.bin` is a 1,932,116-byte
weights-only MBL3 payload: a deterministic counting-Bloom n-gram signature
(78 planes x 4,096 bits = 319,488 bits) plus a binary {-1,+1} linear head
evaluated with XOR + popcount. On the rebuilt ungated held-out
filesystem-label test split it scores **0.947959 fs_accuracy** versus
**0.944888** for the previous wordseq MSQ1 student on the same split.

The corpus, cache, and evaluation split are rebuilt entirely from ungated
public sources (`--no-gated`), so metrics are not comparable to numbers
reported for earlier artifacts on other splits. Both models above are
evaluated on the identical rebuilt split.

## Files

| File | Purpose |
|---|---|
| `train_bloom_head.py` | Bloom signature encoder, binary-head trainer, MBL3 exporter, and integer simulator for the shipped model. |
| `export_bloom_golden.py` | Loads an MBL3 artifact, runs the integer simulator on sample files, and writes `tests/fixtures/bloom-golden.bin` for the Rust parity test. |
| `build_finetune_corpus.py` | Rebuilds the training corpus from The Stack (smol-xl), GitHub repo tarballs, and synthetic ambiguous Markdown/YAML samples. |
| `make_pruned48_config.py` | Generates the pruned 48-label teacher config from the `magika` pip package config. |
| `build_fs_labels.py` | Builds `{split}.fs_labels.mmap` (filesystem truth with teacher fallback) aligned to the cache. |
| `train_magika_qat_student.py` | Legacy wordseq trainer; still used with `--prepare-cache-only` to build the teacher cache (tokens, teacher marginals, v3 units). |
| `train_magika_source_student.py` | Magika teacher loader, byte-window feature extraction, cache iteration helpers. Imported by the cache builder. |
| `eval_50kb_model.py` | Evaluates a legacy MSQ1 `.bin` on a cache split (used for the baseline comparison row). |
| `cache_self_distill.py`, `train_v2_student.py`, `hard_gen_*.py`, `confusion_by_size.py` | Legacy wordseq recipe tooling, kept for reference. |

## Recipe

```
signature:      counting-Bloom, 78 planes x 4096 bits (319,488 bits)
features:       byte n-grams orders 1-8 (begin/end halves), case-folded words
                (2 hash folds), word bigrams/trigrams, line-start words,
                tokenizer-v3 unit n-grams orders 1-4
binarization:   thermometer thresholds over bucket counts (1/2/4/8)
head:           binary {-1,+1} linear, straight-through estimator,
                per-(class, plane) float scale + per-class bias
soft targets:   Magika v3.3 raw head marginals, per-class sigmoid BCE (0.5)
hard targets:   filesystem labels, CE with 0.05 label smoothing (0.5)
optimizer:      Adam 1e-3, 1000 warmup steps, cosine decay, grad-clip
epochs:         30, EMA weights for evaluation, best checkpoint by valid fs
seed:           2
```

## Reproducing the shipped model

1. **Teacher assets** — copy `model.onnx` from the `magika` pip package
   (`magika/models/standard_v3_3/`) and generate the pruned config:

   ```bash
   python3 scripts/make_pruned48_config.py \
     --output /tmp/magika-teacher/config.pruned48.min.json
   ```

2. **Corpus** — build the fully ungated corpus (no HF token needed):

   ```bash
   python3 scripts/build_finetune_corpus.py \
     --output /tmp/betlang-corpus --no-gated
   ```

   `--no-gated` skips the gated `bigcode/the-stack` shards: yaml, json, toml,
   ini, xml, swift, and cobol are harvested from public GitHub repository
   tarballs instead. Files from one repository always land in the same split,
   and per-repository caps keep any single repository from dominating a label.

3. **Cache + labels** — build the teacher cache (tokens, teacher marginals,
   tokenizer-v3 units) and filesystem labels:

   ```bash
   python3 scripts/train_magika_qat_student.py \
     --dataset /tmp/betlang-corpus/files \
     --cache-dir /tmp/betlang-cache \
     --magika-model /tmp/magika-teacher/model.onnx \
     --magika-config /tmp/magika-teacher/config.pruned48.min.json \
     --output /tmp/unused.bin \
     --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
     --unit-tokenizer 3 \
     --head-marginal-targets \
     --min-teacher-head-mass 0.1 \
     --prepare-cache-only
   python3 scripts/build_fs_labels.py \
     --dataset /tmp/betlang-corpus/files \
     --cache-dir /tmp/betlang-cache
   ```

4. **Train + export** — encode Bloom signatures (cached to
   `{split}.bloom319488_v6.mmap` on first run) and train the binary head:

   ```bash
   python3 scripts/train_bloom_head.py \
     --cache-dir /tmp/betlang-cache \
     --output assets/magika/source-bloom.bin \
     --hidden-per-block 0 \
     --epochs 30 \
     --hard-labels fs \
     --soft-loss-weight 0.5 \
     --hard-loss-weight 0.5 \
     --learning-rate 1e-3 \
     --warmup-steps 1000 \
     --augment-fraction 0.15 \
     --threads 8
   ```

   `--augment-fraction 0.15` mixes in short prefix crops (12–1,024 bytes,
   hard labels only) so tiny standalone snippets are represented in training.

   Training runs on CPU (no GPU needed; ~2 hours on 8 cores). After the last
   epoch the script prints test metrics and verifies the exported artifact
   with its integer simulator (`argmax agreement=1.0000` is expected — the
   export is lossless).

5. **Golden vectors** — regenerate the Rust parity fixture whenever the
   artifact changes:

   ```bash
   python3 scripts/export_bloom_golden.py \
     --model assets/magika/source-bloom.bin \
     --output tests/fixtures/bloom-golden.bin \
     tests/fixtures/languages/*
   cargo test
   ```

   `bloom_matches_python_golden_vectors` asserts the Rust runtime reproduces
   the Python simulator's logits on all 48 language fixtures.

## Expected metrics

Printed by `train_bloom_head.py` after the final epoch:

```
test_teacher_parity=0.929505
test_fs_accuracy=0.947959
test_fs_macro_recall=0.923254
```

Baseline for the previous MSQ1 student on the same rebuilt cache:

```bash
python3 scripts/eval_50kb_model.py \
  --checkpoint <legacy source-student-q4.bin> \
  --cache-dir /tmp/betlang-cache \
  --architecture wordseq-b1024-k3-m2048-tiny-3conv-hidden \
  --split test
# test_teacher_parity=0.952408
# test_fs_accuracy=0.944888
# test_fs_macro_recall=0.937713
```

`test_fs_accuracy` is the fraction matching `fs_labels.mmap`
(filesystem-extension labels with teacher fallback for unmapped extensions).
