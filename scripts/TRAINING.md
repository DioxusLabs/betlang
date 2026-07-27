# Training the betlang prompt-detection student model

The model shipped in `assets/magika/source-student-q4.bin` is a 45,448-byte
weights-only MSQ1 wordseq payload trained for the fixed 2-label production
head (`prompt`, `shell_command`). It uses the v3 word-unit tokenizer.

Unlike the earlier source-language student, there is no Magika teacher: the
2-class model trains from scratch on hard corpus labels.

## Files

| File | Purpose |
|---|---|
| `build_prompt_corpus.py` | Downloads public datasets and writes the labelled `prompt` / `shell_command` corpus. |
| `train_prompt_student.py` | Standalone trainer: caches v3 unit ids per split, trains the production architecture with hard labels + QAT, and exports the MSQ1 payload. |
| `train_magika_qat_student.py` | Quantized layer definitions, wordseq architectures, v3 tokenizer, and the metadata-free MSQ1 exporter. Imported by the prompt trainer. |
| `train_magika_source_student.py` | Byte-window feature extraction helpers (`magika_features`) shared with the runtime. |
| `eval_50kb_model.py`, `confusion_by_size.py`, `cache_self_distill.py`, `build_fs_labels.py`, `make_pruned48_config.py`, `hard_gen_*.py` | Tooling from the source-language lineage, kept for reference; not used by the prompt recipe. |

## Recipe

```
arch:           wordseq-b1024-k3-m2048-tiny-3conv-hidden
unit_tokenizer: 3  (punct/digit compression + case-folded words + isolated brackets)
length_buckets: yes (batches sorted by unit length, trimmed per batch)
classes:        fixed 2-label production head
loss:           softmax cross-entropy, label_smoothing 0.05
LR:             cosine 8e-4 -> 5%, AdamW grad-clip 1.0
epochs:         24, qat_start_epoch=12
seed:           2
```

The exported checkpoint is the post-QAT epoch with the best validation
accuracy.

## Reproducing the shipped model

Runs on a laptop CPU in well under an hour; no GPU or teacher assets needed.

1. **Corpus** — downloads real user prompts (oasst1 English first turns,
   ShareGPT first human turns, no_robots, Stack Overflow titles) plus
   instruction datasets (Alpaca, Dolly, awesome-chatgpt-prompts) for the
   prompt class, and NL2Bash one-liners, tldr-pages example commands, agent-trajectory bash tool calls
   (SWE-smith), real user shell history, and
   synthetic quoted-English hard negatives for the shell class (English
   tldr pages only), writing one
   file per sample into deterministic, leakage-aware 90/5/5 splits (split
   assignment hashes a near-duplicate group key, not the raw text):

   ```bash
   python3 scripts/build_prompt_corpus.py --output /tmp/betlang-prompt-corpus
   ```

2. **Train + export** — caches tokens and v3 unit ids per split on first run,
   trains full-precision for the first half of the schedule, enables
   quantization-aware training for the second half, then exports the
   metadata-free MSQ1 payload:

   ```bash
   python3 scripts/train_prompt_student.py \
     --dataset /tmp/betlang-prompt-corpus/files \
     --cache-dir /tmp/betlang-prompt-cache \
     --output assets/magika/source-student-q4.bin
   ```

   The trainer prints `test_accuracy` and per-class recall for the exported
   checkpoint at the end of the run.

## Verifying the trained model

Run the Rust test suite: it loads the embedded payload and checks golden
predictions for both classes against the fixtures in
`tests/fixtures/{prompt,shell_command}/`:

```bash
cargo test
```
