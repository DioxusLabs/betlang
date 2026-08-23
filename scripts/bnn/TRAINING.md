# Shannon-bit binary CNN training pipeline

These scripts produce `assets/bnn/source-bnn2.bin`, the packed "BBN2" ensemble
of binary CNNs that powers `betlang::detect`. All scripts run with Python 3.12,
PyTorch, and NumPy; caches and checkpoints live under `./bnn/` next to the
scripts (`OUT` in `train_bnn.py`).

## Pipeline

1. `build_corpus.py` — assemble the 48-label corpus (The Stack smol / GitHub
   files), with `rebalance.py` / `topup.py` to even out rare classes.
2. `build_cache.py` — extract the raw 2048-byte Magika begin/end windows into
   memory-mapped `{split}.windows.mmap` / `.lengths` / `.labels` caches.
3. `shannon.py` — learn the canonical Shannon code over training byte
   frequencies (add-one smoothing, 12-bit max length, ordered by
   `(length, byte)`), stored in `bnn/codebook.json`. Each window is encoded
   MSB-first into a fixed 16,384-bit stream plus a codeword-boundary plane.
4. `train_bnn.py` / `train_bnn2.py` / `train_bnn3.py` — train binary CNNs
   (binary weights and activations, sign activations, OR-pooling, segmented
   popcount heads) directly on the Shannon bitplanes.
5. `finetune_seg.py`, `finetune_short.py` — segmented-head and short-input
   fine-tuning of the base checkpoints.
6. `build_synth_hard.py`, `add_yaml_synth.py`, `add_yaml_synth2.py` and
   `finetune_synth.py` / `finetune_synth2.py` / `finetune_synth3.py` —
   synthetic hard-pair generation (YAML/Markdown lists and other confusable
   pairs) and fine-tuning for the ensemble members.
7. `weight_search.py` / `weight_valid.py` — grid-search ensemble weights under
   fixture, YAML/Markdown-edge, and benchmark-demo constraints, scored on the
   full and short validation splits.
8. `export_bnn2.py` / `make_fixtures2.py` — quantize (fixed point `FIX=65536`,
   i16 head weights), pack the ensemble into the BBN2 artifact, and emit
   `parity_fixtures2.json` from the same in-memory model state.
9. `eval_export2.py` — exact integer-semantics evaluation of the exported
   ensemble on the valid/test splits.
10. `eval_fixtures.py` / `eval_edge.py` / `eval_combos.py` — fixture-level and
    edge-case evaluation helpers.

## Parity check

```bash
BNN_FIXTURES=path/to/parity_fixtures2.json cargo test --release python_parity
```

verifies the Rust XNOR/popcount inference reproduces the exported integer
logits bit-exactly.
