#!/usr/bin/env python3
"""Evaluate an exported MSQ1 50KB wordseq student against the cached test split.

Reports both:
  - test_teacher_parity   — fraction matching cache labels.mmap (teacher argmax)
  - test_fs_accuracy      — fraction matching cache fs_labels.mmap (file extension
                            truth, with teacher fallback for unmapped extensions).

Loads the .bin via load_exported_layer_weights (decodes int4 packing back to fp32
weights) and sets them on a fresh model built from --architecture. Loads the
matching units_vN.mmap directly without calling convert_splits_to_word_units
(which would rebuild caches for any split arg passed in).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_magika_qat_student import (  # type: ignore
    QAT_ACTIVE,
    architecture_uses_word_units,
    build_word_seq_hashembed_hidden_model,
    load_exported_layer_weights,
    write_confusion_matrix,
    wordseq_config_for_architecture,
)

# At inference, the 2-bit (ternary) in-call quantizer recomputes
# `scale = mean(|w|)` on the loaded weights, which shrinks the scale (loaded
# weights have many zeros). The exported .bin already contains the correct
# discrete representation; bypass the in-call quantization so it is used as-is.
QAT_ACTIVE.assign(False)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--architecture", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--unit-tokenizer", type=int, default=None,
                   help="Tokenizer version of the units cache to read "
                        "({split}.units_v{N}.mmap). Defaults to checkpoint "
                        "metadata, or v2 when metadata is absent.")
    p.add_argument("--confusion-matrix-output", type=Path)
    p.add_argument("--confusion-matrix-top", type=int, default=20)
    args = p.parse_args()

    classes_meta = json.loads((args.cache_dir / f"{args.split}.json").read_text())
    classes = int(classes_meta["classes"])
    n = int(classes_meta["count"])
    token_length = int(classes_meta.get("token_length", 2048))
    print(f"split={args.split} n={n} classes={classes} token_length={token_length}")

    if not architecture_uses_word_units(args.architecture):
        raise SystemExit("only wordseq architectures supported by this eval script")
    cfg = wordseq_config_for_architecture(args.architecture)
    model = build_word_seq_hashembed_hidden_model(classes, bits=4, **cfg)

    # Load weights from exported MSQ1 .bin.
    layer_weights, metadata = load_exported_layer_weights(args.checkpoint)
    metadata_tokenizer = metadata.get("tokenizer_version")
    unit_tokenizer = args.unit_tokenizer
    if unit_tokenizer is None:
        unit_tokenizer = int(metadata_tokenizer or 2)
    print(
        f"checkpoint: bits={metadata.get('bits')} arch={metadata.get('architecture','?')} "
        f"tokenizer_version={metadata_tokenizer or 'legacy-v2'}"
    )
    loaded = 0
    for layer in model.layers:
        if layer.name in layer_weights:
            current = layer.get_weights()
            target = layer_weights[layer.name]
            if len(current) != len(target):
                print(f"warn: layer {layer.name} has {len(current)} weights, exported {len(target)}; skipping",
                      file=sys.stderr)
                continue
            ok = True
            for i, (cur, tgt) in enumerate(zip(current, target)):
                if cur.shape != tgt.shape:
                    print(f"warn: layer {layer.name}[{i}] shape mismatch {cur.shape} vs {tgt.shape}", file=sys.stderr)
                    ok = False
                    break
            if ok:
                layer.set_weights([t.astype(cur.dtype) for cur, t in zip(current, target)])
                loaded += 1
    print(f"loaded weights into {loaded} layers")

    units_path = args.cache_dir / f"{args.split}.units_v{unit_tokenizer}.mmap"
    if not units_path.exists():
        raise SystemExit(f"missing {units_path} — build it via convert_splits_to_word_units "
                         f"with --unit-tokenizer {unit_tokenizer}, or use a different version")
    units = np.memmap(units_path, dtype=np.int32, mode="r", shape=(n, token_length))
    teacher_labels = np.array(np.memmap(args.cache_dir / f"{args.split}.labels.mmap", dtype=np.int64, mode="r",
                                        shape=(n,)))
    fs_labels_path = args.cache_dir / f"{args.split}.fs_labels.mmap"
    fs_labels = None
    if fs_labels_path.exists():
        fs_labels = np.array(np.memmap(fs_labels_path, dtype=np.int64, mode="r", shape=(n,)))
        fsm = json.loads((args.cache_dir / f"{args.split}.fs_labels.json").read_text())
        print(f"fs_labels.json: {fsm}")

    preds = np.empty(n, dtype=np.int64)
    offset = 0
    bs = args.batch_size
    for start in range(0, n, bs):
        end = min(start + bs, n)
        batch = tf.convert_to_tensor(units[start:end], dtype=tf.int32)
        out = model(batch, training=False)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        argmax = tf.argmax(logits, axis=1).numpy().astype(np.int64)
        preds[start:end] = argmax
        offset = end

    teacher_acc = float((preds == teacher_labels).mean())
    print(f"{args.split}_teacher_parity={teacher_acc:.6f}")
    if fs_labels is not None:
        fs_acc = float((preds == fs_labels).mean())
        print(f"{args.split}_fs_accuracy={fs_acc:.6f}")

        # Per-segment breakdown: mapped vs unmapped
        meta = json.loads((args.cache_dir / f"{args.split}.fs_labels.json").read_text())
        unmapped_count = int(meta.get("ext_unmapped", 0))
        # We don't have a per-row mapped-vs-unmapped flag; infer it: unmapped rows
        # have fs_labels == teacher_labels (by construction). Estimate accuracy on
        # the strict-mapped subset as: where fs != teacher, did model predict fs?
        strict_mapped_mask = fs_labels != teacher_labels
        strict_mapped = int(strict_mapped_mask.sum())
        if strict_mapped > 0:
            strict_correct_fs = int((preds[strict_mapped_mask] == fs_labels[strict_mapped_mask]).sum())
            print(f"  on STRICT MAPPED subset (fs_label != teacher_label, n={strict_mapped}):")
            print(f"    model_predicted_fs   = {strict_correct_fs/strict_mapped:.4f}")
            print(f"  on AGREED subset      (fs_label == teacher_label, n={n - strict_mapped}):")
            agreed_mask = ~strict_mapped_mask
            print(f"    model_correct        = {(preds[agreed_mask] == fs_labels[agreed_mask]).mean():.4f}")
    if args.confusion_matrix_output is not None:
        target_labels = fs_labels if fs_labels is not None else teacher_labels
        confusion = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(confusion, (target_labels, preds), 1)
        label_names = metadata.get("labels")
        if not isinstance(label_names, list) or len(label_names) != classes:
            label_names = [str(i) for i in range(classes)]
        write_confusion_matrix(
            args.confusion_matrix_output,
            confusion,
            [str(label) for label in label_names],
            args.confusion_matrix_top,
        )
        print(f"confusion_matrix_output={args.confusion_matrix_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
