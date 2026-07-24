#!/usr/bin/env python3
"""Cache a parent student's predictions as {split}.self_probabilities.mmap.

Loads a parent checkpoint (either a .npz saved via the trainer's
--checkpoint-weights, or an exported MSQ1 .bin) and runs it over the cached
unit-id rows of each split. With --output-mode sigmoid (default) the stored
targets are per-class sigmoid marginals for one-vs-all self-distillation
(--soft-loss-mode bce); with --output-mode softmax they are the normalized
distribution used by the legacy CE self-loss.

Usage:
    python3 scripts/cache_self_distill.py \
      --checkpoint /tmp/parent.npz \
      --architecture wordseq-b1536-k3-m2048-med-3conv-hidden \
      --cache-dir /tmp/betlang-finetune-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_magika_qat_student import (  # noqa: E402
    QAT_ACTIVE,
    SPLITS,
    TOKEN_LENGTH,
    architecture_uses_word_units,
    build_word_seq_hashembed_hidden_model,
    load_exported_layer_weights,
    wordseq_config_for_architecture,
)

QAT_ACTIVE.assign(False)


def load_parent(checkpoint: Path, architecture: str, classes: int) -> tf.keras.Model:
    if not architecture_uses_word_units(architecture):
        raise SystemExit("only wordseq architectures are supported")
    cfg = wordseq_config_for_architecture(architecture)
    model = build_word_seq_hashembed_hidden_model(classes, bits=4, **cfg)
    if checkpoint.suffix == ".npz":
        arrays = np.load(checkpoint)
        loaded = 0
        for layer in model.layers:
            values = []
            index = 0
            while f"{layer.name}|{index}" in arrays:
                values.append(arrays[f"{layer.name}|{index}"])
                index += 1
            if not values:
                continue
            current = layer.get_weights()
            if len(current) != len(values) or any(c.shape != v.shape for c, v in zip(current, values)):
                raise SystemExit(f"layer {layer.name} shape mismatch against checkpoint")
            layer.set_weights([v.astype(c.dtype) for c, v in zip(current, values)])
            loaded += 1
    else:
        layer_weights, _ = load_exported_layer_weights(checkpoint)
        loaded = 0
        for layer in model.layers:
            if layer.name not in layer_weights:
                continue
            current = layer.get_weights()
            target = layer_weights[layer.name]
            if len(current) != len(target) or any(c.shape != t.shape for c, t in zip(current, target)):
                continue
            layer.set_weights([t.astype(c.dtype) for c, t in zip(current, target)])
            loaded += 1
    print(f"loaded weights into {loaded} layers from {checkpoint}", flush=True)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to --cache-dir")
    parser.add_argument("--output-mode", choices=("sigmoid", "softmax"), default="sigmoid")
    parser.add_argument("--unit-tokenizer", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args()

    output_dir = args.output_dir or args.cache_dir
    meta = json.loads((args.cache_dir / f"{args.splits[0]}.json").read_text())
    classes = int(meta["classes"])
    model = load_parent(args.checkpoint, args.architecture, classes)

    suffix = "" if args.unit_tokenizer == 1 else f"_v{args.unit_tokenizer}"
    for split in args.splits:
        split_meta = json.loads((args.cache_dir / f"{split}.json").read_text())
        n = int(split_meta["count"])
        units = np.memmap(
            args.cache_dir / f"{split}.units{suffix}.mmap",
            dtype=np.int32,
            mode="r",
            shape=(n, TOKEN_LENGTH),
        )
        out = np.memmap(
            output_dir / f"{split}.self_probabilities.mmap",
            dtype=np.float32,
            mode="w+",
            shape=(n, classes),
        )
        correct = 0
        labels = np.memmap(args.cache_dir / f"{split}.labels.mmap", dtype=np.int64, mode="r", shape=(n,))
        for start in range(0, n, args.batch_size):
            end = min(start + args.batch_size, n)
            result = model(tf.convert_to_tensor(units[start:end], dtype=tf.int32), training=False)
            logits = result[0] if isinstance(result, (list, tuple)) else result
            if args.output_mode == "sigmoid":
                probs = tf.sigmoid(logits).numpy()
            else:
                probs = tf.nn.softmax(logits, axis=1).numpy()
            out[start:end] = probs.astype(np.float32)
            correct += int((probs.argmax(axis=1) == labels[start:end]).sum())
        out.flush()
        print(f"{split}: n={n} parent_teacher_parity={correct / n:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
