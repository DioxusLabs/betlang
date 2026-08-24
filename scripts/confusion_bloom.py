#!/usr/bin/env python3
"""Render the README confusion-matrix PNGs for an exported Bloom MBL5 model.

Evaluates the artifact on a cache split with the vectorized integer simulator
(gathering the artifact's selected columns from the cached full-resolution
signatures), aligns cache rows to raw file sizes by hashing token windows
against the corpus files (same scheme as confusion_by_size.py), and renders
assets/confusion-overall.png plus assets/confusion-by-size.png.

Usage:
    python3 scripts/confusion_bloom.py \
      --model assets/magika/source-bloom.bin \
      --cache-dir /path/to/cache \
      --dataset /path/to/corpus/files
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from confusion_by_size import (  # noqa: E402
    BUCKETS,
    align_file_sizes,
    render_overall_png,
    render_size_png,
)
from export_bloom_golden import load_mbl5  # noqa: E402
from train_bloom_head import BLOCK, CLASSES, FULL_BITS, FULL_PLANES, FULL_WORDS  # noqa: E402


def selection_columns(model: dict[str, np.ndarray]) -> np.ndarray:
    """Full-resolution column index of each selected signature bit."""
    cols = []
    at = 0
    for group, level, width in model["planes"]:
        plane = FULL_PLANES.index((group, level))
        cols.append(plane * BLOCK + model["buckets"][at:at + width].astype(np.int64))
        at += width
    return np.concatenate(cols)


def batch_logits(model: dict[str, np.ndarray], cols: np.ndarray,
                 signatures: np.ndarray) -> np.ndarray:
    """Vectorized XOR/popcount forward for MBL5 over full-res signatures."""
    n = signatures.shape[0]
    k = int(model["bits"])
    bits = np.unpackbits(
        signatures.view(np.uint8).reshape(n, -1), axis=1, bitorder="little"
    )[:, :FULL_BITS][:, cols]
    w_bits = np.unpackbits(
        model["head"].view(np.uint8).reshape(CLASSES, -1), axis=1, bitorder="little"
    )[:, :k]
    x_pm = 2 * bits.astype(np.int32) - 1
    w_pm = 2 * w_bits.astype(np.int32) - 1
    acc = np.zeros((n, CLASSES), dtype=np.int64)
    at = 0
    for plane, (_, _, width) in enumerate(model["planes"]):
        z = x_pm[:, at:at + width] @ w_pm[:, at:at + width].T
        acc += model["q"][None, :, plane].astype(np.int64) * z
        at += width
    step = model["step"].astype(np.float32)
    return step[None, :] * acc.astype(np.float32) + model["bias"][None, :].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--png-output", type=Path, default=Path("assets/confusion-by-size.png"))
    parser.add_argument("--overall-png-output", type=Path, default=Path("assets/confusion-overall.png"))
    args = parser.parse_args()

    meta = json.loads((args.cache_dir / f"{args.split}.json").read_text())
    count = meta["count"]
    labels = meta["labels"]
    feats = np.memmap(
        args.cache_dir / f"{args.split}.bloom{FULL_BITS}_v6.mmap",
        dtype=np.uint64,
        mode="r",
        shape=(count, FULL_WORDS),
    )
    fs_labels = np.asarray(
        np.memmap(
            args.cache_dir / f"{args.split}.fs_labels.mmap",
            dtype=np.int64,
            mode="r",
            shape=(count,),
        )
    )

    model = load_mbl5(args.model)
    cols = selection_columns(model)
    preds = np.empty(count, dtype=np.int64)
    for start in range(0, count, args.batch_size):
        stop = min(start + args.batch_size, count)
        preds[start:stop] = batch_logits(
            model, cols, np.asarray(feats[start:stop])).argmax(axis=1)
        print(f"eval {stop}/{count}", flush=True)

    fs_accuracy = float((preds == fs_labels).mean())
    print(f"{args.split}_fs_accuracy={fs_accuracy:.6f}")

    overall = np.zeros((CLASSES, CLASSES), dtype=np.int64)
    np.add.at(overall, (fs_labels, preds), 1)
    render_overall_png(args.overall_png_output, overall, labels, fs_accuracy)
    print(f"wrote {args.overall_png_output}")

    sizes, stats = align_file_sizes(args.dataset, args.cache_dir, args.split, count)
    print(f"size alignment: {stats}")
    matrices = []
    for _, low, high in BUCKETS:
        mask = sizes >= low if high is None else (sizes >= low) & (sizes <= high)
        matrix = np.zeros((CLASSES, CLASSES), dtype=np.int64)
        np.add.at(matrix, (fs_labels[mask], preds[mask]), 1)
        matrices.append(matrix)
    render_size_png(args.png_output, matrices, labels, fs_accuracy)
    print(f"wrote {args.png_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
