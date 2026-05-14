#!/usr/bin/env python3
"""Render file-size-bucket confusion matrices for an exported wordseq model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_magika_qat_student import (  # type: ignore
    QAT_ACTIVE,
    architecture_uses_word_units,
    build_word_seq_hashembed_hidden_model,
    load_exported_layer_weights,
    wordseq_config_for_architecture,
)

QAT_ACTIVE.assign(False)

MAGIKA_BEG_SIZE = 1024
MAGIKA_END_SIZE = 1024
MAGIKA_BLOCK_SIZE = 4096
MAGIKA_PADDING_TOKEN = 256
TOKEN_LENGTH = MAGIKA_BEG_SIZE + MAGIKA_END_SIZE

BUCKETS = [
    ("<=128B", 0, 128),
    ("129-512B", 129, 512),
    ("513B-1KiB", 513, 1024),
    ("1-4KiB", 1025, 4096),
    ("4-16KiB", 4097, 16 * 1024),
    ("16-64KiB", 16 * 1024 + 1, 64 * 1024),
    (">64KiB", 64 * 1024 + 1, None),
]


def token_hash(values: np.ndarray | list[int]) -> bytes:
    arr = np.asarray(values, dtype="<u2")
    return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()


def source_paths(split_dir: Path):
    for root, dirs, files in os.walk(split_dir):
        dirs.sort()
        for filename in sorted(files):
            yield Path(root) / filename


def read_window_tokens(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            prefix = file.read(MAGIKA_BLOCK_SIZE)
            if size <= MAGIKA_BLOCK_SIZE:
                suffix = prefix
            else:
                file.seek(max(0, size - MAGIKA_BLOCK_SIZE))
                suffix = file.read(MAGIKA_BLOCK_SIZE)
    except OSError:
        return None
    if size == 0:
        return None

    stripped_beg = prefix[: min(size, MAGIKA_BLOCK_SIZE)].lstrip()
    stripped_end = suffix[-min(size, MAGIKA_BLOCK_SIZE) :].rstrip()
    if len(stripped_beg) < 8:
        return None

    beg = list(stripped_beg[:MAGIKA_BEG_SIZE])
    beg.extend([MAGIKA_PADDING_TOKEN] * (MAGIKA_BEG_SIZE - len(beg)))

    end_data = stripped_end[-MAGIKA_END_SIZE:]
    end = [MAGIKA_PADDING_TOKEN] * (MAGIKA_END_SIZE - len(end_data))
    end.extend(end_data)
    return np.asarray(beg + end, dtype=np.uint16), size


def align_file_sizes(dataset: Path, cache_dir: Path, split: str, n: int) -> tuple[np.ndarray, dict[str, int]]:
    tokens = np.memmap(cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r", shape=(n, TOKEN_LENGTH))
    full_rows: dict[bytes, list[int]] = {}
    prefix_rows: dict[bytes, list[int]] = {}
    for row in range(n):
        full_rows.setdefault(token_hash(tokens[row]), []).append(row)
        prefix_rows.setdefault(token_hash(tokens[row, :MAGIKA_BEG_SIZE]), []).append(row)

    raw_sizes_by_full: dict[bytes, list[int]] = {}
    prefix_size: dict[bytes, int] = {}
    seen_files = 0
    skipped_files = 0
    unmatched_files = 0
    valid_windows = 0
    for path in source_paths(dataset / split):
        seen_files += 1
        window = read_window_tokens(path)
        if window is None:
            skipped_files += 1
            continue
        row_tokens, size = window
        valid_windows += 1
        full_key = token_hash(row_tokens)
        prefix_size.setdefault(token_hash(row_tokens[:MAGIKA_BEG_SIZE]), size)
        if full_key in full_rows:
            raw_sizes_by_full.setdefault(full_key, []).append(size)
        else:
            unmatched_files += 1

    sizes = np.full(n, -1, dtype=np.int64)
    matched_hash_rows = 0
    ambiguous_size_keys = 0
    extra_full_window_matches = 0
    missing_full_window_rows = 0
    for key, rows in full_rows.items():
        raw_sizes = raw_sizes_by_full.get(key, [])
        if not raw_sizes:
            missing_full_window_rows += len(rows)
            continue
        if len(raw_sizes) != len(rows) or len(set(raw_sizes)) > 1:
            ambiguous_size_keys += 1
        if len(raw_sizes) > len(rows):
            extra_full_window_matches += len(raw_sizes) - len(rows)
        for row, size in zip(rows, raw_sizes):
            sizes[row] = size
            matched_hash_rows += 1

    size_collisions = 0
    prefix_fallback_rows = 0
    for row in np.flatnonzero(sizes < 0):
        size = prefix_size.get(token_hash(tokens[row, :MAGIKA_BEG_SIZE]))
        if size is not None:
            sizes[row] = size
            prefix_fallback_rows += 1

    missing_rows = int((sizes < 0).sum())
    if missing_rows:
        raise SystemExit(f"could not align sizes for {missing_rows} cache rows")

    return sizes, {
        "seen_files": seen_files,
        "skipped_files": skipped_files,
        "valid_windows": valid_windows,
        "unmatched_files": unmatched_files,
        "matched_hash_rows": matched_hash_rows,
        "ambiguous_size_keys": ambiguous_size_keys,
        "extra_full_window_matches": extra_full_window_matches,
        "missing_full_window_rows": missing_full_window_rows,
        "prefix_fallback_rows": prefix_fallback_rows,
        "size_collisions": size_collisions,
    }


def load_model(checkpoint: Path, cache_dir: Path, architecture: str, split: str):
    meta = json.loads((cache_dir / f"{split}.json").read_text())
    classes = int(meta["classes"])
    if not architecture_uses_word_units(architecture):
        raise SystemExit("only wordseq architectures are supported")
    cfg = wordseq_config_for_architecture(architecture)
    model = build_word_seq_hashembed_hidden_model(classes, bits=4, **cfg)
    layer_weights, metadata = load_exported_layer_weights(checkpoint)
    loaded = 0
    for layer in model.layers:
        if layer.name not in layer_weights:
            continue
        current = layer.get_weights()
        target = layer_weights[layer.name]
        if len(current) != len(target):
            continue
        if any(cur.shape != tgt.shape for cur, tgt in zip(current, target)):
            continue
        layer.set_weights([tgt.astype(cur.dtype) for cur, tgt in zip(current, target)])
        loaded += 1
    print(f"loaded weights into {loaded} layers", flush=True)
    return model, metadata, meta


def predict(model, units: np.memmap, batch_size: int) -> np.ndarray:
    n = units.shape[0]
    preds = np.empty(n, dtype=np.int64)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = tf.convert_to_tensor(units[start:end], dtype=tf.int32)
        out = model(batch, training=False)
        logits = out[0] if isinstance(out, (list, tuple)) else out
        preds[start:end] = tf.argmax(logits, axis=1).numpy().astype(np.int64)
    return preds


def bucket_index(size: int) -> int:
    for idx, (_, lo, hi) in enumerate(BUCKETS):
        if size >= lo and (hi is None or size <= hi):
            return idx
    raise AssertionError(size)


def build_bucket_matrices(
    labels: np.ndarray,
    preds: np.ndarray,
    sizes: np.ndarray,
    classes: int,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    bucket_ids = np.asarray([bucket_index(int(size)) for size in sizes], dtype=np.int16)
    matrices: list[np.ndarray] = []
    byte_matrices: list[np.ndarray] = []
    for bucket in range(len(BUCKETS)):
        rows = np.flatnonzero(bucket_ids == bucket)
        matrix = np.zeros((classes, classes), dtype=np.int64)
        byte_matrix = np.zeros((classes, classes), dtype=np.int64)
        np.add.at(matrix, (labels[rows], preds[rows]), 1)
        np.add.at(byte_matrix, (labels[rows], preds[rows]), sizes[rows])
        matrices.append(matrix)
        byte_matrices.append(byte_matrix)
    return matrices, byte_matrices, bucket_ids


def top_confusions(matrix: np.ndarray, label_names: list[str], limit: int = 6) -> str:
    pairs: list[tuple[int, str, str]] = []
    for actual in range(matrix.shape[0]):
        for predicted in range(matrix.shape[1]):
            if actual == predicted:
                continue
            count = int(matrix[actual, predicted])
            if count:
                pairs.append((count, label_names[actual], label_names[predicted]))
    pairs.sort(reverse=True)
    return ", ".join(f"{actual}->{predicted} {count}" for count, actual, predicted in pairs[:limit])


def write_csv(path: Path, split: str, matrices: list[np.ndarray], byte_matrices: list[np.ndarray], labels: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "bucket", "actual", "predicted", "count", "bytes"])
        for bucket, matrix in zip(BUCKETS, matrices):
            bucket_name = bucket[0]
            for actual, predicted in zip(*np.nonzero(matrix)):
                writer.writerow([
                    split,
                    bucket_name,
                    labels[int(actual)],
                    labels[int(predicted)],
                    int(matrix[actual, predicted]),
                    int(byte_matrices[BUCKETS.index(bucket)][actual, predicted]),
                ])


def write_markdown(
    path: Path,
    checkpoint: Path,
    cache_dir: Path,
    dataset: Path,
    split: str,
    matrices: list[np.ndarray],
    byte_matrices: list[np.ndarray],
    labels: list[str],
    alignment: dict[str, int],
    teacher_parity: float,
    fs_accuracy: float,
) -> None:
    lines = [
        "# Actual Dataset Confusion By File Size",
        "",
        f"Source cache: `{cache_dir}`",
        f"Raw corpus split: `{dataset / split}`",
        f"Checkpoint: `{checkpoint}`",
        "",
        f"Cached test rows: {sum(int(m.sum()) for m in matrices):,}. "
        f"Raw files scanned for alignment: {alignment['seen_files']:,}. "
        f"Full-window matched rows: {alignment['matched_hash_rows']:,}. "
        f"Prefix fallback rows: {alignment['prefix_fallback_rows']:,}.",
        f"Overall test fs accuracy: {fs_accuracy * 100:.3f}%. Teacher parity: {teacher_parity * 100:.3f}%.",
        "",
        "Labels are the 67 Magika source labels from `test.json`. Actual labels use "
        "`test.fs_labels.mmap`, which is filesystem-extension labels where mapped "
        "and teacher fallback where unmapped.",
        "",
        "## Summary",
        "",
        "| Bucket | Files | Bytes | Accuracy | Byte Accuracy | Top confusions |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for bucket, matrix, byte_matrix in zip(BUCKETS, matrices, byte_matrices):
        total = int(matrix.sum())
        total_bytes = int(byte_matrix.sum())
        correct = int(np.trace(matrix))
        correct_bytes = int(np.trace(byte_matrix))
        acc = correct / total if total else 0.0
        byte_acc = correct_bytes / total_bytes if total_bytes else 0.0
        lines.append(
            f"| {bucket[0]} | {total:,} | {format_bytes(total_bytes)} | "
            f"{acc * 100:.2f}% | {byte_acc * 100:.2f}% | "
            f"{top_confusions(matrix, labels)} |"
        )

    lines.extend([
        "",
        "## Matrices",
        "",
        "Each matrix is count-based. Columns are the top predicted labels in that "
        "bucket; less common predictions are grouped as `other`. The complete "
        "ungrouped cells are in `actual_dataset_confusion_by_size.csv`.",
    ])
    for bucket, matrix in zip(BUCKETS, matrices):
        col_totals = matrix.sum(axis=0)
        top_cols = [int(i) for i in np.argsort(-col_totals)[:14] if col_totals[i] > 0]
        other_cols = [i for i in range(matrix.shape[1]) if i not in top_cols]
        lines.extend(["", f"### {bucket[0]}", ""])
        header = ["actual \\ predicted", *[labels[i] for i in top_cols], "other"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|---" + "|---:" * (len(header) - 1) + "|")
        for actual in range(matrix.shape[0]):
            row_total = int(matrix[actual].sum())
            if row_total == 0:
                continue
            values = [str(int(matrix[actual, col])) for col in top_cols]
            values.append(str(int(matrix[actual, other_cols].sum())))
            lines.append("| " + " | ".join([labels[actual], *values]) + " |")

    path.write_text("\n".join(lines) + "\n")


def format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB"]
    number = float(value)
    for unit in units:
        if number < 1024 or unit == units[-1]:
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024
    raise AssertionError


def render_png(
    path: Path,
    matrices: list[np.ndarray],
    labels: list[str],
    fs_accuracy: float,
) -> None:
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(4, 2, figsize=(24, 30), dpi=160)
    axes_flat = axes.ravel()
    image = None
    for ax, bucket, matrix in zip(axes_flat[: len(BUCKETS)], BUCKETS, matrices):
        row_sums = matrix.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(
                matrix,
                row_sums,
                out=np.zeros_like(matrix, dtype=np.float64),
                where=row_sums != 0,
            )
        masked = np.ma.masked_where(normalized <= 0, normalized)
        image = ax.imshow(masked, cmap=cmap, norm=LogNorm(vmin=0.001, vmax=1.0), interpolation="nearest")
        total = int(matrix.sum())
        acc = float(np.trace(matrix) / total) if total else 0.0
        ax.set_title(f"{bucket[0]}  |  n={total:,}  |  acc={acc * 100:.2f}%", fontsize=13, weight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="#eef2f7", linewidth=0.25)
        ax.tick_params(length=0)
    fig.suptitle("Betlang wordseq confusion matrices by file size", fontsize=24, weight="bold", y=0.995)
    fig.text(
        0.01,
        0.972,
        "Actual labels are rows, predicted labels are columns. Cells are row-normalized shares "
        f"for the held-out bigorig test split. Overall file accuracy: {fs_accuracy * 100:.2f}%.",
        fontsize=11,
        color="#334155",
    )
    fig.text(
        0.01,
        0.956,
        "Off-diagonal cells show where each actual language is confused within that size bucket. "
        "Full raw counts and byte totals are in actual_dataset_confusion_by_size.csv.",
        fontsize=11,
        color="#64748b",
    )
    if image is not None:
        cbar = fig.colorbar(image, cax=axes_flat[-1])
        cbar.set_label("Share of actual label in bucket", fontsize=10)
        cbar.set_ticks([0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0])
        cbar.set_ticklabels(["0.1%", "1%", "5%", "10%", "25%", "50%", "75%", "100%"])
    fig.subplots_adjust(left=0.055, right=0.94, top=0.93, bottom=0.035, hspace=0.34, wspace=0.16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--unit-tokenizer", type=int)
    parser.add_argument("--csv-output", type=Path, default=Path("actual_dataset_confusion_by_size.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("actual_dataset_confusion_by_size.md"))
    parser.add_argument("--png-output", type=Path, default=Path("assets/confusion-by-size.png"))
    args = parser.parse_args()

    split_meta = json.loads((args.cache_dir / f"{args.split}.json").read_text())
    n = int(split_meta["count"])
    labels = [str(label) for label in split_meta["labels"]]
    classes = int(split_meta["classes"])
    print(f"split={args.split} n={n} classes={classes}", flush=True)

    model, metadata, _ = load_model(args.checkpoint, args.cache_dir, args.architecture, args.split)
    tokenizer = args.unit_tokenizer if args.unit_tokenizer is not None else int(metadata.get("tokenizer_version") or 2)
    print(f"checkpoint tokenizer_version={metadata.get('tokenizer_version', 'legacy-v2')} using units_v{tokenizer}", flush=True)
    units = np.memmap(args.cache_dir / f"{args.split}.units_v{tokenizer}.mmap", dtype=np.int32, mode="r", shape=(n, TOKEN_LENGTH))
    preds = predict(model, units, args.batch_size)

    teacher_labels = np.asarray(np.memmap(args.cache_dir / f"{args.split}.labels.mmap", dtype=np.int64, mode="r", shape=(n,)))
    fs_labels_path = args.cache_dir / f"{args.split}.fs_labels.mmap"
    target_labels = (
        np.asarray(np.memmap(fs_labels_path, dtype=np.int64, mode="r", shape=(n,)))
        if fs_labels_path.exists()
        else teacher_labels
    )
    teacher_parity = float((preds == teacher_labels).mean())
    fs_accuracy = float((preds == target_labels).mean())
    print(f"{args.split}_teacher_parity={teacher_parity:.6f}", flush=True)
    print(f"{args.split}_fs_accuracy={fs_accuracy:.6f}", flush=True)

    sizes, alignment = align_file_sizes(args.dataset, args.cache_dir, args.split, n)
    print(f"size_alignment={json.dumps(alignment, sort_keys=True)}", flush=True)

    matrices, byte_matrices, _ = build_bucket_matrices(target_labels, preds, sizes, classes)
    write_csv(args.csv_output, args.split, matrices, byte_matrices, labels)
    write_markdown(
        args.markdown_output,
        args.checkpoint,
        args.cache_dir,
        args.dataset,
        args.split,
        matrices,
        byte_matrices,
        labels,
        alignment,
        teacher_parity,
        fs_accuracy,
    )
    render_png(args.png_output, matrices, labels, fs_accuracy)
    print(f"csv_output={args.csv_output}", flush=True)
    print(f"markdown_output={args.markdown_output}", flush=True)
    print(f"png_output={args.png_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
