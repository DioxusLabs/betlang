#!/usr/bin/env python3
"""Render file-size-bucket confusion matrices for an exported wordseq model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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


@dataclass(frozen=True)
class BlobRecord:
    oid: str


def token_hash(values: np.ndarray | list[int]) -> bytes:
    arr = np.asarray(values, dtype="<u2")
    return hashlib.blake2b(arr.tobytes(), digest_size=16).digest()


def source_paths(split_dir: Path):
    for root, dirs, files in os.walk(split_dir):
        dirs.sort()
        for filename in sorted(files):
            yield Path(root) / filename


def window_tokens(size: int, prefix: bytes, suffix: bytes) -> np.ndarray | None:
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
    return np.asarray(beg + end, dtype=np.uint16)


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
    tokens = window_tokens(size, prefix, suffix)
    if tokens is None:
        return None
    return tokens, size


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


def read_batch_object(stdout, oid: str) -> bytes | None:
    header = stdout.readline()
    if not header:
        return None
    try:
        decoded = header.decode("utf-8", "replace").rstrip("\n")
        parts = decoded.split()
        if len(parts) < 2 or parts[0] != oid:
            return None
        if parts[1] == "missing":
            return None
        if len(parts) < 3 or parts[1] != "blob":
            return None
        size = int(parts[2])
        content = stdout.read(size)
        stdout.read(1)
        return content
    except Exception:
        return None


def load_manifest_records(manifest: Path, split: str) -> dict[str, list[BlobRecord]]:
    records: dict[str, list[BlobRecord]] = {}
    with manifest.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("usage") != split:
                continue
            repo = str(row["repository_dirname"])
            records.setdefault(repo, []).append(BlobRecord(oid=str(row["oid"])))
    return records


def process_blob_repo(repo: str, records: list[BlobRecord], repositories_dir: Path) -> tuple[list[tuple[bytes, int]], dict[str, int]]:
    repo_path = repositories_dir / repo
    stats = {"seen": 0, "missing": 0, "skipped": 0, "kept": 0}
    output: list[tuple[bytes, int]] = []
    try:
        proc = subprocess.Popen(
            ["timeout", "900", "git", "cat-file", "--batch"],
            cwd=repo_path,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        stats["missing"] += len(records)
        return output, stats

    assert proc.stdin is not None
    assert proc.stdout is not None
    try:
        try:
            proc.stdin.write(("".join(record.oid + "\n" for record in records)).encode("ascii"))
            proc.stdin.close()
        except BrokenPipeError:
            stats["missing"] += len(records)
            return output, stats
        for record in records:
            stats["seen"] += 1
            content = read_batch_object(proc.stdout, record.oid)
            if content is None:
                stats["missing"] += 1
                continue
            size = len(content)
            prefix = content[:MAGIKA_BLOCK_SIZE]
            suffix = content if size <= MAGIKA_BLOCK_SIZE else content[-MAGIKA_BLOCK_SIZE:]
            tokens = window_tokens(size, prefix, suffix)
            if tokens is None:
                stats["skipped"] += 1
                continue
            output.append((token_hash(tokens), size))
            stats["kept"] += 1
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

    return output, stats


def align_git_blob_sizes(
    manifest: Path,
    repositories_dir: Path,
    cache_dir: Path,
    split: str,
    n: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, int]]:
    sizes_cache = cache_dir / f"{split}.sizes.mmap"
    if sizes_cache.exists() and sizes_cache.stat().st_size == n * np.dtype(np.int64).itemsize:
        sizes = np.asarray(np.memmap(sizes_cache, dtype=np.int64, mode="r", shape=(n,))).copy()
        if np.any(sizes < 0):
            raise SystemExit(f"{sizes_cache} contains negative size rows")
        return sizes, {
            "source": 1,
            "loaded_sizes_cache": 1,
            "seen_records": 0,
            "missing_records": 0,
            "skipped_records": 0,
            "kept_records": 0,
            "matched_hash_rows": n,
            "ambiguous_size_keys": 0,
            "extra_full_window_matches": 0,
            "missing_full_window_rows": 0,
            "prefix_fallback_rows": 0,
            "size_collisions": 0,
        }

    tokens = np.memmap(cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r", shape=(n, TOKEN_LENGTH))
    full_rows: dict[bytes, list[int]] = {}
    for row in range(n):
        full_rows.setdefault(token_hash(tokens[row]), []).append(row)

    records = load_manifest_records(manifest, split)
    raw_sizes_by_full: dict[bytes, list[int]] = {}
    total_stats = {"seen": 0, "missing": 0, "skipped": 0, "kept": 0}
    worker_count = max(1, workers)
    completed = 0
    print(
        f"aligning sizes from git manifest repos={len(records)} workers={worker_count}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(process_blob_repo, repo, repo_records, repositories_dir)
            for repo, repo_records in records.items()
        ]
        for future in as_completed(futures):
            completed += 1
            pairs, stats = future.result()
            for key, value in stats.items():
                total_stats[key] += value
            for key, size in pairs:
                if key in full_rows:
                    raw_sizes_by_full.setdefault(key, []).append(size)
            if completed % 1000 == 0:
                print(
                    "size_alignment_progress="
                    f"{completed}/{len(futures)} stats={json.dumps(total_stats, sort_keys=True)}",
                    flush=True,
                )

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

    missing_rows = int((sizes < 0).sum())
    if missing_rows:
        raise SystemExit(f"could not align sizes for {missing_rows} cache rows from git manifest")

    sizes_out = np.memmap(sizes_cache, dtype=np.int64, mode="w+", shape=(n,))
    sizes_out[:] = sizes
    sizes_out.flush()
    del sizes_out

    return sizes, {
        "source": 2,
        "loaded_sizes_cache": 0,
        "seen_records": total_stats["seen"],
        "missing_records": total_stats["missing"],
        "skipped_records": total_stats["skipped"],
        "kept_records": total_stats["kept"],
        "matched_hash_rows": matched_hash_rows,
        "ambiguous_size_keys": ambiguous_size_keys,
        "extra_full_window_matches": extra_full_window_matches,
        "missing_full_window_rows": missing_full_window_rows,
        "prefix_fallback_rows": 0,
        "size_collisions": 0,
    }


def load_model(checkpoint: Path, architecture: str):
    layer_weights, metadata = load_exported_layer_weights(checkpoint)
    labels = [str(label) for label in metadata.get("labels", [])]
    classes = len(labels)
    if classes == 0:
        raise SystemExit(f"{checkpoint} metadata does not contain exported labels")
    if not architecture_uses_word_units(architecture):
        raise SystemExit("only wordseq architectures are supported")
    cfg = wordseq_config_for_architecture(architecture)
    model = build_word_seq_hashembed_hidden_model(classes, bits=4, **cfg)
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
    return model, metadata


def label_remap(cache_labels: list[str], model_labels: list[str]) -> np.ndarray:
    cache_index = {label: index for index, label in enumerate(cache_labels)}
    missing = [label for label in model_labels if label not in cache_index]
    if missing:
        raise SystemExit(f"model labels are missing from cache metadata: {', '.join(missing)}")

    old_to_new = np.full(len(cache_labels), -1, dtype=np.int64)
    for new_index, label in enumerate(model_labels):
        old_to_new[cache_index[label]] = new_index
    return old_to_new


def remap_labels(
    labels: np.ndarray,
    old_to_new: np.ndarray,
    cache_labels: list[str],
    name: str,
    *,
    allow_missing: bool = False,
) -> np.ndarray:
    if labels.size and (labels.min() < 0 or labels.max() >= old_to_new.shape[0]):
        raise SystemExit(f"{name} contains label ids outside cache metadata bounds")
    mapped = old_to_new[labels]
    if np.any(mapped < 0) and not allow_missing:
        missing_ids = sorted(set(int(value) for value in labels[mapped < 0]))
        missing_names = [cache_labels[index] for index in missing_ids]
        raise SystemExit(f"{name} contains labels absent from model head: {', '.join(missing_names)}")
    return mapped.astype(np.int64, copy=False)


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
        writer = csv.writer(file, lineterminator="\n")
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
    manifest: Path | None,
    repositories_dir: Path | None,
    split: str,
    matrices: list[np.ndarray],
    byte_matrices: list[np.ndarray],
    labels: list[str],
    alignment: dict[str, int],
    teacher_parity: float,
    teacher_parity_rows: int,
    fs_accuracy: float,
    dropped_labels: list[str],
) -> None:
    teacher_text = "n/a"
    if np.isfinite(teacher_parity):
        teacher_text = f"{teacher_parity * 100:.3f}% over {teacher_parity_rows:,} active-teacher rows"
    dropped_text = "none"
    if dropped_labels:
        dropped_text = ", ".join(f"`{label}`" for label in dropped_labels)
    if "seen_files" in alignment:
        alignment_text = (
            f"Cached test rows: {sum(int(m.sum()) for m in matrices):,}. "
            f"Raw files scanned for alignment: {alignment['seen_files']:,}. "
            f"Full-window matched rows: {alignment['matched_hash_rows']:,}. "
            f"Prefix fallback rows: {alignment['prefix_fallback_rows']:,}."
        )
    elif alignment.get("loaded_sizes_cache"):
        alignment_text = (
            f"Cached test rows: {sum(int(m.sum()) for m in matrices):,}. "
            "File sizes were loaded from the cached size mmap."
        )
    else:
        alignment_text = (
            f"Cached test rows: {sum(int(m.sum()) for m in matrices):,}. "
            f"Manifest blob records scanned for alignment: {alignment['seen_records']:,}. "
            f"Full-window matched rows: {alignment['matched_hash_rows']:,}. "
            f"Skipped blob records: {alignment['skipped_records']:,}."
        )
    source_lines = [
        f"Source cache: `{cache_dir}`",
        f"Checkpoint: `{checkpoint}`",
    ]
    if manifest is not None:
        source_lines.extend([
            f"Alignment manifest: `{manifest}`",
            f"Repositories: `{repositories_dir}`",
        ])
    else:
        source_lines.append(f"Raw corpus split: `{dataset / split}`")
    lines = [
        "# Actual Dataset Confusion By File Size",
        "",
        *source_lines,
        "",
        alignment_text,
        f"Overall test fs accuracy: {fs_accuracy * 100:.3f}%. Teacher parity: {teacher_text}.",
        "",
        f"Labels are the {len(labels)} exported model labels. Cache labels absent "
        f"from this model head are not evaluated or predicted: {dropped_text}. "
        "Actual labels use `test.fs_labels.mmap`, which is filesystem-extension "
        "labels where mapped and teacher fallback where unmapped.",
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


def plot_confusion_panel(ax, title: str, matrix: np.ndarray, labels: list[str], cmap) -> object:
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
    ax.set_title(f"{title}  |  n={total:,}  |  acc={acc * 100:.2f}%", fontsize=13, weight="bold")
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
    return image


COLORBAR_TICKS = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0]
COLORBAR_TICK_LABELS = ["0.1%", "1%", "5%", "10%", "25%", "50%", "75%", "100%"]


def configure_colorbar(fig, image, axes=None, *, cax=None, label: str = "Share of actual label") -> None:
    cbar = fig.colorbar(image, ax=axes, cax=cax)
    cbar.set_label(label, fontsize=10)
    cbar.set_ticks(COLORBAR_TICKS)
    cbar.set_ticklabels(COLORBAR_TICK_LABELS)


def render_size_png(path: Path, matrices: list[np.ndarray], labels: list[str], fs_accuracy: float) -> None:
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, axes = plt.subplots(4, 2, figsize=(24, 30), dpi=160)
    axes_flat = axes.ravel()
    image = None

    for ax, bucket, matrix in zip(axes_flat[: len(BUCKETS)], BUCKETS, matrices):
        image = plot_confusion_panel(ax, bucket[0], matrix, labels, cmap)

    fig.suptitle("Betlang wordseq confusion matrices by file size", fontsize=24, weight="bold", y=0.995)
    fig.text(
        0.01,
        0.972,
        "Actual labels are rows, predicted labels are columns. Cells are row-normalized shares "
        f"for each held-out filesystem-label size bucket. Overall file accuracy: {fs_accuracy * 100:.2f}%.",
        fontsize=11,
        color="#334155",
    )
    fig.text(
        0.01,
        0.956,
        "Off-diagonal cells show where each actual language is confused within that size bucket. "
        "The overall matrix is split out in assets/confusion-overall.png.",
        fontsize=11,
        color="#64748b",
    )
    fig.subplots_adjust(left=0.055, right=0.94, top=0.93, bottom=0.035, hspace=0.34, wspace=0.16)
    if image is not None:
        configure_colorbar(
            fig,
            image,
            cax=axes_flat[len(BUCKETS)],
            label="Share of actual label in bucket",
        )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_overall_png(path: Path, matrix: np.ndarray, labels: list[str], fs_accuracy: float) -> None:
    cmap = plt.get_cmap("magma_r").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(18, 17), dpi=180)
    image = plot_confusion_panel(ax, "Overall", matrix, labels, cmap)
    ax.set_title(
        f"Overall  |  n={int(matrix.sum()):,}  |  acc={fs_accuracy * 100:.2f}%",
        fontsize=16,
        weight="bold",
    )
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    fig.suptitle("Betlang wordseq overall confusion matrix", fontsize=24, weight="bold", y=0.995)
    fig.text(
        0.01,
        0.962,
        "Actual labels are rows, predicted labels are columns. Cells are row-normalized shares "
        "for the held-out filesystem-label test split.",
        fontsize=11,
        color="#334155",
    )
    fig.subplots_adjust(left=0.11, right=0.84, top=0.925, bottom=0.12)
    scale_ax = fig.add_axes([0.875, 0.18, 0.05, 0.65])
    configure_colorbar(fig, image, cax=scale_ax)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--repositories-dir", type=Path)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--size-workers", type=int, default=32)
    parser.add_argument("--unit-tokenizer", type=int)
    parser.add_argument("--csv-output", type=Path, default=Path("actual_dataset_confusion_by_size.csv"))
    parser.add_argument("--markdown-output", type=Path, default=Path("actual_dataset_confusion_by_size.md"))
    parser.add_argument("--png-output", type=Path, default=Path("assets/confusion-by-size.png"))
    parser.add_argument("--overall-png-output", type=Path, default=Path("assets/confusion-overall.png"))
    args = parser.parse_args()

    split_meta = json.loads((args.cache_dir / f"{args.split}.json").read_text())
    n = int(split_meta["count"])
    cache_labels = [str(label) for label in split_meta["labels"]]
    cache_classes = int(split_meta["classes"])

    model, metadata = load_model(args.checkpoint, args.architecture)
    labels = [str(label) for label in metadata["labels"]]
    classes = len(labels)
    old_to_new = label_remap(cache_labels, labels)
    label_set = set(labels)
    dropped_labels = [label for label in cache_labels if label not in label_set]
    print(
        f"split={args.split} n={n} cache_classes={cache_classes} model_classes={classes}",
        flush=True,
    )
    if dropped_labels:
        print(f"dropped_cache_labels={','.join(dropped_labels)}", flush=True)
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
    teacher_labels = remap_labels(teacher_labels, old_to_new, cache_labels, "teacher labels", allow_missing=True)
    target_labels = remap_labels(target_labels, old_to_new, cache_labels, "target labels")
    active_teacher_rows = teacher_labels >= 0
    teacher_parity_rows = int(active_teacher_rows.sum())
    teacher_parity = (
        float((preds[active_teacher_rows] == teacher_labels[active_teacher_rows]).mean())
        if teacher_parity_rows
        else float("nan")
    )
    fs_accuracy = float((preds == target_labels).mean())
    print(f"{args.split}_teacher_parity={teacher_parity:.6f}", flush=True)
    print(f"{args.split}_fs_accuracy={fs_accuracy:.6f}", flush=True)

    if args.manifest is not None:
        if args.repositories_dir is None:
            raise SystemExit("--repositories-dir is required with --manifest")
        sizes, alignment = align_git_blob_sizes(
            args.manifest,
            args.repositories_dir,
            args.cache_dir,
            args.split,
            n,
            args.size_workers,
        )
    else:
        sizes, alignment = align_file_sizes(args.dataset, args.cache_dir, args.split, n)
    print(f"size_alignment={json.dumps(alignment, sort_keys=True)}", flush=True)

    matrices, byte_matrices, _ = build_bucket_matrices(target_labels, preds, sizes, classes)
    write_csv(args.csv_output, args.split, matrices, byte_matrices, labels)
    write_markdown(
        args.markdown_output,
        args.checkpoint,
        args.cache_dir,
        args.dataset,
        args.manifest,
        args.repositories_dir,
        args.split,
        matrices,
        byte_matrices,
        labels,
        alignment,
        teacher_parity,
        teacher_parity_rows,
        fs_accuracy,
        dropped_labels,
    )
    overall_matrix = np.sum(np.stack(matrices, axis=0), axis=0)
    render_size_png(args.png_output, matrices, labels, fs_accuracy)
    render_overall_png(args.overall_png_output, overall_matrix, labels, fs_accuracy)
    print(f"csv_output={args.csv_output}", flush=True)
    print(f"markdown_output={args.markdown_output}", flush=True)
    print(f"png_output={args.png_output}", flush=True)
    print(f"overall_png_output={args.overall_png_output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
