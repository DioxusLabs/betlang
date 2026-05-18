#!/usr/bin/env python3
"""Train a sub-100KB Magika source-code student model.

This script has two distinct phases:

1. Stream the dataset once, run Magika's ONNX teacher in batches, and write
   dense student features plus teacher targets into memory-mapped cache files.
2. Train the compact student from those memmaps with bounded RAM.

The output format is intentionally Rust-friendly: 4-bit packed linear weights
with per-class f32 codebooks and f32 bias.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort


CODEBOOK_SIZE = 16
DEFAULT_HASH_BUCKETS = 2048
MAGIKA_BEG_SIZE = 1024
MAGIKA_END_SIZE = 1024
MAGIKA_BLOCK_SIZE = 4096
MAGIKA_PADDING_TOKEN = 256
MAX_TOKENS = 10_000
MAGIC = b"MSD1\x01\0\0\0"
SPLITS = ("train", "valid", "test")

# Magika label, exported model slug.
SOURCE_LABELS: list[tuple[str, str]] = [
    ("asm", "asm"),
    ("awk", "awk"),
    ("batch", "batch"),
    ("bazel", "bazel"),
    ("c", "c"),
    ("clojure", "clojure"),
    ("cmake", "cmake"),
    ("cobol", "cobol"),
    ("cpp", "cpp"),
    ("cs", "cs"),
    ("csproj", "csproj"),
    ("css", "css"),
    ("dart", "dart"),
    ("diff", "diff"),
    ("dockerfile", "dockerfile"),
    ("elixir", "elixir"),
    ("erb", "erb"),
    ("erlang", "erlang"),
    ("gemfile", "gemfile"),
    ("gemspec", "gemspec"),
    ("go", "go"),
    ("gradle", "gradle"),
    ("groovy", "groovy"),
    ("haskell", "haskell"),
    ("hcl", "hcl"),
    ("html", "html"),
    ("ini", "ini"),
    ("ipynb", "ipynb"),
    ("java", "java"),
    ("javascript", "javascript"),
    ("jinja", "jinja"),
    ("json", "json"),
    ("jsonl", "jsonl"),
    ("julia", "julia"),
    ("kotlin", "kotlin"),
    ("lisp", "lisp"),
    ("lua", "lua"),
    ("markdown", "markdown"),
    ("matlab", "matlab"),
    ("objectivec", "objectivec"),
    ("ocaml", "ocaml"),
    ("perl", "perl"),
    ("php", "php"),
    ("postscript", "postscript"),
    ("powershell", "powershell"),
    ("prolog", "prolog"),
    ("python", "python"),
    ("r", "r"),
    ("ruby", "ruby"),
    ("rust", "rust"),
    ("scala", "scala"),
    ("scss", "scss"),
    ("shell", "shell"),
    ("solidity", "solidity"),
    ("sql", "sql"),
    ("swift", "swift"),
    ("textproto", "textproto"),
    ("toml", "toml"),
    ("typescript", "typescript"),
    ("vba", "vba"),
    ("vcxproj", "vcxproj"),
    ("verilog", "verilog"),
    ("vhdl", "vhdl"),
    ("vue", "vue"),
    ("xml", "xml"),
    ("yaml", "yaml"),
    ("zig", "zig"),
]


@dataclass(frozen=True)
class Teacher:
    session: ort.InferenceSession
    selected_indices: list[int]
    selected_labels: list[str]
    selected_slugs: list[str]


@dataclass(frozen=True)
class SplitCache:
    features: np.memmap
    probabilities: np.memmap
    labels: np.memmap
    count: int
    hash_buckets: int


def source_paths(split_dir: Path, limit: int | None) -> Iterable[Path]:
    count = 0
    for root, dirs, files in os.walk(split_dir):
        dirs.sort()
        for filename in sorted(files):
            if limit is not None and count >= limit:
                return
            count += 1
            yield Path(root) / filename


def count_source_paths(split_dir: Path, limit: int | None) -> int:
    return sum(1 for _ in source_paths(split_dir, limit))


def read_training_windows(path: Path) -> tuple[int, bytes, bytes] | None:
    try:
        size = path.stat().st_size
        with path.open("rb") as file:
            prefix = file.read(max(MAGIKA_BLOCK_SIZE, MAX_TOKENS + 1))
            if size <= MAGIKA_BLOCK_SIZE:
                suffix = prefix
            else:
                file.seek(max(0, size - MAGIKA_BLOCK_SIZE))
                suffix = file.read(MAGIKA_BLOCK_SIZE)
        return size, prefix, suffix
    except OSError:
        return None


def magika_features(size: int, prefix: bytes, suffix: bytes) -> list[int] | None:
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
    return beg + end


def hash_to_bucket(values: Iterable[int], hash_buckets: int) -> int:
    value = 0xCBF29CE484222325
    for byte in values:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value % hash_buckets


def add_feature(buckets: np.ndarray, values: Iterable[int], amount: float = 1.0) -> None:
    buckets[hash_to_bucket(values, len(buckets))] += amount


def normalized_features(buckets: np.ndarray) -> np.ndarray | None:
    norm = max(1.0, float(np.sqrt((buckets * buckets).sum())))
    if norm == 1.0 and not np.any(buckets):
        return None
    buckets /= norm
    return buckets


def byte_bigram_features(prefix: bytes, hash_buckets: int) -> np.ndarray | None:
    if len(prefix) < 2:
        return None

    buckets = np.zeros((hash_buckets,), dtype=np.float32)
    for index in range(min(len(prefix) - 1, MAX_TOKENS)):
        add_feature(buckets, (1, prefix[index], prefix[index + 1]))

    return normalized_features(buckets)


def byte_ngram_features(prefix: bytes, hash_buckets: int) -> np.ndarray | None:
    if len(prefix) < 2:
        return None

    data = prefix[: MAX_TOKENS + 2]
    buckets = np.zeros((hash_buckets,), dtype=np.float32)
    for index, byte in enumerate(data[:MAX_TOKENS]):
        add_feature(buckets, (0, byte), 0.35)
        if index + 1 < len(data):
            add_feature(buckets, (1, byte, data[index + 1]), 1.0)
        if index + 2 < len(data):
            add_feature(buckets, (2, byte, data[index + 1], data[index + 2]), 0.65)

    return normalized_features(buckets)


def is_word_byte(byte: int) -> bool:
    return 48 <= byte <= 57 or 65 <= byte <= 90 or 97 <= byte <= 122 or byte == 95


def token_byte_features(prefix: bytes, hash_buckets: int) -> np.ndarray | None:
    features = byte_ngram_features(prefix, hash_buckets)
    if features is None:
        return None

    data = prefix[:MAX_TOKENS]
    index = 0
    while index < len(data):
        byte = data[index]
        if is_word_byte(byte) or byte in (35, 36):  # # and $ often identify source syntax.
            start = index
            index += 1
            while index < len(data) and is_word_byte(data[index]):
                index += 1
            token = data[start:index].lower()[:32]
            if len(token) >= 2:
                add_feature(features, (3, *token), 1.5)
        else:
            if byte in b"{}[]()<>;:=+-*/%&|!.@":
                add_feature(features, (4, byte), 0.75)
                if index + 1 < len(data) and data[index + 1] in b"{}[]()<>;:=+-*/%&|!.@":
                    add_feature(features, (5, byte, data[index + 1]), 1.0)
            index += 1

    return normalized_features(features)


def student_features(prefix: bytes, hash_buckets: int, feature_mode: str) -> np.ndarray | None:
    if feature_mode == "byte-bigram":
        return byte_bigram_features(prefix, hash_buckets)
    if feature_mode == "byte-ngram":
        return byte_ngram_features(prefix, hash_buckets)
    if feature_mode == "token-byte":
        return token_byte_features(prefix, hash_buckets)
    raise ValueError(f"unknown feature mode: {feature_mode}")


def load_teacher(model_path: Path, config_path: Path) -> Teacher:
    config = json.loads(config_path.read_text())
    target_labels = config["target_labels_space"]
    label_to_index = {label: index for index, label in enumerate(target_labels)}

    selected_indices = []
    selected_labels = []
    selected_slugs = []
    for label, slug in SOURCE_LABELS:
        if label in label_to_index:
            selected_indices.append(label_to_index[label])
            selected_labels.append(label)
            selected_slugs.append(slug)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return Teacher(session, selected_indices, selected_labels, selected_slugs)


def cache_meta_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"{split}.json"


def split_cache_is_current(cache_dir: Path, split: str, classes: int) -> bool:
    meta_path = cache_meta_path(cache_dir, split)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    count = int(meta["count"])
    hash_buckets = int(meta.get("hash_buckets", -1))
    expected = {
        "features": count * hash_buckets * np.dtype(np.float32).itemsize,
        "probabilities": count * classes * np.dtype(np.float32).itemsize,
        "labels": count * np.dtype(np.int64).itemsize,
    }
    return all((cache_dir / f"{split}.{name}.mmap").stat().st_size == size for name, size in expected.items())


def create_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def open_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


def build_split_cache(
    split_dir: Path,
    cache_dir: Path,
    split: str,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
    hash_buckets: int,
    feature_mode: str,
) -> int:
    capacity = count_source_paths(split_dir, limit)
    if capacity == 0:
        raise SystemExit(f"{split} split is empty")

    features_path = cache_dir / f"{split}.features.mmap"
    probabilities_path = cache_dir / f"{split}.probabilities.mmap"
    labels_path = cache_dir / f"{split}.labels.mmap"
    features = create_memmap(features_path, (capacity, hash_buckets), np.float32)
    probabilities = create_memmap(probabilities_path, (capacity, len(teacher.selected_labels)), np.float32)
    labels = create_memmap(labels_path, (capacity,), np.int64)

    pending_features: list[np.ndarray] = []
    pending_teacher_features: list[list[int]] = []
    written = 0
    seen = 0
    skipped = 0

    def flush() -> None:
        nonlocal written
        if not pending_features:
            return
        raw = teacher.session.run(["target_label"], {"bytes": pending_teacher_features})[0].astype(np.float32)
        selected = raw[:, teacher.selected_indices]
        selected_sum = selected.sum(axis=1, keepdims=True)
        keep = selected_sum[:, 0] > 0.0
        selected = selected[keep] / selected_sum[keep]
        kept_features = [feature for feature, should_keep in zip(pending_features, keep) if should_keep]
        if not kept_features:
            pending_features.clear()
            pending_teacher_features.clear()
            return
        end = written + len(kept_features)
        features[written:end] = np.stack(kept_features).astype(np.float32)
        probabilities[written:end] = selected
        labels[written:end] = selected.argmax(axis=1).astype(np.int64)
        written = end
        pending_features.clear()
        pending_teacher_features.clear()

    for path in source_paths(split_dir, limit):
        seen += 1
        windows = read_training_windows(path)
        if windows is None:
            skipped += 1
            continue
        size, prefix, suffix = windows
        student_feature_vector = student_features(prefix[: MAX_TOKENS + 2], hash_buckets, feature_mode)
        teacher_features = magika_features(size, prefix, suffix)
        if student_feature_vector is None or teacher_features is None:
            skipped += 1
            continue
        pending_features.append(student_feature_vector)
        pending_teacher_features.append(teacher_features)
        if len(pending_features) >= teacher_batch_size:
            flush()
        if seen % 10000 == 0:
            print(f"{split}: seen={seen} cached={written} skipped={skipped}", flush=True)

    flush()
    features.flush()
    probabilities.flush()
    labels.flush()

    for path, dtype, cols in (
        (features_path, np.float32, hash_buckets),
        (probabilities_path, np.float32, len(teacher.selected_labels)),
    ):
        with path.open("r+b") as file:
            file.truncate(written * cols * np.dtype(dtype).itemsize)
    with labels_path.open("r+b") as file:
        file.truncate(written * np.dtype(np.int64).itemsize)

    meta = {
        "count": written,
        "seen": seen,
        "skipped": skipped,
        "classes": len(teacher.selected_labels),
        "hash_buckets": hash_buckets,
        "feature_mode": feature_mode,
        "labels": teacher.selected_labels,
        "slugs": teacher.selected_slugs,
    }
    cache_meta_path(cache_dir, split).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"{split}: cached={written} skipped={skipped}", flush=True)
    return written


def ensure_cache(
    dataset: Path,
    cache_dir: Path,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
    rebuild_cache: bool,
    hash_buckets: int,
    feature_mode: str,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        if not rebuild_cache and split_cache_is_current(cache_dir, split, len(teacher.selected_labels)):
            meta = json.loads(cache_meta_path(cache_dir, split).read_text())
            if int(meta.get("hash_buckets", -1)) == hash_buckets and meta.get("feature_mode") == feature_mode:
                counts[split] = int(meta["count"])
                print(f"{split}: using cached {counts[split]} examples", flush=True)
                continue
        counts[split] = build_split_cache(
            dataset / split,
            cache_dir,
            split,
            teacher,
            limit,
            teacher_batch_size,
            hash_buckets,
            feature_mode,
        )
    return counts


def open_split_cache(cache_dir: Path, split: str, classes: int) -> SplitCache:
    meta = json.loads(cache_meta_path(cache_dir, split).read_text())
    count = int(meta["count"])
    hash_buckets = int(meta["hash_buckets"])
    return SplitCache(
        features=open_memmap(cache_dir / f"{split}.features.mmap", (count, hash_buckets), np.float32),
        probabilities=open_memmap(cache_dir / f"{split}.probabilities.mmap", (count, classes), np.float32),
        labels=open_memmap(cache_dir / f"{split}.labels.mmap", (count,), np.int64),
        count=count,
        hash_buckets=hash_buckets,
    )


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted, dtype=np.float32)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def evaluate(split: SplitCache, weight: np.ndarray, bias: np.ndarray, batch_size: int) -> float:
    correct = 0
    for start in range(0, split.count, batch_size):
        end = min(start + batch_size, split.count)
        prediction = split.features[start:end] @ weight + bias
        correct += int((prediction.argmax(axis=1) == split.labels[start:end]).sum())
    return correct / split.count


def train_student(
    train: SplitCache,
    valid: SplitCache,
    classes: int,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    weight = np.zeros((train.hash_buckets, classes), dtype=np.float32)
    bias = np.zeros((classes,), dtype=np.float32)
    weight_m = np.zeros_like(weight)
    weight_v = np.zeros_like(weight)
    bias_m = np.zeros_like(bias)
    bias_v = np.zeros_like(bias)

    best_accuracy = -1.0
    best_weight = None
    best_bias = None
    step = 0
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    weight_decay = 5e-5

    for epoch in range(epochs):
        order = rng.permutation(train.count)
        for start in range(0, train.count, batch_size):
            batch_indices = order[start : start + batch_size]
            x_batch = train.features[batch_indices]
            probabilities_batch = train.probabilities[batch_indices]
            labels_batch = train.labels[batch_indices]

            prediction = x_batch @ weight + bias
            grad_logits = softmax_rows(prediction) - probabilities_batch
            grad_logits += 0.25 * softmax_rows(prediction)
            grad_logits[np.arange(len(labels_batch)), labels_batch] -= 0.25
            grad_logits /= len(labels_batch)

            grad_weight = x_batch.T @ grad_logits + weight_decay * weight
            grad_bias = grad_logits.sum(axis=0)

            step += 1
            weight_m = beta_1 * weight_m + (1.0 - beta_1) * grad_weight
            weight_v = beta_2 * weight_v + (1.0 - beta_2) * (grad_weight * grad_weight)
            bias_m = beta_1 * bias_m + (1.0 - beta_1) * grad_bias
            bias_v = beta_2 * bias_v + (1.0 - beta_2) * (grad_bias * grad_bias)

            weight -= learning_rate * (weight_m / (1.0 - beta_1**step)) / (
                np.sqrt(weight_v / (1.0 - beta_2**step)) + epsilon
            )
            bias -= learning_rate * (bias_m / (1.0 - beta_1**step)) / (
                np.sqrt(bias_v / (1.0 - beta_2**step)) + epsilon
            )

        if epoch % 25 == 0 or epoch == epochs - 1:
            accuracy = evaluate(valid, weight, bias, batch_size)
            print(f"epoch={epoch} valid_teacher_parity={accuracy:.6f}", flush=True)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weight = weight.copy()
                best_bias = bias.copy()

    assert best_weight is not None and best_bias is not None
    return best_weight.astype(np.float32), best_bias.astype(np.float32), best_accuracy


def quantize_columns(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = weight.shape[1]
    codebooks = np.empty((classes, CODEBOOK_SIZE), dtype=np.float32)
    codes = np.empty(weight.shape, dtype=np.uint8)

    for class_index in range(classes):
        column = weight[:, class_index]
        quantiles = np.linspace(100 / (2 * CODEBOOK_SIZE), 100 - 100 / (2 * CODEBOOK_SIZE), CODEBOOK_SIZE)
        centers = np.percentile(column, quantiles).astype(np.float32)
        for _ in range(50):
            assigned = np.argmin((column[:, None] - centers[None, :]) ** 2, axis=1)
            for code in range(CODEBOOK_SIZE):
                if np.any(assigned == code):
                    centers[code] = column[assigned == code].mean()
        centers = np.sort(centers)
        assigned = np.argmin((column[:, None] - centers[None, :]) ** 2, axis=1)
        codebooks[class_index] = centers
        codes[:, class_index] = assigned.astype(np.uint8)

    return codebooks, codes


def dequantize(codebooks: np.ndarray, codes: np.ndarray) -> np.ndarray:
    weight = np.empty(codes.shape, dtype=np.float32)
    for class_index in range(codes.shape[1]):
        weight[:, class_index] = codebooks[class_index, codes[:, class_index]]
    return weight


def pack_codes(codes: np.ndarray) -> bytes:
    flat = codes.reshape(-1)
    packed = bytearray((len(flat) + 1) // 2)
    for index, code in enumerate(flat):
        if index % 2 == 0:
            packed[index // 2] |= int(code) & 0x0F
        else:
            packed[index // 2] |= (int(code) & 0x0F) << 4
    return bytes(packed)


def write_model(
    output: Path,
    codebooks: np.ndarray,
    codes: np.ndarray,
    bias: np.ndarray,
    labels: list[str],
    slugs: list[str],
    hash_buckets: int,
) -> int:
    blob = bytearray(MAGIC)
    blob.extend(len(labels).to_bytes(2, "little"))
    blob.extend(hash_buckets.to_bytes(2, "little"))
    blob.extend(codebooks.astype("<f4").tobytes())
    blob.extend(pack_codes(codes))
    blob.extend(bias.astype("<f4").tobytes())
    label_json = json.dumps({"labels": labels, "slugs": slugs}, separators=(",", ":")).encode()
    blob.extend(len(label_json).to_bytes(4, "little"))
    blob.extend(label_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    return len(blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Directory containing train/valid/test")
    parser.add_argument("--cache-dir", type=Path, required=True, help="Directory for memmapped precomputed features")
    parser.add_argument("--magika-model", type=Path, required=True, help="Path to Magika model.onnx")
    parser.add_argument("--magika-config", type=Path, required=True, help="Path to Magika config.min.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=750)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--hash-buckets", type=int, default=DEFAULT_HASH_BUCKETS)
    parser.add_argument(
        "--feature-mode",
        choices=("byte-bigram", "byte-ngram", "token-byte"),
        default="byte-bigram",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    args = parser.parse_args()

    teacher = load_teacher(args.magika_model, args.magika_config)
    classes = len(teacher.selected_labels)
    expected_size = (
        len(MAGIC)
        + 2
        + 2
        + classes * CODEBOOK_SIZE * 4
        + ((args.hash_buckets * classes + 1) // 2)
        + classes * 4
    )
    print(f"classes={classes}")
    print(f"expected_model_size_without_label_metadata={expected_size}")

    counts = ensure_cache(
        args.dataset,
        args.cache_dir,
        teacher,
        args.limit_per_split,
        args.teacher_batch_size,
        args.rebuild_cache,
        args.hash_buckets,
        args.feature_mode,
    )
    if args.prepare_cache_only:
        print("cache_ready=true")
        return

    train = open_split_cache(args.cache_dir, "train", classes)
    valid = open_split_cache(args.cache_dir, "valid", classes)
    test = open_split_cache(args.cache_dir, "test", classes)
    print(f"train: {counts['train']} examples")
    print(f"valid: {counts['valid']} examples")
    print(f"test: {counts['test']} examples")

    weight, bias, valid_accuracy = train_student(
        train,
        valid,
        classes=classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    codebooks, codes = quantize_columns(weight)
    quantized_weight = dequantize(codebooks, codes)
    test_accuracy = evaluate(test, quantized_weight, bias, args.batch_size)
    size = write_model(args.output, codebooks, codes, bias, teacher.selected_labels, teacher.selected_slugs, train.hash_buckets)

    print(f"valid_teacher_parity={valid_accuracy:.6f}")
    print(f"test_teacher_parity={test_accuracy:.6f}")
    print(f"model_size_bytes={size}")


if __name__ == "__main__":
    main()
