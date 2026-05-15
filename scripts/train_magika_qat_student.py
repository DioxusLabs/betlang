#!/usr/bin/env python3
"""Train the production Betlang wordseq Magika student.

This is intentionally narrow: it only contains the cache, tokenizer, model,
training loop, and MSQ1 export code for the shipped
`wordseq-b1536-k3-m2048-med-3conv-hidden` model.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_magika_source_student import (
    SPLITS,
    Teacher,
    cache_meta_path,
    load_teacher,
    magika_features,
    read_training_windows,
    source_paths,
)


TOKEN_LENGTH = 2048
PADDING_TOKEN = 256
QAT_MAGIC = b"MSQ1\x01\0\0\0"

FINAL_ARCHITECTURE = "wordseq-b1536-k3-m2048-med-3conv-hidden"
TOKENIZER_VERSION = 3

WORDSEQ_BINS = 1536
WORDSEQ_HASH_COUNT = 3
WORDSEQ_MAX_UNITS = 2048
WORDSEQ_EMBED = 28
WORDSEQ_CONV0 = 96
WORDSEQ_CONV1 = 192
WORDSEQ_CONV2 = 192
WORDSEQ_DENSE = 160


@dataclass(frozen=True)
class TokenSplit:
    tokens: np.memmap
    probabilities: np.memmap
    labels: np.memmap
    count: int
    self_probabilities: np.memmap | None = None


QAT_ACTIVE = tf.Variable(True, trainable=False, name="qat_active", dtype=tf.bool)


def _quantize_for_bits(source: tf.Tensor, bits: int) -> tf.Tensor:
    if bits == 2:
        abs_source = tf.abs(source)
        nonzero_abs = tf.boolean_mask(abs_source, abs_source > 0.0)
        scale = tf.cond(
            tf.size(nonzero_abs) > 0,
            lambda: tf.reduce_mean(nonzero_abs),
            lambda: tf.constant(1e-6, dtype=source.dtype),
        )
        scale = tf.maximum(scale, tf.constant(1e-6, dtype=source.dtype))
        threshold = 0.7 * scale
        ternary = tf.where(
            source > threshold,
            scale,
            tf.where(source < -threshold, -scale, tf.zeros_like(source)),
        )
        return source + tf.stop_gradient(ternary - source)
    if bits == 4:
        max_abs = tf.maximum(
            tf.reduce_max(tf.abs(source)), tf.constant(1e-6, dtype=source.dtype)
        )
        scale = max_abs / tf.constant(7.0, dtype=source.dtype)
        quantized = tf.clip_by_value(tf.round(source / scale), -7.0, 7.0) * scale
        return source + tf.stop_gradient(quantized - source)
    raise ValueError(f"unsupported weight bits: {bits}")


def fake_quant_weight(weight: tf.Tensor, bits: int) -> tf.Tensor:
    source = tf.cast(weight, tf.float32)
    quantized = tf.cond(
        QAT_ACTIVE,
        lambda: _quantize_for_bits(source, bits),
        lambda: source,
    )
    return tf.cast(quantized, weight.dtype)


class QEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size: int, dims: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dims = dims
        self.bits = bits

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="embedding",
            shape=(self.vocab_size, self.dims),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.05),
            trainable=True,
        )

    def call(self, inputs, training=False):
        return tf.gather(fake_quant_weight(self.embedding, self.bits), inputs)


class QConv1D(tf.keras.layers.Layer):
    def __init__(self, filters: int, kernel_size: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.bits = bits

    def build(self, input_shape):
        in_channels = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.kernel_size, in_channels, self.filters),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.filters,), initializer="zeros", trainable=True
        )

    def call(self, inputs, training=False):
        output = tf.nn.conv1d(
            inputs,
            fake_quant_weight(self.kernel, self.bits),
            stride=1,
            padding="SAME",
        )
        return output + self.bias


class QDense(tf.keras.layers.Layer):
    def __init__(self, units: int, bits: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.bits = bits

    def build(self, input_shape):
        in_units = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_units, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), initializer="zeros", trainable=True
        )

    def call(self, inputs, training=False):
        return inputs @ fake_quant_weight(self.kernel, self.bits) + self.bias


_WORD_MASK = 0x00FF_FFFF
_PUNCT_FLAG = 0x1000_0000
_INDENT_FLAG = 0x2000_0000
_NUM_FLAG = 0x4000_0000
_BRACKET_FLAG = 0x5000_0000
_V3_BRACKET_BYTES = frozenset((40, 41, 91, 93, 123, 125))  # ( ) [ ] { }


def numpy_word_units_apply_v3(
    tokens_np: np.ndarray, output_length: int = TOKEN_LENGTH
) -> np.ndarray:
    """Production v3 tokenizer: v2 punctuation compression plus case-folding and
    isolated bracket tokens.
    """
    out = np.full((tokens_np.shape[0], output_length), -1, dtype=np.int32)
    prime = 2654435761
    for row in range(tokens_np.shape[0]):
        word: list[int] = []
        number: list[int] = []
        punct: list[int] = []
        out_pos = 0
        tokens = tokens_np[row]
        at_line_start = True
        indent_units = 0

        for col in range(tokens.shape[0]):
            value = int(tokens[col])
            if value >= PADDING_TOKEN:
                break
            if 65 <= value <= 90:
                value += 32
            is_letter = (97 <= value <= 122) or value == 95
            is_digit = 48 <= value <= 57
            is_newline = value == 10
            is_cr = value == 13
            is_space = value == 32 or value == 9
            is_bracket = value in _V3_BRACKET_BYTES

            if not is_letter and word and out_pos < output_length:
                h = 0
                for b in word:
                    h = (h * prime + b) & 0xFFFF_FFFF
                out[row, out_pos] = h & _WORD_MASK
                out_pos += 1
                word.clear()
            if not (is_digit or value == 46) and number and out_pos < output_length:
                h = 0
                for b in number:
                    h = (h * prime + b) & 0xFFFF_FFFF
                out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
                out_pos += 1
                number.clear()
            need_flush_punct = (
                is_letter
                or is_digit
                or is_space
                or is_newline
                or is_cr
                or is_bracket
                or value == 46
            )
            if need_flush_punct and punct and out_pos < output_length:
                h = 0
                for b in punct:
                    h = (h * prime + b) & 0xFFFF_FFFF
                out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
                out_pos += 1
                punct.clear()

            if is_letter:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                word.append(value)
                continue
            if is_digit or value == 46:
                if value == 46 and not number:
                    if at_line_start and indent_units > 0 and out_pos < output_length:
                        out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                        out_pos += 1
                    at_line_start = False
                    indent_units = 0
                    punct.append(value)
                    continue
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                at_line_start = False
                indent_units = 0
                number.append(value)
                continue
            if is_newline:
                if at_line_start and indent_units > 0 and out_pos < output_length:
                    out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                    out_pos += 1
                if out_pos < output_length:
                    out[row, out_pos] = 10 | _PUNCT_FLAG
                    out_pos += 1
                at_line_start = True
                indent_units = 0
                continue
            if is_cr:
                continue
            if at_line_start and is_space:
                indent_units += 1 if value == 32 else 4
                continue
            if at_line_start and indent_units > 0 and out_pos < output_length:
                out[row, out_pos] = min(indent_units, 63) | _INDENT_FLAG
                out_pos += 1
            at_line_start = False
            indent_units = 0
            if is_space:
                if out_pos < output_length:
                    last_was_space = (
                        out_pos > 0 and out[row, out_pos - 1] == (32 | _PUNCT_FLAG)
                    )
                    if not last_was_space:
                        out[row, out_pos] = 32 | _PUNCT_FLAG
                        out_pos += 1
                continue
            if is_bracket:
                if out_pos < output_length:
                    out[row, out_pos] = value | _BRACKET_FLAG
                    out_pos += 1
                continue
            punct.append(value)

        if word and out_pos < output_length:
            h = 0
            for b in word:
                h = (h * prime + b) & 0xFFFF_FFFF
            out[row, out_pos] = h & _WORD_MASK
            out_pos += 1
        if number and out_pos < output_length:
            h = 0
            for b in number:
                h = (h * prime + b) & 0xFFFF_FFFF
            out[row, out_pos] = (h & _WORD_MASK) | _NUM_FLAG
            out_pos += 1
        if punct and out_pos < output_length:
            h = 0
            for b in punct:
                h = (h * prime + b) & 0xFFFF_FFFF
            out[row, out_pos] = (h & _WORD_MASK) | _PUNCT_FLAG
            out_pos += 1
    return out


def hash_unit_indices(
    unit_ids: tf.Tensor, bins: int, hash_count: int, max_units: int
) -> tf.Tensor:
    truncated = unit_ids[:, :max_units]
    safe = tf.where(truncated >= 0, truncated, tf.zeros_like(truncated))
    safe64 = tf.cast(safe, tf.int64)
    primes = (2654435761, 2246822519, 3266489917, 668265263)
    mask32 = tf.constant(0xFFFF_FFFF, dtype=tf.int64)
    parts = []
    for hi in range(hash_count):
        p1 = tf.constant(primes[hi % len(primes)], dtype=tf.int64)
        p2 = tf.constant(primes[(hi + 1) % len(primes)], dtype=tf.int64)
        h = tf.bitwise.bitwise_and(safe64 * p1, mask32)
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 13))
        h = tf.bitwise.bitwise_and(h * p2, mask32)
        parts.append(tf.cast(tf.math.floormod(h, bins), tf.int32))
    return tf.stack(parts, axis=-1)


def build_final_model(classes: int, bits: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
    indices = tf.keras.layers.Lambda(
        lambda value: hash_unit_indices(
            value, WORDSEQ_BINS, WORDSEQ_HASH_COUNT, WORDSEQ_MAX_UNITS
        ),
        name=f"hash_indices_b{WORDSEQ_BINS}_h{WORDSEQ_HASH_COUNT}_m{WORDSEQ_MAX_UNITS}",
        dtype="int32",
    )(inputs)
    valid_mask = tf.keras.layers.Lambda(
        lambda value: tf.cast(value[:, :WORDSEQ_MAX_UNITS] >= 0, tf.float32),
        name=f"unit_mask_m{WORDSEQ_MAX_UNITS}",
    )(inputs)

    embed_per_hash = QEmbedding(
        WORDSEQ_BINS, WORDSEQ_EMBED, 4, name="q_hash_embedding"
    )(indices)
    x = tf.keras.layers.Lambda(
        lambda value: tf.reduce_sum(value, axis=-2), name="hash_embed_sum"
    )(embed_per_hash)
    x = tf.keras.layers.Lambda(
        lambda args: args[0] * tf.cast(tf.expand_dims(args[1], -1), args[0].dtype),
        name="apply_pad_mask",
    )([x, valid_mask])
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    x = QConv1D(WORDSEQ_CONV0, 7, 2, name="q_conv_0")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=4)(x)
    x = QConv1D(WORDSEQ_CONV1, 5, 2, name="q_conv_1")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = QConv1D(WORDSEQ_CONV2, 3, 2, name="q_conv_2")(x)
    x = tf.keras.layers.Activation(tf.nn.gelu)(x)

    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x)
    pooled = tf.keras.layers.Concatenate()([max_pool, avg_pool])
    hidden = QDense(512, bits, name="q_hidden_project")(pooled)
    feat = QDense(WORDSEQ_DENSE, 2, name="q_dense_0")(pooled)
    feat = tf.keras.layers.Activation(tf.nn.gelu)(feat)
    outputs = QDense(classes, 4, name="q_output")(feat)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, hidden])


def count_paths(split_dir: Path, limit: int | None) -> int:
    return sum(1 for _ in source_paths(split_dir, limit))


def cache_is_current(cache_dir: Path, split: str, classes: int) -> bool:
    meta_path = cache_meta_path(cache_dir, split)
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    count = int(meta["count"])
    if int(meta.get("classes", classes)) != classes:
        return False
    if int(meta.get("token_length", TOKEN_LENGTH)) != TOKEN_LENGTH:
        return False
    expected = {
        "tokens": count * TOKEN_LENGTH * np.dtype(np.uint16).itemsize,
        "probabilities": count * classes * np.dtype(np.float32).itemsize,
        "labels": count * np.dtype(np.int64).itemsize,
    }
    for name, size in expected.items():
        path = cache_dir / f"{split}.{name}.mmap"
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


def build_split_cache(
    split_dir: Path,
    cache_dir: Path,
    split: str,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
) -> int:
    capacity = count_paths(split_dir, limit)
    if capacity == 0:
        raise RuntimeError(
            f"Cache rebuild for {split!r} would create empty mmaps because "
            f"{split_dir} contains no files. Pass --dataset <root>/files."
        )

    tokens_path = cache_dir / f"{split}.tokens.mmap"
    probabilities_path = cache_dir / f"{split}.probabilities.mmap"
    labels_path = cache_dir / f"{split}.labels.mmap"
    tokens = np.memmap(
        tokens_path, dtype=np.uint16, mode="w+", shape=(capacity, TOKEN_LENGTH)
    )
    probabilities = np.memmap(
        probabilities_path,
        dtype=np.float32,
        mode="w+",
        shape=(capacity, len(teacher.selected_labels)),
    )
    labels = np.memmap(labels_path, dtype=np.int64, mode="w+", shape=(capacity,))

    pending_tokens: list[list[int]] = []
    written = 0
    seen = 0
    skipped = 0

    def flush() -> None:
        nonlocal written
        if not pending_tokens:
            return
        raw = teacher.session.run(["target_label"], {"bytes": pending_tokens})[0].astype(
            np.float32
        )
        selected = raw[:, teacher.selected_indices]
        selected_sum = selected.sum(axis=1, keepdims=True)
        keep = selected_sum[:, 0] > 0.0
        selected = selected[keep] / selected_sum[keep]
        kept_tokens = [
            window for window, should_keep in zip(pending_tokens, keep) if should_keep
        ]
        if kept_tokens:
            end = written + len(kept_tokens)
            tokens[written:end] = np.asarray(kept_tokens, dtype=np.uint16)
            probabilities[written:end] = selected
            labels[written:end] = selected.argmax(axis=1).astype(np.int64)
            written = end
        pending_tokens.clear()

    for path in source_paths(split_dir, limit):
        seen += 1
        windows = read_training_windows(path)
        if windows is None:
            skipped += 1
            continue
        size, prefix, suffix = windows
        token_window = magika_features(size, prefix, suffix)
        if token_window is None:
            skipped += 1
            continue
        pending_tokens.append(token_window)
        if len(pending_tokens) >= teacher_batch_size:
            flush()
        if seen % 10000 == 0:
            print(f"{split}: seen={seen} cached={written} skipped={skipped}", flush=True)

    flush()
    tokens.flush()
    probabilities.flush()
    labels.flush()
    with tokens_path.open("r+b") as file:
        file.truncate(written * TOKEN_LENGTH * np.dtype(np.uint16).itemsize)
    with probabilities_path.open("r+b") as file:
        file.truncate(written * len(teacher.selected_labels) * np.dtype(np.float32).itemsize)
    with labels_path.open("r+b") as file:
        file.truncate(written * np.dtype(np.int64).itemsize)

    meta = {
        "count": written,
        "seen": seen,
        "skipped": skipped,
        "classes": len(teacher.selected_labels),
        "token_length": TOKEN_LENGTH,
        "labels": teacher.selected_labels,
        "slugs": teacher.selected_slugs,
        "hidden_dim": None,
        "hidden_output": None,
    }
    cache_meta_path(cache_dir, split).write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"{split}: cached={written} skipped={skipped}", flush=True)
    return written


def ensure_cache(
    dataset: Path,
    cache_dir: Path,
    teacher: Teacher,
    limit: int | None,
    teacher_batch_size: int,
    rebuild_cache: bool,
) -> dict[str, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in SPLITS:
        meta_path = cache_meta_path(cache_dir, split)
        if not rebuild_cache and cache_is_current(
            cache_dir, split, len(teacher.selected_labels)
        ):
            meta = json.loads(meta_path.read_text())
            counts[split] = int(meta["count"])
            print(f"{split}: using cached {counts[split]} examples", flush=True)
            continue
        counts[split] = build_split_cache(
            dataset / split, cache_dir, split, teacher, limit, teacher_batch_size
        )
    return counts


def open_split(
    cache_dir: Path,
    split: str,
    classes: int,
    self_probabilities_dir: Path | None = None,
) -> TokenSplit:
    meta = json.loads(cache_meta_path(cache_dir, split).read_text())
    count = int(meta["count"])
    self_probabilities = None
    if self_probabilities_dir is not None:
        self_path = self_probabilities_dir / f"{split}.self_probabilities.mmap"
        if not self_path.exists():
            raise FileNotFoundError(
                f"--self-probabilities was set but {self_path} is missing"
            )
        self_probabilities = np.memmap(
            self_path, dtype=np.float32, mode="r", shape=(count, classes)
        )
    return TokenSplit(
        tokens=np.memmap(
            cache_dir / f"{split}.tokens.mmap",
            dtype=np.uint16,
            mode="r",
            shape=(count, TOKEN_LENGTH),
        ),
        probabilities=np.memmap(
            cache_dir / f"{split}.probabilities.mmap",
            dtype=np.float32,
            mode="r",
            shape=(count, classes),
        ),
        labels=np.memmap(
            cache_dir / f"{split}.labels.mmap",
            dtype=np.int64,
            mode="r",
            shape=(count,),
        ),
        count=count,
        self_probabilities=self_probabilities,
    )


def convert_splits_to_word_units(
    cache_dir: Path,
    train: TokenSplit,
    valid: TokenSplit,
    test: TokenSplit,
) -> tuple[TokenSplit, TokenSplit, TokenSplit]:
    out: list[TokenSplit] = []
    suffix = f"_v{TOKENIZER_VERSION}"
    for name, split in (("train", train), ("valid", valid), ("test", test)):
        units_path = cache_dir / f"{name}.units{suffix}.mmap"
        expected = split.count * TOKEN_LENGTH * np.dtype(np.int32).itemsize
        if units_path.exists() and units_path.stat().st_size == expected:
            print(f"{name}: using cached unit ids (v{TOKENIZER_VERSION})")
        else:
            print(
                f"{name}: building unit-id cache v{TOKENIZER_VERSION} "
                f"({split.count} examples)...",
                flush=True,
            )
            started = time.perf_counter()
            block = 8192
            units_mm = np.memmap(
                units_path, dtype=np.int32, mode="w+", shape=(split.count, TOKEN_LENGTH)
            )
            for start in range(0, split.count, block):
                end = min(start + block, split.count)
                tokens_block = np.asarray(split.tokens[start:end], dtype=np.int32)
                units_mm[start:end] = numpy_word_units_apply_v3(
                    tokens_block, TOKEN_LENGTH
                )
            units_mm.flush()
            del units_mm
            print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)
        units_view = np.memmap(
            units_path, dtype=np.int32, mode="r", shape=(split.count, TOKEN_LENGTH)
        )
        out.append(
            TokenSplit(
                tokens=units_view,
                probabilities=split.probabilities,
                labels=split.labels,
                count=split.count,
                self_probabilities=split.self_probabilities,
            )
        )
    return out[0], out[1], out[2]


def _np_cutmix(
    tokens: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    self_probs: np.ndarray,
    cutmix_prob: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bs, length = tokens.shape
    if bs < 2 or cutmix_prob <= 0.0:
        return tokens, probabilities, labels, self_probs
    shift = int(rng.integers(1, bs))
    partner = (np.arange(bs) + shift) % bs
    k = rng.integers(8, length - 8, size=bs)
    alpha = (k / length).astype(np.float32)
    apply = rng.random(bs) < cutmix_prob
    pos = np.arange(length)[None, :]
    mask = pos < k[:, None]

    mixed_tokens = np.where(mask, tokens, tokens[partner]).astype(tokens.dtype)
    mixed_probs = (
        alpha[:, None] * probabilities + (1.0 - alpha[:, None]) * probabilities[partner]
    )
    mixed_labels = np.where(alpha > 0.5, labels, labels[partner]).astype(labels.dtype)
    apply2 = apply[:, None]
    tokens = np.where(apply2, mixed_tokens, tokens)
    probabilities = np.where(apply2, mixed_probs, probabilities).astype(np.float32)
    labels = np.where(apply, mixed_labels, labels)
    if self_probs.shape[-1] > 0:
        mixed_self = (
            alpha[:, None] * self_probs + (1.0 - alpha[:, None]) * self_probs[partner]
        )
        self_probs = np.where(apply2, mixed_self, self_probs).astype(np.float32)
    return tokens, probabilities, labels, self_probs


_LENGTH_BUCKETS = (128, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048)


def compute_unit_lengths(units: np.ndarray | np.memmap) -> np.ndarray:
    count, _ = units.shape
    out = np.empty(count, dtype=np.int32)
    block = 8192
    for start in range(0, count, block):
        end = min(start + block, count)
        out[start:end] = (np.asarray(units[start:end]) >= 0).sum(axis=1)
    return out


def bucket_for(length: int) -> int:
    for bucket in _LENGTH_BUCKETS:
        if length <= bucket:
            return bucket
    return _LENGTH_BUCKETS[-1]


def batches(
    split: TokenSplit,
    batch_size: int,
    shuffle: bool,
    seed: int,
    cutmix_prob: float = 0.0,
    unit_lengths: np.ndarray | None = None,
    max_batches: int | None = None,
):
    rng = np.random.default_rng(seed)
    yielded = 0
    if unit_lengths is not None and shuffle:
        buckets_for_each = np.asarray(
            [bucket_for(int(length)) for length in unit_lengths], dtype=np.int32
        )
        unique_buckets = sorted(set(int(bucket) for bucket in buckets_for_each))
        all_batches: list[tuple[int, np.ndarray]] = []
        for bucket in unique_buckets:
            indices = np.where(buckets_for_each == bucket)[0]
            indices = indices.copy()
            rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                all_batches.append((bucket, indices[start : start + batch_size]))
        order = rng.permutation(len(all_batches))
        for item in order:
            bucket, indices = all_batches[item]
            tokens = split.tokens[indices].astype(np.int32)[:, :bucket]
            probabilities = split.probabilities[indices]
            labels = split.labels[indices]
            self_probs = (
                split.self_probabilities[indices]
                if split.self_probabilities is not None
                else np.zeros((len(indices), 0), dtype=np.float32)
            )
            if cutmix_prob > 0.0 and len(indices) >= 2:
                tokens, probabilities, labels, self_probs = _np_cutmix(
                    tokens, probabilities, labels, self_probs, cutmix_prob, rng
                )
            yield tokens, probabilities, labels, self_probs
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                return
        return

    order = rng.permutation(split.count) if shuffle else np.arange(split.count)
    for start in range(0, split.count, batch_size):
        indices = order[start : start + batch_size]
        tokens = split.tokens[indices].astype(np.int32)
        probabilities = split.probabilities[indices]
        labels = split.labels[indices]
        self_probs = (
            split.self_probabilities[indices]
            if split.self_probabilities is not None
            else np.zeros((len(indices), 0), dtype=np.float32)
        )
        if shuffle and cutmix_prob > 0.0 and len(indices) >= 2:
            tokens, probabilities, labels, self_probs = _np_cutmix(
                tokens, probabilities, labels, self_probs, cutmix_prob, rng
            )
        yield tokens, probabilities, labels, self_probs
        yielded += 1
        if max_batches is not None and yielded >= max_batches:
            return


def self_probs_or_none(self_probs_batch: tf.Tensor | None) -> tf.Tensor | None:
    if self_probs_batch is None:
        return None
    if self_probs_batch.shape.rank is not None and self_probs_batch.shape[-1] == 0:
        return None
    return self_probs_batch


def model_logits(
    model: tf.keras.Model, token_batch: tf.Tensor, training: bool
) -> tf.Tensor:
    prediction = model(token_batch, training=training)
    if isinstance(prediction, (list, tuple)):
        return prediction[0]
    return prediction


def distillation_loss(
    probability_batch: tf.Tensor,
    label_batch: tf.Tensor,
    logits: tf.Tensor,
    temperature: float,
    hard_loss_weight: float,
    self_probabilities_batch: tf.Tensor | None = None,
    self_loss_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> tf.Tensor:
    if temperature > 1.0:
        softened = tf.pow(
            tf.maximum(probability_batch, tf.constant(1e-8, dtype=probability_batch.dtype)),
            1.0 / temperature,
        )
        soft_targets = softened / tf.reduce_sum(softened, axis=1, keepdims=True)
        soft_logits = logits / temperature
        soft_scale = temperature * temperature
    else:
        soft_targets = probability_batch
        soft_logits = logits
        soft_scale = 1.0
    soft_per_example = (
        tf.keras.losses.categorical_crossentropy(
            soft_targets, soft_logits, from_logits=True
        )
        * soft_scale
    )
    if label_smoothing > 0.0:
        classes = int(logits.shape[-1])
        one_hot = tf.one_hot(label_batch, classes, dtype=tf.float32)
        smoothed = one_hot * (1.0 - label_smoothing) + (label_smoothing / classes)
        hard_per_example = tf.keras.losses.categorical_crossentropy(
            smoothed, logits, from_logits=True
        )
    else:
        hard_per_example = tf.keras.losses.sparse_categorical_crossentropy(
            label_batch, logits, from_logits=True
        )
    soft_loss = tf.reduce_mean(tf.cast(soft_per_example, tf.float32))
    hard_loss = tf.reduce_mean(tf.cast(hard_per_example, tf.float32))
    hard_loss_weight = tf.cast(hard_loss_weight, tf.float32)
    loss = (1.0 - hard_loss_weight) * soft_loss + hard_loss_weight * hard_loss

    if self_loss_weight > 0.0 and self_probabilities_batch is not None:
        if temperature > 1.0:
            self_softened = tf.pow(
                tf.maximum(
                    self_probabilities_batch,
                    tf.constant(1e-8, dtype=self_probabilities_batch.dtype),
                ),
                1.0 / temperature,
            )
            self_targets = self_softened / tf.reduce_sum(
                self_softened, axis=1, keepdims=True
            )
            self_logits = logits / temperature
            self_scale = temperature * temperature
        else:
            self_targets = self_probabilities_batch
            self_logits = logits
            self_scale = 1.0
        self_per_example = (
            tf.keras.losses.categorical_crossentropy(
                self_targets, self_logits, from_logits=True
            )
            * self_scale
        )
        self_loss = tf.reduce_mean(tf.cast(self_per_example, tf.float32))
        loss = loss + tf.cast(self_loss_weight, tf.float32) * self_loss
    return loss


def make_train_step(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    temperature: float,
    hard_loss_weight: float,
    jit_compile: bool,
    self_loss_weight: float = 0.0,
    label_smoothing: float = 0.0,
):
    @tf.function(jit_compile=jit_compile)
    def train_step(token_batch, probability_batch, label_batch, self_probs_batch):
        self_probs = self_probs_or_none(self_probs_batch)
        with tf.GradientTape() as tape:
            logits = model_logits(model, token_batch, training=True)
            loss = distillation_loss(
                probability_batch,
                label_batch,
                logits,
                temperature,
                hard_loss_weight,
                self_probabilities_batch=self_probs,
                self_loss_weight=self_loss_weight,
                label_smoothing=label_smoothing,
            )
        gradients = tape.gradient(loss, model.trainable_variables)
        grads_and_vars = [
            (gradient, variable)
            for gradient, variable in zip(gradients, model.trainable_variables)
            if gradient is not None
        ]
        optimizer.apply_gradients(grads_and_vars)
        return loss, tf.shape(label_batch)[0]

    return train_step


def make_eval_step(
    model: tf.keras.Model,
    temperature: float,
    hard_loss_weight: float,
    jit_compile: bool,
    self_loss_weight: float = 0.0,
):
    @tf.function(jit_compile=jit_compile)
    def eval_step(token_batch, probability_batch, label_batch, self_probs_batch):
        self_probs = self_probs_or_none(self_probs_batch)
        logits = model_logits(model, token_batch, training=False)
        loss = distillation_loss(
            probability_batch,
            label_batch,
            logits,
            temperature,
            hard_loss_weight,
            self_probabilities_batch=self_probs,
            self_loss_weight=self_loss_weight,
        )
        predictions = tf.argmax(logits, axis=1, output_type=label_batch.dtype)
        correct = tf.reduce_sum(tf.cast(tf.equal(predictions, label_batch), tf.int64))
        return loss, correct, tf.shape(label_batch)[0]

    return eval_step


def evaluate_dataset(
    model: tf.keras.Model,
    eval_step,
    split: TokenSplit,
    batch_size: int,
    classes: int,
    collect_confusion: bool = False,
) -> tuple[float, float, float, np.ndarray | None]:
    correct = 0
    loss_sum = 0.0
    seen = 0
    confusion = np.zeros((classes, classes), dtype=np.int64) if collect_confusion else None
    started = time.perf_counter()
    for token_batch, probability_batch, label_batch, self_probs_batch in batches(
        split, batch_size, shuffle=False, seed=0
    ):
        loss, batch_correct, batch_size_tensor = eval_step(
            token_batch, probability_batch, label_batch, self_probs_batch
        )
        batch_n = int(batch_size_tensor.numpy())
        seen += batch_n
        correct += int(batch_correct.numpy())
        loss_sum += float(loss.numpy()) * batch_n
        if confusion is not None:
            logits = model_logits(model, token_batch, training=False)
            predictions = logits.numpy().argmax(axis=1)
            np.add.at(confusion, (label_batch, predictions), 1)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return correct / split.count, loss_sum / split.count, seen / elapsed, confusion


def confusion_summary(
    confusion: np.ndarray, labels: list[str], limit: int
) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    for actual_index, actual in enumerate(labels):
        total = int(confusion[actual_index].sum())
        if total == 0:
            continue
        correct = int(confusion[actual_index, actual_index])
        for predicted_index, predicted in enumerate(labels):
            if predicted_index == actual_index:
                continue
            count = int(confusion[actual_index, predicted_index])
            if count == 0:
                continue
            rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "actual_total": total,
                    "actual_recall": correct / total,
                    "share_of_actual": count / total,
                }
            )
    rows.sort(key=lambda row: int(row["count"]), reverse=True)
    return rows[:limit]


def write_confusion_matrix(
    path: Path, confusion: np.ndarray, labels: list[str], limit: int
) -> None:
    payload = {
        "labels": labels,
        "matrix": confusion.tolist(),
        "top_confusions": confusion_summary(confusion, labels, limit),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def pack_int4(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int8)
    encoded = np.clip(flat + 8, 0, 15).astype(np.uint8)
    packed = bytearray((len(encoded) + 1) // 2)
    for index, value in enumerate(encoded):
        if index % 2 == 0:
            packed[index // 2] |= int(value)
        else:
            packed[index // 2] |= int(value) << 4
    return bytes(packed)


def pack_ternary(values: np.ndarray) -> bytes:
    flat = values.reshape(-1).astype(np.int8)
    encoded = np.where(flat < 0, 0, np.where(flat > 0, 2, 1)).astype(np.uint8)
    packed = bytearray((len(encoded) + 3) // 4)
    for index, value in enumerate(encoded):
        packed[index // 4] |= int(value) << ((index % 4) * 2)
    return bytes(packed)


def quantize_int4(weight: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = max(float(np.max(np.abs(weight))), 1e-6)
    scale = max_abs / 7.0
    quantized = np.clip(np.rint(weight / scale), -7, 7).astype(np.int8)
    return quantized, scale


def quantize_weight(weight: np.ndarray, bits: int) -> tuple[bytes, float, str]:
    if not np.all(np.isfinite(weight)):
        raise ValueError("cannot export non-finite weight tensor")
    if bits == 2:
        abs_weight = np.abs(weight)
        nonzero_abs = abs_weight[abs_weight > 0.0]
        scale = max(float(np.mean(nonzero_abs)) if nonzero_abs.size else 1e-6, 1e-6)
        threshold = 0.7 * scale
        quantized = np.where(
            weight > threshold, 1, np.where(weight < -threshold, -1, 0)
        ).astype(np.int8)
        return pack_ternary(quantized), scale, "ternary"
    if bits == 4:
        quantized, scale = quantize_int4(weight)
        return pack_int4(quantized), scale, "int4"
    raise ValueError(f"unsupported weight bits: {bits}")


def export_model(
    output: Path,
    model: tf.keras.Model,
    labels: list[str],
    slugs: list[str],
    bits: int,
) -> int:
    metadata = {
        "bits": bits,
        "token_length": TOKEN_LENGTH,
        "architecture": FINAL_ARCHITECTURE,
        "tokenizer_version": TOKENIZER_VERSION,
        "labels": labels,
        "slugs": slugs,
        "layers": [],
    }
    blob = bytearray(QAT_MAGIC)
    for layer in model.layers:
        if isinstance(layer, QEmbedding):
            weights = [("embedding", layer.embedding.numpy())]
            biases = []
        elif isinstance(layer, QConv1D):
            weights = [("kernel", layer.kernel.numpy())]
            biases = [("bias", layer.bias.numpy())]
        elif isinstance(layer, QDense):
            if layer.name == "q_hidden_project":
                continue
            weights = [("kernel", layer.kernel.numpy())]
            biases = [("bias", layer.bias.numpy())]
        else:
            continue

        layer_meta = {"name": layer.name, "weights": [], "biases": []}
        layer_bits = getattr(layer, "bits", bits)
        for name, value in weights:
            payload, scale, encoding = quantize_weight(value, layer_bits)
            layer_meta["weights"].append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "scale": scale,
                    "bits": layer_bits,
                    "encoding": encoding,
                    "bytes": len(payload),
                }
            )
            blob.extend(payload)
        for name, value in biases:
            data = value.astype("<f4").tobytes()
            layer_meta["biases"].append(
                {"name": name, "shape": list(value.shape), "bytes": len(data)}
            )
            blob.extend(data)
        metadata["layers"].append(layer_meta)

    metadata_json = json.dumps(metadata, separators=(",", ":")).encode()
    blob.extend(len(metadata_json).to_bytes(4, "little"))
    blob.extend(metadata_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(blob)
    return len(blob)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--magika-model", type=Path, required=True)
    parser.add_argument("--magika-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--weight-bits", type=int, default=4, choices=[4])
    parser.add_argument("--qat-start-epoch", type=int, default=45)
    parser.add_argument("--architecture", default=FINAL_ARCHITECTURE, choices=[FINAL_ARCHITECTURE])
    parser.add_argument("--hard-loss-weight", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--prepare-cache-only", action="store_true")
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--eval-initial", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--confusion-matrix-output", type=Path)
    parser.add_argument("--confusion-matrix-top", type=int, default=25)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--xla", action="store_true")
    parser.add_argument("--self-probabilities", type=Path, default=None)
    parser.add_argument("--self-loss-weight", type=float, default=0.5)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--unit-tokenizer", type=int, default=TOKENIZER_VERSION, choices=[TOKENIZER_VERSION])
    parser.add_argument("--length-buckets", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--global-clipnorm", type=float, default=1.0)
    args = parser.parse_args()

    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    if not 0.0 <= args.hard_loss_weight <= 1.0:
        raise ValueError("--hard-loss-weight must be in [0, 1]")
    if not 0.0 <= args.cutmix_prob <= 1.0:
        raise ValueError("--cutmix-prob must be in [0, 1]")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")

    tf.keras.utils.set_random_seed(args.seed)
    teacher = load_teacher(args.magika_model, args.magika_config)
    classes = len(teacher.selected_labels)
    counts = ensure_cache(
        args.dataset,
        args.cache_dir,
        teacher,
        args.limit_per_split,
        args.teacher_batch_size,
        args.rebuild_cache,
    )
    if args.prepare_cache_only:
        print("cache_ready=true")
        return

    self_probabilities_dir = args.self_probabilities
    if self_probabilities_dir is not None and args.self_loss_weight <= 0.0:
        print(
            "warning: --self-probabilities provided but --self-loss-weight=0.0; "
            "self-distillation term will be skipped",
            flush=True,
        )
    if self_probabilities_dir is None and args.self_loss_weight > 0.0:
        raise SystemExit("--self-loss-weight > 0 requires --self-probabilities <dir>")

    train = open_split(args.cache_dir, "train", classes, self_probabilities_dir)
    valid = open_split(args.cache_dir, "valid", classes, self_probabilities_dir)
    test = open_split(args.cache_dir, "test", classes, self_probabilities_dir)
    if self_probabilities_dir is not None:
        print(
            f"self_probabilities_dir={self_probabilities_dir} "
            f"self_loss_weight={args.self_loss_weight}",
            flush=True,
        )

    train, valid, test = convert_splits_to_word_units(args.cache_dir, train, valid, test)
    train_unit_lengths = None
    if args.length_buckets:
        train_unit_lengths = compute_unit_lengths(train.tokens)
        from collections import Counter

        bucket_counts = Counter(bucket_for(int(length)) for length in train_unit_lengths)
        print(f"length_buckets={dict(sorted(bucket_counts.items()))}", flush=True)

    model = build_final_model(classes, args.weight_bits)
    learning_rate = args.learning_rate
    if args.cosine_decay:
        if not 0.0 <= args.min_learning_rate_ratio <= 1.0:
            raise ValueError("--min-learning-rate-ratio must be in [0, 1]")
        steps_per_epoch = max(1, math.ceil(train.count / args.batch_size))
        if args.max_train_batches is not None:
            steps_per_epoch = min(steps_per_epoch, max(1, args.max_train_batches))
        decay_steps = max(1, args.epochs * steps_per_epoch)
        learning_rate = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=decay_steps,
            alpha=args.min_learning_rate_ratio,
        )
        print(
            f"learning_rate_schedule=cosine initial={args.learning_rate:.8g} "
            f"min_ratio={args.min_learning_rate_ratio:.6f} decay_steps={decay_steps}",
            flush=True,
        )

    optimizer_kwargs = {"learning_rate": learning_rate, "weight_decay": args.weight_decay}
    if args.global_clipnorm > 0.0:
        optimizer_kwargs["global_clipnorm"] = args.global_clipnorm
    optimizer = tf.keras.optimizers.AdamW(**optimizer_kwargs)
    train_step = make_train_step(
        model,
        optimizer,
        args.distill_temperature,
        args.hard_loss_weight,
        args.xla,
        self_loss_weight=args.self_loss_weight,
        label_smoothing=args.label_smoothing,
    )
    eval_step = make_eval_step(
        model,
        args.distill_temperature,
        args.hard_loss_weight,
        args.xla,
        self_loss_weight=args.self_loss_weight,
    )

    best_valid = -1.0
    best_weights = None
    checks_without_improvement = 0
    if args.qat_start_epoch > 0:
        QAT_ACTIVE.assign(False)
        print(f"qat_active=False (will enable at epoch {args.qat_start_epoch})", flush=True)
    else:
        QAT_ACTIVE.assign(True)

    if args.eval_initial:
        initial_export_phase = args.qat_start_epoch == 0 or args.qat_start_epoch >= args.epochs
        valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
            model, eval_step, valid, args.batch_size, classes
        )
        print(
            f"epoch=initial valid_loss={valid_loss:.6f} "
            f"valid_teacher_parity={valid_accuracy:.6f} "
            f"valid_examples_per_sec={valid_examples_per_sec:.1f}",
            flush=True,
        )
        if initial_export_phase:
            best_valid = valid_accuracy
            best_weights = model.get_weights()
            size = export_model(
                args.output,
                model,
                teacher.selected_labels,
                teacher.selected_slugs,
                args.weight_bits,
            )
            print(f"best_checkpoint_model_size_bytes={size}", flush=True)

    for epoch in range(args.epochs):
        if args.qat_start_epoch > 0 and epoch == args.qat_start_epoch:
            QAT_ACTIVE.assign(True)
            print(f"qat_active=True (enabled at epoch {epoch})", flush=True)
        loss_sum = 0.0
        seen = 0
        started = time.perf_counter()
        for token_batch, probability_batch, label_batch, self_probs_batch in batches(
            train,
            args.batch_size,
            shuffle=True,
            seed=args.seed + epoch,
            cutmix_prob=args.cutmix_prob,
            unit_lengths=train_unit_lengths,
            max_batches=args.max_train_batches,
        ):
            loss, batch_size_tensor = train_step(
                token_batch, probability_batch, label_batch, self_probs_batch
            )
            batch_n = int(batch_size_tensor.numpy())
            loss_value = float(loss.numpy())
            if not math.isfinite(loss_value):
                raise FloatingPointError(
                    f"non-finite training loss at epoch={epoch} "
                    f"examples_seen={seen + batch_n}; aborting"
                )
            seen += batch_n
            loss_sum += loss_value * batch_n
        train_seconds = max(time.perf_counter() - started, 1e-9)
        train_loss = loss_sum / max(seen, 1)
        train_examples_per_sec = seen / train_seconds

        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
                model, eval_step, valid, args.batch_size, classes
            )
            print(
                f"epoch={epoch} loss={train_loss:.6f} "
                f"train_examples_per_sec={train_examples_per_sec:.1f} "
                f"valid_loss={valid_loss:.6f} "
                f"valid_teacher_parity={valid_accuracy:.6f} "
                f"valid_examples_per_sec={valid_examples_per_sec:.1f}",
                flush=True,
            )
            qat_phase = args.qat_start_epoch == 0 or epoch >= args.qat_start_epoch
            export_phase = qat_phase or args.qat_start_epoch >= args.epochs
            if export_phase and valid_accuracy > best_valid:
                best_valid = valid_accuracy
                best_weights = model.get_weights()
                checks_without_improvement = 0
                size = export_model(
                    args.output,
                    model,
                    teacher.selected_labels,
                    teacher.selected_slugs,
                    args.weight_bits,
                )
                print(f"best_checkpoint_model_size_bytes={size}", flush=True)
            elif not qat_phase:
                pass
            else:
                checks_without_improvement += 1
                if (
                    args.early_stop_patience > 0
                    and checks_without_improvement >= args.early_stop_patience
                ):
                    print(
                        f"early_stopping=true epoch={epoch} "
                        f"checks_without_improvement={checks_without_improvement}",
                        flush=True,
                    )
                    break

    if best_weights is not None:
        model.set_weights(best_weights)
    valid_accuracy, valid_loss, valid_examples_per_sec, _ = evaluate_dataset(
        model, eval_step, valid, args.batch_size, classes
    )
    test_accuracy, test_loss, test_examples_per_sec, test_confusion = evaluate_dataset(
        model,
        eval_step,
        test,
        args.batch_size,
        classes,
        collect_confusion=args.confusion_matrix_output is not None or args.confusion_matrix_top > 0,
    )
    size = export_model(
        args.output,
        model,
        teacher.selected_labels,
        teacher.selected_slugs,
        args.weight_bits,
    )
    print(f"train: {counts['train']} examples")
    print(f"valid: {counts['valid']} examples")
    print(f"test: {counts['test']} examples")
    print(f"valid_teacher_parity={best_valid:.6f}")
    print(f"valid_loss={valid_loss:.6f}")
    print(f"valid_examples_per_sec={valid_examples_per_sec:.1f}")
    print(f"test_teacher_parity={test_accuracy:.6f}")
    print(f"test_loss={test_loss:.6f}")
    print(f"test_examples_per_sec={test_examples_per_sec:.1f}")
    if test_confusion is not None:
        top_confusions = confusion_summary(
            test_confusion, teacher.selected_labels, args.confusion_matrix_top
        )
        for row in top_confusions:
            print(
                "confusion "
                f"actual={row['actual']} predicted={row['predicted']} "
                f"count={row['count']} actual_total={row['actual_total']} "
                f"actual_recall={row['actual_recall']:.6f} "
                f"share_of_actual={row['share_of_actual']:.6f}"
            )
        if args.confusion_matrix_output:
            write_confusion_matrix(
                args.confusion_matrix_output,
                test_confusion,
                teacher.selected_labels,
                args.confusion_matrix_top,
            )
            print(f"confusion_matrix_output={args.confusion_matrix_output}")
    print(f"model_size_bytes={size}")


if __name__ == "__main__":
    main()
