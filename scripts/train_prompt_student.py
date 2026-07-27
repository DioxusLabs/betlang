#!/usr/bin/env python3
"""Train the natural-language vs LLM-prompt wordseq student.

Trains the production `wordseq-b1024-k3-m2048-tiny-3conv-hidden` architecture
from scratch on the corpus produced by `build_prompt_corpus.py`, with hard
labels only (there is no teacher model for this task), and exports the
weights-only MSQ1 payload consumed by the Rust runtime.

Usage:

    python3 scripts/train_prompt_student.py \
        --dataset /tmp/betlang-prompt-corpus/files \
        --cache-dir /tmp/betlang-prompt-cache \
        --output assets/magika/source-student-q4.bin
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from train_magika_qat_student import (
    FIXED_EXPORT_ARCHITECTURE,
    FIXED_EXPORT_LABELS,
    FIXED_EXPORT_SLUGS,
    FIXED_EXPORT_TOKENIZER_VERSION,
    QAT_ACTIVE,
    TOKEN_LENGTH,
    build_wordseq_by_name,
    export_model,
    numpy_word_units_apply_v3,
)
from train_magika_source_student import SPLITS, magika_features, read_training_windows, source_paths

BATCH_SIZE = 128
MIN_BATCH_UNITS = 64


def build_split_cache(dataset: Path, cache_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    units_path = cache_dir / f"{split}.units_v3.npy"
    labels_path = cache_dir / f"{split}.labels.npy"
    if units_path.exists() and labels_path.exists():
        return np.load(units_path, mmap_mode="r"), np.load(labels_path)

    tokens_rows: list[list[int]] = []
    labels: list[int] = []
    for label_index, label in enumerate(FIXED_EXPORT_LABELS):
        for path in source_paths(dataset / split / label, limit=None):
            window = read_training_windows(path)
            if window is None:
                continue
            features = magika_features(*window)
            if features is None:
                continue
            tokens_rows.append(features)
            labels.append(label_index)

    tokens = np.asarray(tokens_rows, dtype=np.int32)
    units = numpy_word_units_apply_v3(tokens, output_length=TOKEN_LENGTH)
    labels_np = np.asarray(labels, dtype=np.int32)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(units_path, units)
    np.save(labels_path, labels_np)
    return np.load(units_path, mmap_mode="r"), labels_np


def unit_lengths(units: np.ndarray) -> np.ndarray:
    return (np.asarray(units) >= 0).sum(axis=1).astype(np.int64)


def length_bucketed_batches(units: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    order = np.argsort(unit_lengths(units), kind="stable")
    batches = [order[start : start + BATCH_SIZE] for start in range(0, len(order), BATCH_SIZE)]
    rng.shuffle(batches)
    return batches


def batch_inputs(units: np.ndarray, rows: np.ndarray) -> np.ndarray:
    batch = np.asarray(units[np.sort(rows)])
    max_len = int(max(MIN_BATCH_UNITS, unit_lengths(batch).max()))
    max_len = min(TOKEN_LENGTH, ((max_len + 7) // 8) * 8)
    return batch[:, :max_len]


def evaluate(model: tf.keras.Model, units: np.ndarray, labels: np.ndarray) -> float:
    correct = 0
    for start in range(0, len(labels), BATCH_SIZE):
        rows = np.arange(start, min(start + BATCH_SIZE, len(labels)))
        logits = model(batch_inputs(units, rows), training=False)[0]
        correct += int((np.argmax(logits.numpy(), axis=-1) == labels[rows]).sum())
    return correct / max(1, len(labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--qat-start-epoch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    splits = {split: build_split_cache(args.dataset, args.cache_dir, split) for split in SPLITS}
    for split, (units, labels) in splits.items():
        print(f"{split}: {len(labels)} samples")

    model = build_wordseq_by_name(len(FIXED_EXPORT_LABELS), 4, FIXED_EXPORT_ARCHITECTURE)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=args.learning_rate, global_clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=args.label_smoothing)

    train_units, train_labels = splits["train"]
    valid_units, valid_labels = splits["valid"]
    steps_per_epoch = math.ceil(len(train_labels) / BATCH_SIZE)
    total_steps = steps_per_epoch * args.epochs

    @tf.function(reduce_retracing=True)
    def train_step(inputs: tf.Tensor, targets: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = model(inputs, training=True)[0]
            loss = loss_fn(targets, logits)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_valid = -1.0
    best_weights = None
    step = 0
    for epoch in range(args.epochs):
        QAT_ACTIVE.assign(epoch >= args.qat_start_epoch)
        epoch_loss = 0.0
        batches = length_bucketed_batches(train_units, rng)
        for rows in batches:
            progress = step / max(1, total_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = args.min_learning_rate_ratio + (1.0 - args.min_learning_rate_ratio) * cosine
            optimizer.learning_rate.assign(args.learning_rate * ratio)
            inputs = batch_inputs(train_units, rows)
            targets = tf.one_hot(train_labels[np.sort(rows)], len(FIXED_EXPORT_LABELS))
            epoch_loss += float(train_step(tf.constant(inputs), targets))
            step += 1
        valid_accuracy = evaluate(model, valid_units, valid_labels)
        quantized = epoch >= args.qat_start_epoch
        print(
            f"epoch={epoch} loss={epoch_loss / len(batches):.4f} "
            f"valid_accuracy={valid_accuracy:.6f} qat={quantized}"
        )
        if quantized and valid_accuracy > best_valid:
            best_valid = valid_accuracy
            best_weights = [weight.copy() for weight in model.get_weights()]

    if best_weights is None:
        raise RuntimeError("no quantized epoch produced a checkpoint; lower --qat-start-epoch")
    model.set_weights(best_weights)
    QAT_ACTIVE.assign(True)

    test_units, test_labels = splits["test"]
    test_accuracy = evaluate(model, test_units, test_labels)
    per_class = {}
    predictions: list[int] = []
    for start in range(0, len(test_labels), BATCH_SIZE):
        rows = np.arange(start, min(start + BATCH_SIZE, len(test_labels)))
        logits = model(batch_inputs(test_units, rows), training=False)[0]
        predictions.extend(np.argmax(logits.numpy(), axis=-1).tolist())
    predictions_np = np.asarray(predictions)
    for index, label in enumerate(FIXED_EXPORT_LABELS):
        mask = test_labels == index
        recall = float((predictions_np[mask] == index).mean()) if mask.any() else float("nan")
        per_class[label] = recall
    print(f"best_valid_accuracy={best_valid:.6f}")
    print(f"test_accuracy={test_accuracy:.6f}")
    print("per_class_recall=" + json.dumps(per_class))

    size = export_model(
        args.output,
        model,
        FIXED_EXPORT_LABELS,
        FIXED_EXPORT_SLUGS,
        4,
        FIXED_EXPORT_ARCHITECTURE,
        tokenizer_version=FIXED_EXPORT_TOKENIZER_VERSION,
    )
    print(f"exported {size} bytes to {args.output}")


if __name__ == "__main__":
    main()
