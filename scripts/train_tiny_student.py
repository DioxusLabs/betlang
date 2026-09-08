#!/usr/bin/env python3
"""Train/evaluate a <=5,000-byte lexical student on files/{train,valid,test}/label.

Checkpoint selection uses validation only; the report measures the reloaded
quantized export. This experiment does not replace the production model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

from tiny_student import BINS, HIDDEN, MAGIC, MODEL_BYTES, features, load_model, logits, window
from train_magika_qat_student import (
    FIXED_EXPORT_LABELS,
    QAT_ACTIVE,
    QDense,
    quantize_weight,
)


def featurize_file(path: Path) -> tuple[np.ndarray, bytes]:
    source = path.read_bytes()
    return features(source), hashlib.sha256(window(source)).digest()


def prepare(dataset: Path, cache: Path, workers: int) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    seen = set()
    for split in ("train", "valid", "test"):
        paths = sorted(path for path in (dataset / split).glob("*/*") if path.is_file())
        if not paths:
            raise ValueError(f"no files in {dataset / split}")
        labels = np.array([FIXED_EXPORT_LABELS.index(p.parent.name) for p in paths], np.int64)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(featurize_file, paths, chunksize=128))
        keep = []
        for i, (_, digest) in enumerate(rows):
            if digest not in seen:
                keep.append(i)
                seen.add(digest)
        inputs = np.stack([rows[i][0] for i in keep])
        labels = labels[keep]
        paths = [paths[i] for i in keep]
        if any("\n" in str(p) or "\t" in str(p) for p in paths):
            raise ValueError("manifest paths cannot contain tabs or newlines")
        np.savez(cache / f"{split}.npz", inputs=inputs, labels=labels)
        (cache / f"{split}.paths.txt").write_text("".join(f"{p.resolve()}\n" for p in paths))
        print(f"{split}: {len(paths)} files, {len(np.unique(labels))} labels", flush=True)


def read_split(cache: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache / f"{split}.npz", allow_pickle=False) as data:
        return data["inputs"], data["labels"]


def export(path: Path, model: tf.keras.Model) -> int:
    payload = bytearray()
    scales = []
    for layer in model.layers:
        if isinstance(layer, QDense):
            packed, scale, _ = quantize_weight(layer.kernel.numpy(), 4)
            scales.append(scale)
            payload.extend(packed)
            payload.extend(layer.bias.numpy().astype("<f4").tobytes())
    blob = MAGIC + np.asarray(scales, dtype="<f4").tobytes() + payload
    if len(blob) != MODEL_BYTES or len(blob) > 5000:
        raise ValueError(f"model exceeds fixed byte budget: {len(blob)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return len(blob)


def evaluate(cache: Path, model_path: Path) -> dict:
    inputs, labels = read_split(cache, "test")
    predictions = logits(inputs, load_model(model_path)).argmax(axis=1)
    confusion = np.zeros((48, 48), np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    totals = confusion.sum(axis=1)
    recalls = np.divide(confusion.diagonal(), totals, out=np.zeros(48), where=totals > 0)
    return {
        "bytes": model_path.stat().st_size,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "test_files": len(labels),
        "accuracy": float(np.mean(predictions == labels)),
        "macro_recall_present_labels": float(recalls[totals > 0].mean()),
        "missing_test_labels": [s for s, n in zip(FIXED_EXPORT_LABELS, totals) if n == 0],
        "per_label": {
            slug: {"files": int(n), "recall": float(r)}
            for slug, n, r in zip(FIXED_EXPORT_LABELS, totals, recalls)
        },
        "confusion": confusion.tolist(),
    }


def train(args: argparse.Namespace) -> None:
    tf.keras.utils.set_random_seed(args.seed)
    inputs, labels = read_split(args.cache, "train")
    valid_x, valid_y = read_split(args.cache, "valid")
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(BINS,)),
        QDense(HIDDEN, 4),
        tf.keras.layers.ReLU(),
        QDense(len(FIXED_EXPORT_LABELS), 4),
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)
    targets = np.eye(48, dtype=np.float32)[labels]

    @tf.function(input_signature=[
        tf.TensorSpec([None, BINS], tf.float32), tf.TensorSpec([None, 48], tf.float32)
    ])
    def step(x, y):
        with tf.GradientTape() as tape:
            loss = loss_fn(y, model(x, training=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    rng = np.random.default_rng(args.seed)
    best = -1.0
    history = []
    for epoch in range(args.epochs):
        started = time.perf_counter()
        QAT_ACTIVE.assign(epoch >= args.qat_start)
        optimizer.learning_rate.assign(args.learning_rate * (
            0.05 + 0.95 * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
        ))
        order = rng.permutation(len(inputs))
        losses = []
        for offset in range(0, len(order), args.batch_size):
            ids = order[offset:offset + args.batch_size]
            losses.append(float(step(inputs[ids], targets[ids])))
        QAT_ACTIVE.assign(True)
        predictions = np.concatenate([
            model(valid_x[start:start + 2048], training=False).numpy().argmax(axis=1)
            for start in range(0, len(valid_x), 2048)
        ])
        accuracy = float(np.mean(predictions == valid_y))
        if accuracy > best:
            best = accuracy
            export(args.output, model)
        row = {"epoch": epoch + 1, "loss": float(np.mean(losses)),
               "valid_accuracy": accuracy, "seconds": time.perf_counter() - started}
        history.append(row)
        print(json.dumps(row), flush=True)
    report = evaluate(args.cache, args.output)
    report.update({"best_valid_accuracy": best, "seed": args.seed, "history": history})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("history", "confusion", "per_label")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--qat-start", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        if args.dataset is None:
            parser.error("--prepare requires --dataset")
        prepare(args.dataset, args.cache, args.workers)
    elif args.evaluate:
        print(json.dumps(evaluate(args.cache, args.output), indent=2))
    else:
        train(args)


if __name__ == "__main__":
    main()
