#!/usr/bin/env python3
"""Export golden vectors for the Rust MBL5 Bloom runtime parity test.

For each sample source file this script mirrors the full production pipeline
in Python — byte window construction, tokenizer-v3 word units, counting-Bloom
buckets, selected-column signature gather, and integer inference over the
exported MBL5 artifact — and writes the raw source bytes plus expected logits
to a fixture consumed by `bloom_matches_python_golden_vectors` in
src/model/tests.rs.

Fixture layout (little endian):
  magic  b"BGL1"
  u32    sample count
  per sample:
    u32  source byte length
    ...  source bytes
    f32  x 48 expected logits

Usage:
  python scripts/export_bloom_golden.py \
      --model assets/magika/source-bloom.bin \
      --output tests/fixtures/bloom-golden.bin \
      tests/fixtures/languages/*
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from train_bloom_head import (  # noqa: E402
    BLOCK,
    CLASSES,
    FULL_PLANES,
    GROUPS,
    PAD,
    TOKEN_LENGTH,
    encode_counts,
    zobrist_tables,
)
from train_magika_qat_student import numpy_word_units_apply_v3  # noqa: E402

MAGIKA_BLOCK_SIZE = 4096
HALF = TOKEN_LENGTH // 2
ASCII_WHITESPACE = b"\t\n\x0c\r "


def build_token_window(source: bytes) -> np.ndarray | None:
    """Mirror src/model/bloom.rs build_token_window exactly."""
    if not source:
        return None
    block = min(len(source), MAGIKA_BLOCK_SIZE)
    stripped_beg = source[:block].lstrip(ASCII_WHITESPACE)
    if len(stripped_beg) < 8:
        return None
    stripped_end = source[len(source) - block:].rstrip(ASCII_WHITESPACE)

    tokens = np.full(TOKEN_LENGTH, PAD, dtype=np.uint16)
    beg_len = min(len(stripped_beg), HALF)
    tokens[:beg_len] = np.frombuffer(stripped_beg[:beg_len], dtype=np.uint8)
    end_len = min(len(stripped_end), HALF)
    if end_len:
        end_src = stripped_end[len(stripped_end) - end_len:]
        tokens[TOKEN_LENGTH - end_len:] = np.frombuffer(end_src, dtype=np.uint8)
    return tokens


def load_mbl5(path: Path) -> dict[str, np.ndarray]:
    blob = path.read_bytes()
    assert blob[:4] == b"MBL5", "bad magic"
    bits, plane_count, classes = struct.unpack_from("<III", blob, 4)
    assert classes == CLASSES
    cur = 16
    planes = []
    for _ in range(plane_count):
        group, level, width = struct.unpack_from("<BBH", blob, cur)
        assert group < GROUPS and width % 64 == 0 and width <= BLOCK
        assert (group, level) in FULL_PLANES, "unknown (group, level) plane"
        planes.append((group, level, width))
        cur += 4
    assert sum(width for _, _, width in planes) == bits
    buckets = np.frombuffer(blob, dtype="<u2", count=bits, offset=cur)
    assert int(buckets.max()) < BLOCK
    cur += 2 * bits
    q = np.frombuffer(blob, dtype=np.int8, count=classes * plane_count,
                      offset=cur).reshape(classes, plane_count)
    cur += classes * plane_count
    step = np.frombuffer(blob, dtype="<f4", count=classes, offset=cur)
    cur += 4 * classes
    bias = np.frombuffer(blob, dtype="<f4", count=classes, offset=cur)
    cur += 4 * classes
    words = bits // 64
    head = np.frombuffer(blob, dtype="<u8", count=classes * words, offset=cur)
    cur += 8 * classes * words
    assert cur == len(blob), f"trailing bytes: {len(blob) - cur}"
    return {"bits": bits, "planes": planes, "buckets": buckets, "q": q,
            "step": step, "bias": bias, "head": head.reshape(classes, words)}


def selected_bits(model: dict[str, np.ndarray], counts: np.ndarray) -> np.ndarray:
    """Selected-column signature bits from (GROUPS, BLOCK) bucket counts."""
    bits = np.empty(model["bits"], dtype=np.uint8)
    at = 0
    for group, level, width in model["planes"]:
        ids = model["buckets"][at:at + width]
        bits[at:at + width] = counts[group][ids] >= level
        at += width
    return bits


def forward(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """Integer forward pass identical to BloomModel::logits."""
    bits = int(model["bits"])
    w_bits = np.unpackbits(
        model["head"].view(np.uint8).reshape(CLASSES, -1), axis=1, bitorder="little"
    )[:, :bits]
    acc = np.zeros(CLASSES, dtype=np.int64)
    at = 0
    for plane, (_, _, width) in enumerate(model["planes"]):
        mism = (x[None, at:at + width] != w_bits[:, at:at + width]).sum(axis=1)
        z = width - 2 * mism
        acc += model["q"][:, plane].astype(np.int64) * z
        at += width
    step = model["step"].astype(np.float32)
    return step * acc.astype(np.float32) + model["bias"].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    model = load_mbl5(args.model)
    tables = zobrist_tables()

    blob = bytearray(b"BGL1")
    samples = []
    for source_path in args.sources:
        source = source_path.read_bytes()
        tokens = build_token_window(source)
        if tokens is None:
            print(f"skip {source_path} (window rejected)")
            continue
        units = numpy_word_units_apply_v3(tokens.astype(np.int64)[None, :])
        counts = encode_counts(tokens[None, :], units, tables)[0]
        logits = forward(model, selected_bits(model, counts))
        samples.append((source, logits))
        top = int(logits.argmax())
        print(f"{source_path.name}: argmax={top} logit={logits[top]:.3f}")

    blob += struct.pack("<I", len(samples))
    for source, logits in samples:
        blob += struct.pack("<I", len(source))
        blob += source
        blob += logits.astype("<f4").tobytes()
    args.output.write_bytes(bytes(blob))
    print(f"wrote {len(samples)} samples ({len(blob)} bytes) to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
