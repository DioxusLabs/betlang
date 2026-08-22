#!/usr/bin/env python3
"""Export golden vectors for the Rust MBL3 Bloom runtime parity test.

For each sample source file this script mirrors the full production pipeline
in Python — byte window construction, tokenizer-v3 word units, Bloom signature
encoding, and integer inference over the exported MBL3 artifact — and writes
the raw source bytes plus expected logits to a fixture consumed by
`bloom_matches_python_golden_vectors` in src/model/tests.rs.

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
    BITS,
    BLOCK_BITS,
    BLOCKS,
    CLASSES,
    PAD,
    TOKEN_LENGTH,
    encode_windows,
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


def load_mbl3(path: Path) -> dict[str, np.ndarray | int]:
    blob = path.read_bytes()
    assert blob[:4] == b"MBL3", "bad magic"
    block_bits, planes, hidden, classes = struct.unpack_from("<IIII", blob, 4)
    assert block_bits == BLOCK_BITS and planes == BLOCKS and classes == CLASSES
    cur = 20
    scale = np.frombuffer(blob, dtype="<f4", count=classes * planes,
                          offset=cur).reshape(classes, planes)
    cur += 4 * classes * planes
    bias = np.frombuffer(blob, dtype="<f4", count=classes, offset=cur)
    cur += 4 * classes
    model: dict[str, np.ndarray | int] = {"hidden": hidden, "scale": scale, "bias": bias}
    if hidden > 0:
        units = planes * hidden
        rows = units * (BLOCK_BITS // 64)
        model["hidden_w"] = np.frombuffer(blob, dtype="<u8", count=rows, offset=cur)
        cur += 8 * rows
        model["hidden_thr"] = np.frombuffer(blob, dtype="<i2", count=units, offset=cur)
        cur += 2 * units
        flip_words = (units + 63) // 64
        model["hidden_flip"] = np.frombuffer(blob, dtype="<u8", count=flip_words, offset=cur)
        cur += 8 * flip_words
        head_bits = units
    else:
        head_bits = BITS
    head_words = (head_bits + 63) // 64
    model["head_w"] = np.frombuffer(blob, dtype="<u8", count=classes * head_words, offset=cur)
    cur += 8 * classes * head_words
    assert cur == len(blob), f"trailing bytes: {len(blob) - cur}"
    model["head_bits"] = head_bits
    model["head_words"] = head_words
    return model


def unpack_bits(words: np.ndarray, nbits: int) -> np.ndarray:
    raw = np.unpackbits(words.view(np.uint8), bitorder="little")
    return raw[:nbits]


def forward(model: dict[str, np.ndarray | int], signature: np.ndarray) -> np.ndarray:
    """Integer forward pass identical to BloomModel::logits."""
    x = unpack_bits(signature, BITS)
    hidden = int(model["hidden"])
    if hidden > 0:
        units = BLOCKS * hidden
        block_words = BLOCK_BITS // 64
        hw = np.asarray(model["hidden_w"]).reshape(units, block_words)
        thr = np.asarray(model["hidden_thr"])
        flips = unpack_bits(np.asarray(model["hidden_flip"]), units)
        xb = x.reshape(BLOCKS, BLOCK_BITS)
        w_bits = np.unpackbits(
            hw.view(np.uint8).reshape(units, -1), axis=1, bitorder="little"
        )[:, :BLOCK_BITS]
        mism = (xb[np.arange(units) // hidden] != w_bits).sum(axis=1)
        head_in = ((mism <= thr).astype(np.uint8) ^ flips).astype(np.uint8)
    else:
        head_in = x
    head_bits = int(model["head_bits"])
    head_words = int(model["head_words"])
    block_width = hidden if hidden > 0 else BLOCK_BITS
    head = np.asarray(model["head_w"]).reshape(CLASSES, head_words)
    w_bits = np.unpackbits(
        head.view(np.uint8).reshape(CLASSES, -1), axis=1, bitorder="little"
    )[:, :head_bits]
    hb = head_in.reshape(BLOCKS, block_width)
    wb = w_bits.reshape(CLASSES, BLOCKS, block_width)
    mism = (hb[None, :, :] != wb).sum(axis=2)  # (CLASSES, BLOCKS)
    z = block_width - 2 * mism
    scale = np.asarray(model["scale"], dtype=np.float64)
    bias = np.asarray(model["bias"], dtype=np.float64)
    return ((scale * z).sum(axis=1) + bias).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    model = load_mbl3(args.model)
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
        signature = encode_windows(tokens[None, :], units, tables)[0]
        logits = forward(model, signature)
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
