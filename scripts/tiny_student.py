"""Experimental 4,752-byte lexical classifier (BTQ1 format).

Only NumPy is needed for inference. The format fixes the 48-label production
order, 512 input features, 16 hidden units and symmetric four-bit weights.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

BINS = 512
HIDDEN = 16
CLASSES = 48
MAGIC = b"BTQ1\x01\0\0\0"
MODEL_BYTES = 4752
TOKENS = re.compile(rb"[a-zA-Z_\x80-\xff][a-zA-Z_0-9\x80-\xff]*|[0-9]+|\n|[^\s]")
ASCII_SPACE = b" \t\n\r\x0c"


def window(source: bytes) -> bytes:
    beginning = source[:4096].lstrip(ASCII_SPACE)
    if len(beginning) < 8:
        return b""
    if len(beginning) < 1024:
        return beginning
    end = source[-4096:].rstrip(ASCII_SPACE)
    if len(end) < 1024:
        return beginning[:1024]
    return beginning[:1024] + end[-1024:]


def hash_bytes(value: bytes) -> int:
    result = 2166136261
    for byte in value:
        result = ((result ^ byte) * 16777619) & 0xFFFFFFFF
    return result


def features(source: bytes) -> np.ndarray:
    counts = np.zeros(BINS, dtype=np.float32)
    previous = 0
    for match in TOKENS.finditer(window(source)):
        token = match.group()
        if token[:1].isdigit():
            token = b"0"
        value = hash_bytes(token.lower())
        counts[value % BINS] += 1
        if previous:
            pair = ((previous * 16777619) ^ value) & 0xFFFFFFFF
            counts[pair % BINS] += 1
        previous = value
    counts = np.log1p(counts)
    norm = np.linalg.norm(counts)
    if norm:
        counts /= norm
    return counts


def load_model(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    blob = path.read_bytes()
    if len(blob) != MODEL_BYTES or blob[:8] != MAGIC:
        raise ValueError("expected a 4,752-byte BTQ1 model")
    scales = struct.unpack_from("<2f", blob, 8)
    if not all(np.isfinite(s) and s > 0 for s in scales):
        raise ValueError("invalid weight scales")
    cursor = 16
    result = []
    for rows, cols, scale in ((BINS, HIDDEN, scales[0]), (HIDDEN, CLASSES, scales[1])):
        count = rows * cols
        packed = np.frombuffer(blob, dtype=np.uint8, count=count // 2, offset=cursor)
        cursor += count // 2
        values = np.empty(count, dtype=np.float32)
        values[::2] = (packed & 15).astype(np.int16) - 8
        values[1::2] = (packed >> 4).astype(np.int16) - 8
        result.append(values.reshape(rows, cols) * scale)
        bias = np.frombuffer(blob, dtype="<f4", count=cols, offset=cursor).copy()
        if not np.isfinite(bias).all():
            raise ValueError("non-finite bias")
        result.append(bias)
        cursor += cols * 4
    assert cursor == len(blob)
    return tuple(result)


def logits(inputs: np.ndarray, weights: tuple) -> np.ndarray:
    kernel, bias, output, output_bias = weights
    return np.maximum(inputs @ kernel + bias, 0) @ output + output_bias
