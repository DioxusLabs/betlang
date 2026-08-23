#!/usr/bin/env python3
"""Canonical Shannon code over raw window bytes.

Code lengths: L_b = clamp(ceil(-log2 p_b), 1, MAX_LEN) from the train-split
byte distribution (add-one smoothing). Ceil guarantees Kraft <= 1 before the
cap; the cap only shortens codes for symbols with p < 2^-MAX_LEN whose total
mass is far below the slack, so Kraft still holds. Codewords are assigned
canonically: sort symbols by (length, byte value), assign lexicographically
increasing codewords MSB-first.
"""
from __future__ import annotations

import math

import numpy as np

MAX_LEN = 12


def code_lengths(byte_counts: np.ndarray) -> np.ndarray:
    counts = byte_counts.astype(np.float64) + 1.0
    probs = counts / counts.sum()
    lengths = np.ceil(-np.log2(probs)).astype(np.int32)
    lengths = np.clip(lengths, 1, MAX_LEN)
    kraft = float(np.sum(2.0 ** (-lengths)))
    while kraft > 1.0:  # extend the longest-and-rarest codes until Kraft holds
        order = np.lexsort((probs, -lengths))
        for idx in order:
            if lengths[idx] < MAX_LEN:
                kraft -= 2.0 ** (-lengths[idx])
                lengths[idx] += 1
                kraft += 2.0 ** (-lengths[idx])
                break
        else:
            raise AssertionError("cannot satisfy Kraft inequality")
    return lengths


def canonical_codes(lengths: np.ndarray) -> np.ndarray:
    """Returns [256, 2] array of (codeword, length); codeword MSB-first."""
    order = sorted(range(256), key=lambda b: (int(lengths[b]), b))
    codes = np.zeros((256, 2), dtype=np.uint32)
    code = 0
    prev_len = 0
    for b in order:
        length = int(lengths[b])
        code <<= length - prev_len
        codes[b] = (code, length)
        code += 1
        prev_len = length
    return codes


def encode_bits(data: bytes, codes: np.ndarray, out_bits: int) -> np.ndarray:
    """Encode into a fixed-size uint8 bit array (truncate/zero-pad)."""
    bits = np.zeros(out_bits, dtype=np.uint8)
    pos = 0
    for byte in data:
        word, length = int(codes[byte, 0]), int(codes[byte, 1])
        if pos + length > out_bits:
            break
        for i in range(length - 1, -1, -1):
            bits[pos] = (word >> i) & 1
            pos += 1
    return bits


def encode_bits_bulk(windows: np.ndarray, lengths_arr: np.ndarray,
                     codes: np.ndarray, out_bits: int) -> np.ndarray:
    """Vectorized batch encoder: windows uint8 [N, W], lengths [N] -> [N, out_bits]."""
    n, w = windows.shape
    sym_len = codes[:, 1].astype(np.int64)
    sym_code = codes[:, 0].astype(np.int64)
    mask = np.arange(w)[None, :] < lengths_arr[:, None]
    lens = np.where(mask, sym_len[windows], 0)
    starts = np.cumsum(lens, axis=1) - lens
    out = np.zeros((n, out_bits), dtype=np.uint8)
    # expand each symbol's bits; loop over bit index within codeword (<= MAX_LEN)
    for k in range(MAX_LEN):
        sel = (lens > k) & (starts + lens <= out_bits)
        rows, cols = np.nonzero(sel)
        sym = windows[rows, cols]
        bit = (sym_code[sym] >> (sym_len[sym] - 1 - k)) & 1
        out[rows, starts[rows, cols] + k] = bit
    return out


def encode_planes_bulk(windows: np.ndarray, lengths_arr: np.ndarray,
                       codes: np.ndarray, out_bits: int) -> np.ndarray:
    """Batch encoder emitting two bitplanes: [N, 2, out_bits].
    Plane 0: Shannon code bits. Plane 1: codeword-boundary markers (1 at the
    first bit of each encoded codeword)."""
    n, w = windows.shape
    sym_len = codes[:, 1].astype(np.int64)
    sym_code = codes[:, 0].astype(np.int64)
    mask = np.arange(w)[None, :] < lengths_arr[:, None]
    lens = np.where(mask, sym_len[windows], 0)
    starts = np.cumsum(lens, axis=1) - lens
    out = np.zeros((n, 2, out_bits), dtype=np.uint8)
    fits = (lens > 0) & (starts + lens <= out_bits)
    rows0, cols0 = np.nonzero(fits)
    out[rows0, 1, starts[rows0, cols0]] = 1
    for k in range(MAX_LEN):
        sel = (lens > k) & (starts + lens <= out_bits)
        rows, cols = np.nonzero(sel)
        sym = windows[rows, cols]
        bit = (sym_code[sym] >> (sym_len[sym] - 1 - k)) & 1
        out[rows, 0, starts[rows, cols] + k] = bit
    return out


def expected_bits(byte_counts: np.ndarray, lengths: np.ndarray) -> float:
    probs = byte_counts / byte_counts.sum()
    return float((probs * lengths).sum())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    counts = rng.integers(0, 10000, 256).astype(np.int64)
    lengths = code_lengths(counts)
    codes = canonical_codes(lengths)
    # prefix-freeness check
    seen = set()
    for b in range(256):
        word, length = int(codes[b, 0]), int(codes[b, 1])
        s = format(word, f"0{length}b")
        for other in seen:
            assert not s.startswith(other) and not other.startswith(s)
        seen.add(s)
    data = bytes(rng.integers(0, 256, 500).astype(np.uint8))
    single = encode_bits(data, codes, 4096)
    bulk = encode_bits_bulk(
        np.frombuffer(data, dtype=np.uint8)[None, :].copy(),
        np.array([len(data)]), codes, 4096)
    assert np.array_equal(single, bulk[0])
    print("shannon self-test ok; mean bits/byte:",
          expected_bits(counts.astype(np.float64), lengths))
