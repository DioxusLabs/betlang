#!/usr/bin/env python3
"""Train the betlang Shannon n-gram Bloom binary head (selected columns).

Stage 1 (deterministic, no learned weights): the Magika byte window is encoded
into counting-Bloom buckets. For every byte position and every n-gram order n
in ORDERS, a Zobrist hash (XOR of per-offset random 64-bit codes, which is a
Shannon-style random binary code for each (byte, offset) symbol) is folded
into 4,096 buckets per (half, order) group and *counted*; word-unit and
tokenizer-v3 unit streams get their own groups. Bucket counts pass through
per-plane thermometer thresholds (quantized log-frequency, i.e. Shannon
surprisal), giving 319,488 candidate signature bits computed with table
lookups + XOR + integer compares only.

Stage 2 (column selection): computing candidate bits is nearly free at
inference, but every kept column costs 48 head bits plus a 16-bit bucket id in
the artifact. A saliency score — how much each column's class-conditional
activation deviation aligns with a full-width trained head's weights — ranks
all 319,488 columns, and a per-plane knapsack keeps the top BITS columns in
64-column chunks so plane segments stay u64-aligned.

Stage 3 (learned): a binary {-1,+1} linear head over the selected columns,
trained with a straight-through estimator against Magika teacher marginals
(BCE) plus hard filesystem labels. Inference is XNOR + popcount per class with
a per-(class, plane) int8 scale, followed by one float multiply per class.

Usage (three stages; the full run and selection are cached):
  python scripts/train_bloom_head.py full --cache-dir CACHE --scorer full.pt
  python scripts/train_bloom_head.py select --cache-dir CACHE \
      --scorer full.pt --k 4608 --selection sel.npz
  python scripts/train_bloom_head.py train --cache-dir CACHE \
      --selection sel.npz --output model.bin
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train_magika_qat_student import numpy_word_units_apply_v3

SYMBOLS = 257
PAD = 256
TOKEN_LENGTH = 2048
HALF = 1024
CLASSES = 48

ORDERS = (1, 2, 3, 4, 5, 6, 7, 8)
BLOCK = 4096
NGRAM_GROUPS = 2 * len(ORDERS)  # (half, order) counting groups
# Word-unit groups (per half): casefolded identifier unigrams (two
# independent hash folds), bigrams, trigrams, and line-start unigrams.
WORD_KIND_LEVELS = (
    (1, 2, 4, 8),  # unigram, first fold
    (1, 2),        # bigram
    (1, 2, 4, 8),  # unigram, second fold
    (1, 2),        # trigram
    (1, 2, 4, 8),  # line-start unigram
)
WORD_GROUPS = 2 * len(WORD_KIND_LEVELS)
# Wordseq-unit n-gram groups over the production tokenizer-v3 unit stream
# (words / punct runs / numbers / brackets / indents), one group per order.
UNIT_ORDERS = (1, 2, 3, 4)
UNIT_LEVELS = ((1, 2, 4, 8), (1, 2), (1,), (1,))
UNIT_GROUPS = len(UNIT_ORDERS)
GROUPS = NGRAM_GROUPS + WORD_GROUPS + UNIT_GROUPS
COUNT_BUCKETS = GROUPS * BLOCK
# Count thermometer levels per order: low orders carry frequency information
# (log-TF), high orders are presence-only.
ORDER_LEVELS = tuple(
    (1, 2, 4, 8) if order <= 3 else (1, 2) if order <= 5 else (1,)
    for order in ORDERS
)
# Full-resolution candidate plane table: (group, count threshold) per plane,
# 4,096 candidate columns each. Selection picks columns from these planes.
FULL_PLANES: list[tuple[int, int]] = []
for _oi, _order in enumerate(ORDERS):
    for _half in range(2):
        for _level in ORDER_LEVELS[_oi]:
            FULL_PLANES.append((2 * _oi + _half, _level))
for _kind, _levels in enumerate(WORD_KIND_LEVELS):
    for _half in range(2):
        for _level in _levels:
            FULL_PLANES.append((NGRAM_GROUPS + 2 * _kind + _half, _level))
for _oi, _levels in enumerate(UNIT_LEVELS):
    for _level in _levels:
        FULL_PLANES.append((NGRAM_GROUPS + WORD_GROUPS + _oi, _level))
FULL_BLOCKS = len(FULL_PLANES)  # 78 planes
FULL_BITS = FULL_BLOCKS * BLOCK  # 319488
FULL_WORDS = FULL_BITS // 64
# Columns are selected in 64-column chunks so every plane segment of the
# packed signature stays u64-aligned (no masked popcounts in the runtime).
CHUNK_COLS = 64

ZOBRIST_SEED = 0xBE7A_1AB5_5EED_0001
UNIT_SEED = 0xBE7A_1AB5_5EED_0002


def xorshift64(state: int, count: int) -> np.ndarray:
    """Deterministic xorshift64* stream, reproducible in Rust."""
    mask = 0xFFFFFFFFFFFFFFFF
    out = np.empty(count, dtype=np.uint64)
    x = state & mask
    for i in range(count):
        x ^= x >> 12
        x = (x ^ (x << 25)) & mask
        x ^= x >> 27
        out[i] = (x * 0x2545F4914F6CDD1D) & mask
    return out


def zobrist_tables() -> np.ndarray:
    """Per-offset random codes: table[offset][symbol] -> u64 (last = words)."""
    max_order = max(ORDERS)
    flat = xorshift64(ZOBRIST_SEED, (max_order + 1) * SYMBOLS)
    return flat.reshape(max_order + 1, SYMBOLS)


def casefold_symbols() -> np.ndarray:
    """Symbol -> casefolded symbol (ASCII upper -> lower), PAD unchanged."""
    fold = np.arange(SYMBOLS, dtype=np.int64)
    fold[ord("A"):ord("Z") + 1] += 32
    return fold


def word_symbols() -> np.ndarray:
    """Symbol -> bool: part of an identifier word (alnum or underscore)."""
    is_word = np.zeros(SYMBOLS, dtype=bool)
    for c in range(256):
        ch = chr(c)
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            is_word[c] = True
    return is_word


def rotl64(x: np.ndarray, k: int) -> np.ndarray:
    k &= 63
    if k == 0:
        return x
    return ((x << np.uint64(k)) | (x >> np.uint64(64 - k))) & np.uint64(0xFFFFFFFFFFFFFFFF)


def splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer; must match the Rust runtime exactly."""
    z = (x + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return z ^ (z >> np.uint64(31))


def unit_salts() -> np.ndarray:
    """Per-offset salts for wordseq-unit n-gram codes."""
    return xorshift64(UNIT_SEED, max(UNIT_ORDERS))


def encode_counts(tokens: np.ndarray, units: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """Count n-gram/word/unit hashes into (n, GROUPS, BLOCK) Bloom buckets.

    n-grams never cross the begin/end half boundary and any n-gram containing
    the PAD token is skipped. Unit n-grams run over the tokenizer-v3 unit
    stream (-1 = padding) without halving.
    """
    n = tokens.shape[0]
    tok = tokens.astype(np.int64)
    valid = tok != PAD
    stride = COUNT_BUCKETS + 1  # one trash slot per row for invalid n-grams
    indices = []
    fold = casefold_symbols()
    is_word_sym = word_symbols()
    word_table = tables[max(ORDERS)]
    rows_flat = np.arange(n, dtype=np.int64) * stride

    for half in range(2):
        lo, hi = half * HALF, (half + 1) * HALF
        htok = tok[:, lo:hi]
        hvalid = valid[:, lo:hi]
        # hashes[o] at position i = XOR of tables[j][b_{i-j}] for j<=o
        acc = np.zeros((n, HALF), dtype=np.uint64)
        acc_valid = np.ones((n, HALF), dtype=bool)
        for order_index, order in enumerate(ORDERS):
            depth = order - 1
            shifted = np.empty_like(htok)
            shifted[:, depth:] = htok[:, : HALF - depth]
            shifted[:, :depth] = PAD
            svalid = np.zeros_like(hvalid)
            svalid[:, depth:] = hvalid[:, : HALF - depth]
            acc ^= tables[depth][shifted]
            acc_valid &= svalid
            group = 2 * order_index + half
            folded = rotl64(acc, 7 * order_index + half)
            bucket = (folded & np.uint64(BLOCK - 1)).astype(np.int64)
            bucket += group * BLOCK
            bucket = np.where(acc_valid, bucket, COUNT_BUCKETS)
            rows = np.arange(n)[:, None] * stride
            indices.append((rows + bucket).reshape(-1))

        # Word-unit scan: rotate-XOR chain over identifier bytes, emitting
        # unigrams (two folds), bigrams, trigrams, and line-start unigrams at
        # each word end.
        folded_tok = fold[htok]
        wsym = is_word_sym[htok]
        wcode = word_table[folded_tok]
        acc_w = np.zeros(n, dtype=np.uint64)
        prev_hash = np.zeros(n, dtype=np.uint64)
        prev2_hash = np.zeros(n, dtype=np.uint64)
        have_prev = np.zeros(n, dtype=bool)
        have_prev2 = np.zeros(n, dtype=bool)
        at_linestart = np.zeros(n, dtype=bool)
        uni_group = NGRAM_GROUPS + half
        bi_group = NGRAM_GROUPS + 2 + half
        uni2_group = NGRAM_GROUPS + 4 + half
        tri_group = NGRAM_GROUPS + 6 + half
        ls_group = NGRAM_GROUPS + 8 + half

        def fold_bucket(hashes: np.ndarray, rot: int, group: int,
                        emit: np.ndarray) -> np.ndarray:
            bucket = (rotl64(hashes, rot) & np.uint64(BLOCK - 1)).astype(np.int64)
            bucket += group * BLOCK
            return rows_flat + np.where(emit, bucket, COUNT_BUCKETS)

        for i in range(HALF):
            iw = wsym[:, i]
            prev_iw = wsym[:, i - 1] if i > 0 else np.zeros(n, dtype=bool)
            starts = iw & ~prev_iw
            if i == 0:
                start_is_nl = np.ones(n, dtype=bool)
            else:
                prev_byte = htok[:, i - 1]
                start_is_nl = (prev_byte == 10) | (prev_byte == 13)
            at_linestart = np.where(starts, start_is_nl, at_linestart)
            acc_w = np.where(iw, rotl64(acc_w, 1) ^ wcode[:, i], np.uint64(0))
            at_end = iw & (~wsym[:, i + 1] if i + 1 < HALF else True)
            indices.append(fold_bucket(acc_w, 23 + half, uni_group, at_end))
            indices.append(fold_bucket(acc_w, 41 + half, uni2_group, at_end))
            bg = rotl64(prev_hash, 17) ^ acc_w
            indices.append(fold_bucket(bg, 29 + half, bi_group, at_end & have_prev))
            tg = rotl64(prev2_hash, 34) ^ bg
            indices.append(fold_bucket(tg, 47 + half, tri_group, at_end & have_prev2))
            indices.append(fold_bucket(acc_w, 53 + half, ls_group, at_end & at_linestart))
            prev2_hash = np.where(at_end, prev_hash, prev2_hash)
            prev_hash = np.where(at_end, acc_w, prev_hash)
            have_prev2 |= at_end & have_prev
            have_prev |= at_end

    # Wordseq-unit n-grams: SplitMix64 codes salted per offset, XOR-combined.
    salts = unit_salts()
    uvalid = units >= 0
    u64 = units.astype(np.int64).astype(np.uint64)
    length = units.shape[1]
    acc_u = np.zeros((n, length), dtype=np.uint64)
    acc_uvalid = np.ones((n, length), dtype=bool)
    rows = np.arange(n)[:, None] * stride
    for order_index, order in enumerate(UNIT_ORDERS):
        depth = order - 1
        shifted = np.zeros_like(u64)
        shifted[:, depth:] = u64[:, : length - depth]
        svalid = np.zeros_like(uvalid)
        svalid[:, depth:] = uvalid[:, : length - depth]
        acc_u ^= splitmix64(shifted ^ salts[depth])
        acc_uvalid &= svalid
        group = NGRAM_GROUPS + WORD_GROUPS + order_index
        folded = rotl64(acc_u, 11 * order_index + 5)
        bucket = (folded & np.uint64(BLOCK - 1)).astype(np.int64)
        bucket += group * BLOCK
        bucket = np.where(acc_uvalid, bucket, COUNT_BUCKETS)
        indices.append((rows + bucket).reshape(-1))

    flat = np.concatenate(indices)
    counts = np.bincount(flat, minlength=n * stride).reshape(n, stride)
    return counts[:, :COUNT_BUCKETS].reshape(n, GROUPS, BLOCK)


def encode_windows(tokens: np.ndarray, units: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """Encode (n, 2048) token windows into packed full-resolution signatures.

    Column p * BLOCK + b of the 319,488-bit candidate signature is
    `counts[group(p)][b] >= level(p)` for full-resolution plane p.
    """
    n = tokens.shape[0]
    counts = encode_counts(tokens, units, tables)
    bits = np.zeros((n, FULL_BITS), dtype=np.uint8)
    for plane_index, (group, level) in enumerate(FULL_PLANES):
        lo = plane_index * BLOCK
        bits[:, lo:lo + BLOCK] = counts[:, group] >= level
    packed = np.ascontiguousarray(np.packbits(bits, axis=1, bitorder="little"))
    return packed.view(np.uint64).reshape(n, FULL_WORDS)


@dataclass
class Split:
    tokens: np.ndarray
    units: np.ndarray
    marginals: np.ndarray
    labels: np.ndarray
    fs_labels: np.ndarray | None
    count: int


def open_split(cache_dir: Path, split: str) -> Split:
    meta = json.loads((cache_dir / f"{split}.json").read_text())
    count = meta["count"]
    if not meta.get("head_marginal_targets"):
        raise SystemExit(f"{split}: cache must be built with --head-marginal-targets")
    tokens = np.memmap(cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r",
                       shape=(count, TOKEN_LENGTH))
    units_path = cache_dir / f"{split}.units_v3.mmap"
    if not units_path.exists():
        raise SystemExit(f"{split}: build units_v3 cache first (scripts/build_units.py)")
    units = np.memmap(units_path, dtype=np.int32, mode="r", shape=(count, TOKEN_LENGTH))
    probs = np.memmap(cache_dir / f"{split}.probabilities.mmap", dtype=np.float32,
                      mode="r", shape=(count, CLASSES))
    labels = np.memmap(cache_dir / f"{split}.labels.mmap", dtype=np.int64, mode="r",
                       shape=(count,))
    fs_path = cache_dir / f"{split}.fs_labels.mmap"
    fs = None
    if fs_path.exists():
        fs = np.memmap(fs_path, dtype=np.int64, mode="r", shape=(count,))
    return Split(tokens, units, probs, labels, fs, count)


ASCII_WS = (9, 10, 12, 13, 32)
AUG_MIN_BYTES = 12
AUG_MAX_BYTES = 1024


def build_crop_windows(tokens: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Token windows for prefix crops of the beg half (short-file augmentation).

    A crop of the first L bytes of a file behaves like a standalone short file:
    both window halves are built from the same bytes, mirroring
    build_token_window (begin half lstripped, end half rstripped).
    """
    n = tokens.shape[0]
    out = np.full((n, TOKEN_LENGTH), PAD, dtype=tokens.dtype)
    for i in range(n):
        beg = tokens[i, :HALF]
        available = int((beg != PAD).sum())
        crop_len = min(int(lengths[i]), available)
        crop = beg[:crop_len]
        out[i, :crop_len] = crop
        tail_len = crop_len
        while tail_len > 0 and int(crop[tail_len - 1]) in ASCII_WS:
            tail_len -= 1
        tail = crop[:tail_len][-HALF:]
        if tail.size:
            out[i, TOKEN_LENGTH - tail.size:] = tail
    return out


def ensure_augmented(cache_dir: Path, train: Split, count: int, fraction: float,
                     labels: np.ndarray, seed: int,
                     batch: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Compute (or load) short prefix-crop augmentation features + hard labels.

    Crop lengths are log-uniform in [AUG_MIN_BYTES, AUG_MAX_BYTES] so tiny
    files are well represented; the filesystem label of the source file is
    kept as the hard target (crops carry no teacher marginals).
    """
    n_aug = int(count * fraction)
    tag = f"train.bloomaug{FULL_BITS}_v1_s{seed}_n{n_aug}"
    feat_path = cache_dir / f"{tag}.feats.mmap"
    label_path = cache_dir / f"{tag}.labels.mmap"
    if feat_path.exists() and label_path.exists():
        return (np.memmap(feat_path, dtype=np.uint64, mode="r",
                          shape=(n_aug, FULL_WORDS)),
                np.memmap(label_path, dtype=np.int64, mode="r", shape=(n_aug,)))
    rng = np.random.default_rng(seed ^ 0xA06)
    src = np.sort(rng.choice(count, size=n_aug, replace=n_aug > count))
    lengths = np.exp(rng.uniform(math.log(AUG_MIN_BYTES), math.log(AUG_MAX_BYTES),
                                 size=n_aug)).astype(np.int64)
    tables = zobrist_tables()
    tmp = feat_path.with_suffix(".tmp")
    out = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(n_aug, FULL_WORDS))
    for start in range(0, n_aug, batch):
        stop = min(start + batch, n_aug)
        windows = build_crop_windows(
            np.asarray(train.tokens[src[start:stop]]), lengths[start:stop])
        units = numpy_word_units_apply_v3(windows.astype(np.int64))
        out[start:stop] = encode_windows(windows, units, tables)
        print(f"augment: encoded {stop}/{n_aug}", flush=True)
    out.flush()
    del out
    labels[src].astype(np.int64).tofile(label_path)
    tmp.rename(feat_path)
    return (np.memmap(feat_path, dtype=np.uint64, mode="r", shape=(n_aug, FULL_WORDS)),
            np.memmap(label_path, dtype=np.int64, mode="r", shape=(n_aug,)))


def ensure_features(cache_dir: Path, split: str, data: Split, batch: int = 1024) -> np.ndarray:
    """Compute (or load) packed full-resolution Bloom signatures for a split."""
    tag = f"{split}.bloom{FULL_BITS}_v6.mmap"
    path = cache_dir / tag
    if path.exists():
        return np.memmap(path, dtype=np.uint64, mode="r", shape=(data.count, FULL_WORDS))
    tables = zobrist_tables()
    tmp = path.with_suffix(".tmp")
    out = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(data.count, FULL_WORDS))
    for start in range(0, data.count, batch):
        stop = min(start + batch, data.count)
        out[start:stop] = encode_windows(
            np.asarray(data.tokens[start:stop]),
            np.asarray(data.units[start:stop]), tables)
        print(f"{split}: encoded {stop}/{data.count}", flush=True)
    out.flush()
    del out
    tmp.rename(path)
    return np.memmap(path, dtype=np.uint64, mode="r", shape=(data.count, FULL_WORDS))


def ensure_class_stats(cache_dir: Path, feats: np.ndarray,
                       labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class column activation sums over train (cached)."""
    path = cache_dir / "train.colstats_v1.npz"
    if path.exists():
        data = np.load(path)
        return data["sums"], data["counts"]
    n = feats.shape[0]
    sums = np.zeros((CLASSES, FULL_BITS), dtype=np.int64)
    counts = np.bincount(labels, minlength=CLASSES).astype(np.int64)
    for start in range(0, n, 1024):
        stop = min(start + 1024, n)
        packed = np.asarray(feats[start:stop])
        bits = np.unpackbits(packed.view(np.uint8).reshape(stop - start, -1),
                             axis=1, bitorder="little")[:, :FULL_BITS]
        lab = labels[start:stop]
        for c in np.unique(lab):
            sums[c] += bits[lab == c].sum(axis=0, dtype=np.int64)
        if (start // 1024) % 20 == 0:
            print(f"stats: {stop}/{n}", flush=True)
    np.savez(path, sums=sums, counts=counts)
    return sums, counts


def load_scorer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Full-width scorer weights: per-(class, plane) eff scale + sign bits.

    Accepts a `full` stage checkpoint (.pt) or a legacy full-width MBL3
    artifact.
    """
    if path.suffix == ".pt":
        state = torch.load(path, map_location="cpu")["state"]
        latent = state["latent"].numpy()
        block_scale = state["block_scale"].numpy()
        eff = block_scale / math.sqrt(FULL_BITS)
        return eff, (latent >= 0).astype(np.uint8)
    blob = path.read_bytes()
    assert blob[:4] == b"MBL3", "scorer must be a .pt checkpoint or MBL3 artifact"
    block_bits, blocks, hidden, classes = struct.unpack_from("<IIII", blob, 4)
    assert (block_bits, blocks, hidden, classes) == (BLOCK, FULL_BLOCKS, 0, CLASSES)
    cur = 20
    eff = np.frombuffer(blob, dtype="<f4", count=CLASSES * FULL_BLOCKS,
                        offset=cur).reshape(CLASSES, FULL_BLOCKS).copy()
    cur += CLASSES * FULL_BLOCKS * 4 + CLASSES * 4  # skip biases
    packed = np.frombuffer(blob, dtype=np.uint8, offset=cur).reshape(CLASSES, -1)
    w = np.unpackbits(packed, axis=1, bitorder="little")[:, :FULL_BITS]
    return eff, w


def align_scores(sums: np.ndarray, counts: np.ndarray, eff: np.ndarray,
                 w: np.ndarray) -> np.ndarray:
    """Saliency of each candidate column for the trained full-width head.

    score_j = sum_c pi_c * (p_{j|c} - p_j) * eff[c, plane(j)] * sign[c, j]:
    how much the column's class-conditional activation deviation aligns with
    the weights the full-width head learned for it. Negative scores mean the
    column mostly cancels and is not worth artifact bytes.
    """
    pi = counts / counts.sum()
    p_jc = sums / counts[:, None]
    p_j = (pi[:, None] * p_jc).sum(axis=0)
    plane_of = np.repeat(np.arange(FULL_BLOCKS), BLOCK)
    u = eff[:, plane_of] * (2.0 * w - 1.0)
    dev = p_jc - p_j[None, :]
    return np.maximum((pi[:, None] * dev * u).sum(axis=0), 0.0)


def knapsack_select(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep k columns as per-plane 64-column chunks with the best score mass.

    Returns (planes, buckets): planes rows are (full plane index, width) and
    buckets holds each plane's selected bucket ids, ascending, concatenated in
    plane order.
    """
    assert k % CHUNK_COLS == 0
    per_plane = scores.reshape(FULL_BLOCKS, BLOCK)
    order = np.argsort(-per_plane, axis=1)
    ranked = np.take_along_axis(per_plane, order, axis=1)
    chunk_vals = ranked.reshape(FULL_BLOCKS, BLOCK // CHUNK_COLS, CHUNK_COLS).sum(axis=2)
    take = np.sort(np.argsort(-chunk_vals.ravel())[: k // CHUNK_COLS])
    sel: dict[int, list[np.ndarray]] = {}
    for chunk in take:
        plane, ci = divmod(int(chunk), BLOCK // CHUNK_COLS)
        sel.setdefault(plane, []).append(
            order[plane, ci * CHUNK_COLS:(ci + 1) * CHUNK_COLS])
    planes = np.array([(p, sum(len(b) for b in sel[p])) for p in sorted(sel)],
                      dtype=np.int64)
    buckets = np.concatenate([np.sort(np.concatenate(sel[p])) for p in sorted(sel)])
    return planes, buckets


def selection_columns(planes: np.ndarray, buckets: np.ndarray) -> np.ndarray:
    """Global candidate-column indices of a selection, in signature order."""
    cols = []
    at = 0
    for plane, width in planes:
        cols.append(plane * BLOCK + buckets[at:at + width])
        at += width
    return np.concatenate(cols)


def ensure_selected(cache_dir: Path, split: str, feats: np.ndarray,
                    cols: np.ndarray, tag: str, batch: int = 1024) -> np.ndarray:
    """Gather selected columns from full-resolution features (cached)."""
    k = cols.size
    words = k // 64
    path = cache_dir / f"{split}.{tag}.mmap"
    if path.exists():
        return np.memmap(path, dtype=np.uint64, mode="r",
                         shape=(feats.shape[0], words))
    n = feats.shape[0]
    tmp = path.with_suffix(".tmp")
    out = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(n, words))
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        packed = np.asarray(feats[start:stop])
        bits = np.unpackbits(packed.view(np.uint8).reshape(stop - start, -1),
                             axis=1, bitorder="little")[:, :FULL_BITS]
        sel_bits = np.ascontiguousarray(bits[:, cols])
        repacked = np.ascontiguousarray(
            np.packbits(sel_bits, axis=1, bitorder="little"))
        out[start:stop] = repacked.view(np.uint64).reshape(stop - start, words)
    out.flush()
    del out
    tmp.rename(path)
    print(f"gathered {split} -> {path.name}", flush=True)
    return np.memmap(path, dtype=np.uint64, mode="r", shape=(n, words))


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1.0).to(grad.dtype)


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    return SignSTE.apply(x)


class BloomHead(nn.Module):
    """Binary {-1,+1} linear head over a (selected or full) Bloom signature.

    The head uses per-(class, plane) scales (XNOR-net style): the per-plane
    bipolar dot products (computed with XOR + popcount at inference) are
    combined with a small scale per plane, which recovers most of the accuracy
    lost by weight binarization at negligible cost (the scales quantize to
    int8 with one float step per class for export).
    """

    def __init__(self, widths: list[int], float_head: bool = False) -> None:
        super().__init__()
        self.float_head = float_head
        self.widths = widths
        self.bits = sum(widths)
        self.nplanes = len(widths)
        self.latent = nn.Parameter(torch.empty(CLASSES, self.bits))
        nn.init.uniform_(self.latent, -0.1, 0.1)
        self.block_scale = nn.Parameter(torch.full((CLASSES, self.nplanes), 4.0))
        self.bias = nn.Parameter(torch.zeros(CLASSES))
        bounds = [0]
        for width in widths:
            bounds.append(bounds[-1] + width)
        self.bounds = [(bounds[i], bounds[i + 1]) for i in range(self.nplanes)]

    def eff_scale(self) -> torch.Tensor:
        """Effective per-(class, plane) scale, fan-in normalized."""
        return self.block_scale / math.sqrt(self.bits)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Float head is a diagnostic-only upper bound for the binary head.
        w = self.latent if self.float_head else sign_ste(self.latent)
        # Per-plane bipolar dot products (planes have unequal widths).
        dots = [feats[:, lo:hi] @ w[:, lo:hi].T for lo, hi in self.bounds]
        d = torch.stack(dots, dim=2)  # (n, CLASSES, planes)
        return (d * self.eff_scale().unsqueeze(0)).sum(dim=2) + self.bias


def unpack_batch(packed: np.ndarray, bits: int) -> torch.Tensor:
    """Packed u64 -> float ±1 tensor of shape (n, bits)."""
    raw = np.unpackbits(packed.view(np.uint8).reshape(packed.shape[0], bits // 8),
                        axis=1, bitorder="little")
    return torch.from_numpy(raw.astype(np.float32) * 2.0 - 1.0)


def evaluate(model: BloomHead, feats: np.ndarray, data: Split,
             batch: int) -> tuple[float, float]:
    model.eval()
    correct_teacher = 0
    correct_fs = 0
    with torch.no_grad():
        for start in range(0, data.count, batch):
            stop = min(start + batch, data.count)
            x = unpack_batch(np.asarray(feats[start:stop]), model.bits)
            pred = model(x).argmax(dim=1).numpy()
            correct_teacher += int((pred == data.labels[start:stop]).sum())
            if data.fs_labels is not None:
                correct_fs += int((pred == data.fs_labels[start:stop]).sum())
    return correct_teacher / data.count, correct_fs / data.count


def train_head(args: argparse.Namespace, widths: list[int], train: Split,
               valid: Split, train_feats: np.ndarray, valid_feats: np.ndarray,
               aug_feats: np.ndarray | None,
               aug_labels: np.ndarray | None) -> BloomHead:
    rng = np.random.default_rng(args.seed)
    count = train.count
    labels = np.asarray(train.fs_labels[:count])
    marginals = np.asarray(train.marginals[:count])
    n_aug = 0 if aug_feats is None else aug_feats.shape[0]
    total = count + n_aug

    model = BloomHead(widths, args.float_head)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if "latent" in name or "block_scale" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": 1e-3}],
        lr=args.learning_rate,
    )
    steps_per_epoch = max(1, total // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()
           if v.dtype.is_floating_point}
    best_state = None
    best_fs = -1.0

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(total)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            idx = np.sort(order[step * args.batch_size:(step + 1) * args.batch_size])
            real_idx = idx[idx < count]
            crop_idx = idx[idx >= count] - count
            parts = []
            hard_parts = []
            if real_idx.size:
                parts.append(np.asarray(train_feats[real_idx]))
                hard_parts.append(labels[real_idx])
            if crop_idx.size:
                assert aug_feats is not None and aug_labels is not None
                parts.append(np.asarray(aug_feats[crop_idx]))
                hard_parts.append(np.asarray(aug_labels[crop_idx]))
            x = unpack_batch(np.concatenate(parts) if len(parts) > 1 else parts[0],
                             model.bits)
            hard = torch.from_numpy(np.concatenate(hard_parts).copy())

            logits = model(x)
            loss = logits.sum() * 0.0
            if real_idx.size:
                soft = torch.from_numpy(marginals[real_idx].copy())
                loss = loss + args.soft_loss_weight * F.binary_cross_entropy_with_logits(
                    logits[:real_idx.size], soft)
            loss = loss + args.hard_loss_weight * F.cross_entropy(logits, hard,
                                                                  label_smoothing=0.05)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                if not args.float_head:
                    for name, param in model.named_parameters():
                        if "latent" in name:
                            param.clamp_(-1.0, 1.0)
                for key, value in model.state_dict().items():
                    if key in ema:
                        ema[key].mul_(args.ema_decay).add_(value, alpha=1 - args.ema_decay)
            epoch_loss += float(loss.detach())

        raw_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        eval_state = dict(raw_state)
        eval_state.update({k: v.clone() for k, v in ema.items()})
        model.load_state_dict(eval_state)
        parity, fs_acc = evaluate(model, valid_feats, valid, args.batch_size)
        star = ""
        if fs_acc > best_fs:
            best_fs = fs_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            star = " *"
        model.load_state_dict(raw_state)
        print(f"epoch {epoch + 1}/{args.epochs} loss={epoch_loss / steps_per_epoch:.4f} "
              f"valid_parity={parity:.6f} valid_fs={fs_acc:.6f}{star}", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def report_test(model: BloomHead, test: Split, test_feats: np.ndarray,
                batch: int) -> None:
    parity, fs_acc = evaluate(model, test_feats, test, batch)
    print(f"test_teacher_parity={parity:.6f}", flush=True)
    print(f"test_fs_accuracy={fs_acc:.6f}", flush=True)
    if test.fs_labels is None:
        return
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, test.count, batch):
            stop = min(start + batch, test.count)
            x = unpack_batch(np.asarray(test_feats[start:stop]), model.bits)
            preds.append(model(x).argmax(dim=1).numpy())
    pred = np.concatenate(preds)
    fs = np.asarray(test.fs_labels)
    recalls = [float((pred[fs == c] == c).mean()) for c in range(CLASSES)
               if (fs == c).any()]
    print(f"test_fs_macro_recall={np.mean(recalls):.6f}", flush=True)


def pack_rows(bits01: np.ndarray) -> bytes:
    """(rows, nbits) 0/1 -> little-endian packed u64 words per row."""
    rows, nbits = bits01.shape
    padded = nbits + (-nbits) % 64
    buf = np.zeros((rows, padded), dtype=np.uint8)
    buf[:, :nbits] = bits01
    return np.packbits(buf, axis=1, bitorder="little").tobytes()


def quantized_scales(model: BloomHead) -> tuple[np.ndarray, np.ndarray]:
    """Per-(class, plane) int8 scales plus the per-class f32 step."""
    eff = model.eff_scale().detach().numpy().astype(np.float64)
    step = np.abs(eff).max(axis=1) / 127.0
    step = np.where(step <= 0.0, 1e-12, step)
    q = np.clip(np.round(eff / step[:, None]), -127, 127).astype(np.int8)
    return q, step.astype(np.float32)


def export(model: BloomHead, planes: np.ndarray, buckets: np.ndarray,
           output: Path) -> None:
    """MBL5: header, plane table, bucket ids, int8 scales, biases, head rows."""
    model.eval()
    blob = bytearray(b"MBL5")
    blob += struct.pack("<III", model.bits, len(planes), CLASSES)
    for plane, width in planes:
        group, level = FULL_PLANES[int(plane)]
        blob += struct.pack("<BBH", group, level, int(width))
    blob += buckets.astype("<u2").tobytes()
    q, step = quantized_scales(model)
    blob += q.tobytes()
    blob += step.astype("<f4").tobytes()
    blob += model.bias.detach().numpy().astype("<f4").tobytes()
    out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
    blob += pack_rows(out_w)

    output.write_bytes(bytes(blob))
    print(f"exported {len(blob)} bytes to {output}", flush=True)


def simulate(model: BloomHead, feats: np.ndarray) -> np.ndarray:
    """Integer XOR/popcount inference over packed selected signatures.

    Mirrors the Rust runtime: per-plane bipolar dot products accumulate as
    int8-scale * int, with one float multiply per class at the end.
    """
    n = feats.shape[0]
    q, step = quantized_scales(model)
    bias = model.bias.detach().numpy().astype(np.float32)
    x_bits = np.unpackbits(feats.view(np.uint8).reshape(n, model.bits // 8), axis=1,
                           bitorder="little").astype(np.uint8)
    out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
    acc = np.zeros((n, CLASSES), dtype=np.int64)
    for plane, (lo, hi) in enumerate(model.bounds):
        mismatches = (x_bits[:, None, lo:hi] != out_w[None, :, lo:hi]).sum(axis=2)
        z = (hi - lo) - 2 * mismatches
        acc += q[None, :, plane].astype(np.int64) * z
    return step[None, :] * acc.astype(np.float32) + bias[None, :]


def add_train_args(p: argparse.ArgumentParser, epochs: int) -> None:
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--soft-loss-weight", type=float, default=0.5)
    p.add_argument("--hard-loss-weight", type=float, default=0.5)
    p.add_argument("--augment-fraction", type=float, default=0.35,
                   help="extra short prefix-crop samples as a fraction of train "
                        "count (hard fs labels only)")
    p.add_argument("--float-head", action="store_true",
                   help="diagnostic: float output weights (no export)")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)


def cmd_full(args: argparse.Namespace) -> None:
    """Train the full-width (319,488-column) scorer head; saves a checkpoint."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train = open_split(args.cache_dir, "train")
    valid = open_split(args.cache_dir, "valid")
    test = open_split(args.cache_dir, "test")
    train_feats = ensure_features(args.cache_dir, "train", train)
    valid_feats = ensure_features(args.cache_dir, "valid", valid)
    test_feats = ensure_features(args.cache_dir, "test", test)
    print(f"train={train.count} valid={valid.count} test={test.count} "
          f"bits={FULL_BITS}", flush=True)

    aug_feats = aug_labels = None
    if args.augment_fraction > 0:
        labels = np.asarray(train.fs_labels[:train.count])
        aug_feats, aug_labels = ensure_augmented(
            args.cache_dir, train, train.count, args.augment_fraction, labels,
            args.seed)
        print(f"augmented={aug_feats.shape[0]}", flush=True)

    model = train_head(args, [BLOCK] * FULL_BLOCKS, train, valid,
                       train_feats, valid_feats, aug_feats, aug_labels)
    report_test(model, test, test_feats, args.batch_size)
    torch.save({"state": model.state_dict()}, args.scorer)
    print(f"saved scorer checkpoint to {args.scorer}", flush=True)


def cmd_select(args: argparse.Namespace) -> None:
    """Score all candidate columns and write the selection table."""
    train = open_split(args.cache_dir, "train")
    train_feats = ensure_features(args.cache_dir, "train", train)
    labels = np.asarray(train.fs_labels[:train.count])
    sums, counts = ensure_class_stats(args.cache_dir, train_feats, labels)
    eff, w = load_scorer(args.scorer)
    scores = align_scores(sums, counts, eff, w)
    planes, buckets = knapsack_select(scores, args.k)
    np.savez(args.selection, planes=planes, buckets=buckets)
    print(f"selection: {len(planes)} planes, {buckets.size} columns "
          f"-> {args.selection}", flush=True)


def cmd_train(args: argparse.Namespace) -> None:
    """Train the binary head over the selected columns and export MBL5."""
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    sel = np.load(args.selection)
    planes, buckets = sel["planes"], sel["buckets"]
    cols = selection_columns(planes, buckets)
    widths = [int(width) for _, width in planes]
    k = cols.size
    tag = f"bloomsel{k}_v1"
    print(f"selection: {len(planes)} planes, {k} columns", flush=True)

    train = open_split(args.cache_dir, "train")
    valid = open_split(args.cache_dir, "valid")
    test = open_split(args.cache_dir, "test")
    train_feats = ensure_selected(
        args.cache_dir, "train", ensure_features(args.cache_dir, "train", train),
        cols, tag)
    valid_feats = ensure_selected(
        args.cache_dir, "valid", ensure_features(args.cache_dir, "valid", valid),
        cols, tag)
    test_feats = ensure_selected(
        args.cache_dir, "test", ensure_features(args.cache_dir, "test", test),
        cols, tag)
    print(f"train={train.count} valid={valid.count} test={test.count} bits={k}",
          flush=True)

    aug_feats = aug_labels = None
    if args.augment_fraction > 0:
        labels = np.asarray(train.fs_labels[:train.count])
        full_aug, aug_labels = ensure_augmented(
            args.cache_dir, train, train.count, args.augment_fraction, labels,
            args.seed)
        aug_feats = ensure_selected(
            args.cache_dir, f"aug{full_aug.shape[0]}", full_aug, cols, tag)
        print(f"augmented={aug_feats.shape[0]}", flush=True)

    model = train_head(args, widths, train, valid, train_feats, valid_feats,
                       aug_feats, aug_labels)
    report_test(model, test, test_feats, args.batch_size)

    if args.float_head:
        return
    export(model, planes, buckets, args.output)

    sample = np.asarray(test_feats[:512])
    sim_logits = simulate(model, sample)
    model.eval()
    with torch.no_grad():
        torch_logits = model(unpack_batch(sample, model.bits)).numpy()
    agree = float((sim_logits.argmax(1) == torch_logits.argmax(1)).mean())
    max_err = float(np.abs(sim_logits - torch_logits).max())
    print(f"simulator: argmax agreement={agree:.4f} max_logit_err={max_err:.4f}",
          flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    pf = sub.add_parser("full", help="train the full-width scorer head")
    pf.add_argument("--cache-dir", type=Path, required=True)
    pf.add_argument("--scorer", type=Path, required=True)
    add_train_args(pf, epochs=30)
    pf.set_defaults(fn=cmd_full)

    ps = sub.add_parser("select", help="write the column-selection table")
    ps.add_argument("--cache-dir", type=Path, required=True)
    ps.add_argument("--scorer", type=Path, required=True,
                    help="full-stage checkpoint (.pt) or legacy MBL3 artifact")
    ps.add_argument("--k", type=int, default=4608,
                    help="selected columns (multiple of 64)")
    ps.add_argument("--selection", type=Path, required=True)
    ps.set_defaults(fn=cmd_select)

    pt = sub.add_parser("train", help="train the selected head and export MBL5")
    pt.add_argument("--cache-dir", type=Path, required=True)
    pt.add_argument("--selection", type=Path, required=True)
    pt.add_argument("--output", type=Path, required=True)
    add_train_args(pt, epochs=300)
    pt.set_defaults(fn=cmd_train)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
