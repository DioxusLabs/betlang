#!/usr/bin/env python3
"""Train the betlang Shannon n-gram Bloom binary head.

Stage 1 (deterministic, no learned weights): the Magika byte window is encoded
into a fixed-width binary signature. For every byte position and every n-gram
order n in ORDERS, a Zobrist hash (XOR of per-offset random 64-bit codes,
which is a Shannon-style random binary code for each (byte, offset) symbol) is
folded into a per-(half, order) counting-Bloom block; bucket counts pass
through fixed thermometer thresholds (quantized log-frequency, i.e. Shannon
surprisal). The result is a BITS-wide binary feature vector computed with
table lookups + XOR + integer compares only.

Stage 2 (learned): a binary {-1,+1} linear head over the signature, trained
with a straight-through estimator against Magika teacher marginals (BCE) plus
hard labels. Inference is XNOR + popcount per class, followed by an affine
calibration for probabilities.

Usage:
  python scripts/train_bloom_head.py --cache-dir CACHE --output model.bin
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
BLOCK_BITS = 4096
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
COUNT_BUCKETS = GROUPS * BLOCK_BITS
# Count thermometer levels per order: low orders carry frequency information
# (log-TF), high orders are presence-only.
ORDER_LEVELS = tuple(
    (1, 2, 4, 8) if order <= 3 else (1, 2) if order <= 5 else (1,)
    for order in ORDERS
)
PLANES: list[tuple[int, int]] = []  # (group, count threshold)
for _oi, _order in enumerate(ORDERS):
    for _half in range(2):
        for _level in ORDER_LEVELS[_oi]:
            PLANES.append((2 * _oi + _half, _level))
for _kind, _levels in enumerate(WORD_KIND_LEVELS):
    for _half in range(2):
        for _level in _levels:
            PLANES.append((NGRAM_GROUPS + 2 * _kind + _half, _level))
for _oi, _levels in enumerate(UNIT_LEVELS):
    for _level in _levels:
        PLANES.append((NGRAM_GROUPS + WORD_GROUPS + _oi, _level))
BLOCKS = len(PLANES)  # 78 planes
BITS = BLOCKS * BLOCK_BITS  # 319488
WORDS = BITS // 64

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


def encode_windows(tokens: np.ndarray, units: np.ndarray, tables: np.ndarray) -> np.ndarray:
    """Encode (n, 2048) token windows into packed (n, WORDS) u64 signatures.

    Each (half, order) group is a counting Bloom block: n-gram Zobrist hashes
    are folded into BLOCK_BITS buckets and *counted*; the signature exposes the
    counts through fixed thermometer thresholds (1/2/4/8 for orders <= 3,
    presence for higher orders). n-grams never cross the begin/end half
    boundary and any n-gram containing the PAD token is skipped. Unit n-grams
    run over the tokenizer-v3 unit stream (-1 = padding) without halving.
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
            bucket = (folded & np.uint64(BLOCK_BITS - 1)).astype(np.int64)
            bucket += group * BLOCK_BITS
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
            bucket = (rotl64(hashes, rot) & np.uint64(BLOCK_BITS - 1)).astype(np.int64)
            bucket += group * BLOCK_BITS
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
        bucket = (folded & np.uint64(BLOCK_BITS - 1)).astype(np.int64)
        bucket += group * BLOCK_BITS
        bucket = np.where(acc_uvalid, bucket, COUNT_BUCKETS)
        indices.append((rows + bucket).reshape(-1))

    flat = np.concatenate(indices)
    counts = np.bincount(flat, minlength=n * stride).reshape(n, stride)
    counts = counts[:, :COUNT_BUCKETS].reshape(n, GROUPS, BLOCK_BITS)

    bits = np.zeros((n, BITS), dtype=np.uint8)
    for plane_index, (group, level) in enumerate(PLANES):
        lo = plane_index * BLOCK_BITS
        bits[:, lo:lo + BLOCK_BITS] = counts[:, group] >= level

    packed = np.packbits(bits, axis=1, bitorder="little")
    return packed.view(np.uint64).reshape(n, WORDS)


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
    tag = f"train.bloomaug{BITS}_v1_s{seed}_n{n_aug}"
    feat_path = cache_dir / f"{tag}.feats.mmap"
    label_path = cache_dir / f"{tag}.labels.mmap"
    if feat_path.exists() and label_path.exists():
        return (np.memmap(feat_path, dtype=np.uint64, mode="r", shape=(n_aug, WORDS)),
                np.memmap(label_path, dtype=np.int64, mode="r", shape=(n_aug,)))
    rng = np.random.default_rng(seed ^ 0xA06)
    src = np.sort(rng.choice(count, size=n_aug, replace=n_aug > count))
    lengths = np.exp(rng.uniform(math.log(AUG_MIN_BYTES), math.log(AUG_MAX_BYTES),
                                 size=n_aug)).astype(np.int64)
    tables = zobrist_tables()
    tmp = feat_path.with_suffix(".tmp")
    out = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(n_aug, WORDS))
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
    return (np.memmap(feat_path, dtype=np.uint64, mode="r", shape=(n_aug, WORDS)),
            np.memmap(label_path, dtype=np.int64, mode="r", shape=(n_aug,)))


def ensure_features(cache_dir: Path, split: str, data: Split, batch: int = 1024) -> np.ndarray:
    """Compute (or load) packed Bloom signatures for a split."""
    tag = f"{split}.bloom{BITS}_v6.mmap"
    path = cache_dir / tag
    if path.exists():
        return np.memmap(path, dtype=np.uint64, mode="r", shape=(data.count, WORDS))
    tables = zobrist_tables()
    tmp = path.with_suffix(".tmp")
    out = np.memmap(tmp, dtype=np.uint64, mode="w+", shape=(data.count, WORDS))
    for start in range(0, data.count, batch):
        stop = min(start + batch, data.count)
        out[start:stop] = encode_windows(
            np.asarray(data.tokens[start:stop]),
            np.asarray(data.units[start:stop]), tables)
        print(f"{split}: encoded {stop}/{data.count}", flush=True)
    out.flush()
    del out
    tmp.rename(path)
    return np.memmap(path, dtype=np.uint64, mode="r", shape=(data.count, WORDS))


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
    """Optional block-diagonal binary hidden layer + binary linear output.

    With hidden units per block H > 0, each of the BLOCKS feature blocks gets H
    binary units connected only to that block's BLOCK_BITS presence bits
    (XNOR/popcount + BN-folded threshold at inference), and the output layer is
    binary over the 2*BLOCKS*H hidden bits. With H == 0 this is a plain binary
    linear head over the raw signature.

    The output layer uses per-(class, block) scales (XNOR-net style): the
    per-block bipolar dot products (computed with XOR + popcount at inference)
    are combined with a small float scale per block, which recovers most of
    the accuracy lost by weight binarization at negligible float cost
    (CLASSES*BLOCKS multiply-adds per file).
    """

    def __init__(self, hidden_per_block: int = 0, float_head: bool = False) -> None:
        super().__init__()
        self.hidden_per_block = hidden_per_block
        self.float_head = float_head
        if hidden_per_block > 0:
            self.hidden_latent = nn.Parameter(
                torch.empty(BLOCKS, BLOCK_BITS, hidden_per_block))
            nn.init.uniform_(self.hidden_latent, -0.1, 0.1)
            hidden = BLOCKS * hidden_per_block
            self.bn = nn.BatchNorm1d(hidden)
            self.latent = nn.Parameter(torch.empty(CLASSES, hidden))
            self.width = hidden
            self.block_width = hidden_per_block
        else:
            self.bn = None
            self.latent = nn.Parameter(torch.empty(CLASSES, BITS))
            self.width = BITS
            self.block_width = BLOCK_BITS
        nn.init.uniform_(self.latent, -0.1, 0.1)
        self.block_scale = nn.Parameter(torch.full((CLASSES, BLOCKS), 4.0))
        self.bias = nn.Parameter(torch.zeros(CLASSES))

    def eff_scale(self) -> torch.Tensor:
        """Effective per-(class, block) scale, fan-in normalized."""
        return self.block_scale / math.sqrt(self.width)

    def features(self, feats: torch.Tensor) -> torch.Tensor:
        if self.hidden_per_block == 0:
            return feats
        blocks = feats.view(-1, BLOCKS, BLOCK_BITS).transpose(0, 1)
        w = sign_ste(self.hidden_latent)
        z = torch.bmm(blocks, w)  # (BLOCKS, n, H)
        z = z.transpose(0, 1).reshape(feats.shape[0], -1)
        return sign_ste(self.bn(z))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.features(feats)
        # Float head is a diagnostic-only upper bound for the binary head.
        w = self.latent if self.float_head else sign_ste(self.latent)
        # Per-block bipolar dot products: (BLOCKS, n, K) @ (BLOCKS, K, CLASSES)
        xb = x.view(-1, BLOCKS, self.block_width).transpose(0, 1)
        wb = w.view(CLASSES, BLOCKS, self.block_width).permute(1, 2, 0)
        d = torch.bmm(xb, wb).permute(1, 2, 0)  # (n, CLASSES, BLOCKS)
        return (d * self.eff_scale().unsqueeze(0)).sum(dim=2) + self.bias


def unpack_batch(packed: np.ndarray) -> torch.Tensor:
    """Packed u64 -> float ±1 tensor of shape (n, BITS)."""
    raw = np.unpackbits(packed.view(np.uint8).reshape(packed.shape[0], BITS // 8),
                        axis=1, bitorder="little")
    bits = raw.astype(np.float32)
    return torch.from_numpy(bits * 2.0 - 1.0)


def reestimate_bn(model: BloomHead, feats: np.ndarray, count: int, batch: int,
                  batches: int, rng: np.random.Generator) -> None:
    """Recompute BN running stats (binary nets drift from the training EMA)."""
    if model.bn is None:
        return
    model.bn.reset_running_stats()
    model.bn.momentum = None
    model.train()
    with torch.no_grad():
        for _ in range(batches):
            idx = np.sort(rng.choice(count, size=min(batch, count), replace=False))
            model.features(unpack_batch(np.asarray(feats[idx])))
    model.bn.momentum = 0.1


def evaluate(model: BloomHead, feats: np.ndarray, data: Split, batch: int) -> tuple[float, float]:
    model.eval()
    correct_teacher = 0
    correct_fs = 0
    with torch.no_grad():
        for start in range(0, data.count, batch):
            stop = min(start + batch, data.count)
            x = unpack_batch(np.asarray(feats[start:stop]))
            pred = model(x).argmax(dim=1).numpy()
            correct_teacher += int((pred == data.labels[start:stop]).sum())
            if data.fs_labels is not None:
                correct_fs += int((pred == data.fs_labels[start:stop]).sum())
    return correct_teacher / data.count, correct_fs / data.count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=2e-3)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--ema-decay", type=float, default=0.998)
    p.add_argument("--soft-loss-weight", type=float, default=0.7)
    p.add_argument("--hard-loss-weight", type=float, default=0.3)
    p.add_argument("--hard-labels", choices=("teacher", "fs"), default="fs",
                   help="hard CE targets: teacher argmax or filesystem labels")
    p.add_argument("--hidden-per-block", type=int, default=32)
    p.add_argument("--augment-fraction", type=float, default=0.0,
                   help="extra short prefix-crop samples as a fraction of train "
                        "count (hard fs labels only)")
    p.add_argument("--float-head", action="store_true",
                   help="diagnostic: float output weights (no export)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    train = open_split(args.cache_dir, "train")
    valid = open_split(args.cache_dir, "valid")
    test = open_split(args.cache_dir, "test")
    train_feats = ensure_features(args.cache_dir, "train", train)
    valid_feats = ensure_features(args.cache_dir, "valid", valid)
    test_feats = ensure_features(args.cache_dir, "test", test)

    count = train.count if args.limit is None else min(args.limit, train.count)
    print(f"train={count} valid={valid.count} test={test.count} bits={BITS}", flush=True)

    marginals = np.asarray(train.marginals[:count])
    if args.hard_labels == "fs" and train.fs_labels is not None:
        labels = np.asarray(train.fs_labels[:count])
    else:
        labels = np.asarray(train.labels[:count])

    aug_feats: np.ndarray | None = None
    aug_labels: np.ndarray | None = None
    n_aug = 0
    if args.augment_fraction > 0:
        aug_feats, aug_labels = ensure_augmented(
            args.cache_dir, train, count, args.augment_fraction, labels, args.seed)
        n_aug = aug_feats.shape[0]
        print(f"augmented={n_aug}", flush=True)
    total = count + n_aug

    model = BloomHead(args.hidden_per_block, args.float_head)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        (no_decay if "latent" in name or "block_scale" in name else decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": 1e-3}],
        lr=args.learning_rate,
    )
    steps_per_epoch = max(1, (count + n_aug) // args.batch_size)
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
            x = unpack_batch(np.concatenate(parts) if len(parts) > 1 else parts[0])
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
        reestimate_bn(model, train_feats, count, args.batch_size, 32, rng)
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
    parity, fs_acc = evaluate(model, test_feats, test, args.batch_size)
    print(f"test_teacher_parity={parity:.6f}", flush=True)
    print(f"test_fs_accuracy={fs_acc:.6f}", flush=True)

    if test.fs_labels is not None:
        preds = []
        model.eval()
        with torch.no_grad():
            for start in range(0, test.count, args.batch_size):
                stop = min(start + args.batch_size, test.count)
                x = unpack_batch(np.asarray(test_feats[start:stop]))
                preds.append(model(x).argmax(dim=1).numpy())
        pred = np.concatenate(preds)
        fs = np.asarray(test.fs_labels)
        recalls = []
        for c in range(CLASSES):
            mask = fs == c
            if mask.any():
                recalls.append(float((pred[mask] == c).mean()))
        print(f"test_fs_macro_recall={np.mean(recalls):.6f}", flush=True)

    if args.float_head:
        return
    export(model, args.output)

    sample = np.asarray(test_feats[:512])
    sim_logits = simulate(model, sample)
    model.eval()
    with torch.no_grad():
        torch_logits = model(unpack_batch(sample)).numpy()
    agree = float((sim_logits.argmax(1) == torch_logits.argmax(1)).mean())
    max_err = float(np.abs(sim_logits - torch_logits).max())
    print(f"simulator: argmax agreement={agree:.4f} max_logit_err={max_err:.4f}", flush=True)


def pack_rows(bits01: np.ndarray) -> bytes:
    """(rows, nbits) 0/1 -> little-endian packed u64 words per row."""
    rows, nbits = bits01.shape
    padded = nbits + (-nbits) % 64
    buf = np.zeros((rows, padded), dtype=np.uint8)
    buf[:, :nbits] = bits01
    return np.packbits(buf, axis=1, bitorder="little").tobytes()


def export(model: BloomHead, output: Path) -> None:
    """MBL3: header, per-block calibration, folded hidden layer, packed output weights."""
    model.eval()
    hidden = model.hidden_per_block
    blob = bytearray(b"MBL3")
    blob += struct.pack("<IIII", BLOCK_BITS, BLOCKS, hidden, CLASSES)
    blob += model.eff_scale().detach().numpy().astype("<f4").tobytes()
    blob += model.bias.detach().numpy().astype("<f4").tobytes()

    if hidden > 0:
        # (BLOCKS, BLOCK_BITS, H) -> unit-major rows (BLOCKS*H, BLOCK_BITS)
        w = (model.hidden_latent.detach().numpy() >= 0).astype(np.uint8)
        rows = w.transpose(0, 2, 1).reshape(BLOCKS * hidden, BLOCK_BITS)
        bn = model.bn
        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        mu = bn.running_mean.detach().numpy()
        sigma = np.sqrt(bn.running_var.detach().numpy() + bn.eps)
        # sign(gamma*(z-mu)/sigma + beta), z = B - 2*mismatch
        tau = mu - beta * sigma / np.where(np.abs(gamma) < 1e-12, 1e-12, gamma)
        thr = np.floor((BLOCK_BITS - tau) / 2.0).astype(np.int64)
        flips = (gamma < 0).astype(np.uint8)
        flip_thr = np.ceil((BLOCK_BITS - tau) / 2.0).astype(np.int64) - 1
        thr = np.clip(np.where(flips == 1, flip_thr, thr), -1, BLOCK_BITS)
        blob += pack_rows(rows)
        blob += thr.astype("<i2").tobytes()
        blob += pack_rows(flips.reshape(1, -1))
        out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
        blob += pack_rows(out_w)
    else:
        out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
        blob += pack_rows(out_w)

    output.write_bytes(bytes(blob))
    print(f"exported {len(blob)} bytes to {output}", flush=True)


def simulate(model: BloomHead, feats: np.ndarray) -> np.ndarray:
    """Pure-integer inference over packed signatures; returns logits."""
    n = feats.shape[0]
    hidden = model.hidden_per_block
    scale = model.eff_scale().detach().numpy().astype(np.float64)
    bias = model.bias.detach().numpy().astype(np.float64)
    x_bits = np.unpackbits(feats.view(np.uint8).reshape(n, BITS // 8), axis=1,
                           bitorder="little").astype(np.uint8)
    if hidden > 0:
        w = (model.hidden_latent.detach().numpy() >= 0).astype(np.uint8)
        rows = w.transpose(0, 2, 1).reshape(BLOCKS * hidden, BLOCK_BITS)
        bn = model.bn
        gamma = bn.weight.detach().numpy()
        beta = bn.bias.detach().numpy()
        mu = bn.running_mean.detach().numpy()
        sigma = np.sqrt(bn.running_var.detach().numpy() + bn.eps)
        tau = mu - beta * sigma / np.where(np.abs(gamma) < 1e-12, 1e-12, gamma)
        thr = np.floor((BLOCK_BITS - tau) / 2.0).astype(np.int64)
        flips = (gamma < 0).astype(np.uint8)
        flip_thr = np.ceil((BLOCK_BITS - tau) / 2.0).astype(np.int64) - 1
        thr = np.clip(np.where(flips == 1, flip_thr, thr), -1, BLOCK_BITS)

        xb = x_bits.reshape(n, BLOCKS, BLOCK_BITS)
        wb = rows.reshape(BLOCKS, hidden, BLOCK_BITS)
        mismatch = np.empty((n, BLOCKS, hidden), dtype=np.int64)
        for b in range(BLOCKS):
            mismatch[:, b] = (xb[:, b, None, :] != wb[None, b]).sum(axis=2)
        bits = (mismatch.reshape(n, -1) <= thr[None, :]).astype(np.uint8)
        bits ^= flips[None, :]
        h_bits = bits
        out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
        block_width = hidden
    else:
        h_bits = x_bits
        out_w = (model.latent.detach().numpy() >= 0).astype(np.uint8)
        block_width = BLOCK_BITS
    hb = h_bits.reshape(n, BLOCKS, block_width)
    wb_out = out_w.reshape(CLASSES, BLOCKS, block_width)
    mismatches = np.empty((n, CLASSES, BLOCKS), dtype=np.int64)
    for b in range(BLOCKS):
        mismatches[:, :, b] = (hb[:, None, b, :] != wb_out[None, :, b, :]).sum(axis=2)
    z = block_width - 2 * mismatches
    return (scale[None, :, :] * z).sum(axis=2) + bias[None, :]


if __name__ == "__main__":
    main()
