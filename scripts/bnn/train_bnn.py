#!/usr/bin/env python3
"""Binary CNN (XNOR/popcount-compatible) over Shannon-coded raw window bytes.

Pipeline: raw 2048-byte Magika window -> canonical Shannon code (lengths from
train byte histogram) -> fixed 16384-bit input -> binary conv stack -> integer
popcount logits. All inference-time ops are binary/integer: XNOR + popcount +
integer thresholds (folded BN) + integer classifier bias.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from shannon import code_lengths, canonical_codes, encode_bits_bulk

CACHE = Path.home() / "work/cache"
OUT = Path.home() / "work/bnn"
BITS = 16384
WINDOW = 2048
CLASSES = 48
STEM_K = 32
STEM_STRIDE = 4
C = 192
SEED = 2


def load_split(split: str):
    meta = json.loads((CACHE / f"{split}.json").read_text())
    n = meta["count"]
    windows = np.memmap(CACHE / f"{split}.windows.mmap", dtype=np.uint8, mode="r")
    windows = windows.reshape(-1, WINDOW)[:n]
    lengths = np.memmap(CACHE / f"{split}.lengths.mmap", dtype=np.int32, mode="r")[:n]
    labels = np.memmap(CACHE / f"{split}.labels.mmap", dtype=np.int16, mode="r")[:n]
    teacher = np.memmap(CACHE / f"{split}.teacher.mmap", dtype=np.float32, mode="r")
    teacher = teacher.reshape(-1, CLASSES)[:n]
    return windows, lengths, labels, teacher


def build_bit_cache(split: str, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    packed_path = OUT / f"{split}.bits.npy"
    windows, lengths, labels, teacher = load_split(split)
    if packed_path.exists():
        packed = np.load(packed_path, mmap_mode="r")
        return packed, np.asarray(labels), np.asarray(teacher)
    n = len(lengths)
    packed = np.zeros((n, BITS // 8), dtype=np.uint8)
    step = 2048
    for i in range(0, n, step):
        j = min(i + step, n)
        bits = encode_bits_bulk(np.asarray(windows[i:j]), np.asarray(lengths[i:j]),
                                codes, BITS)
        packed[i:j] = np.packbits(bits, axis=1)
        print(f"{split} bits: {j}/{n}", flush=True)
    np.save(packed_path, packed)
    return np.load(packed_path, mmap_mode="r"), np.asarray(labels), np.asarray(teacher)


class SignSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * (x.abs() <= 1).float()


def binarize(x):
    return SignSTE.apply(x)


class RSign(nn.Module):
    """Learnable-threshold sign; sparse init mimics ReLU pattern detectors.
    Exports as: integer accumulator >= integer threshold."""

    def __init__(self, c, bias=1.5):
        super().__init__()
        self.b = nn.Parameter(torch.full((1, c, 1), bias))

    def forward(self, z):
        return binarize(z - self.b)


class MaybeBinConv1d(nn.Module):
    """Float conv whose weights are progressively annealed to binary
    (XNOR-Net scale folds into the following BN threshold at export)."""

    def __init__(self, cin, cout, k, stride=1, padding=0):
        super().__init__()
        conv = nn.Conv1d(cin, cout, k, stride=stride, padding=padding, bias=False)
        self.weight = conv.weight
        self.stride = stride
        self.padding = padding
        self.binw = 0.0

    def forward(self, x):
        w = self.weight
        if self.binw > 0:
            wb = binarize(w) * w.abs().mean(dim=(1, 2), keepdim=True)
            w = (1 - self.binw) * w + self.binw * wb
        return F.conv1d(x, w, stride=self.stride, padding=self.padding)


class BNN(nn.Module):
    """Binary CNN over the Shannon bitstream.

    Inference graph is entirely bitwise/integer: XNOR+popcount convs,
    integer thresholds (folded BN+RSign), OR-pooling (word-OR), popcount
    head counts, and a fixed-point integer classifier."""

    def __init__(self):
        super().__init__()
        self.stem = MaybeBinConv1d(1, C, STEM_K, stride=STEM_STRIDE,
                                   padding=STEM_K // 2)
        self.bn0 = nn.BatchNorm1d(C)
        self.s0 = RSign(C)
        self.conv1 = MaybeBinConv1d(C, C, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(C)
        self.s1 = RSign(C)
        self.conv2 = MaybeBinConv1d(C, C, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(C)
        self.s2 = RSign(C)
        self.hb = nn.BatchNorm1d(C * 2)
        self.fc = nn.Linear(C * 2, CLASSES)

    def set_binw(self, v: float):
        for m in (self.stem, self.conv1, self.conv2):
            m.binw = v

    def forward(self, x):  # x: [B, 1, BITS] in {-1,+1}
        h = self.s0(self.bn0(self.stem(x)))           # [B, C, 4096]
        h = F.max_pool1d(h, 4)                        # OR-pool -> 1024
        h = self.s1(self.bn1(self.conv1(h)))
        h = F.max_pool1d(h, 4)                        # 256
        h = self.s2(self.bn2(self.conv2(h)))          # 256
        half = h.shape[2] // 2
        feats = torch.cat([(h[:, :, :half] + 1).sum(2),
                           (h[:, :, half:] + 1).sum(2)], 1) / 2  # [B, 2C] ints
        return self.fc(self.hb(feats))


def unpack_batch(packed: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    bits = np.unpackbits(packed[idx], axis=1)
    return torch.from_numpy(bits.astype(np.float32) * 2 - 1).unsqueeze(1)


def evaluate(model, packed, labels, device, batch=256):
    model.eval()
    correct = np.zeros(CLASSES); total = np.zeros(CLASSES)
    with torch.no_grad():
        for i in range(0, len(labels), batch):
            idx = np.arange(i, min(i + batch, len(labels)))
            x = unpack_batch(packed, idx).to(device)
            pred = model(x).argmax(1).cpu().numpy()
            lab = labels[idx]
            for c in range(CLASSES):
                m = lab == c
                total[c] += m.sum()
                correct[c] += (pred[m] == c).sum()
    acc = correct.sum() / max(total.sum(), 1)
    recalls = correct[total > 0] / total[total > 0]
    return acc, recalls.mean()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # Shannon codebook from train byte histogram
    cb_path = OUT / "codebook.json"
    if cb_path.exists():
        lengths256 = np.array(json.loads(cb_path.read_text())["lengths"], dtype=np.int32)
    else:
        windows, lengths, _, _ = load_split("train")
        hist = np.zeros(256, dtype=np.int64)
        step = 4096
        for i in range(0, len(lengths), step):
            j = min(i + step, len(lengths))
            w = np.asarray(windows[i:j])
            mask = np.arange(WINDOW)[None, :] < np.asarray(lengths[i:j])[:, None]
            hist += np.bincount(w[mask], minlength=256)
        lengths256 = code_lengths(hist)
        cb_path.write_text(json.dumps({"lengths": lengths256.tolist(),
                                       "hist": hist.tolist()}))
    codes = canonical_codes(lengths256)
    print("mean code bits/byte:", flush=True)

    tr_bits, tr_labels, tr_teacher = build_bit_cache("train", codes)
    va_bits, va_labels, _ = build_bit_cache("valid", codes)

    model = BNN().to(device)
    float_epochs = 24
    anneal_epochs = 10
    bin_epochs = 14
    epochs = float_epochs + anneal_epochs + bin_epochs
    batch = 96
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=5e-5)
    n = len(tr_labels)
    best = 0.0
    tr_teacher_t = torch.from_numpy(np.asarray(tr_teacher))
    tr_labels_t = torch.from_numpy(np.asarray(tr_labels).astype(np.int64))

    for epoch in range(epochs):
        if epoch < float_epochs:
            model.set_binw(0.0)
        elif epoch < float_epochs + anneal_epochs:
            model.set_binw((epoch - float_epochs + 1) / anneal_epochs)
        else:
            model.set_binw(1.0)
        model.train()
        perm = np.random.permutation(n)
        t0 = time.time()
        tot_loss = 0.0
        for i in range(0, n - batch + 1, batch):
            sidx = np.sort(perm[i:i + batch])
            x = unpack_batch(tr_bits, sidx).to(device)
            y = tr_labels_t[sidx].to(device)
            t = tr_teacher_t[sidx].to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            logt = torch.log(t.clamp_min(1e-9))
            kl = F.kl_div(F.log_softmax(logits / 1.5, dim=1),
                          F.softmax(logt / 1.5, dim=1), reduction="batchmean")
            loss = 0.5 * ce + 0.5 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += float(loss.detach())
        sched.step()
        acc, macro = evaluate(model, va_bits, va_labels, device)
        dt = time.time() - t0
        binw = model.stem.binw
        print(f"epoch {epoch}: binw {binw:.2f} loss {tot_loss / (n // batch):.4f} "
              f"valid acc {acc:.4f} macro {macro:.4f} ({dt:.0f}s)", flush=True)
        if binw >= 1.0 and acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "best.pt")
    print(f"best fully-binary valid acc {best:.4f}")


if __name__ == "__main__":
    main()
