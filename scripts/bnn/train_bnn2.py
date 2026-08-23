#!/usr/bin/env python3
"""Binary CNN v2 over Shannon-coded raw window bytes.

Changes vs v1: binary weights from epoch 0 (no annealing shock), two input
bitplanes (code bits + codeword-boundary markers), multi-basis binary stem
(ABC-Net style: sum of M XNOR-popcount branches with per-channel fixed-point
scales), wider C. Inference remains entirely bitwise/integer.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes, encode_planes_bulk
from train_bnn import CACHE, OUT, BITS, WINDOW, CLASSES, SEED, load_split

STEM_K = 32
STEM_STRIDE = 4
C = 256
STEM_M = 3


def build_plane_cache(split: str, codes: np.ndarray):
    packed_path = OUT / f"{split}.planes.npy"
    windows, lengths, labels, teacher = load_split(split)
    if packed_path.exists():
        packed = np.load(packed_path, mmap_mode="r")
        return packed, np.asarray(labels), np.asarray(teacher)
    n = len(lengths)
    packed = np.zeros((n, 2, BITS // 8), dtype=np.uint8)
    step = 2048
    for i in range(0, n, step):
        j = min(i + step, n)
        planes = encode_planes_bulk(np.asarray(windows[i:j]),
                                    np.asarray(lengths[i:j]), codes, BITS)
        packed[i:j] = np.packbits(planes, axis=2)
        print(f"{split} planes: {j}/{n}", flush=True)
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
    def __init__(self, c, bias=1.5):
        super().__init__()
        self.b = nn.Parameter(torch.full((1, c, 1), bias))

    def forward(self, z):
        return binarize(z - self.b)


class BinConvM(nn.Module):
    """M-basis binary conv: z = sum_m alpha_m * conv(x, sign(W_m)).
    Exports as M XNOR-popcount passes + fixed-point per-channel combine."""

    def __init__(self, cin, cout, k, m=1, stride=1, padding=0):
        super().__init__()
        self.latent = nn.Parameter(torch.empty(m, cout, cin, k))
        nn.init.kaiming_normal_(self.latent.view(m * cout, cin, k))
        shifts = torch.linspace(-0.6, 0.6, m).view(m, 1) if m > 1 else torch.zeros(1, 1)
        self.shift = nn.Parameter(shifts.clone())  # [m, 1] per-basis latent shift
        self.alpha = nn.Parameter(torch.full((m, cout), 1.0 / m))
        self.stride = stride
        self.padding = padding
        self.m = m
        self.cout = cout

    def forward(self, x):
        m, cout, cin, k = self.latent.shape
        std = self.latent.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        wb = binarize(self.latent - self.shift.view(m, 1, 1, 1) * std)
        w = (wb * self.alpha.view(m, cout, 1, 1)).sum(0)
        return F.conv1d(x, w, stride=self.stride, padding=self.padding)


class BNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = BinConvM(2, C, STEM_K, m=STEM_M, stride=STEM_STRIDE,
                             padding=STEM_K // 2)
        self.bn0 = nn.BatchNorm1d(C)
        self.s0 = RSign(C)
        self.conv1 = BinConvM(C, C, 3, m=1, padding=1)
        self.bn1 = nn.BatchNorm1d(C)
        self.s1 = RSign(C)
        self.conv2 = BinConvM(C, C, 3, m=1, padding=1)
        self.bn2 = nn.BatchNorm1d(C)
        self.s2 = RSign(C)
        self.hb = nn.BatchNorm1d(C * 2)
        self.fc = nn.Linear(C * 2, CLASSES)

    def forward(self, x):  # x: [B, 2, BITS] in {-1,+1}
        h = self.s0(self.bn0(self.stem(x)))
        h = F.max_pool1d(h, 4)
        h = self.s1(self.bn1(self.conv1(h)))
        h = F.max_pool1d(h, 4)
        h = self.s2(self.bn2(self.conv2(h)))
        half = h.shape[2] // 2
        feats = torch.cat([(h[:, :, :half] + 1).sum(2),
                           (h[:, :, half:] + 1).sum(2)], 1) / 2
        return self.fc(self.hb(feats))


def unpack_batch(packed: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    bits = np.unpackbits(packed[idx], axis=2)
    return torch.from_numpy(bits.astype(np.float32) * 2 - 1)


def evaluate(model, packed, labels, device, batch=128):
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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    tr_bits, tr_labels, tr_teacher = build_plane_cache("train", codes)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)

    model = BNN2().to(device)
    epochs = 40
    batch = 64
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=5e-5)
    n = len(tr_labels)
    best = 0.0
    tr_teacher_t = torch.from_numpy(np.asarray(tr_teacher))
    tr_labels_t = torch.from_numpy(np.asarray(tr_labels).astype(np.int64))

    for epoch in range(epochs):
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
        print(f"epoch {epoch}: loss {tot_loss / (n // batch):.4f} "
              f"valid acc {acc:.4f} macro {macro:.4f} ({dt:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "best2.pt")
    print(f"best fully-binary valid acc {best:.4f}")


if __name__ == "__main__":
    main()
