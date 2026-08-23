#!/usr/bin/env python3
"""Complementary binary model for ensembling: proven BNN2 architecture,
different seed, grad clipping, CE-heavy final epochs."""
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
from shannon import canonical_codes
from train_bnn import OUT, CLASSES
from train_bnn2 import BNN2, build_plane_cache, unpack_batch

SEED3 = 7


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
    torch.manual_seed(SEED3)
    np.random.seed(SEED3)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    tr_bits, tr_labels, tr_teacher = build_plane_cache("train", codes)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)

    model = BNN2().to(device)
    epochs = 40
    batch = 64
    start_epoch = 0
    resume = OUT / "best3.pt"
    if resume.exists():
        model.load_state_dict(torch.load(resume, map_location=device))
        start_epoch = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=3e-5)
    for _ in range(start_epoch):
        sched.step()
    n = len(tr_labels)
    best = 0.0
    if start_epoch:
        acc, macro = evaluate(model, va_bits, va_labels, device)
        best = acc
        print(f"m3 resume @ {start_epoch}: valid acc {acc:.4f} macro {macro:.4f}",
              flush=True)
    tr_teacher_t = torch.from_numpy(np.asarray(tr_teacher))
    tr_labels_t = torch.from_numpy(np.asarray(tr_labels).astype(np.int64))

    for epoch in range(start_epoch, epochs):
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
            w = 0.5 if epoch < 30 else 0.1  # CE-heavy at the end (teacher ceiling)
            loss = (1 - w) * ce + w * kl if epoch >= 30 else 0.5 * ce + 0.5 * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot_loss += float(loss.detach())
        sched.step()
        acc, macro = evaluate(model, va_bits, va_labels, device)
        dt = time.time() - t0
        print(f"m3 {epoch}: loss {tot_loss / (n // batch):.4f} "
              f"valid acc {acc:.4f} macro {macro:.4f} ({dt:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save(model.state_dict(), OUT / "best3.pt")
    print(f"best m3 valid acc {best:.4f}")


if __name__ == "__main__":
    main()
