#!/usr/bin/env python3
"""Short-file robustness fine-tune: continue training best2seg + best3 with
random-truncation augmentation (simulates small source files, which the
corpus underrepresents but betlang's fixture suite exercises heavily)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes, encode_planes_bulk
from train_bnn import OUT, BITS, CLASSES, load_split
from train_bnn2 import BNN2, build_plane_cache, unpack_batch
from finetune_seg import BNN2Seg

AUG_P = 0.4
LMIN, LMAX = 16, 1024


def build_short_valid(codes, windows, lengths):
    path = OUT / "valid.short.planes.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    n = len(lengths)
    rng = np.random.default_rng(123)
    Ls = np.exp(rng.uniform(np.log(LMIN), np.log(LMAX), n)).astype(np.int64)
    Ls = np.maximum(np.minimum(Ls, np.asarray(lengths)), 8)
    packed = np.zeros((n, 2, BITS // 8), dtype=np.uint8)
    step = 2048
    for i in range(0, n, step):
        j = min(i + step, n)
        planes = encode_planes_bulk(np.asarray(windows[i:j]), Ls[i:j], codes, BITS)
        packed[i:j] = np.packbits(planes, axis=2)
    np.save(path, packed)
    return np.load(path, mmap_mode="r")


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


def finetune(tag, model, ckpt_out, codes, device, epochs=3, lr=3e-4):
    tr_bits, tr_labels, tr_teacher = build_plane_cache("train", codes)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)
    tr_windows, tr_lengths, _, _ = load_split("train")
    va_windows, va_lengths, _, _ = load_split("valid")
    va_short = build_short_valid(codes, va_windows, va_lengths)

    acc0, mac0 = evaluate(model, va_bits, va_labels, device)
    sacc0, smac0 = evaluate(model, va_short, va_labels, device)
    print(f"{tag} start: full {acc0:.4f}/{mac0:.4f} short {sacc0:.4f}/{smac0:.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=3e-5)
    tr_teacher_t = torch.from_numpy(np.asarray(tr_teacher))
    tr_labels_t = torch.from_numpy(np.asarray(tr_labels).astype(np.int64))
    n = len(tr_labels)
    batch = 64
    rng = np.random.default_rng(hash(tag) % (2 ** 31))
    best = acc0 * 0.5 + sacc0 * 0.5

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        t0 = time.time()
        tot = 0.0
        for i in range(0, n - batch + 1, batch):
            sidx = np.sort(perm[i:i + batch])
            bits = np.unpackbits(tr_bits[sidx], axis=2).astype(np.float32)
            aug = rng.random(batch) < AUG_P
            if aug.any():
                widx = sidx[aug]
                Ls = np.exp(rng.uniform(np.log(LMIN), np.log(LMAX),
                                        aug.sum())).astype(np.int64)
                Ls = np.maximum(np.minimum(
                    Ls, np.asarray(tr_lengths)[widx].astype(np.int64)), 8)
                planes = encode_planes_bulk(np.asarray(tr_windows[widx]),
                                            Ls, codes, BITS)
                bits[aug] = planes.astype(np.float32)
            x = torch.from_numpy(bits * 2 - 1).to(device)
            y = tr_labels_t[sidx].to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            keep = torch.from_numpy(~aug).to(device)
            if keep.any():
                t = tr_teacher_t[sidx].to(device)[keep]
                logt = torch.log(t.clamp_min(1e-9))
                kl = F.kl_div(F.log_softmax(logits[keep] / 1.5, dim=1),
                              F.softmax(logt / 1.5, dim=1),
                              reduction="batchmean")
            else:
                kl = torch.zeros((), device=device)
            loss = 0.8 * ce + 0.2 * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
        sched.step()
        acc, mac = evaluate(model, va_bits, va_labels, device)
        sacc, smac = evaluate(model, va_short, va_labels, device)
        score = acc * 0.5 + sacc * 0.5
        dt = time.time() - t0
        print(f"{tag} ep{epoch}: loss {tot / (n // batch):.4f} "
              f"full {acc:.4f}/{mac:.4f} short {sacc:.4f}/{smac:.4f} "
              f"({dt:.0f}s)", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt_out)
            print(f"{tag} saved (score {score:.4f})", flush=True)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)

    resume = len(sys.argv) > 1 and sys.argv[1] == "resume"
    m0 = BNN2Seg()
    m0.load_state_dict(torch.load(
        OUT / ("best2seg_short.pt" if resume else "best2seg.pt"),
        map_location="cpu"))
    m0.to(device)
    finetune("m0", m0, OUT / "best2seg_short.pt", codes, device,
             epochs=5 if resume else 3, lr=1.5e-4 if resume else 3e-4)

    m1 = BNN2()
    m1.load_state_dict(torch.load(
        OUT / ("best3_short.pt" if resume else "best3.pt"),
        map_location="cpu"))
    m1.to(device)
    finetune("m1", m1, OUT / "best3_short.pt", codes, device,
             epochs=5 if resume else 3, lr=1.5e-4 if resume else 3e-4)


if __name__ == "__main__":
    main()
