#!/usr/bin/env python3
"""Fine-tune with synthetic hard-pair samples (repo hard_gen_* style) mixed
into short-truncation augmented batches."""
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
from train_bnn import OUT, BITS, load_split
from train_bnn2 import BNN2, build_plane_cache
from finetune_seg import BNN2Seg
from finetune_short import build_short_valid, evaluate

AUG_P = 0.35
SYNTH_PER_BATCH = 16
LMIN, LMAX = 16, 1024


def finetune(tag, model, ckpt_out, codes, device, epochs=3, lr=1.2e-4):
    tr_bits, tr_labels, tr_teacher = build_plane_cache("train", codes)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)
    tr_windows, tr_lengths, _, _ = load_split("train")
    va_windows, va_lengths, _, _ = load_split("valid")
    va_short = build_short_valid(codes, va_windows, va_lengths)

    sw = np.load(OUT / "synth_hard.windows.npy")
    sl = np.load(OUT / "synth_hard.lengths.npy")
    sy = np.load(OUT / "synth_hard.labels.npy")
    print(f"{tag}: {len(sy)} synth rows", flush=True)

    acc0, mac0 = evaluate(model, va_bits, va_labels, device)
    sacc0, smac0 = evaluate(model, va_short, va_labels, device)
    print(f"{tag} start: full {acc0:.4f}/{mac0:.4f} short {sacc0:.4f}/{smac0:.4f}",
          flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-5)
    tr_teacher_t = torch.from_numpy(np.asarray(tr_teacher).copy())
    tr_labels_np = np.asarray(tr_labels).astype(np.int64)
    n = len(tr_labels)
    batch = 48
    rng = np.random.default_rng((hash(tag) + 7) % (2 ** 31))

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        t0 = time.time()
        tot = 0.0
        nb = 0
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
            sidx2 = rng.choice(len(sy), SYNTH_PER_BATCH, replace=False)
            splanes = encode_planes_bulk(sw[sidx2], sl[sidx2].astype(np.int64),
                                         codes, BITS).astype(np.float32)
            x = torch.from_numpy(
                np.concatenate([bits, splanes]) * 2 - 1).to(device)
            y = torch.from_numpy(
                np.concatenate([tr_labels_np[sidx], sy[sidx2]])).to(device)
            logits = model(x)
            ce = F.cross_entropy(logits, y, label_smoothing=0.05)
            keep = torch.from_numpy(
                np.concatenate([~aug, np.zeros(SYNTH_PER_BATCH, bool)])).to(device)
            if keep.any():
                t = tr_teacher_t[torch.from_numpy(sidx)].to(device)
                t = t[torch.from_numpy(~aug).to(device)]
                logt = torch.log(t.clamp_min(1e-9))
                kl = F.kl_div(F.log_softmax(logits[keep] / 1.5, dim=1),
                              F.softmax(logt / 1.5, dim=1),
                              reduction="batchmean")
            else:
                kl = torch.zeros((), device=device)
            loss = 0.85 * ce + 0.15 * kl
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        acc, mac = evaluate(model, va_bits, va_labels, device)
        sacc, smac = evaluate(model, va_short, va_labels, device)
        dt = time.time() - t0
        print(f"{tag} ep{epoch}: loss {tot / nb:.4f} "
              f"full {acc:.4f}/{mac:.4f} short {sacc:.4f}/{smac:.4f} "
              f"({dt:.0f}s)", flush=True)
        torch.save(model.state_dict(), ckpt_out.with_suffix(f".ep{epoch}.pt"))


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)

    m0 = BNN2Seg()
    m0.load_state_dict(torch.load(OUT / "best2seg_short.pt", map_location="cpu"))
    m0.to(device)
    finetune("m0s2", m0, OUT / "best2seg_synth.pt", codes, device)

    m1 = BNN2()
    m1.load_state_dict(torch.load(OUT / "best3_short.pt", map_location="cpu"))
    m1.to(device)
    finetune("m1s2", m1, OUT / "best3_synth.pt", codes, device)


if __name__ == "__main__":
    main()
