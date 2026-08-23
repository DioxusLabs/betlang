#!/usr/bin/env python3
"""Phase-2 fine-tune: upgrade head from 2 to 8 pooled count segments
(function-preserving warm start), continue training at low LR with EMA."""
from __future__ import annotations

import copy
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
from train_bnn import OUT, CLASSES, SEED
from train_bnn2 import BNN2, BinConvM, C, build_plane_cache, unpack_batch
from finetune_bnn2 import Ema, upgrade_conv

SEGS = 8


class BNN2Seg(nn.Module):
    def __init__(self):
        super().__init__()
        base = BNN2()
        base.conv1 = upgrade_conv(base.conv1)
        base.conv2 = upgrade_conv(base.conv2)
        self.stem = base.stem
        self.bn0 = base.bn0
        self.s0 = base.s0
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.s1 = base.s1
        self.conv2 = base.conv2
        self.bn2 = base.bn2
        self.s2 = base.s2
        self.hb = nn.BatchNorm1d(C * SEGS)
        self.fc = nn.Linear(C * SEGS, CLASSES)

    def load_from_ft(self, sd):
        own = self.state_dict()
        for k, v in sd.items():
            if k.startswith(("hb.", "fc.")):
                continue
            own[k].copy_(v)
        # warm-start head: old halves -> replicate over 4 segments each
        rep = SEGS // 2
        with torch.no_grad():
            self.fc.weight.copy_(sd["fc.weight"].repeat_interleave(1, 0)
                                 .reshape(CLASSES, 2, C).repeat_interleave(rep, 1)
                                 .reshape(CLASSES, C * SEGS) / rep)
            self.fc.bias.copy_(sd["fc.bias"])
            for name in ("weight", "bias"):
                getattr(self.hb, name).copy_(
                    sd[f"hb.{name}"].reshape(2, C).repeat_interleave(rep, 0).reshape(-1))
            self.hb.running_mean.copy_(
                sd["hb.running_mean"].reshape(2, C).repeat_interleave(rep, 0).reshape(-1) / rep)
            self.hb.running_var.copy_(
                sd["hb.running_var"].reshape(2, C).repeat_interleave(rep, 0).reshape(-1) / (rep * rep))

    def forward(self, x):
        h = self.s0(self.bn0(self.stem(x)))
        h = F.max_pool1d(h, 4)
        h = self.s1(self.bn1(self.conv1(h)))
        h = F.max_pool1d(h, 4)
        h = self.s2(self.bn2(self.conv2(h)))
        seg = h.shape[2] // SEGS
        feats = torch.cat([(h[:, :, i * seg:(i + 1) * seg] + 1).sum(2)
                           for i in range(SEGS)], 1) / 2
        return self.fc(self.hb(feats))


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
    torch.manual_seed(SEED + 2)
    np.random.seed(SEED + 2)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    tr_bits, tr_labels, tr_teacher = build_plane_cache("train", codes)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)

    model = BNN2Seg()
    model.load_from_ft(torch.load(OUT / "best2ft.pt", map_location="cpu"))
    model = model.to(device)
    va = evaluate(model, va_bits, va_labels, device)
    print(f"warm-start valid: {va[0]:.4f}/{va[1]:.4f}", flush=True)
    ema = Ema(model)

    epochs = 12
    batch = 64
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-4, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=2e-5)
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
            ema.update(model)
            tot_loss += float(loss.detach())
        sched.step()
        acc, macro = evaluate(model, va_bits, va_labels, device)
        eacc, emacro = evaluate(ema.shadow, va_bits, va_labels, device)
        dt = time.time() - t0
        print(f"seg {epoch}: loss {tot_loss / (n // batch):.4f} "
              f"acc {acc:.4f}/{macro:.4f} ema {eacc:.4f}/{emacro:.4f} ({dt:.0f}s)",
              flush=True)
        for tag, a, m_ in (("raw", acc, model), ("ema", eacc, ema.shadow)):
            if a > best:
                best = a
                torch.save(m_.state_dict(), OUT / "best2seg.pt")
                print(f"  saved {tag} {a:.4f}", flush=True)
    print(f"best seg valid acc {best:.4f}")


if __name__ == "__main__":
    main()
