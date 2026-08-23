#!/usr/bin/env python3
"""Exact batched evaluation of exported v2 integer semantics.

Per-basis convs on {-1,+1} with +/-1 weights are exact in f32 (|z'| <= 768);
the fixed-point combine and thresholds run in f64 on CPU (exact for i32
range). Supports single models and the ensemble (sum of logits)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import CLASSES, OUT
from train_bnn2 import C, build_plane_cache
from export_bnn2 import IntModel, load_models

import os

DEV = "cpu" if os.environ.get("EVAL_CPU") else (
    "mps" if torch.backends.mps.is_available() else "cpu")


class FastInt:
    def __init__(self, im: IntModel):
        self.segs = im.segs
        self.layers = []
        for layer in (im.l0, im.l1, im.l2):
            m, cout, _ = layer.w.shape
            w = torch.from_numpy(layer.w.astype(np.float32) * 2 - 1)
            w = w.reshape(m, cout, layer.k, layer.cin).permute(0, 1, 3, 2).contiguous()
            self.layers.append({
                "w": w.to(DEV),                              # [m, cout, cin, k]
                "aq": torch.from_numpy(layer.aq.astype(np.float64)),  # [m, cout]
                "sg": torch.from_numpy(layer.sg.astype(np.int64)),
                "th": torch.from_numpy(layer.th.astype(np.float64)),
                "stride": layer.stride, "pad": layer.pad,
            })
        self.hq = torch.from_numpy(im.hq.astype(np.float64))
        self.hs = torch.from_numpy(im.hs.astype(np.float64))
        self.hb = torch.from_numpy(im.hb_.astype(np.float64))

    def conv_fire(self, x, layer):
        # x: [B, cin, L] in {-1,+1} f32 on DEV
        zq = None
        for j in range(layer["w"].shape[0]):
            zj = F.conv1d(x, layer["w"][j], stride=layer["stride"],
                          padding=layer["pad"]).cpu().double()
            term = layer["aq"][j].view(1, -1, 1) * zj
            zq = term if zq is None else zq + term
        fire = torch.where(layer["sg"].view(1, -1, 1) == 1,
                           zq >= layer["th"].view(1, -1, 1),
                           zq <= layer["th"].view(1, -1, 1))
        return fire.float().to(DEV)

    def logits(self, x):  # x: [B, 2, BITS] {-1,+1}
        h = self.conv_fire(x, self.layers[0])
        h = F.max_pool1d(h, 4)
        h = self.conv_fire(h * 2 - 1, self.layers[1])
        h = F.max_pool1d(h, 4)
        h = self.conv_fire(h * 2 - 1, self.layers[2])
        segs = self.segs
        seg = h.shape[2] // segs
        counts = torch.cat([h[:, :, i * seg:(i + 1) * seg].sum(2)
                            for i in range(segs)], 1).cpu().double()
        return counts @ self.hq.T * self.hs[None, :] + self.hb[None, :]


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    models = [FastInt(IntModel(m, w)) for m, w in load_models()]
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    bits, labels, _ = build_plane_cache(split, codes)
    n = len(labels)
    batch = 128
    all_logits = np.zeros((len(models), n, CLASSES))
    with torch.no_grad():
        for i in range(0, n, batch):
            idx = np.arange(i, min(i + batch, n))
            xb = np.unpackbits(np.asarray(bits[idx]), axis=2).astype(np.float32) * 2 - 1
            x = torch.from_numpy(xb).to(DEV)
            for mi, fm in enumerate(models):
                all_logits[mi, idx] = fm.logits(x).numpy()

    def report(name, logits):
        pred = logits.argmax(1)
        correct = np.zeros(CLASSES); total = np.zeros(CLASSES)
        for c in range(CLASSES):
            m = labels == c
            total[c] = m.sum()
            correct[c] = (pred[m] == c).sum()
        acc = correct.sum() / total.sum()
        macro = (correct[total > 0] / total[total > 0]).mean()
        print(f"{split} {name}: acc {acc:.6f} macro {macro:.6f} n={n}")

    for mi in range(len(models)):
        report(f"model{mi}", all_logits[mi])
    if len(models) > 1:
        report("ensemble", all_logits.sum(0))
    np.save(OUT / f"{split}.intlogits.npy", all_logits)


if __name__ == "__main__":
    main()
