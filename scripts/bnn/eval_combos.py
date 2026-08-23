#!/usr/bin/env python3
"""Compare ensemble combinations on full-valid and short-valid splits."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import OUT, CLASSES, load_split
from train_bnn2 import BNN2, build_plane_cache, unpack_batch
from finetune_seg import BNN2Seg
from finetune_short import build_short_valid

device = "mps" if torch.backends.mps.is_available() else "cpu"


def all_logits(model, packed, n, batch=128):
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n, batch):
            idx = np.arange(i, min(i + batch, n))
            x = unpack_batch(packed, idx).to(device)
            outs.append(model(x).cpu().numpy())
    return np.concatenate(outs)


def metrics(logits, labels):
    pred = logits.argmax(1)
    acc = (pred == labels).mean()
    rec = []
    for c in range(CLASSES):
        m = labels == c
        if m.sum():
            rec.append((pred[m] == c).mean())
    return acc, float(np.mean(rec))


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)
    va_windows, va_lengths, _, _ = load_split("valid")
    va_short = build_short_valid(codes, va_windows, va_lengths)
    n = len(va_labels)

    if len(sys.argv) > 1:
        specs = [(BNN2Seg if "2seg" in ck else BNN2, ck) for ck in sys.argv[1:]]
        names = [f"m{i}" for i in range(len(specs))]
    else:
        specs = [(BNN2Seg, "best2seg.pt"), (BNN2, "best3.pt"),
                 (BNN2Seg, "best2seg_short.pt"), (BNN2, "best3_short.pt")]
        names = ["m0", "m1", "m0s", "m1s"]
    models = []
    for cls, ck in specs:
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.to(device)
        models.append(m)

    lf = [all_logits(m, va_bits, n) for m in models]
    ls = [all_logits(m, va_short, n) for m in models]
    k = len(models)
    for r in range(1, k + 1):
        for combo in itertools.combinations(range(k), r):
            f = sum(lf[i] for i in combo)
            s = sum(ls[i] for i in combo)
            fa, fm = metrics(f, va_labels)
            sa, sm = metrics(s, va_labels)
            print(f"{'+'.join(names[i] for i in combo):16s} "
                  f"full {fa:.4f}/{fm:.4f} short {sa:.4f}/{sm:.4f}", flush=True)


if __name__ == "__main__":
    main()
