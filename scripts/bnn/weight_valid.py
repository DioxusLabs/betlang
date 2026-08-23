#!/usr/bin/env python3
"""Score fixture/edge-passing weightings on full-valid and short-valid."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import OUT, load_split
from train_bnn2 import build_plane_cache
from finetune_short import build_short_valid
from eval_combos import all_logits, metrics
from train_bnn2 import BNN2
from finetune_seg import BNN2Seg
from weight_search import CKPTS

device = "mps" if torch.backends.mps.is_available() else "cpu"


def main():
    passing = [tuple(map(float, line.strip("()\n").split(",")))
               for line in Path(sys.argv[1]).read_text().splitlines()
               if line.startswith("(")]
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    va_bits, va_labels, _ = build_plane_cache("valid", codes)
    va_windows, va_lengths, _, _ = load_split("valid")
    va_short = build_short_valid(codes, va_windows, va_lengths)
    n = len(va_labels)

    models = []
    for ck in CKPTS:
        cls = BNN2Seg if "2seg" in ck else BNN2
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.to(device)
        models.append(m)
    lf = np.stack([all_logits(m, va_bits, n) for m in models])
    ls = np.stack([all_logits(m, va_short, n) for m in models])

    results = []
    for ws in passing:
        w = np.array(ws)[:, None, None]
        fa, fm = metrics((lf * w).sum(0), va_labels)
        sa, sm = metrics((ls * w).sum(0), va_labels)
        results.append((fa + fm + 0.5 * (sa + sm), ws, fa, fm, sa, sm))
    results.sort(reverse=True)
    for score, ws, fa, fm, sa, sm in results[:15]:
        print(f"{ws} full {fa:.4f}/{fm:.4f} short {sa:.4f}/{sm:.4f}")


if __name__ == "__main__":
    main()
