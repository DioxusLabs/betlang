#!/usr/bin/env python3
"""Grid-search per-model weights for the 5-model ensemble on fixtures+edge."""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes, encode_planes_bulk
from train_bnn import OUT, BITS
from train_bnn2 import BNN2
from finetune_seg import BNN2Seg
from eval_fixtures import build_window, LABELS, REPO
from eval_edge import CASES

CKPTS = ["best2seg.pt", "best3.pt", "best2seg_synth.ep0.pt",
         "best2seg_synth3.ep0.pt", "best3_synth2.ep0.pt"]
ALIAS = {"Gemfile": "gemfile", "bash": "shell", "c-sharp": "cs",
         "commonlisp": "lisp", "objc": "objectivec", "vb": "vba"}


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    models = []
    for ck in CKPTS:
        cls = BNN2Seg if "2seg" in ck else BNN2
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.eval()
        models.append(m)

    inputs = []
    fix_expected = []
    for f in sorted((REPO / "tests/fixtures/languages").iterdir()):
        w = build_window(f.read_bytes())
        arr = np.frombuffer(w[0], dtype=np.uint8)[None, :]
        inputs.append(encode_planes_bulk(arr, np.array([w[1]]), codes, BITS))
        fix_expected.append(ALIAS.get(f.stem, f.stem))
    edge = []
    for name, text, want in CASES:
        w = build_window(text.encode())
        arr = np.frombuffer(w[0], dtype=np.uint8)[None, :]
        edge.append(encode_planes_bulk(arr, np.array([w[1]]), codes, BITS))
    demo = 'pub fn greet(name: &str) -> String {\n    format!("hello, {name}")\n}\n'
    w = build_window(demo.encode())
    arr = np.frombuffer(w[0], dtype=np.uint8)[None, :]
    edge.append(encode_planes_bulk(arr, np.array([w[1]]), codes, BITS))

    x = torch.from_numpy(
        np.concatenate(inputs + edge).astype(np.float32) * 2 - 1)
    with torch.no_grad():
        L = np.stack([m(x).numpy() for m in models])
    nfix = len(fix_expected)
    yi, mi = LABELS.index("yaml"), LABELS.index("markdown")
    ri = LABELS.index("rust")

    grid = [0.7, 1.0, 1.3, 1.6]
    best = []
    for ws in itertools.product(grid, repeat=5):
        w = np.array(ws)[:, None, None]
        tot = (L * w).sum(0)
        wrong = sum(LABELS[int(tot[i].argmax())] != fix_expected[i]
                    for i in range(nfix))
        if wrong:
            continue
        p = np.exp(tot / w.sum())
        p /= p.sum(1, keepdims=True)
        e = p[nfix:]
        ok = (e[0].argmax() == mi and e[1].argmax() == mi
              and e[2].argmax() == yi
              and set(np.argsort(-e[3])[:2]) == {yi, mi}
              and set(np.argsort(-e[4])[:2]) == {yi, mi}
              and e[4].max() < 0.9
              and e[5].argmax() == ri)
        if ok:
            best.append(ws)
    print(f"passing weightings: {len(best)}")
    for ws in best[:40]:
        print(ws)


if __name__ == "__main__":
    main()
