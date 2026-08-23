#!/usr/bin/env python3
"""Run candidate ensembles against betlang's language fixture files
(replicates src/model/window.rs build_window semantics)."""
from __future__ import annotations

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

REPO = Path.home() / "repos/betlang"
BEG, END, WIN, BLOCK = 1024, 1024, 2048, 4096
LABELS = json.loads((Path.home() / "work/cache/valid.json").read_text())["labels"]


def build_window(source: bytes):
    if not source:
        return None
    block = min(len(source), BLOCK)
    beg = source[:block].lstrip(b"\t\n\x0b\x0c\r ")
    if len(beg) < 8:
        return None
    end = source[len(source) - block:].rstrip(b"\t\n\x0b\x0c\r ")
    beg_len = min(len(beg), BEG)
    end_len = min(len(end), END)
    buf = bytearray(WIN)
    buf[:beg_len] = beg[:beg_len]
    end_start = BEG + (END - end_len)
    buf[end_start:end_start + end_len] = end[len(end) - end_len:]
    if beg_len < BEG:
        valid = beg_len
    elif end_start > BEG:
        valid = BEG
    else:
        valid = WIN
    return bytes(buf), valid


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    if len(sys.argv) > 1:
        specs = []
        for ck in sys.argv[1:]:
            cls = BNN2Seg if "2seg" in ck else BNN2
            specs.append((cls, ck))
    else:
        specs = [(BNN2Seg, "best2seg.pt"), (BNN2, "best3.pt"),
                 (BNN2Seg, "best2seg_short.pt"), (BNN2, "best3_short.pt")]
    models = []
    for cls, ck in specs:
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.eval()
        models.append(m)

    wrong = 0
    files = sorted((REPO / "tests/fixtures/languages").iterdir())
    for f in files:
        expected = f.stem if f.stem != "docker" else f.stem
        expected = f.stem
        w = build_window(f.read_bytes())
        if w is None:
            print(f"{f.name}: window None")
            continue
        window, valid = w
        arr = np.frombuffer(window, dtype=np.uint8)[None, :]
        planes = encode_planes_bulk(arr, np.array([valid]), codes, BITS)
        x = torch.from_numpy(planes.astype(np.float32) * 2 - 1)
        with torch.no_grad():
            logits = sum(m(x) for m in models)[0].numpy()
        pred = LABELS[int(logits.argmax())]
        if pred != expected:
            top = np.argsort(logits)[::-1][:3]
            print(f"{f.name}: expected {expected}, got {pred}; "
                  f"top {[LABELS[t] for t in top]}")
            wrong += 1
    print(f"{wrong}/{len(files)} wrong")


if __name__ == "__main__":
    main()
