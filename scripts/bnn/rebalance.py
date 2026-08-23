#!/usr/bin/env python3
"""Rebalance corpus splits to ~70/10/20 per label by moving files (deterministic)."""
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
splits = ("train", "valid", "test")

labels = sorted({d.name for s in splits for d in (root / s).iterdir() if d.is_dir()})
for label in labels:
    files = []
    for s in splits:
        d = root / s / label
        if d.is_dir():
            files.extend(d.iterdir())
    files = [f for f in files if f.is_file()]
    n = len(files)
    n_valid = max(1, round(n * 0.10)) if n >= 10 else 0
    n_test = max(1, round(n * 0.20)) if n >= 10 else 0
    keyed = sorted(files, key=lambda f: hashlib.sha1(f.name.encode()).hexdigest())
    assign = {}
    for i, f in enumerate(keyed):
        if i < n_valid:
            assign[f] = "valid"
        elif i < n_valid + n_test:
            assign[f] = "test"
        else:
            assign[f] = "train"
    moved = 0
    for f, target in assign.items():
        cur = f.parent.parent.name
        if cur != target:
            dest = root / target / label
            dest.mkdir(parents=True, exist_ok=True)
            f.rename(dest / f.name)
            moved += 1
    print(f"{label}: n={n} moved={moved}")
