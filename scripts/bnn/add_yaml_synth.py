#!/usr/bin/env python3
"""Append synthetic yaml (sequence-of-mappings) and markdown (bullet lists)
samples to the synth hard corpus."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

import build_synth_hard as bs

W = bs.W1


def synth_yaml(rng: random.Random) -> bytes:
    style = rng.random()
    lines = []
    if style < 0.5:
        for _ in range(rng.randint(2, 6)):
            lines.append(f"- {rng.choice(['name', 'id', 'key', 'task'])}: "
                         f"{rng.choice(W)}")
            for _ in range(rng.randint(1, 3)):
                k = rng.choice(["run", "uses", "value", "with", "cmd", "path"])
                lines.append(f"  {k}: {rng.choice(W)} {rng.choice(W)}")
    elif style < 0.8:
        lines.append(f"{rng.choice(W)}:")
        for _ in range(rng.randint(2, 6)):
            lines.append(f"- {rng.choice(['name', 'key'])}: {rng.choice(W)}")
            if rng.random() < 0.5:
                lines.append(f"  {rng.choice(['run', 'value'])}: {rng.choice(W)}")
    else:
        for _ in range(rng.randint(2, 5)):
            lines.append(f"{rng.choice(W)}:")
            for _ in range(rng.randint(1, 3)):
                lines.append(f"  {rng.choice(W)}: "
                             f"{rng.choice([rng.choice(W), str(rng.randint(0, 99)), 'true', 'false'])}")
    return ("\n".join(lines) + "\n").encode()


def synth_markdown(rng: random.Random) -> bytes:
    lines = []
    if rng.random() < 0.8:
        lines.append(f"{'#' * rng.randint(1, 3)} {rng.choice(W).title()}"
                     f"{rng.choice(['', ' ' + rng.choice(W).title()])}")
        lines.append("")
    for _ in range(rng.randint(3, 8)):
        item = rng.choice(W)
        if rng.random() < 0.4:
            item = item.title()
        if rng.random() < 0.3:
            item += " " + rng.choice(W)
        lines.append(f"{rng.choice(['-', '*'])} {item}")
    if rng.random() < 0.3:
        lines.append("")
        lines.append(" ".join(rng.choice(W) for _ in range(rng.randint(5, 12))))
    return ("\n".join(lines) + "\n").encode()


def main():
    rng = random.Random(7)
    labels = bs.LABELS
    rows, lens, labs = [], [], []
    seen = set()
    for label, gen, count in [("yaml", synth_yaml, 2000),
                              ("markdown", synth_markdown, 1200)]:
        lid = labels.index(label)
        made = 0
        attempts = 0
        while made < count and attempts < count * 60:
            attempts += 1
            data = gen(rng)
            if data in seen:
                continue
            seen.add(data)
            w = bs.build_window(data)
            if w is None:
                continue
            rows.append(w[0])
            lens.append(w[1])
            labs.append(lid)
            made += 1
        print(label, made)
    out = Path(bs.OUT)
    windows = np.concatenate([np.load(out / "synth_hard.windows.npy"),
                              np.stack(rows)])
    lengths = np.concatenate([np.load(out / "synth_hard.lengths.npy"),
                              np.array(lens, dtype=np.int32)])
    lab = np.concatenate([np.load(out / "synth_hard.labels.npy"),
                          np.array(labs, dtype=np.int64)])
    np.save(out / "synth_hard.windows.npy", windows)
    np.save(out / "synth_hard.lengths.npy", lengths)
    np.save(out / "synth_hard.labels.npy", lab)
    print("total", len(lab))


if __name__ == "__main__":
    main()
