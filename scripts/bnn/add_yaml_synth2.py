#!/usr/bin/env python3
"""More realistic short yaml sequence-of-mappings samples (CI-style)."""
from __future__ import annotations

import random

import numpy as np

import build_synth_hard as bs

NAMES = ["build", "test", "lint", "deploy", "check", "fmt", "docs", "bench",
         "setup", "install", "release", "clean", "package", "verify"]
CMDS = ["make", "make test", "make build", "cargo test", "cargo build",
        "npm test", "npm run build", "pytest", "go test ./...", "mvn verify",
        "./configure", "bash ci.sh", "python setup.py test", "cmake --build ."]


def synth_yaml_ci(rng: random.Random) -> bytes:
    lines = []
    for _ in range(rng.randint(2, 5)):
        lines.append(f"- name: {rng.choice(NAMES)}")
        for _ in range(rng.randint(1, 2)):
            k = rng.choice(["run", "run", "cmd", "script"])
            lines.append(f"  {k}: {rng.choice(CMDS)}")
    return ("\n".join(lines) + "\n").encode()


def main():
    rng = random.Random(99)
    lid = bs.LABELS.index("yaml")
    rows, lens, labs = [], [], []
    seen = set()
    made = 0
    attempts = 0
    while made < 2000 and attempts < 200000:
        attempts += 1
        data = synth_yaml_ci(rng)
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
    out = bs.OUT
    windows = np.concatenate([np.load(out / "synth_hard.windows.npy"),
                              np.stack(rows)])
    lengths = np.concatenate([np.load(out / "synth_hard.lengths.npy"),
                              np.array(lens, dtype=np.int32)])
    lab = np.concatenate([np.load(out / "synth_hard.labels.npy"),
                          np.array(labs, dtype=np.int64)])
    np.save(out / "synth_hard.windows.npy", windows)
    np.save(out / "synth_hard.lengths.npy", lengths)
    np.save(out / "synth_hard.labels.npy", lab)
    print("yaml_ci", made, "total", len(lab))


if __name__ == "__main__":
    main()
