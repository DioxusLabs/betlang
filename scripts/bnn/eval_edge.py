#!/usr/bin/env python3
"""Check the ambiguous yaml/markdown behavioral tests from src/model/tests.rs."""
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
from eval_fixtures import build_window

CASES = [
    ("md_list", "# Heading\n\n- first\n- second\n- third\n- fourth\n- fifth",
     "markdown top"),
    ("md_caps", "# Names\n\n- Alice\n- Bob\n- Carol\n- Dave", "markdown top"),
    ("yaml_seq", "- name: build\n  run: make\n- name: test\n  run: make test",
     "yaml top"),
    ("commented", "# comment\nitems:\n- first\n- second\n- third",
     "yaml+markdown top2"),
    ("bare_dash", "- first\n- second\n- third\n- fourth\n- fifth",
     "yaml+markdown top2, p<0.9"),
]


def main():
    labels = json.loads((OUT.parent / "cache/valid.json").read_text())["labels"]
    lengths256 = np.array(
        json.loads((OUT / "codebook.json").read_text())["lengths"], dtype=np.int32)
    codes = canonical_codes(lengths256)
    models = []
    for ck in sys.argv[1:]:
        cls = BNN2Seg if "2seg" in ck else BNN2
        m = cls()
        m.load_state_dict(torch.load(OUT / ck, map_location="cpu"))
        m.eval()
        models.append(m)
    for name, text, want in CASES:
        w = build_window(text.encode())
        if w is None:
            print(f"{name}: window None")
            continue
        arr = np.frombuffer(w[0], dtype=np.uint8)[None, :]
        planes = encode_planes_bulk(arr, np.array([w[1]]), codes, BITS)
        x = torch.from_numpy(planes.astype(np.float32) * 2 - 1)
        tot = None
        with torch.no_grad():
            for m in models:
                lg = m(x)[0]
                tot = lg if tot is None else tot + lg
        p = torch.softmax(tot / len(models), dim=0).numpy()
        order = np.argsort(-p)
        top = [(labels[i], round(float(p[i]), 4)) for i in order[:3]]
        print(f"{name}: want[{want}] got {top}")


if __name__ == "__main__":
    main()
