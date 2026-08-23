#!/usr/bin/env python3
"""Second synth fine-tune pass (with yaml/markdown samples added)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import OUT
from train_bnn2 import BNN2
from finetune_seg import BNN2Seg
from finetune_synth import finetune


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)

    m0 = BNN2Seg()
    m0.load_state_dict(torch.load(OUT / "best2seg_synth.ep0.pt", map_location="cpu"))
    m0.to(device)
    finetune("m0s3", m0, OUT / "best2seg_synth2.pt", codes, device,
             epochs=2, lr=8e-5)


if __name__ == "__main__":
    main()
