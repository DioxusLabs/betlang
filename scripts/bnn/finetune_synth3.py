#!/usr/bin/env python3
"""Third synth pass (yaml_ci samples added) on the m0 synth2 checkpoint."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from shannon import canonical_codes
from train_bnn import OUT
from finetune_seg import BNN2Seg
from finetune_synth import finetune


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)

    m = BNN2Seg()
    m.load_state_dict(torch.load(OUT / "best2seg_synth2.ep1.pt",
                                 map_location="cpu"))
    m.to(device)
    finetune("m0s4", m, OUT / "best2seg_synth3.pt", codes, device,
             epochs=1, lr=5e-5)


if __name__ == "__main__":
    main()
