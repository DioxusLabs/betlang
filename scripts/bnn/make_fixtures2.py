#!/usr/bin/env python3
"""Write the BBN2 artifact and matching Rust parity fixtures atomically
(from one in-memory model load, so trainers rewriting checkpoints on disk
cannot desynchronize them)."""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from train_bnn import BITS, CLASSES, OUT, load_split
from export_bnn2 import MAGIC, IntModel, load_models
from shannon import canonical_codes, encode_planes_bulk


def main():
    lengths256 = np.array(json.loads((OUT / "codebook.json").read_text())["lengths"],
                          dtype=np.int32)
    codes = canonical_codes(lengths256)
    ints = [IntModel(m, w) for m, w in load_models()]

    buf = bytearray()
    buf += struct.pack("<5I", MAGIC, 2, BITS, CLASSES, len(ints))
    buf += lengths256.astype(np.uint8).tobytes()
    for im in ints:
        im.write(buf)
    (OUT / "source-bnn2.bin").write_bytes(bytes(buf))
    print(f"wrote {OUT / 'source-bnn2.bin'} ({len(buf)} bytes)")
    windows, lengths, _, _ = load_split("valid")
    cases = []
    picks = [0, len(lengths) // 2, len(lengths) - 1]
    for i in picks:
        w = np.asarray(windows[i])
        n = int(lengths[i])
        cases.append(bytes(w[:n].tobytes()))
    cases.append(b"fn main() {}")
    out = []
    for data in cases:
        arr = np.frombuffer(data, dtype=np.uint8)[None, :]
        planes = encode_planes_bulk(arr, np.array([len(data)]), codes, BITS)[0]
        logits = sum(im.logits(planes) for im in ints) / len(ints)
        out.append({"window_hex": data.hex(),
                    "logits": [float(x) for x in logits]})
    (OUT / "parity_fixtures2.json").write_text(json.dumps(out, indent=1))
    print("wrote", OUT / "parity_fixtures2.json")


if __name__ == "__main__":
    main()
