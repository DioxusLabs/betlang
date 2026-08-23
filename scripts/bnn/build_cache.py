#!/usr/bin/env python3
"""Cache builder: raw Magika windows + fs labels + Magika teacher marginals.

Writes per split:
  {split}.windows.mmap  uint8  [N, 2048]  (zero padded)
  {split}.lengths.mmap  int32  [N]
  {split}.labels.mmap   int16  [N]
  {split}.teacher.mmap  float32 [N, 48]   (teacher head-label marginals)
  {split}.json          metadata
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path.home() / "repos/betlang/scripts"))
from betlang_tokenizer import build_window, MAGIKA_WINDOW_SIZE

LABELS = [
    "asm", "batch", "c", "clojure", "cmake", "cobol", "cpp", "cs", "css",
    "dart", "dockerfile", "elixir", "erlang", "gemfile", "gemspec", "go",
    "gradle", "groovy", "haskell", "html", "ini", "java", "javascript",
    "json", "julia", "kotlin", "lisp", "lua", "markdown", "objectivec",
    "ocaml", "perl", "php", "powershell", "python", "r", "ruby", "rust",
    "scala", "shell", "sql", "swift", "toml", "typescript", "vba",
    "verilog", "xml", "yaml",
]
LABEL_INDEX = {label: i for i, label in enumerate(LABELS)}

MAGIKA_BEG_SIZE = 1024
MAGIKA_END_SIZE = 1024
MAGIKA_BLOCK_SIZE = 4096
MAGIKA_PADDING_TOKEN = 256


def magika_features(data: bytes) -> list[int] | None:
    size = len(data)
    if size == 0:
        return None
    stripped_beg = data[: min(size, MAGIKA_BLOCK_SIZE)].lstrip()
    stripped_end = data[-min(size, MAGIKA_BLOCK_SIZE):].rstrip()
    if len(stripped_beg) < 8:
        return None
    beg = list(stripped_beg[:MAGIKA_BEG_SIZE])
    beg.extend([MAGIKA_PADDING_TOKEN] * (MAGIKA_BEG_SIZE - len(beg)))
    end_data = stripped_end[-MAGIKA_END_SIZE:]
    end = [MAGIKA_PADDING_TOKEN] * (MAGIKA_END_SIZE - len(end_data))
    end.extend(end_data)
    return beg + end


def load_teacher():
    import onnxruntime as ort
    import magika

    model_dir = Path(magika.__file__).parent / "models" / "standard_v3_3"
    config = json.loads((model_dir / "config.min.json").read_text())
    target = config["target_labels_space"]
    label_to_index = {label: i for i, label in enumerate(target)}
    cols = [label_to_index[label] for label in LABELS]
    session = ort.InferenceSession(
        str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    return session, cols


def main() -> int:
    corpus = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    session, cols = load_teacher()
    input_name = session.get_inputs()[0].name

    for split in ("train", "valid", "test"):
        meta_path = out / f"{split}.json"
        if meta_path.exists():
            print(f"{split}: cached", flush=True)
            continue
        rows = []
        for label_dir in sorted((corpus / split).iterdir()):
            if not label_dir.is_dir() or label_dir.name not in LABEL_INDEX:
                continue
            for file in sorted(label_dir.iterdir()):
                rows.append((file, LABEL_INDEX[label_dir.name]))
        n = len(rows)
        print(f"{split}: {n} candidate files", flush=True)
        windows = np.memmap(out / f"{split}.windows.mmap", dtype=np.uint8,
                            mode="w+", shape=(n, MAGIKA_WINDOW_SIZE))
        lengths = np.memmap(out / f"{split}.lengths.mmap", dtype=np.int32,
                            mode="w+", shape=(n,))
        labels = np.memmap(out / f"{split}.labels.mmap", dtype=np.int16,
                           mode="w+", shape=(n,))
        teacher = np.memmap(out / f"{split}.teacher.mmap", dtype=np.float32,
                            mode="w+", shape=(n, len(LABELS)))
        count = 0
        batch_feats, batch_idx = [], []

        def flush():
            nonlocal batch_feats, batch_idx
            if not batch_feats:
                return
            arr = np.asarray(batch_feats, dtype=np.int32)
            probs = session.run(None, {input_name: arr})[0]
            teacher[batch_idx] = probs[:, cols]
            batch_feats, batch_idx = [], []

        for file, label in rows:
            data = file.read_bytes()
            window = build_window(data)
            feats = magika_features(data)
            if window is None or feats is None:
                continue
            windows[count, :len(window)] = np.frombuffer(window, dtype=np.uint8)
            lengths[count] = len(window)
            labels[count] = label
            batch_feats.append(feats)
            batch_idx.append(count)
            count += 1
            if len(batch_feats) == 256:
                flush()
                if count % 10240 == 0:
                    print(f"{split}: {count}/{n}", flush=True)
        flush()
        windows.flush(); lengths.flush(); labels.flush(); teacher.flush()
        meta_path.write_text(json.dumps({
            "count": count, "total": n, "labels": LABELS,
            "window": MAGIKA_WINDOW_SIZE,
        }))
        print(f"{split}: wrote {count} rows", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
