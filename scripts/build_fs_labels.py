#!/usr/bin/env python3
"""Build {split}.fs_labels.mmap files for a corpus + teacher cache.

The fs label for each cached row is the "filesystem truth": the label implied
by where the file lives on disk, with teacher fallback when no mapping exists.
Rows are aligned to the cache by walking the corpus in the same order as the
cache builder and matching each file's Magika byte window against
{split}.tokens.mmap, so files the cache builder dropped are skipped here too.

Label resolution order per file:
1. Parent directory name, when it is one of the cache labels (corpora built by
   scripts/build_finetune_corpus.py place files under files/{split}/{label}/).
2. Extension/filename mapping (see EXTENSION_LABELS / FILENAME_LABELS).
3. Teacher argmax from {split}.labels.mmap (counted as ext_unmapped).

Usage:
    python3 scripts/build_fs_labels.py \
      --dataset /path/to/corpus/files --cache-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_magika_source_student import (  # noqa: E402
    SPLITS,
    magika_features,
    read_training_windows,
    source_paths,
)

TOKEN_LENGTH = 2048

FILENAME_LABELS = {
    "Gemfile": "gemfile",
    "Dockerfile": "dockerfile",
    "Containerfile": "dockerfile",
    "CMakeLists.txt": "cmake",
}

EXTENSION_LABELS = {
    "asm": "asm", "s": "asm",
    "bat": "batch", "cmd": "batch",
    "c": "c",
    "clj": "clojure", "cljs": "clojure", "cljc": "clojure",
    "cmake": "cmake",
    "cob": "cobol", "cbl": "cobol", "cpy": "cobol",
    "cpp": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp", "hh": "cpp", "hxx": "cpp",
    "cs": "cs",
    "css": "css",
    "dart": "dart",
    "dockerfile": "dockerfile",
    "ex": "elixir", "exs": "elixir",
    "erl": "erlang", "hrl": "erlang",
    "gemspec": "gemspec",
    "go": "go",
    "gradle": "gradle",
    "groovy": "groovy", "gvy": "groovy",
    "hs": "haskell", "lhs": "haskell",
    "html": "html", "htm": "html", "xhtml": "html",
    "ini": "ini", "cfg": "ini",
    "java": "java",
    "js": "javascript", "mjs": "javascript", "cjs": "javascript", "jsx": "javascript",
    "json": "json",
    "jl": "julia",
    "kt": "kotlin", "kts": "kotlin",
    "lisp": "lisp", "lsp": "lisp", "cl": "lisp",
    "lua": "lua",
    "md": "markdown", "markdown": "markdown", "mdx": "markdown",
    "m": "objectivec", "mm": "objectivec",
    "ml": "ocaml", "mli": "ocaml",
    "pl": "perl", "pm": "perl",
    "php": "php", "phtml": "php",
    "ps1": "powershell", "psm1": "powershell", "psd1": "powershell",
    "py": "python", "pyi": "python", "pyw": "python",
    "r": "r",
    "rb": "ruby", "rake": "ruby",
    "rs": "rust",
    "scala": "scala", "sbt": "scala",
    "sh": "shell", "bash": "shell", "zsh": "shell", "ksh": "shell",
    "sql": "sql",
    "swift": "swift",
    "toml": "toml",
    "ts": "typescript", "tsx": "typescript", "mts": "typescript", "cts": "typescript",
    "vb": "vba", "vbs": "vba",
    "v": "verilog", "sv": "verilog", "svh": "verilog",
    "xml": "xml", "xsd": "xml", "xsl": "xml", "xslt": "xml", "svg": "xml",
    "yaml": "yaml", "yml": "yaml",
}


def fs_label_for(path: Path, label_to_index: dict[str, int]) -> int | None:
    parent = path.parent.name
    if parent in label_to_index:
        return label_to_index[parent]
    name = path.name
    if name in FILENAME_LABELS:
        return label_to_index.get(FILENAME_LABELS[name])
    extension = path.suffix.lstrip(".").lower()
    label = EXTENSION_LABELS.get(extension)
    if label is None:
        return None
    return label_to_index.get(label)


def build_split(dataset: Path, cache_dir: Path, split: str) -> None:
    meta = json.loads((cache_dir / f"{split}.json").read_text())
    n = int(meta["count"])
    labels = [str(label) for label in meta["labels"]]
    label_to_index = {label: index for index, label in enumerate(labels)}

    tokens = np.memmap(cache_dir / f"{split}.tokens.mmap", dtype=np.uint16, mode="r", shape=(n, TOKEN_LENGTH))
    teacher_labels = np.memmap(cache_dir / f"{split}.labels.mmap", dtype=np.int64, mode="r", shape=(n,))
    fs_labels = np.full(n, -1, dtype=np.int64)

    row = 0
    mapped = 0
    unmapped = 0
    dropped = 0
    for path in source_paths(dataset / split, None):
        windows = read_training_windows(path)
        if windows is None:
            continue
        size, prefix, suffix = windows
        features = magika_features(size, prefix, suffix)
        if features is None:
            continue
        if row >= n:
            raise SystemExit(f"{split}: more corpus windows than cache rows; wrong --dataset?")
        if not np.array_equal(np.asarray(features, dtype=np.uint16), tokens[row]):
            dropped += 1
            continue
        label_index = fs_label_for(path, label_to_index)
        if label_index is None:
            fs_labels[row] = int(teacher_labels[row])
            unmapped += 1
        else:
            fs_labels[row] = label_index
            mapped += 1
        row += 1

    if row != n:
        raise SystemExit(f"{split}: aligned {row} rows but cache has {n}; wrong --dataset?")

    out = np.memmap(cache_dir / f"{split}.fs_labels.mmap", dtype=np.int64, mode="w+", shape=(n,))
    out[:] = fs_labels
    out.flush()
    stats = {
        "count": n,
        "ext_mapped": mapped,
        "ext_unmapped": unmapped,
        "cache_dropped_files": dropped,
    }
    (cache_dir / f"{split}.fs_labels.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"{split}: {json.dumps(stats, sort_keys=True)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    args = parser.parse_args()

    for split in args.splits:
        build_split(args.dataset, args.cache_dir, split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
