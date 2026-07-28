#!/usr/bin/env python3
"""Build and score an out-of-distribution benchmark for the prompt vs
shell-command classifier.

Sources are disjoint from the training corpus builder:

- ``prompt``: nvidia/HelpSteer2 prompts, Anthropic/hh-rlhf first human turns,
  and NL2SH-ALFA test-set instructions.
- ``shell_command``: NL2SH-ALFA test-set commands, InterCode-Corrections gold
  commands, a slice of real bash history (spignelon/bash_history), and
  command lines from held-out GitHub shell-script repositories
  (the-stack-smol-xl repos hashed into the benchmark bucket, which the
  corpus builder never trains on).

Any sample whose exact content or near-duplicate group key (same keys as
``build_prompt_corpus.py``) appears in the training corpus is excluded, so
every scored sample is genuinely out-of-distribution. Junk that does not
survive normalization (fewer than 8 bytes) is dropped.

Usage:

    python3 scripts/ood_benchmark.py \
        --corpus /tmp/betlang-corpus/files \
        --output /tmp/betlang-ood-benchmark \
        --detect-binary target/release/examples/detect
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

from datasets import load_dataset

from build_prompt_corpus import (
    normalize,
    prompt_group_key,
    shell_group_key,
    stack_shell_commands,
)

LABELS = ("prompt", "shell_command")
SEED = 11


def training_index(corpus: Path) -> tuple[set[str], dict[str, set[str]]]:
    hashes: set[str] = set()
    keys: dict[str, set[str]] = {label: set() for label in LABELS}
    for path in corpus.rglob("*.txt"):
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
        label = path.parent.name
        if label in keys:
            key = prompt_group_key(text) if label == "prompt" else shell_group_key(text)
            keys[label].add(key)
    return hashes, keys


def first_human_turn(conversation: str) -> str | None:
    match = re.search(r"Human: (.*?)(?:\n\nAssistant:|$)", conversation, re.DOTALL)
    return match.group(1).strip() if match else None


def collect(
    limit_prompts: int, limit_shell_history: int, limit_stack_shell: int
) -> list[tuple[str, str]]:
    samples: list[tuple[str, str]] = []

    helpsteer = load_dataset("nvidia/HelpSteer2", split="train")
    seen: set[str] = set()
    for row in helpsteer:
        prompt = row["prompt"].split("<extra_id_1>")[0].strip()
        if len(prompt) >= 8 and prompt not in seen:
            seen.add(prompt)
            samples.append(("prompt", prompt))
        if len(seen) >= limit_prompts:
            break

    hh = load_dataset("Anthropic/hh-rlhf", split="test")
    count = 0
    for row in hh:
        turn = first_human_turn(row["chosen"])
        if turn and len(turn) >= 8:
            samples.append(("prompt", turn))
            count += 1
        if count >= 500:
            break

    alfa = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
    for row in alfa:
        samples.append(("prompt", row["nl"].strip()))
        samples.append(("shell_command", row["bash"].strip()))
        samples.append(("shell_command", row["bash2"].strip()))

    intercode = load_dataset("westenfelder/InterCode-Corrections", split="train")
    for row in intercode:
        samples.append(("shell_command", row["gold"].strip()))
        samples.append(("shell_command", row["gold2"].strip()))

    history = load_dataset("spignelon/bash_history", split="train")
    pool = sorted({row["text"].strip() for row in history if len(row["text"].strip()) >= 8})
    random.Random(SEED).shuffle(pool)
    samples.extend(("shell_command", command) for command in pool[:limit_shell_history])

    samples.extend(
        ("shell_command", command)
        for command in stack_shell_commands(limit_stack_shell, heldout=True)
    )
    return samples


def build(
    output: Path,
    corpus: Path,
    limit_prompts: int,
    limit_shell_history: int,
    limit_stack_shell: int,
) -> None:
    hashes, keys = training_index(corpus)
    shutil.rmtree(output, ignore_errors=True)
    written = {label: 0 for label in LABELS}
    seen: set[str] = set()
    for label, text in collect(limit_prompts, limit_shell_history, limit_stack_shell):
        normalized = normalize(text)
        if normalized is None:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen or digest in hashes:
            continue
        key = prompt_group_key(normalized) if label == "prompt" else shell_group_key(normalized)
        if key in keys[label]:
            continue
        seen.add(digest)
        path = output / label / f"{digest[:16]}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8")
        written[label] += 1
    print("benchmark sizes:", json.dumps(written))


def score(output: Path, detect_binary: Path) -> None:
    stats = {}
    total = correct = 0
    for label in LABELS:
        directory = output / label
        result = subprocess.run(
            [str(detect_binary), str(directory)], capture_output=True, text=True, check=True
        )
        n = c = 0
        for line in result.stdout.splitlines():
            parts = line.split()
            for index, part in enumerate(parts):
                if part.endswith(".txt") and index + 1 < len(parts):
                    n += 1
                    if parts[index + 1] == label:
                        c += 1
        stats[label] = {"n": n, "recall": round(c / n, 4) if n else None}
        total += n
        correct += c
    stats["accuracy"] = round(correct / total, 4) if total else None
    print("betlang_ood=" + json.dumps(stats))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detect-binary", type=Path, default=None)
    parser.add_argument("--prompt-limit", type=int, default=4_000)
    parser.add_argument("--shell-history-limit", type=int, default=4_000)
    parser.add_argument("--stack-shell-limit", type=int, default=4_000)
    args = parser.parse_args()

    build(
        args.output,
        args.corpus,
        args.prompt_limit,
        args.shell_history_limit,
        args.stack_shell_limit,
    )
    if args.detect_binary is not None:
        score(args.output, args.detect_binary)


if __name__ == "__main__":
    main()
