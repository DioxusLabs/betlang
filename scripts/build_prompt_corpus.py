"""Build the shell-command vs LLM-prompt corpus.

Downloads public datasets and writes one file per sample into
``<output>/files/{train,valid,test}/{prompt,shell_command}/`` for
``train_prompt_student.py``.

Labels:

- ``prompt``: text a user would send to an AI assistant. Real user prompts
  from OpenAssistant (oasst1 first turns), ShareGPT first human turns, and
  HuggingFaceH4/no_robots, plus instruction-style prompts from Alpaca, Dolly,
  and the awesome-chatgpt-prompts personas.
- ``shell_command``: real shell one-liners from the NL2Bash corpus and
  example commands from tldr-pages (placeholders like ``{{path}}`` are
  flattened to ``path``).

Within each label samples are deduplicated by content hash. Splits use a
deterministic seed so rebuilds are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import urllib.request
from pathlib import Path

from datasets import load_dataset

SPLITS = ("train", "valid", "test")
SPLIT_WEIGHTS = (0.90, 0.05, 0.05)
MIN_BYTES = 8
MAX_BYTES = 8192
SEED = 2

NL2BASH_URL = "https://raw.githubusercontent.com/TellinaTool/nl2bash/master/data/bash/all.cm"
TLDR_TARBALL = "https://github.com/tldr-pages/tldr/archive/refs/heads/main.tar.gz"


def normalize(text: str) -> str | None:
    text = text.replace("\r\n", "\n").strip()
    if len(text.encode("utf-8", "ignore")) < MIN_BYTES:
        return None
    encoded = text.encode("utf-8", "ignore")[:MAX_BYTES]
    return encoded.decode("utf-8", "ignore").strip()


def assign_split(rng: random.Random) -> str:
    return rng.choices(SPLITS, weights=SPLIT_WEIGHTS, k=1)[0]


def write_samples(output: Path, label: str, source: str, texts: list[str], rng: random.Random) -> int:
    seen: set[str] = set()
    written = 0
    for text in texts:
        normalized = normalize(text)
        if normalized is None:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        split = assign_split(rng)
        path = output / "files" / split / label / f"{source}-{digest[:16]}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8")
        written += 1
    return written


def prompt_texts(limit_alpaca: int, limit_dolly: int, limit_oasst: int, limit_sharegpt: int) -> list[str]:
    texts: list[str] = []

    no_robots = load_dataset("HuggingFaceH4/no_robots", split="train")
    texts.extend(row["prompt"] for row in no_robots)

    oasst = load_dataset("OpenAssistant/oasst1", split="train")
    count = 0
    for row in oasst:
        if row["role"] != "prompter" or row["parent_id"] is not None or row["lang"] != "en":
            continue
        texts.append(row["text"])
        count += 1
        if count >= limit_oasst:
            break

    sharegpt = load_dataset("RyokoAI/ShareGPT52K", split="train")
    count = 0
    for row in sharegpt:
        conversations = row["conversations"]
        if not conversations:
            continue
        first = conversations[0]
        if first.get("from") != "human":
            continue
        text = first.get("value", "")
        # ShareGPT bodies sometimes contain raw HTML from the scrape.
        if "<div" in text or "<p>" in text:
            continue
        texts.append(text)
        count += 1
        if count >= limit_sharegpt:
            break

    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    for row in alpaca.select(range(min(limit_alpaca, len(alpaca)))):
        instruction = row["instruction"].strip()
        task_input = (row["input"] or "").strip()
        texts.append(f"{instruction}\n\n{task_input}" if task_input else instruction)

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train")
    for index, row in enumerate(dolly.select(range(min(limit_dolly, len(dolly))))):
        instruction = row["instruction"].strip()
        context = (row["context"] or "").strip()
        # Include some instruction+context pairs: a real prompt often pastes
        # reference text after the request.
        if context and index % 4 == 0:
            texts.append(f"{instruction}\n\n{context}")
        else:
            texts.append(instruction)

    persona = load_dataset("fka/awesome-chatgpt-prompts", split="train")
    texts.extend(row["prompt"] for row in persona)

    return texts


def nl2bash_commands() -> list[str]:
    with urllib.request.urlopen(NL2BASH_URL) as response:
        body = response.read().decode("utf-8", "ignore")
    return [line.strip() for line in body.splitlines() if line.strip()]


def tldr_commands(pages_dir: Path) -> list[str]:
    """Extract example command lines from a tldr-pages checkout.

    Example lines look like ``\\`command --flag {{argument}}\\``; the braces
    are stripped so the command reads like real input.
    """
    commands: list[str] = []
    for page in pages_dir.glob("pages*/**/*.md"):
        for line in page.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not (line.startswith("`") and line.endswith("`")):
                continue
            command = line[1:-1]
            command = re.sub(r"\{\{(.*?)\}\}", r"\1", command)
            commands.append(command)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tldr-dir", type=Path, default=None, help="existing tldr-pages checkout (skips download)")
    parser.add_argument("--alpaca-limit", type=int, default=20_000)
    parser.add_argument("--dolly-limit", type=int, default=15_000)
    parser.add_argument("--oasst-limit", type=int, default=12_000)
    parser.add_argument("--sharegpt-limit", type=int, default=20_000)
    args = parser.parse_args()

    rng = random.Random(SEED)

    prompt_count = write_samples(
        args.output,
        "prompt",
        "prompt",
        prompt_texts(args.alpaca_limit, args.dolly_limit, args.oasst_limit, args.sharegpt_limit),
        rng,
    )

    shell: list[str] = nl2bash_commands()
    if args.tldr_dir is None:
        import io
        import tarfile
        import tempfile

        with urllib.request.urlopen(TLDR_TARBALL) as response:
            payload = response.read()
        with tempfile.TemporaryDirectory() as tmp:
            with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
                tar.extractall(tmp, filter="data")
            shell.extend(tldr_commands(Path(tmp)))
    else:
        shell.extend(tldr_commands(args.tldr_dir))

    shell_count = write_samples(args.output, "shell_command", "shell", shell, rng)

    print(json.dumps({"prompt": prompt_count, "shell_command": shell_count}))


if __name__ == "__main__":
    main()
