"""Build the shell-command vs LLM-prompt corpus.

Downloads public datasets and writes one file per sample into
``<output>/files/{train,valid,test}/{prompt,shell_command}/`` for
``train_prompt_student.py``.

Labels:

- ``prompt``: text a user would send to an AI assistant. Real user prompts
  from OpenAssistant (oasst1 first turns), ShareGPT first human turns, and
  HuggingFaceH4/no_robots, developer questions from Stack Overflow titles,
  plus instruction-style prompts from Alpaca, Dolly, and the
  awesome-chatgpt-prompts personas.
- ``shell_command``: real shell one-liners from the NL2Bash corpus, example
  commands from tldr-pages (placeholders like ``{{path}}`` are flattened to
  ``path``), bash tool calls mined from agent RL/SFT trajectories
  (SWE-bench/SWE-smith-trajectories), real user shell history
  (spignelon/bash_history), plus synthetic hard negatives that embed English
  phrases inside quoted command arguments (``git commit -m "..."``,
  ``echo "..."``, ``grep -r "..."``) so quoted prose does not flip the
  label.

Within each label samples are deduplicated by content hash. Split assignment
is leakage-aware: every sample is assigned to train/valid/test by hashing a
group key rather than the raw text, so near-duplicates land in the same
split. Prompts group by case/punctuation-normalized text, shell commands by
their command template (quoted strings, numbers, and paths collapsed), and
each synthetic hard negative follows the split of the prompt its phrase came
from. Only English tldr pages are used. Rebuilds are deterministic.
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
    """Canonicalize text with the same transforms as the Rust runtime
    (``src/model/normalize.rs``): newline normalization, tab/NBSP to space,
    zero-width/BOM and control-character removal, space-run collapse, trim."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\t", " ").replace("\u00a0", " ")
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = re.sub(r"[\x00-\x09\x0b-\x1f]", "", text)
    text = re.sub(r" {2,}", " ", text).strip()
    if len(text.encode("utf-8", "ignore")) < MIN_BYTES:
        return None
    encoded = text.encode("utf-8", "ignore")[:MAX_BYTES]
    return encoded.decode("utf-8", "ignore").strip()


def split_for_key(key: str) -> str:
    """Deterministic split from a group key: near-duplicates share a key and
    therefore always land in the same split."""
    fraction = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0x1_0000_0000
    bound = 0.0
    for split, weight in zip(SPLITS, SPLIT_WEIGHTS):
        bound += weight
        if fraction < bound:
            return split
    return SPLITS[-1]


def prompt_group_key(text: str) -> str:
    """Group prompts by case/whitespace/punctuation-normalized text so
    trivial variants share a split."""
    return re.sub(r"[^a-z0-9 ]", "", " ".join(text.lower().split()))


def shell_group_key(command: str) -> str:
    """Group shell commands by their template: quoted strings, numbers, and
    path-like tokens are collapsed so variants of one command share a split."""
    template = re.sub(r"\"[^\"]*\"|'[^']*'", "S", command.lower())
    template = re.sub(r"\S*/\S*", "P", template)
    template = re.sub(r"\d+", "N", template)
    return " ".join(template.split())


def write_samples(output: Path, label: str, source: str, samples: list[tuple[str, str]]) -> int:
    """Write deduplicated ``(text, split)`` samples for one label."""
    seen: set[str] = set()
    written = 0
    for text, split in samples:
        normalized = normalize(text)
        if normalized is None:
            continue
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        path = output / "files" / split / label / f"{source}-{digest[:16]}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(normalized, encoding="utf-8")
        written += 1
    return written


def prompt_texts(
    limit_alpaca: int,
    limit_dolly: int,
    limit_oasst: int,
    limit_sharegpt: int,
    limit_stackoverflow: int,
) -> list[str]:
    texts: list[str] = []

    stackoverflow = load_dataset("pacovaldez/stackoverflow-questions", split="train", streaming=True)
    count = 0
    for row in stackoverflow:
        title = row["title"].strip()
        if len(title) < 16:
            continue
        texts.append(title)
        count += 1
        if count >= limit_stackoverflow:
            break

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


QUOTED_ARG_TEMPLATES = (
    'git commit -m "{}"',
    'git commit --amend -m "{}"',
    'echo "{}"',
    'echo "{}" >> notes.txt',
    'grep -r "{}" .',
    'grep -in "{}" src/main.rs',
    'rg "{}" --type py',
    'notify-send "{}"',
    'curl -X POST https://api.example.com/messages -d \'{{"text": "{}"}}\'',
    'osascript -e \'display notification "{}"\'',
    'sed -i "s/TODO/{}/" README.md',
    'gh pr create --title "{}"',
    'wall "{}"',
    'say "{}"',
    'figlet "{}"',
    'cowsay "{}"',
    'tmux display-message "{}"',
    'git tag -a v1.2.0 -m "{}"',
    'logger -p user.notice "{}"',
)


def quoted_arg_commands(
    phrases: list[str], rng: random.Random, limit: int
) -> list[tuple[str, str]]:
    """Hard negatives: shell commands whose quoted argument is English prose.

    Each command inherits the split of the prompt its phrase came from, so
    no phrase text crosses split boundaries.
    """
    commands: list[tuple[str, str]] = []
    for phrase in phrases:
        phrase = " ".join(phrase.split())
        if not (16 <= len(phrase) <= 90) or '"' in phrase or "\\" in phrase:
            continue
        command = rng.choice(QUOTED_ARG_TEMPLATES).format(phrase)
        commands.append((command, split_for_key(prompt_group_key(phrase))))
        if len(commands) >= limit:
            break
    return commands


def swe_smith_bash_commands(limit: int) -> list[str]:
    """Bash tool calls mined from agent trajectories (SWE-smith)."""
    trajectories = load_dataset("SWE-bench/SWE-smith-trajectories", split="tool", streaming=True)
    commands: list[str] = []
    for row in trajectories:
        for message in json.loads(row["messages"]):
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                if function.get("name") != "bash":
                    continue
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    continue
                command = (arguments.get("command") or "").strip()
                if command:
                    commands.append(command)
                    if len(commands) >= limit:
                        return commands
    return commands


def bash_history_commands(limit: int) -> list[str]:
    """Real user shell history: messy, typo-ridden, realistic commands."""
    history = load_dataset("spignelon/bash_history", split="train")
    commands: list[str] = []
    seen: set[str] = set()
    for row in history:
        command = row["text"].strip()
        if len(command) < MIN_BYTES or command in seen:
            continue
        seen.add(command)
        commands.append(command)
        if len(commands) >= limit:
            break
    return commands


def nl2sh_alfa_train(limit: int) -> tuple[list[str], list[str]]:
    """NL2SH-ALFA training pairs: imperative sysadmin English (prompt class)
    and the matching bash commands (shell class). Hard positives for the
    prompt side: terse verb-first requests full of shell vocabulary."""
    rows = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
    instructions: list[str] = []
    commands: list[str] = []
    for row in rows:
        instructions.append(row["nl"].strip())
        commands.append(row["bash"].strip())
        if len(instructions) >= limit:
            break
    return instructions, commands


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
    for page in pages_dir.glob("**/pages/**/*.md"):
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
    parser.add_argument("--stackoverflow-limit", type=int, default=50_000)
    parser.add_argument("--quoted-arg-limit", type=int, default=20_000)
    parser.add_argument("--agent-bash-limit", type=int, default=60_000)
    parser.add_argument("--bash-history-limit", type=int, default=60_000)
    parser.add_argument("--alfa-limit", type=int, default=30_000)
    args = parser.parse_args()

    rng = random.Random(SEED)

    prompts = prompt_texts(
        args.alpaca_limit,
        args.dolly_limit,
        args.oasst_limit,
        args.sharegpt_limit,
        args.stackoverflow_limit,
    )
    alfa_instructions, alfa_commands = nl2sh_alfa_train(args.alfa_limit)
    prompts.extend(alfa_instructions)
    prompt_count = write_samples(
        args.output,
        "prompt",
        "prompt",
        [(text, split_for_key(prompt_group_key(text))) for text in prompts],
    )

    shell: list[str] = nl2bash_commands()
    shell.extend(alfa_commands)
    shell.extend(swe_smith_bash_commands(args.agent_bash_limit))
    shell.extend(bash_history_commands(args.bash_history_limit))
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

    shell_samples = [(command, split_for_key(shell_group_key(command))) for command in shell]

    hard_negative_source = list(prompts)
    rng.shuffle(hard_negative_source)
    shell_samples.extend(quoted_arg_commands(hard_negative_source, rng, args.quoted_arg_limit))

    shell_count = write_samples(args.output, "shell_command", "shell", shell_samples)

    print(json.dumps({"prompt": prompt_count, "shell_command": shell_count}))


if __name__ == "__main__":
    main()
