#!/usr/bin/env python3
"""Build the natural-language vs LLM-prompt training corpus.

Downloads public Hugging Face datasets and writes one text file per sample
under `--output/files/{train,valid,test}/{natural_language,prompt}/`.

Class definitions:

- `prompt`: text a user would type at a language model to make it do
  something — instructions, task requests, role-play setups, questions
  addressed to an assistant. Sources: Alpaca and Dolly instructions (with
  and without their task input/context blocks) and the awesome-chatgpt-prompts
  persona prompts.
- `natural_language`: prose that is not addressed to a model — encyclopedia
  paragraphs, news, reviews, and other narrative text. Sources: WikiText-103
  paragraphs, AG News, IMDB, and Yelp reviews.

Every sample is randomly assigned to train/valid/test (90/5/5) with a
deterministic seed so rebuilds are reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
from pathlib import Path

from datasets import load_dataset

SPLITS = ("train", "valid", "test")
SPLIT_WEIGHTS = (0.90, 0.05, 0.05)
MIN_BYTES = 8
MAX_BYTES = 8192
SEED = 2


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


def sentence_chunks(text: str, rng: random.Random) -> list[str]:
    """Emit the full text plus a short 1-2 sentence slice.

    Prompts skew short while prose sources skew long; without short prose
    samples the model learns length instead of style, so every long natural
    sample also contributes a sentence-level counterexample.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    chunks = [text]
    if len(sentences) >= 2:
        start = rng.randrange(len(sentences))
        count = rng.choice((1, 2))
        chunk = " ".join(sentences[start : start + count])
        if len(chunk) >= MIN_BYTES:
            chunks.append(chunk)
    return chunks


def prompt_texts(limit_alpaca: int, limit_dolly: int) -> list[str]:
    texts: list[str] = []

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


def natural_texts(
    limit_wiki: int, limit_news: int, limit_imdb: int, limit_yelp: int, rng: random.Random
) -> list[str]:
    texts: list[str] = []

    wiki_count = 0
    wikitext = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    for row in wikitext:
        line = row["text"].strip()
        # Skip headings and stub lines; keep real paragraphs.
        if line.startswith("=") or len(line) < 120:
            continue
        texts.extend(sentence_chunks(line, rng))
        wiki_count += 1
        if wiki_count >= limit_wiki:
            break

    news = load_dataset("fancyzhx/ag_news", split="train")
    texts.extend(row["text"] for row in news.select(range(min(limit_news, len(news)))))

    imdb = load_dataset("stanfordnlp/imdb", split="train")
    for row in imdb.select(range(min(limit_imdb, len(imdb)))):
        texts.extend(sentence_chunks(row["text"].replace("<br />", "\n"), rng))

    yelp = load_dataset("Yelp/yelp_review_full", split="train")
    for row in yelp.select(range(min(limit_yelp, len(yelp)))):
        texts.extend(sentence_chunks(row["text"].replace("\\n", "\n"), rng))

    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpaca-limit", type=int, default=24000)
    parser.add_argument("--dolly-limit", type=int, default=15000)
    parser.add_argument("--wiki-limit", type=int, default=16000)
    parser.add_argument("--news-limit", type=int, default=10000)
    parser.add_argument("--imdb-limit", type=int, default=6000)
    parser.add_argument("--yelp-limit", type=int, default=6000)
    args = parser.parse_args()

    rng = random.Random(SEED)
    prompt_count = write_samples(args.output, "prompt", "prompt", prompt_texts(args.alpaca_limit, args.dolly_limit), rng)
    natural_count = write_samples(
        args.output,
        "natural_language",
        "natural",
        natural_texts(args.wiki_limit, args.news_limit, args.imdb_limit, args.yelp_limit, rng),
        rng,
    )
    print(f"prompt={prompt_count} natural_language={natural_count}")


if __name__ == "__main__":
    main()
