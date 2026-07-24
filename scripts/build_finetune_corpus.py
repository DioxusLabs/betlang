#!/usr/bin/env python3
"""Build a fine-tuning corpus for the betlang wordseq student.

The original `bigorig` corpus (GitHub partial-clone blob index, guesslang-style
repository selection) is not redistributable and no longer exists locally, so
this script rebuilds a corpus with the same layout — `files/{train,valid,test}/
{label}/...` — from public sources:

1. `bigcode/the-stack-smol-xl` (ungated) for most programming languages.
2. `bigcode/the-stack` (gated; needs an HF token with accepted terms) for the
   config/data formats missing from smol-xl: yaml, json, toml, ini, xml,
   swift, cobol, plus extra ruby rows filtered for Gemfile/gemspec.
3. GitHub repo tarballs for objectivec (.m) and gradle (.gradle), which have
   no per-language subset in The Stack.
4. A small synthetic set targeting the Markdown/YAML ambiguity from issue #5:
   heading + single-word dash-list Markdown files, and YAML keyed-sequence
   counterexamples. Teacher probabilities are still assigned by the Magika
   teacher when the training cache is built, so these only steer coverage.

Files from one repository always land in the same split (80/10/10 by repo-name
hash), mirroring the guesslang rule that train/valid/test repositories are
disjoint.

Usage:
    python3 scripts/build_finetune_corpus.py --output /tmp/betlang-finetune-corpus
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

SMOL_XL = "bigcode/the-stack-smol-xl"
THE_STACK = "bigcode/the-stack"

# smol-xl language directory -> betlang label.
SMOL_XL_LANGS = {
    "assembly": "asm",
    "batchfile": "batch",
    "c": "c",
    "c++": "cpp",
    "c-sharp": "cs",
    "clojure": "clojure",
    "cmake": "cmake",
    "common-lisp": "lisp",
    "css": "css",
    "dart": "dart",
    "dockerfile": "dockerfile",
    "elixir": "elixir",
    "erlang": "erlang",
    "go": "go",
    "groovy": "groovy",
    "haskell": "haskell",
    "html": "html",
    "java": "java",
    "javascript": "javascript",
    "julia": "julia",
    "kotlin": "kotlin",
    "lua": "lua",
    "markdown": "markdown",
    "ocaml": "ocaml",
    "perl": "perl",
    "php": "php",
    "powershell": "powershell",
    "python": "python",
    "r": "r",
    "ruby": "ruby",
    "rust": "rust",
    "scala": "scala",
    "shell": "shell",
    "sql": "sql",
    "systemverilog": "verilog",
    "typescript": "typescript",
    "verilog": "verilog",
    "visual-basic": "vba",
}

# the-stack language directory -> betlang label (first shard only).
STACK_LANGS = {
    "yaml": "yaml",
    "json": "json",
    "toml": "toml",
    "ini": "ini",
    "xml": "xml",
    "swift": "swift",
    "cobol": "cobol",
}

# GitHub tarballs: label -> (repos, extensions). Repos are chosen so that the
# repo-name hash spreads each label across all three splits.
GITHUB_SOURCES = {
    "objectivec": (
        [
            "AFNetworking/AFNetworking",
            "SDWebImage/SDWebImage",
            "CocoaLumberjack/CocoaLumberjack",
            "jdg/MBProgressHUD",
            "ibireme/YYKit",
            "gnustep/libs-base",
            "facebookarchive/three20",
            "facebookarchive/AsyncDisplayKit",
            "BradLarson/GPUImage",
            "ccgus/fmdb",
            "Mantle/Mantle",
            "magicalpanda/MagicalRecord",
            "robbiehanson/CocoaAsyncSocket",
            "TTTAttributedLabel/TTTAttributedLabel",
            "jessesquires/JSQMessagesViewController",
        ],
        (".m",),
    ),
    "gradle": (
        [
            "gradle/gradle",
            "android/architecture-samples",
            "android/compose-samples",
            "spring-projects/spring-boot",
            "spring-projects/spring-framework",
            "micronaut-projects/micronaut-core",
            "spockframework/spock",
            "nebula-plugins/gradle-lint-plugin",
        ],
        (".gradle",),
    ),
}

# Single raw-file fetches: label -> (repos, path inside the repo).
RAW_FILE_SOURCES = {
    "gemfile": (
        [
            "rails/rails", "jekyll/jekyll", "fastlane/fastlane", "rubocop/rubocop",
            "sinatra/sinatra", "rack/rack", "sidekiq/sidekiq", "heartcombo/devise",
            "kaminari/kaminari", "activeadmin/activeadmin", "ruby-grape/grape",
            "pry/pry", "capistrano/capistrano", "lostisland/faraday",
            "octokit/octokit.rb", "teamcapybara/capybara", "sparklemotion/nokogiri",
            "varvet/pundit", "doorkeeper-gem/doorkeeper", "drapergem/draper",
            "slim-template/slim", "haml/haml", "middleman/middleman", "ruby/rake",
            "rubygems/rubygems", "puma/puma", "resque/resque",
            "mperham/connection_pool", "thoughtbot/factory_bot",
            "carrierwaveuploader/carrierwave", "omniauth/omniauth",
            "rspec/rspec-core", "rspec/rspec-rails", "ankane/searchkick",
        ],
        "Gemfile",
    ),
}

TRAIN_CAP = 3000
VALID_CAP = 375
TEST_CAP = 750
SPLIT_CAPS = {"train": TRAIN_CAP, "valid": VALID_CAP, "test": TEST_CAP}
MIN_BYTES = 8
MAX_BYTES = 256 * 1024

SYNTHETIC_MARKDOWN = {"train": 2000, "valid": 250, "test": 500}
SYNTHETIC_YAML = {"train": 800, "valid": 100, "test": 200}
SYNTHETIC_GEMFILE = {"train": 800, "valid": 100, "test": 200}
# Bare `- item` lists with no heading are valid Markdown and valid YAML; the
# teacher is genuinely split (yaml ~0.6 / markdown ~0.2 after renormalizing to
# the 48-label head), so these rows teach the student *uncertainty* rather
# than a label. Train-only so the valid/test metrics keep measuring
# unambiguous files. Bare `*` lists are excluded: the teacher keeps ~2% of its
# mass in the head there and the renormalized targets are junk.
SYNTHETIC_BARE_LIST = {"train": 1500}

# Synthetic hard-boundary variants for the student-error clusters seen in the
# confusion matrix. Generators live in the hard_gen_* modules (c<->cpp,
# javascript<->typescript, ini, tsx-markup, systemverilog) and are vetted
# against the teacher: the Magika teacher labels these samples as intended
# while the pre-fine-tune student errs on up to 58% of them. Train-only:
# (module, function, label, extension, count).
#
# The list is intentionally empty for the shipped recipe. Three A/B
# fine-tunes showed that at this model capacity (~50 KB) every hard-pair mix
# that meaningfully moved its target confusion cells paid for it elsewhere:
# - c/cpp + javascript/typescript sets only relocated errors inside those
#   genuinely arbitrary boundaries (every JS program is valid TS;
#   C-compatible sources are valid C++) and cost ~0.4 pp fs accuracy.
# - The INI set (setup.cfg/editorconfig-flavored styles) shifted the
#   ini/toml boundary enough to misread real TOML as INI (+30 errors).
# - The tsx/verilog set alone produced no reliable gain over run-to-run
#   seed variance (target cells move by +/-10 counts between seeds).
# The generators are kept for future experiments with larger heads or
# per-pair loss weighting.
SYNTHETIC_HARD_PAIRS: list[tuple[str, str, str, str, int]] = []

NAMES = (
    "Alice Bob Carol Dave Erin Frank Grace Heidi Ivan Judy Mallory Oscar "
    "Peggy Trent Victor Wendy London Paris Tokyo Berlin Madrid Rome Oslo "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January "
    "February March April June July August September October November "
    "December Mercury Venus Earth Mars Jupiter Saturn Uranus Neptune"
).split()

GEM_NAMES = (
    "rails rake rspec minitest puma pg sqlite3 redis sidekiq devise pundit "
    "kaminari nokogiri faraday webmock vcr rubocop pry byebug debug bootsnap "
    "jbuilder turbo-rails stimulus-rails importmap-rails sassc-rails "
    "capybara selenium-webdriver factory_bot_rails simplecov yard rdoc "
    "activerecord activesupport actionpack sinatra rack thin unicorn "
    "aws-sdk-s3 stripe httparty rest-client oj json multi_json dotenv"
).split()

WORDS = (
    "first second third fourth fifth alpha beta gamma delta epsilon apples "
    "bananas carrots oranges grapes monday tuesday wednesday thursday friday "
    "setup install configure build deploy test release docs cleanup refactor "
    "todo done pending blocked review draft merged closed open backlog "
    "red green blue yellow purple orange black white silver golden "
    "alice bob carol dave erin frank grace heidi ivan judy "
    "parser lexer tokenizer compiler linker runtime kernel driver daemon "
    "notes ideas goals tasks items steps points topics sections examples"
).split()

EXT_FOR_LABEL = {
    "asm": ".s", "batch": ".bat", "c": ".c", "clojure": ".clj", "cmake": ".cmake",
    "cobol": ".cob", "cpp": ".cpp", "cs": ".cs", "css": ".css", "dart": ".dart",
    "dockerfile": ".dockerfile", "elixir": ".ex", "erlang": ".erl",
    "gemfile": "", "gemspec": ".gemspec", "go": ".go", "gradle": ".gradle",
    "groovy": ".groovy", "haskell": ".hs", "html": ".html", "ini": ".ini",
    "java": ".java", "javascript": ".js", "json": ".json", "julia": ".jl",
    "kotlin": ".kt", "lisp": ".lisp", "lua": ".lua", "markdown": ".md",
    "objectivec": ".m", "ocaml": ".ml", "perl": ".pl", "php": ".php",
    "powershell": ".ps1", "python": ".py", "r": ".r", "ruby": ".rb",
    "rust": ".rs", "scala": ".scala", "shell": ".sh", "sql": ".sql",
    "swift": ".swift", "toml": ".toml", "typescript": ".ts", "vba": ".vb",
    "verilog": ".v", "xml": ".xml", "yaml": ".yaml",
}


def read_hf_token(explicit: str | None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    raise SystemExit("no HF token: pass --hf-token, set HF_TOKEN, or run `hf auth login`")


def split_for_repo(repo: str) -> str:
    bucket = int.from_bytes(hashlib.md5(repo.encode()).digest()[:4], "little") % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "valid"
    return "test"


class CorpusWriter:
    def __init__(self, root: Path):
        self.root = root
        self.seen: set[bytes] = set()
        self.counts: dict[tuple[str, str], int] = defaultdict(int)

    def full(self, label: str, caps: dict[str, int] | None = None) -> bool:
        caps = caps or SPLIT_CAPS
        return all(self.counts[(split, label)] >= cap for split, cap in caps.items())

    def add(
        self,
        label: str,
        split: str,
        content: bytes,
        ext: str,
        caps: dict[str, int] | None = None,
    ) -> bool:
        caps = caps or SPLIT_CAPS
        if not (MIN_BYTES <= len(content) <= MAX_BYTES):
            return False
        if len(content.lstrip()) < MIN_BYTES:
            return False
        if self.counts[(split, label)] >= caps[split]:
            return False
        digest = hashlib.sha1(content).digest()
        if digest in self.seen:
            return False
        self.seen.add(digest)
        index = self.counts[(split, label)]
        directory = self.root / split / label
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{index:06d}{ext}").write_bytes(content)
        self.counts[(split, label)] += 1
        return True

    def summary(self) -> str:
        labels = sorted({label for _, label in self.counts})
        lines = [f"{'label':<12} {'train':>6} {'valid':>6} {'test':>6}"]
        for label in labels:
            lines.append(
                f"{label:<12} {self.counts[('train', label)]:>6} "
                f"{self.counts[('valid', label)]:>6} {self.counts[('test', label)]:>6}"
            )
        return "\n".join(lines)


def hf_download(repo: str, filename: str, token: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        repo_type="dataset",
        token=token,
        local_dir=dest.parent / "hf",
    )
    return Path(path)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                yield json.loads(line)


def harvest_smol_xl(writer: CorpusWriter, token: str, scratch: Path, rng: random.Random) -> None:
    for lang, label in SMOL_XL_LANGS.items():
        marker = scratch / f"smolxl.{lang}.done"
        if marker.exists():
            print(f"smol-xl {lang}: already harvested", flush=True)
            continue
        print(f"smol-xl {lang} -> {label}: downloading", flush=True)
        data = hf_download(SMOL_XL, f"data/{lang}/data.json", token, scratch / f"{lang}.json")
        rows = list(iter_jsonl(data))
        rng.shuffle(rows)
        added = 0
        for row in rows:
            repo = row.get("max_stars_repo_name") or row["hexsha"]
            path = row.get("max_stars_repo_path") or ""
            content = row["content"].encode("utf-8", errors="replace")
            split = split_for_repo(repo)
            row_label = label
            base = os.path.basename(path)
            if label == "ruby":
                if base == "Gemfile":
                    row_label = "gemfile"
                elif base.endswith(".gemspec"):
                    row_label = "gemspec"
            ext = EXT_FOR_LABEL[row_label]
            if writer.add(row_label, split, content, ext):
                added += 1
        print(f"smol-xl {lang} -> {label}: added {added}", flush=True)
        data.unlink()
        marker.write_text("done\n")


def harvest_stack(writer: CorpusWriter, token: str, scratch: Path, rng: random.Random) -> None:
    import pyarrow.parquet as pq

    stack_langs = dict(STACK_LANGS)
    stack_langs["ruby"] = "ruby"  # extra pass filtered to Gemfile/gemspec rows
    for lang, label in stack_langs.items():
        marker = scratch / f"stack.{lang}.done"
        if marker.exists():
            print(f"the-stack {lang}: already harvested", flush=True)
            continue
        listing = json.loads(
            urllib.request.urlopen(
                urllib.request.Request(
                    f"https://huggingface.co/api/datasets/{THE_STACK}/tree/main/data/{lang}",
                    headers={"Authorization": f"Bearer {token}"},
                )
            ).read()
        )
        shards = sorted(item["path"] for item in listing if item["path"].endswith(".parquet"))
        if not shards:
            raise SystemExit(f"the-stack {lang}: no parquet shards found")
        print(f"the-stack {lang} -> {label}: downloading {shards[0]}", flush=True)
        shard = hf_download(THE_STACK, shards[0], token, scratch / f"{lang}.parquet")
        table = pq.read_table(
            shard,
            columns=["content", "max_stars_repo_name", "max_stars_repo_path"],
        )
        rows = list(range(table.num_rows))
        rng.shuffle(rows)
        contents = table.column("content")
        repos = table.column("max_stars_repo_name")
        paths = table.column("max_stars_repo_path")
        added = 0
        for i in rows:
            repo = repos[i].as_py() or f"row{i}"
            path = paths[i].as_py() or ""
            base = os.path.basename(path)
            if lang == "ruby":
                if base == "Gemfile":
                    row_label = "gemfile"
                elif base.endswith(".gemspec"):
                    row_label = "gemspec"
                else:
                    continue
            else:
                row_label = label
            content = contents[i].as_py().encode("utf-8", errors="replace")
            split = split_for_repo(repo)
            if writer.add(row_label, split, content, EXT_FOR_LABEL[row_label]):
                added += 1
            if lang != "ruby" and writer.full(label):
                break
        print(f"the-stack {lang} -> {label}: added {added}", flush=True)
        shard.unlink()
        marker.write_text("done\n")


def harvest_github(writer: CorpusWriter, scratch: Path) -> None:
    for label, (repos, extensions) in GITHUB_SOURCES.items():
        for repo in repos:
            marker = scratch / f"github.{repo.replace('/', '__')}.done"
            if marker.exists():
                print(f"github {repo}: already harvested", flush=True)
                continue
            split = split_for_repo(repo)
            url = f"https://codeload.github.com/{repo}/tar.gz/HEAD"
            print(f"github {repo} -> {label} ({split}): downloading", flush=True)
            added = 0
            with urllib.request.urlopen(url) as response:
                stream = io.BufferedReader(response, buffer_size=1 << 20)
                with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile() or member.size > MAX_BYTES:
                            continue
                        if not member.name.endswith(extensions):
                            continue
                        extracted = tar.extractfile(member)
                        if extracted is None:
                            continue
                        content = extracted.read()
                        ext = EXT_FOR_LABEL[label]
                        if writer.add(label, split, content, ext):
                            added += 1
            print(f"github {repo} -> {label}: added {added}", flush=True)
            marker.write_text("done\n")


def harvest_raw_files(writer: CorpusWriter, scratch: Path) -> None:
    for label, (repos, filename) in RAW_FILE_SOURCES.items():
        for repo in repos:
            marker = scratch / f"raw.{repo.replace('/', '__')}.done"
            if marker.exists():
                continue
            split = split_for_repo(repo)
            url = f"https://raw.githubusercontent.com/{repo}/HEAD/{filename}"
            try:
                with urllib.request.urlopen(url) as response:
                    content = response.read()
            except urllib.error.HTTPError as error:
                print(f"raw {repo}/{filename}: {error.code}", flush=True)
                marker.write_text("error\n")
                continue
            added = writer.add(label, split, content, EXT_FOR_LABEL[label])
            print(f"raw {repo}/{filename} -> {label} ({split}): added {int(added)}", flush=True)
            marker.write_text("done\n")


def synth_gemfile(rng: random.Random) -> bytes:
    lines = ['source "https://rubygems.org"']
    if rng.random() < 0.3:
        lines.append("")
        lines.append("gemspec")
    lines.append("")
    for _ in range(rng.randint(2, 8)):
        gem = rng.choice(GEM_NAMES)
        version = rng.random()
        if version < 0.4:
            lines.append(f'gem "{gem}"')
        elif version < 0.7:
            major, minor = rng.randint(0, 7), rng.randint(0, 12)
            lines.append(f'gem "{gem}", "~> {major}.{minor}"')
        else:
            lines.append(f'gem "{gem}", ">= {rng.randint(1, 5)}.0"')
    if rng.random() < 0.6:
        group = rng.choice([":development", ":test", ":development, :test"])
        lines.append("")
        lines.append(f"group {group} do")
        for _ in range(rng.randint(1, 3)):
            lines.append(f'  gem "{rng.choice(GEM_NAMES)}"')
        lines.append("end")
    return ("\n".join(lines) + "\n").encode()


def synth_markdown(rng: random.Random) -> bytes:
    lines: list[str] = []
    heading_level = rng.choice(["#", "#", "#", "##"])
    words = rng.sample(WORDS, k=rng.randint(1, 3))
    if rng.random() < 0.85:
        lines.append(f"{heading_level} {' '.join(w.capitalize() for w in words)}")
        lines.append("")
    bullet = rng.choice(["-", "-", "-", "-", "*", "+"])
    style = rng.choice(["lower", "lower", "capital", "title", "title", "names"])
    for _ in range(rng.randint(3, 10)):
        if style == "names":
            item = " ".join(rng.sample(NAMES, k=1 if rng.random() < 0.8 else 2))
        else:
            item_words = rng.sample(WORDS, k=rng.randint(1, 2) if rng.random() < 0.8 else 3)
            item = " ".join(item_words)
            if style == "capital":
                item = item.capitalize()
            elif style == "title":
                item = item.title()
        lines.append(f"{bullet} {item}")
    if rng.random() < 0.25:
        lines.append("")
        lines.append(f"## {rng.choice(WORDS).capitalize()}")
        lines.append("")
        for _ in range(rng.randint(2, 5)):
            lines.append(f"{bullet} {rng.choice(WORDS)}")
    return ("\n".join(lines) + ("\n" if rng.random() < 0.5 else "")).encode()


def synth_bare_list(rng: random.Random) -> bytes:
    style = rng.choice(["lower", "lower", "names", "capital", "two-word"])
    lines = []
    for _ in range(rng.randint(3, 10)):
        if style == "names":
            item = rng.choice(NAMES)
        elif style == "capital":
            item = rng.choice(WORDS).capitalize()
        elif style == "two-word":
            item = f"{rng.choice(WORDS)} {rng.choice(WORDS)}"
        else:
            item = rng.choice(WORDS)
        lines.append(f"- {item}")
    return ("\n".join(lines) + ("\n" if rng.random() < 0.5 else "")).encode()


def synth_yaml(rng: random.Random) -> bytes:
    lines: list[str] = []
    if rng.random() < 0.6:
        lines.append(f"# {' '.join(rng.sample(WORDS, k=rng.randint(1, 3)))}")
    key = rng.choice(["items", "steps", "names", "packages", "tags", "jobs", "hosts"])
    lines.append(f"{key}:")
    indent = rng.choice(["", "  "])
    for _ in range(rng.randint(3, 8)):
        lines.append(f"{indent}- {rng.choice(WORDS)}")
    if rng.random() < 0.5:
        other = rng.choice(["enabled", "count", "version", "name", "region"])
        value = rng.choice(["true", "false", "3", "1.2.0", "us-east-1", rng.choice(WORDS)])
        lines.append(f"{other}: {value}")
    return ("\n".join(lines) + "\n").encode()


def synth_fill(writer: CorpusWriter, label: str, split: str, count: int, ext: str, generator, caps) -> int:
    added = 0
    for _ in range(count * 100):
        if added >= count or writer.counts[(split, label)] >= caps[split]:
            break
        if writer.add(label, split, generator(), ext, caps=caps):
            added += 1
    return added


def harvest_hard_pairs(writer: CorpusWriter, rng: random.Random) -> None:
    import importlib

    for module_name, fn_name, label, ext, count in SYNTHETIC_HARD_PAIRS:
        generator = getattr(importlib.import_module(module_name), fn_name)
        caps = {**SPLIT_CAPS, "train": TRAIN_CAP + 20000}
        added = synth_fill(writer, label, "train", count, ext, lambda gen=generator: gen(rng), caps)
        print(f"hard pairs {fn_name} -> {label}: added {added}", flush=True)


def harvest_synthetic(writer: CorpusWriter, rng: random.Random) -> None:
    for split, count in SYNTHETIC_MARKDOWN.items():
        caps = {**SPLIT_CAPS, split: SPLIT_CAPS[split] + count + SYNTHETIC_BARE_LIST.get(split, 0)}
        synth_fill(writer, "markdown", split, count, ".md", lambda: synth_markdown(rng), caps)
    for split, count in SYNTHETIC_BARE_LIST.items():
        caps = {**SPLIT_CAPS, split: SPLIT_CAPS[split] + count + SYNTHETIC_MARKDOWN.get(split, 0)}
        synth_fill(writer, "markdown", split, count, ".md", lambda: synth_bare_list(rng), caps)
    for split, count in SYNTHETIC_YAML.items():
        caps = {**SPLIT_CAPS, split: SPLIT_CAPS[split] + count}
        synth_fill(writer, "yaml", split, count, ".yaml", lambda: synth_yaml(rng), caps)
    for split, count in SYNTHETIC_GEMFILE.items():
        synth_fill(writer, "gemfile", split, count, "", lambda: synth_gemfile(rng), SPLIT_CAPS)
    print("synthetic markdown/yaml/gemfile: added", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()

    token = read_hf_token(args.hf_token)
    rng = random.Random(args.seed)
    files_root = args.output / "files"
    scratch = args.output / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    writer = CorpusWriter(files_root)

    # Rebuild dedup/counters from any files already on disk (resume support).
    for split_dir in files_root.glob("*/"):
        for label_dir in split_dir.glob("*/"):
            for file in label_dir.iterdir():
                writer.seen.add(hashlib.sha1(file.read_bytes()).digest())
                writer.counts[(split_dir.name, label_dir.name)] += 1

    harvest_smol_xl(writer, token, scratch, rng)
    harvest_stack(writer, token, scratch, rng)
    harvest_github(writer, scratch)
    harvest_raw_files(writer, scratch)
    if not (scratch / "synthetic.done").exists():
        harvest_synthetic(writer, rng)
        (scratch / "synthetic.done").write_text("done\n")
    if not (scratch / "synthetic_hard_pairs.done").exists():
        harvest_hard_pairs(writer, rng)
        (scratch / "synthetic_hard_pairs.done").write_text("done\n")

    print(writer.summary())
    (args.output / "corpus_summary.txt").write_text(writer.summary() + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
