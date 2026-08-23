#!/usr/bin/env python3
"""Build a synthetic hard-pair corpus (windows + labels) reusing betlang's
hard_gen_* generators, plus small cmake/gemspec/r/ruby/toml/ocaml/haskell
generators for the remaining confusable short-file pairs."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "repos/betlang/scripts"))
sys.path.insert(0, str(Path(__file__).parent))
from hard_gen_ini import synth_hard_ini
from hard_gen_js_ts import synth_hard_javascript, synth_hard_typescript
from train_bnn import OUT, WINDOW

LABELS = json.loads((Path.home() / "work/cache/valid.json").read_text())["labels"]

W1 = ("alpha beta gamma delta core util math data net io fmt log cfg http "
      "json test demo file cache queue stack list map set node tree lex "
      "parse token scan").split()


def _n(rng):
    return rng.choice(W1) + (rng.choice(["", "_" + rng.choice(W1)]))


def synth_cmake(rng: random.Random) -> bytes:
    n = _n(rng)
    lines = []
    if rng.random() < 0.5:
        lines.append(f"cmake_minimum_required(VERSION 3.{rng.randint(10, 28)})")
    lines.append(f"project({n}{rng.choice(['', ' C', ' CXX', ' VERSION 1.0'])})")
    for _ in range(rng.randint(1, 5)):
        kind = rng.random()
        tgt = _n(rng)
        srcs = " ".join(f"{_n(rng)}.{rng.choice(['c', 'cpp', 'cc'])}"
                        for _ in range(rng.randint(1, 3)))
        if kind < 0.35:
            lines.append(f"add_library({tgt} {srcs})")
        elif kind < 0.7:
            lines.append(f"add_executable({tgt} {srcs})")
        elif kind < 0.85:
            lines.append(f"target_link_libraries({tgt} PRIVATE {_n(rng)})")
        else:
            lines.append(f"set({tgt.upper()}_SOURCES {srcs})")
    if rng.random() < 0.6:
        lines.append(f"install(TARGETS {_n(rng)})")
    if rng.random() < 0.4:
        lines.append(f"include_directories({rng.choice(['include', 'src'])})")
    if rng.random() < 0.3:
        lines.insert(rng.randint(0, len(lines)), f"# {_n(rng)} build rules")
    return ("\n".join(lines) + "\n").encode()


def synth_gemspec(rng: random.Random) -> bytes:
    n = _n(rng).replace("_", "-")
    var = rng.choice(["spec", "s", "gem"])
    lines = []
    if rng.random() < 0.3:
        lines.append("# frozen_string_literal: true")
        lines.append("")
    if rng.random() < 0.4:
        lines.append(f'require_relative "lib/{n.replace("-", "/")}/version"')
        lines.append("")
    lines.append(f"Gem::Specification.new do |{var}|")
    lines.append(f'  {var}.name = "{n}"')
    lines.append(f'  {var}.version = "{rng.randint(0, 3)}.{rng.randint(0, 9)}.'
                 f'{rng.randint(0, 9)}"')
    if rng.random() < 0.7:
        lines.append(f'  {var}.summary = "A {rng.choice(W1)} {rng.choice(W1)} library"')
    if rng.random() < 0.6:
        lines.append(f'  {var}.authors = ["{rng.choice(W1).title()}"]')
    if rng.random() < 0.5:
        lines.append(f'  {var}.files = Dir["lib/**/*.rb"]')
    if rng.random() < 0.5:
        lines.append(f'  {var}.required_ruby_version = ">= {rng.randint(2, 3)}.'
                     f'{rng.randint(0, 4)}"')
    if rng.random() < 0.4:
        lines.append(f'  {var}.license = "{rng.choice(["MIT", "Apache-2.0"])}"')
    for _ in range(rng.randint(0, 3)):
        dep = rng.choice(["json", "rake", "rspec", "minitest", "nokogiri", "thor"])
        kind = rng.choice(["add_dependency", "add_development_dependency",
                           "add_runtime_dependency"])
        lines.append(f'  {var}.{kind} "{dep}", "~> {rng.randint(1, 13)}.'
                     f'{rng.randint(0, 9)}"')
    lines.append("end")
    return ("\n".join(lines) + "\n").encode()


def synth_ruby(rng: random.Random) -> bytes:
    n = _n(rng)
    cls = "".join(p.title() for p in n.split("_"))
    lines = []
    if rng.random() < 0.3:
        lines.append("# frozen_string_literal: true")
        lines.append("")
    style = rng.random()
    if style < 0.4:
        lines.append(f"class {cls}")
        lines.append(f"  def initialize({_n(rng)})")
        lines.append(f"    @{_n(rng)} = {_n(rng)}")
        lines.append("  end")
        lines.append("")
        lines.append(f"  def {_n(rng)}")
        lines.append(f"    @{_n(rng)}.to_s")
        lines.append("  end")
        lines.append("end")
    elif style < 0.7:
        lines.append(f"module {cls}")
        lines.append(f"  def self.{_n(rng)}(items)")
        lines.append(f"    items.map {{ |x| x.{rng.choice(['to_s', 'upcase', 'strip'])} }}")
        lines.append("  end")
        lines.append("end")
    else:
        lines.append(f"{_n(rng)} = [{rng.randint(1, 9)}, {rng.randint(10, 99)}]")
        lines.append(f"{_n(rng)}.each do |v|")
        lines.append('  puts "#{v}"')
        lines.append("end")
    return ("\n".join(lines) + "\n").encode()


def synth_r(rng: random.Random) -> bytes:
    lines = []
    if rng.random() < 0.4:
        lines.append(f"library({rng.choice(['dplyr', 'ggplot2', 'stats'])})")
    n = _n(rng)
    lines.append(f"{n} <- function(x, y = {rng.randint(1, 9)}) {{")
    lines.append(f"  z <- x * y + {rng.randint(1, 20)}")
    lines.append("  return(z)")
    lines.append("}")
    if rng.random() < 0.6:
        lines.append(f"{_n(rng)} <- c({rng.randint(1, 5)}, {rng.randint(6, 9)})")
    if rng.random() < 0.5:
        lines.append(f"print({n}({rng.randint(1, 5)}))")
    return ("\n".join(lines) + "\n").encode()


GENERATORS = {
    "cmake": synth_cmake,
    "gemspec": synth_gemspec,
    "ruby": synth_ruby,
    "r": synth_r,
    "ini": synth_hard_ini,
    "javascript": synth_hard_javascript,
    "typescript": synth_hard_typescript,
}
COUNTS = {"cmake": 3000, "gemspec": 3000, "ruby": 1500, "r": 1500,
          "ini": 3000, "javascript": 2000, "typescript": 3000}


def build_window(source: bytes):
    ws = b"\t\n\x0b\x0c\r "
    beg = source[:4096].lstrip(ws)
    if len(beg) < 8:
        return None
    end = source[-4096:].rstrip(ws)
    beg_len = min(len(beg), 1024)
    buf = np.zeros(WINDOW, dtype=np.uint8)
    buf[:beg_len] = np.frombuffer(beg[:beg_len], dtype=np.uint8)
    end_len = min(len(end), 1024)
    end_start = 1024 + (1024 - end_len)
    buf[end_start:end_start + end_len] = np.frombuffer(
        end[len(end) - end_len:], dtype=np.uint8)
    if beg_len < 1024:
        valid = beg_len
    elif end_start > 1024:
        valid = 1024
    else:
        valid = WINDOW
    return buf, valid


def main():
    rng = random.Random(42)
    rows, lens, labs = [], [], []
    seen = set()
    for label, gen in GENERATORS.items():
        lid = LABELS.index(label)
        made = 0
        attempts = 0
        while made < COUNTS[label] and attempts < COUNTS[label] * 50:
            attempts += 1
            data = gen(rng)
            if data in seen:
                continue
            seen.add(data)
            w = build_window(data)
            if w is None:
                continue
            rows.append(w[0])
            lens.append(w[1])
            labs.append(lid)
            made += 1
        print(f"{label}: {made} samples", flush=True)
    windows = np.stack(rows)
    np.save(OUT / "synth_hard.windows.npy", windows)
    np.save(OUT / "synth_hard.lengths.npy", np.array(lens, dtype=np.int32))
    np.save(OUT / "synth_hard.labels.npy", np.array(labs, dtype=np.int64))
    print("total", len(labs))


if __name__ == "__main__":
    main()
