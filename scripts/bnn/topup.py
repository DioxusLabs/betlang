#!/usr/bin/env python3
"""Top-up harvester for starved labels (ini, cobol). Splits files across
train/valid/test by path hash; dedupes by content sha1 against whole corpus."""
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

root = Path(sys.argv[1])  # corpus/files
MIN_BYTES, MAX_BYTES = 64, 262144

SOURCES = {
    "cobol": ([
        "eclipse-che4z/che-che4z-lsp-for-cobol", "OCamlPro/superbol-studio-oss",
        "GnuCOBOL/GnuCOBOL", "simonsobol? no", "MicroFocus/CICS-Banking-Sample-Application-CBSA",
    ], (".cbl", ".cob", ".CBL", ".COB", ".cpy"), 4200),
}

seen = set()
for split_dir in root.glob("*/"):
    for label_dir in split_dir.glob("*/"):
        for f in label_dir.iterdir():
            seen.add(hashlib.sha1(f.read_bytes()).digest())
print(f"seen {len(seen)} existing files", flush=True)

for label, (repos, exts, cap) in SOURCES.items():
    counts = {s: sum(1 for _ in (root / s / label).glob("*")) for s in ("train", "valid", "test")}
    added_total = 0
    for repo in repos:
        total = sum(counts.values())
        if total >= cap:
            break
        url = f"https://codeload.github.com/{repo}/tar.gz/HEAD"
        print(f"{label} {repo}: downloading", flush=True)
        added = 0
        try:
            with urllib.request.urlopen(url) as response:
                stream = io.BufferedReader(response, buffer_size=1 << 20)
                with tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    for member in tar:
                        if not member.isfile() or member.size > MAX_BYTES:
                            continue
                        if not member.name.endswith(exts):
                            continue
                        fobj = tar.extractfile(member)
                        if fobj is None:
                            continue
                        content = fobj.read()
                        if not (MIN_BYTES <= len(content) <= MAX_BYTES):
                            continue
                        if len(content.lstrip()) < MIN_BYTES:
                            continue
                        digest = hashlib.sha1(content).digest()
                        if digest in seen:
                            continue
                        seen.add(digest)
                        bucket = int(hashlib.sha1(member.name.encode()).hexdigest(), 16) % 10
                        split = "valid" if bucket == 0 else ("test" if bucket <= 2 else "train")
                        dest = root / split / label
                        dest.mkdir(parents=True, exist_ok=True)
                        (dest / f"topup_{digest.hex()[:16]}").write_bytes(content)
                        counts[split] += 1
                        added += 1
                        if sum(counts.values()) >= cap:
                            break
        except Exception as err:
            print(f"{label} {repo}: FAILED {err}", flush=True)
            continue
        added_total += added
        print(f"{label} {repo}: added {added}", flush=True)
    print(f"{label}: total added {added_total}, counts {counts}", flush=True)
