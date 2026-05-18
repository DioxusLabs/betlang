#!/usr/bin/env python3
"""Training driver for `assets/magika/source-student-q4.bin`.

It shells out to `train_magika_qat_student.py` with the shipped architecture,
cache paths, and hyperparameters.

Recipe:

  Architecture       wordseq-b1024-k3-m2048-tiny-3conv-hidden (~50 KB exported)
                     - K=3 HashEmbedding (1024 bins x 24 dim, 4-bit) ~12 KB
                     - 3 conv stages (64 -> 128 -> 128 ch, 2-bit) with
                       MaxPool(4) then MaxPool(2)
                     - Dense 96 (2-bit) -> output 48 (4-bit)
                     - Export budget: 50 KB.

  Cache              fixed 48-label production cache
                     - units_v3.mmap (v3 tokenizer)
                     - labels.mmap / probabilities.mmap in the fixed head order
                     - self_probabilities.mmap (student self-distillation)

  Soft target (0.5)  self_probabilities.mmap.
  Hard target (0.5)  cache labels.mmap.
  Distill temperature 3.

  Label smoothing    0.05 - combats the train-loss-collapse-and-overfit pattern.
  CutMix prob        0.5 - Magika v3.1 augmentation.

  LR schedule        cosine 8e-4 → 5% over 60 epochs (AdamW grad-clip 1.0).
  QAT                4-bit weights from epoch 45; early-stop patience 6.
  Throughput         length-buckets ~2x; mixed-precision.

Expected shipped artifact:
  test_fs_accuracy    0.965238 (manifest-aligned filesystem-label split)
  macro_recall        0.965411
  exported size       47,840 bytes

Usage:
    python scripts/train_v2_student.py                  # uses local cache defaults
    python scripts/train_v2_student.py --output other.bin
    python scripts/train_v2_student.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "scripts" / "train_magika_qat_student.py"

DEFAULTS = {
    "python": os.environ.get("PY", "python3"),
    "dataset": "/tmp/magika-source-corpus-bigorig/files",
    "cache_dir": "/tmp/magika-source-qat-cache-guesslang-direct-fs-pruned48",
    "magika_model": "/tmp/magika/assets/models/standard_v3_3/model.onnx",
    "magika_config": "/tmp/magika/assets/models/standard_v3_3/config.pruned48.min.json",
    "self_probabilities": "/tmp/magika-source-qat-cache-guesslang-direct-fs-pruned48",
    "output": str(REPO_ROOT / "assets" / "magika" / "source-student-q4.bin"),
    "confusion_output": str(REPO_ROOT / "assets" / "magika" / "source-student-q4-confusion.json"),
    "max_export_bytes": 50000,
}

ARCHITECTURE = "wordseq-b1024-k3-m2048-tiny-3conv-hidden"


def build_cmd(a: argparse.Namespace) -> list[str]:
    return [
        a.python,
        str(TRAINER),
        "--dataset", a.dataset,
        "--cache-dir", a.cache_dir,
        "--magika-model", a.magika_model,
        "--magika-config", a.magika_config,
        "--output", a.output,
        "--architecture", ARCHITECTURE,
        "--length-buckets",
        "--epochs", str(a.epochs),
        "--batch-size", str(a.batch_size),
        "--learning-rate", str(a.learning_rate),
        "--cosine-decay",
        "--min-learning-rate-ratio", "0.05",
        "--weight-bits", "4",
        "--qat-start-epoch", str(a.qat_start_epoch),
        "--distill-temperature", "3",
        "--hard-loss-weight", str(a.hard_loss_weight),
        "--self-probabilities", a.self_probabilities,
        "--self-loss-weight", str(a.self_loss_weight),
        "--label-smoothing", str(a.label_smoothing),
        "--cutmix-prob", str(a.cutmix_prob),
        "--early-stop-patience", str(a.early_stop_patience),
        "--seed", str(a.seed),
        "--mixed-precision",
        "--eval-every", str(a.eval_every),
        "--confusion-matrix-output", a.confusion_output,
        "--confusion-matrix-top", "25",
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--python", default=DEFAULTS["python"])
    p.add_argument("--dataset", default=DEFAULTS["dataset"])
    p.add_argument("--cache-dir", default=DEFAULTS["cache_dir"])
    p.add_argument("--magika-model", default=DEFAULTS["magika_model"])
    p.add_argument("--magika-config", default=DEFAULTS["magika_config"])
    p.add_argument("--self-probabilities", default=DEFAULTS["self_probabilities"])
    p.add_argument("--output", default=DEFAULTS["output"])
    p.add_argument("--confusion-output", default=DEFAULTS["confusion_output"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=8e-4)
    p.add_argument("--qat-start-epoch", type=int, default=45)
    p.add_argument("--early-stop-patience", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--hard-loss-weight", type=float, default=0.5)
    p.add_argument("--self-loss-weight", type=float, default=0.5)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--cutmix-prob", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--max-export-bytes", type=int, default=DEFAULTS["max_export_bytes"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cmd = build_cmd(args)
    print("$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    final_metrics: dict[str, float] = {}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = re.match(r"^(test_teacher_parity|test_loss|valid_teacher_parity|valid_loss)=([0-9.]+)\s*$", line)
        if m:
            final_metrics[m.group(1)] = float(m.group(2))
    rc = proc.wait()
    if rc != 0:
        print(f"trainer exited with code {rc}", file=sys.stderr)
        return rc

    out_path = Path(args.output)
    if not out_path.exists():
        print(f"missing output file {out_path}", file=sys.stderr)
        return 2
    size = out_path.stat().st_size
    print()
    print("=" * 64)
    print(f"exported_model_path={out_path}")
    print(f"exported_model_size_bytes={size}")
    if "test_teacher_parity" in final_metrics:
        print(f"test_teacher_parity={final_metrics['test_teacher_parity']:.6f}")
    print("=" * 64)
    if size > args.max_export_bytes:
        print(f"FAIL: exported model {size} bytes exceeds max {args.max_export_bytes}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
