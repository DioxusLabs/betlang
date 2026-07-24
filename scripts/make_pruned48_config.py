#!/usr/bin/env python3
"""Generate the pruned 48-label Magika teacher config.

`train_magika_source_student.load_teacher` selects teacher output columns by
looking up each `SOURCE_LABELS` entry in the config's `target_labels_space`.
The indices must keep matching the ONNX output columns, so this script does
not remove entries: it renames the source labels that are outside the fixed
48-label production head so they no longer match `SOURCE_LABELS`.

Usage:
    python3 scripts/make_pruned48_config.py \
      --magika-config /path/to/standard_v3_3/config.min.json \
      --output /path/to/standard_v3_3/config.pruned48.min.json

Without --magika-config the config is located inside the installed `magika`
pip package.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_magika_qat_student import FIXED_EXPORT_LABELS  # noqa: E402
from train_magika_source_student import SOURCE_LABELS  # noqa: E402


def default_config_path() -> Path:
    import magika

    return Path(magika.__file__).parent / "models" / "standard_v3_3" / "config.min.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--magika-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.magika_config or default_config_path()
    config = json.loads(config_path.read_text())
    target_labels = config["target_labels_space"]

    source_label_set = {label for label, _ in SOURCE_LABELS}
    fixed_label_set = set(FIXED_EXPORT_LABELS)
    pruned = sorted(source_label_set - fixed_label_set)
    missing = sorted(fixed_label_set - set(target_labels))
    if missing:
        raise SystemExit(f"teacher config is missing fixed head labels: {missing}")

    config["target_labels_space"] = [
        f"pruned:{label}" if label in pruned else label for label in target_labels
    ]

    kept = [
        label
        for label, _ in SOURCE_LABELS
        if label in fixed_label_set and label in set(config["target_labels_space"])
    ]
    if kept != FIXED_EXPORT_LABELS:
        raise SystemExit(
            "pruned config selects labels that do not match the fixed head:\n"
            f"selected={kept}\nexpected={FIXED_EXPORT_LABELS}"
        )

    args.output.write_text(json.dumps(config, separators=(",", ":")) + "\n")
    print(f"source_config={config_path}")
    print(f"pruned_labels={','.join(pruned)}")
    print(f"selected_labels={len(kept)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
