"""Download, clean, split. The first stage of the DVC pipeline.

python scripts/prepare_data.py            # uses the cache if present
python scripts/prepare_data.py --force    # re-download and re-clean
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.load import build_clean, write_distribution_markdown  # noqa: E402
from arabic_nlp.data.splits import distribution, make_splits, save_splits  # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    settings = load_settings()
    frame, report = build_clean(settings, force=args.force)

    print(report.render())
    write_distribution_markdown(report, ROOT / "data" / "class_distribution.md")

    splits = make_splits(frame, settings)
    save_splits(splits)

    print()
    print("SPLITS")
    print("| Split | rows | " + " | ".join(sorted(settings.dataset.label_names)) + " |")
    print("|---" * (len(settings.dataset.label_names) + 2) + "|")
    for name, part in splits.items():
        shares = distribution(part)
        cells = " | ".join(
            f"{shares.get(k, 0.0):.3f}" for k in sorted(settings.dataset.label_names)
        )
        print(f"| {name} | {len(part)} | {cells} |")
    print()
    print("The shares match across splits because the split is stratified.")

    (ROOT / "data" / "split_report.json").write_text(
        json.dumps(
            {
                "seed": settings.split.seed,
                "test_size": settings.split.test_size,
                "validation_size": settings.split.validation_size,
                "max_train_rows": settings.split.max_train_rows,
                "rows": {name: len(part) for name, part in splits.items()},
                "distribution": {name: distribution(part) for name, part in splits.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
