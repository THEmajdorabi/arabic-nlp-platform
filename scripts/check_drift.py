"""Build the reference distribution, compare the request log against it, alert.

    python scripts/check_drift.py --build-reference     # once, after training
    python scripts/check_drift.py                       # the scheduled job

Exits 1 when the drift share crosses the alert threshold, so a scheduler or a
CI job can act on it rather than only writing a file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split  # noqa: E402
from arabic_nlp.serving.monitor import (  # noqa: E402
    REFERENCE_PATH,
    compute_drift,
    load_request_log,
    save_report,
)


def build_reference(settings, model_dir: str) -> pd.DataFrame:
    """Reference = the model's own behaviour on training-distribution text.

    Not the ground-truth label distribution: the service never sees labels, so
    a reference built from them would flag as drift the model's ordinary error
    pattern.
    """
    from arabic_nlp.serving.predictor import Predictor

    frame = load_split("train").head(settings.drift.reference_rows).copy()
    predictor = Predictor(model_dir, settings)
    predictions = predictor.predict(frame["text"].tolist())

    frame["chars"] = frame["text"].str.len()
    frame["words"] = frame["text"].str.split().str.len()
    frame["label"] = [p.label for p in predictions]
    frame["confidence"] = [p.confidence for p in predictions]
    return frame[["chars", "words", "label", "confidence"]]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--log", default="")
    parser.add_argument("--window", type=int, default=0)
    args = parser.parse_args()

    settings = load_settings()

    if args.build_reference:
        if not args.model_dir:
            print("--build-reference needs --model-dir")
            return 1
        reference = build_reference(settings, args.model_dir)
        REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        reference.to_parquet(REFERENCE_PATH, index=False)
        print(f"wrote {REFERENCE_PATH} with {len(reference)} rows")
        return 0

    if not REFERENCE_PATH.exists():
        print(f"no reference at {REFERENCE_PATH}. Run with --build-reference first.")
        return 1

    reference = pd.read_parquet(REFERENCE_PATH)
    current = load_request_log(
        args.log or (ROOT / settings.serving.request_log),
        limit=args.window or settings.drift.window_rows,
    )
    if len(current) < 30:
        print(f"only {len(current)} logged predictions; too few to judge drift. Not alerting.")
        return 0

    report = compute_drift(current, reference, settings)
    print(report.render())
    save_report(report)

    if report.alert:
        print()
        print("ALERT: the input distribution has moved. See the retraining policy in the README.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
