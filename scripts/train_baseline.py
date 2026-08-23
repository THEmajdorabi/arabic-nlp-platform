"""Train and evaluate the TF-IDF baseline. No GPU, no API key, a few minutes.

This runs before the transformer and produces the number it has to beat.

    python scripts/train_baseline.py                    # validation only
    python scripts/train_baseline.py --final            # ALSO scores the test set
    python scripts/train_baseline.py --no-normalize     # the normalization ablation

The test set is read only with `--final`, and the flag exists so that reading
it is a deliberate act that shows up in shell history.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split, load_test  # noqa: E402
from arabic_nlp.evaluate import (  # noqa: E402
    confusion_pairs,
    evaluate,
    save_calibration_image,
    save_confusion_image,
    save_result,
    worst_errors,
)
from arabic_nlp.models.baseline import top_features, train_baseline  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "baseline"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="also score the test set")
    parser.add_argument("--no-normalize", action="store_true", help="ablate Arabic normalization")
    parser.add_argument("--tag", default="", help="suffix for the artifact filenames")
    args = parser.parse_args()

    settings = load_settings()
    if args.no_normalize:
        settings.baseline.normalize_arabic = False

    labels = settings.dataset.label_names
    index = {name: i for i, name in enumerate(labels)}

    train = load_split("train")
    validation = load_split("validation")

    started = time.perf_counter()
    model, info = train_baseline(train["text"].tolist(), train["label"].tolist(), settings)
    train_seconds = time.perf_counter() - started

    tag = args.tag or ("raw" if not settings.baseline.normalize_arabic else "normalized")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS / f"model_{tag}.joblib")

    def score(frame, split_name: str):  # noqa: ANN202
        texts = frame["text"].tolist()
        y_true = np.array([index[label] for label in frame["label"]])
        probabilities = model.predict_proba(texts)
        y_pred = probabilities.argmax(axis=1)
        result = evaluate(
            y_true,
            y_pred,
            probabilities,
            labels,
            extra={
                "split": split_name,
                "model": "tfidf+logreg",
                "normalize_arabic": settings.baseline.normalize_arabic,
                "train_seconds": round(train_seconds, 1),
                **info,
            },
        )
        return result, texts, y_true, y_pred, probabilities

    result, texts, y_true, y_pred, probabilities = score(validation, "validation")
    print(
        result.render(
            f"BASELINE (tfidf+logreg, normalize_arabic={settings.baseline.normalize_arabic}) on VALIDATION"
        )
    )
    save_result(result, ARTIFACTS / f"validation_{tag}.json")
    save_confusion_image(
        result, ARTIFACTS / f"confusion_validation_{tag}.png", "baseline, validation"
    )
    save_calibration_image(
        result, ARTIFACTS / f"calibration_validation_{tag}.png", "baseline, validation"
    )

    print()
    print("MOST CONFUSED PAIRS (share of the true class)")
    for pair in confusion_pairs(result):
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['rate']:.3f}  ({pair['n']} rows)")

    print()
    print("TOP FEATURES PER CLASS")
    for label, features in top_features(model).items():
        print(f"  {label}: {', '.join(features)}")

    errors = worst_errors(texts, y_true, y_pred, probabilities, labels)
    (ARTIFACTS / f"worst_errors_{tag}.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.final:
        print()
        print("!! Reading the test set. This should happen once, at the end.")
        test_result, *_ = score(load_test(), "test")
        print(test_result.render("BASELINE on TEST"))
        save_result(test_result, ARTIFACTS / f"test_{tag}.json")
        save_confusion_image(test_result, ARTIFACTS / f"confusion_test_{tag}.png", "baseline, test")

    print()
    print(f"trained in {train_seconds:.1f}s on {info['rows']} rows; artifacts in {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
