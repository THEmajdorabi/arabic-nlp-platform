"""Score the test set. Run this ONCE, at the end.

Every other script in this repo reports validation numbers. This is the only
one that reads `data/splits/test.parquet`, and the README records how many
times it has been run.

    python scripts/final_eval.py --model-dir <dir> --baseline artifacts/baseline/model_raw.joblib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_test  # noqa: E402
from arabic_nlp.evaluate import (  # noqa: E402
    bootstrap_macro_f1,
    confusion_pairs,
    evaluate,
    paired_macro_f1_difference,
    save_calibration_image,
    save_confusion_image,
    save_result,
    worst_errors,
)

ARTIFACTS = ROOT / "artifacts" / "final"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="the fine-tuned model directory")
    parser.add_argument("--baseline", default="", help="the baseline .joblib, for the comparison")
    parser.add_argument("--backend", default="torch", choices=["torch", "onnx"])
    parser.add_argument(
        "--i-know-this-touches-the-test-set",
        action="store_true",
        help="required. Reading the test set should be a deliberate act.",
    )
    args = parser.parse_args()

    if not args.i_know_this_touches_the_test_set:
        print(
            "Refusing to read the test set without --i-know-this-touches-the-test-set.\n"
            "This is not bureaucracy: a test set read casually during development is a\n"
            "test set you have tuned against, and every number from it afterwards is\n"
            "optimistic by an unknown amount."
        )
        return 2

    settings = load_settings()
    labels = settings.dataset.label_names
    index = {name: i for i, name in enumerate(labels)}

    test = load_test()
    texts = test["text"].tolist()
    y_true = np.array([index[label] for label in test["label"]])
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    from arabic_nlp.serving.predictor import Predictor

    predictor = Predictor(args.model_dir, settings, backend=args.backend)
    probabilities = predictor.probabilities(texts)
    y_pred = probabilities.argmax(axis=1)

    interval = bootstrap_macro_f1(y_true, y_pred, labels)
    result = evaluate(
        y_true,
        y_pred,
        probabilities,
        labels,
        extra={
            "split": "test",
            "model_dir": args.model_dir,
            "backend": args.backend,
            "macro_f1_ci": interval,
        },
    )
    print(result.render("FINE-TUNED -- TEST"))
    print(
        f"macro F1 {interval['macro_f1']:.4f} [{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
    )
    save_result(result, ARTIFACTS / "test_transformer.json")
    save_confusion_image(result, ARTIFACTS / "confusion_test.png", "MARBERTv2, test")
    save_calibration_image(result, ARTIFACTS / "calibration_test.png", "MARBERTv2, test")
    (ARTIFACTS / "worst_errors_test.json").write_text(
        json.dumps(
            worst_errors(texts, y_true, y_pred, probabilities, labels), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )

    print()
    print("MOST CONFUSED PAIRS")
    for pair in confusion_pairs(result):
        print(f"  {pair['true']} -> {pair['predicted']}: {pair['rate']:.3f} ({pair['n']} rows)")

    if args.baseline:
        from arabic_nlp.models.baseline import BaselineModel

        baseline = BaselineModel.load(args.baseline)
        baseline_probabilities = baseline.predict_proba(texts)
        baseline_pred = baseline_probabilities.argmax(axis=1)

        baseline_result = evaluate(
            y_true, baseline_pred, baseline_probabilities, labels, extra={"split": "test"}
        )
        print()
        print(baseline_result.render("BASELINE (tfidf+logreg) -- TEST"))
        save_result(baseline_result, ARTIFACTS / "test_baseline.json")

        paired = paired_macro_f1_difference(y_true, baseline_pred, y_pred, labels)
        print()
        print("FINE-TUNED vs BASELINE, paired over the same test examples")
        print(f"  baseline macro F1   {baseline_result.macro_f1:.4f}")
        print(f"  fine-tuned macro F1 {result.macro_f1:.4f}")
        print(
            f"  difference {paired['difference']:+.4f} "
            f"[95% CI {paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]"
        )
        print(
            "  the interval excludes zero: the difference is real."
            if paired["significant"]
            else "  the interval includes zero: this test set cannot separate them."
        )
        (ARTIFACTS / "comparison.json").write_text(
            json.dumps(
                {
                    "baseline_macro_f1": baseline_result.macro_f1,
                    "transformer_macro_f1": result.macro_f1,
                    "paired": paired,
                    "baseline_per_class_f1": {m.label: m.f1 for m in baseline_result.per_class},
                    "transformer_per_class_f1": {m.label: m.f1 for m in result.per_class},
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    print()
    print("The test set has now been read. Record it in the README.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
