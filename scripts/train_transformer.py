"""Fine-tune MARBERTv2, track it in MLflow, evaluate it properly.

    python scripts/train_transformer.py                       # one seed, config default
    python scripts/train_transformer.py --seeds 3             # the reported result
    python scripts/train_transformer.py --imbalance none      # the ablation
    python scripts/train_transformer.py --smoke                # CI: 200 rows, loss must fall

Validation only. The test set is scored once, by `scripts/final_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split  # noqa: E402
from arabic_nlp.evaluate import (  # noqa: E402
    bootstrap_macro_f1,
    confusion_pairs,
    evaluate,
    save_calibration_image,
    save_confusion_image,
    save_result,
    worst_errors,
)
from arabic_nlp.models.transformer import predict_proba, train  # noqa: E402

ARTIFACTS = ROOT / "artifacts" / "transformer"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, default=1, help="how many of the config's seeds to run"
    )
    parser.add_argument(
        "--imbalance", default="", choices=["", "class_weights", "oversample", "none"]
    )
    parser.add_argument(
        "--smoke", action="store_true", help="200 rows, 30 steps; asserts loss falls"
    )
    parser.add_argument(
        "--only-seeds",
        default="",
        help="comma-separated seeds to run, e.g. 20260824,20260825. Overrides --seeds.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a seed whose validation.json is already on disk",
    )
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    settings = load_settings()
    strategy = args.imbalance or settings.transformer.imbalance
    labels = settings.dataset.label_names
    index = {name: i for i, name in enumerate(labels)}

    train_frame = load_split("train")
    validation_frame = load_split("validation")

    if args.smoke:
        # A training smoke test for CI: tiny, CPU-friendly, and it asserts the
        # thing that actually breaks -- that the loss goes down. It catches a
        # broken collator, a label mismatch or a frozen model without a GPU.
        train_frame = train_frame.head(200)
        validation_frame = validation_frame.head(100)
        settings.transformer.epochs = 1
        settings.transformer.batch_size = 8
        settings.transformer.fp16 = False

    train_texts = train_frame["text"].tolist()
    train_labels = [index[label] for label in train_frame["label"]]
    validation_texts = validation_frame["text"].tolist()
    validation_labels = [index[label] for label in validation_frame["label"]]
    y_true = np.array(validation_labels)

    # Seed runs are independent, so they are resumable. A machine that dies
    # eighteen minutes into seed two should not cost the forty minutes already
    # spent on seed one -- and the summary below is rebuilt from whatever is on
    # disk, however it got there.
    if args.only_seeds:
        seeds = [int(part) for part in args.only_seeds.split(",") if part.strip()]
    else:
        seeds = settings.transformer.seeds[: max(1, args.seeds)]
    tag = args.tag or strategy

    if args.skip_existing:
        remaining = [
            s for s in seeds if not (ARTIFACTS / f"{tag}_seed{s}" / "validation.json").exists()
        ]
        for seed in set(seeds) - set(remaining):
            print(f"  seed {seed}: validation.json exists, skipping")
        seeds = remaining

    runs: list[dict] = []

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    mlflow.set_experiment(settings.mlflow.experiment)

    for seed in seeds:
        output_dir = ARTIFACTS / f"{tag}_seed{seed}"
        started = time.perf_counter()

        with mlflow.start_run(run_name=f"{tag}-seed{seed}") as run:
            mlflow.log_params(
                {
                    "base_model": settings.transformer.base_model,
                    "seed": seed,
                    "imbalance": strategy,
                    "epochs": settings.transformer.epochs,
                    "learning_rate": settings.transformer.learning_rate,
                    "batch_size": settings.transformer.batch_size,
                    "max_length": settings.transformer.max_length,
                    "train_rows": len(train_texts),
                    "normalize_arabic": settings.preprocess.normalize_arabic,
                    "dataset": settings.dataset.id,
                    "split_seed": settings.split.seed,
                }
            )

            trained = train(
                train_texts,
                train_labels,
                validation_texts,
                validation_labels,
                settings,
                seed=seed,
                output_dir=output_dir,
                imbalance=strategy,
                max_steps=30 if args.smoke else -1,
            )

            if args.smoke:
                # The assertion CI actually cares about.
                if trained.metrics["train_loss"] >= 2.0:
                    print(
                        f"SMOKE FAILED: training loss {trained.metrics['train_loss']:.3f} "
                        f"did not fall below 2.0 (random is about {np.log(len(labels)):.2f})"
                    )
                    return 1
                print(f"SMOKE PASSED: training loss {trained.metrics['train_loss']:.3f}")
                return 0

            probabilities = predict_proba(trained.directory, validation_texts, settings)
            y_pred = probabilities.argmax(axis=1)
            interval = bootstrap_macro_f1(y_true, y_pred, labels)

            result = evaluate(
                y_true,
                y_pred,
                probabilities,
                labels,
                extra={
                    "split": "validation",
                    "model": settings.transformer.base_model,
                    "seed": seed,
                    "imbalance": strategy,
                    "mlflow_run_id": run.info.run_id,
                    "wall_s": round(time.perf_counter() - started, 1),
                    "macro_f1_ci": interval,
                    **trained.metrics,
                },
            )

            print(result.render(f"MARBERTv2 ({strategy}, seed {seed}) -- VALIDATION"))
            print(
                f"macro F1 {interval['macro_f1']:.4f} "
                f"[{interval['ci_low']:.4f}, {interval['ci_high']:.4f}]"
            )

            save_result(result, output_dir / "validation.json")
            save_confusion_image(result, output_dir / "confusion.png", f"MARBERTv2 seed {seed}")
            save_calibration_image(result, output_dir / "calibration.png", f"MARBERTv2 seed {seed}")
            (output_dir / "worst_errors.json").write_text(
                json.dumps(
                    worst_errors(validation_texts, y_true, y_pred, probabilities, labels),
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            np.save(output_dir / "validation_predictions.npy", y_pred)
            np.save(output_dir / "validation_probabilities.npy", probabilities)

            mlflow.log_metrics(
                {
                    "macro_f1": result.macro_f1,
                    "weighted_f1": result.weighted_f1,
                    "ece": result.expected_calibration_error,
                    "macro_f1_ci_low": interval["ci_low"],
                    "macro_f1_ci_high": interval["ci_high"],
                    **{f"f1_{m.label}": m.f1 for m in result.per_class},
                }
            )
            mlflow.log_artifact(str(output_dir / "confusion.png"))
            mlflow.log_artifact(str(output_dir / "calibration.png"))
            mlflow.log_artifact(str(output_dir / "validation.json"))

            runs.append(
                {
                    "seed": seed,
                    "macro_f1": result.macro_f1,
                    "weighted_f1": result.weighted_f1,
                    "ece": result.expected_calibration_error,
                    "per_class_f1": {m.label: m.f1 for m in result.per_class},
                    "mlflow_run_id": run.info.run_id,
                    "directory": str(trained.directory),
                    "wall_s": round(time.perf_counter() - started, 1),
                }
            )

        print()
        print("MOST CONFUSED PAIRS")
        for pair in confusion_pairs(result):
            print(f"  {pair['true']} -> {pair['predicted']}: {pair['rate']:.3f} ({pair['n']} rows)")
        print()

    # Rebuild from disk, not from this process's `runs`. Three seeds run in
    # three invocations must produce the same summary as three in one.
    on_disk: list[dict] = []
    for path in sorted(ARTIFACTS.glob(f"{tag}_seed*/validation.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        on_disk.append(
            {
                "seed": record.get("seed"),
                "macro_f1": record["macro_f1"],
                "weighted_f1": record["weighted_f1"],
                "ece": record["expected_calibration_error"],
                "per_class_f1": {k: v["f1"] for k, v in record["per_class"].items()},
                "macro_f1_ci": record.get("macro_f1_ci"),
                "mlflow_run_id": record.get("mlflow_run_id"),
                "directory": str(path.parent / "model"),
                "wall_s": record.get("wall_s"),
            }
        )

    scores = [r["macro_f1"] for r in on_disk]
    if not scores:
        print("no completed seeds on disk; nothing to summarise")
        return 1

    summary = {
        "strategy": strategy,
        "seeds": [r["seed"] for r in on_disk],
        "macro_f1_mean": round(statistics.mean(scores), 4),
        "macro_f1_sd": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "weighted_f1_mean": round(statistics.mean(r["weighted_f1"] for r in on_disk), 4),
        "ece_mean": round(statistics.mean(r["ece"] for r in on_disk), 4),
        "per_class_f1_mean": {
            label: round(statistics.mean(r["per_class_f1"][label] for r in on_disk), 4)
            for label in labels
        },
        "per_class_f1_sd": {
            label: (
                round(statistics.stdev(r["per_class_f1"][label] for r in on_disk), 4)
                if len(on_disk) > 1
                else 0.0
            )
            for label in labels
        },
        "runs": on_disk,
    }
    (ARTIFACTS / f"summary_{tag}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("ACROSS SEEDS")
    print(
        f"  macro F1 mean {summary['macro_f1_mean']:.4f}  sd {summary['macro_f1_sd']:.4f}  "
        f"over {len(scores)} run(s)"
    )
    if len(scores) == 1:
        print("  One run is not a result. Run with --seeds 3 for the reported number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
