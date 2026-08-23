"""Evaluation. Macro F1, per-class numbers, calibration, and error analysis.

Accuracy is not reported anywhere in this repo. On a corpus that is 39%
Egyptian, a model that predicts Egyptian for everything scores 39% accuracy and
0.11 macro F1, and only one of those two numbers tells you it is useless.

Calibration is here because this classifier feeds a routing decision in repo
02. A router that is 95% confident and 70% right will send questions down the
wrong path with no signal that anything is wrong, and expected calibration
error is the number that catches it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class EvaluationResult:
    macro_f1: float
    weighted_f1: float
    per_class: list[ClassMetrics]
    confusion: list[list[int]]
    labels: list[str]
    expected_calibration_error: float
    calibration_bins: list[dict[str, float]] = field(default_factory=list)
    n: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "expected_calibration_error": round(self.expected_calibration_error, 4),
            "per_class": {
                m.label: {
                    "precision": round(m.precision, 4),
                    "recall": round(m.recall, 4),
                    "f1": round(m.f1, 4),
                    "support": m.support,
                }
                for m in self.per_class
            },
            "confusion": self.confusion,
            "labels": self.labels,
            "calibration_bins": self.calibration_bins,
            **self.extra,
        }

    def render(self, title: str = "") -> str:
        lines = []
        if title:
            lines += [title, ""]
        lines += [
            f"macro F1 {self.macro_f1:.4f}   weighted F1 {self.weighted_f1:.4f}   "
            f"ECE {self.expected_calibration_error:.4f}   n={self.n}",
            "",
            "| Class | precision | recall | F1 | support |",
            "|---|---|---|---|---|",
        ]
        for m in sorted(self.per_class, key=lambda m: -m.support):
            lines.append(
                f"| {m.label} | {m.precision:.3f} | {m.recall:.3f} | {m.f1:.3f} | {m.support} |"
            )
        lines += ["", "Confusion (rows true, columns predicted)", ""]
        lines.append("| | " + " | ".join(self.labels) + " |")
        lines.append("|---" * (len(self.labels) + 1) + "|")
        for label, row in zip(self.labels, self.confusion, strict=True):
            lines.append(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
        return "\n".join(lines)


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, bins: int = 10
) -> tuple[float, list[dict[str, float]]]:
    """ECE with equal-width bins.

    Within each confidence bin, compare the mean confidence to the actual
    accuracy; ECE is the support-weighted mean absolute gap. A model that says
    "0.9" and is right 90% of the time in that bin contributes nothing.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidences)
    error = 0.0
    rows: list[dict[str, float]] = []

    for low, high in zip(edges[:-1], edges[1:], strict=True):
        # Upper-inclusive on the last bin so confidence == 1.0 is counted.
        mask = (confidences > low) & (confidences <= high) if high < 1.0 else (confidences > low)
        count = int(mask.sum())
        if count == 0:
            continue
        bin_confidence = float(confidences[mask].mean())
        bin_accuracy = float(correct[mask].mean())
        error += (count / total) * abs(bin_confidence - bin_accuracy)
        rows.append(
            {
                "bin_low": round(float(low), 2),
                "bin_high": round(float(high), 2),
                "n": count,
                "confidence": round(bin_confidence, 4),
                "accuracy": round(bin_accuracy, 4),
                "gap": round(bin_accuracy - bin_confidence, 4),
            }
        )
    return error, rows


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    extra: dict[str, Any] | None = None,
) -> EvaluationResult:
    indices = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    per_class = [
        ClassMetrics(
            label=labels[i],
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        )
        for i in indices
    ]

    confidences = probabilities.max(axis=1)
    correct = (y_pred == y_true).astype(float)
    ece, bins = expected_calibration_error(confidences, correct)

    return EvaluationResult(
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=indices, zero_division=0)),
        weighted_f1=float(
            f1_score(y_true, y_pred, average="weighted", labels=indices, zero_division=0)
        ),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=indices).tolist(),
        labels=labels,
        expected_calibration_error=ece,
        calibration_bins=bins,
        n=len(y_true),
        extra=extra or {},
    )


def save_confusion_image(result: EvaluationResult, path: str | Path, title: str = "") -> None:
    """Row-normalised confusion matrix as a PNG.

    Row-normalised, not raw counts: with a 5x imbalance the raw matrix is a
    picture of the class distribution, and the confusion pattern, which is
    the thing worth looking at, is invisible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array(result.confusion, dtype=float)
    normalised = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    figure, axes = plt.subplots(figsize=(5.5, 4.8))
    image = axes.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    axes.set_xticks(range(len(result.labels)), result.labels)
    axes.set_yticks(range(len(result.labels)), result.labels)
    axes.set_xlabel("predicted")
    axes.set_ylabel("true")
    axes.set_title(title or f"macro F1 {result.macro_f1:.3f}")

    for i in range(len(result.labels)):
        for j in range(len(result.labels)):
            axes.text(
                j,
                i,
                f"{normalised[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if normalised[i, j] > 0.5 else "black",
                fontsize=9,
            )

    figure.colorbar(image, ax=axes, fraction=0.046)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_calibration_image(result: EvaluationResult, path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = result.calibration_bins
    if not bins:
        return
    x = [b["confidence"] for b in bins]
    y = [b["accuracy"] for b in bins]

    figure, axes = plt.subplots(figsize=(4.8, 4.6))
    axes.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="perfect")
    axes.plot(x, y, "o-", label="model")
    axes.set_xlabel("mean confidence")
    axes.set_ylabel("accuracy")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    axes.set_title(title or f"ECE {result.expected_calibration_error:.3f}")
    axes.legend()
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def worst_errors(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    top_k: int = 30,
) -> list[dict[str, Any]]:
    """The most confident mistakes.

    Sorted by the confidence of the WRONG prediction, so the list is the
    model's most assured errors rather than its most uncertain ones. Those are
    the ones that tell you something: a low-confidence mistake is a hard
    example, a high-confidence mistake is a wrong belief.
    """
    wrong = np.where(y_pred != y_true)[0]
    confidences = probabilities.max(axis=1)
    ordered = wrong[np.argsort(confidences[wrong])[::-1][:top_k]]
    return [
        {
            "text": texts[i],
            "true": labels[int(y_true[i])],
            "predicted": labels[int(y_pred[i])],
            "confidence": round(float(confidences[i]), 4),
            "chars": len(texts[i]),
        }
        for i in ordered
    ]


def confusion_pairs(result: EvaluationResult, top_k: int = 6) -> list[dict[str, Any]]:
    """The most-confused ordered class pairs, as a share of the true class."""
    matrix = np.array(result.confusion, dtype=float)
    totals = np.clip(matrix.sum(axis=1, keepdims=True), 1, None)
    rates = matrix / totals
    pairs: list[dict[str, Any]] = [
        {
            "true": result.labels[i],
            "predicted": result.labels[j],
            "rate": round(float(rates[i, j]), 4),
            "n": int(matrix[i, j]),
        }
        for i in range(len(result.labels))
        for j in range(len(result.labels))
        if i != j
    ]
    return sorted(pairs, key=lambda p: -float(p["rate"]))[:top_k]


def save_result(result: EvaluationResult, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# ---- uncertainty ---------------------------------------------------------
#
# A macro F1 without an interval is a point estimate presented as a fact. On a
# 22k validation set the interval is narrow, which is exactly why it is worth
# printing: it turns "0.820 beats 0.816" from an argument into an arithmetic
# question, and on this data the answer is usually "no".


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260823,
) -> dict[str, float]:
    """Percentile bootstrap interval for macro F1, resampling examples."""
    rng = np.random.default_rng(seed)
    indices = list(range(len(labels)))
    n = len(y_true)
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        picked = rng.integers(0, n, size=n)
        draws[i] = f1_score(
            y_true[picked], y_pred[picked], average="macro", labels=indices, zero_division=0
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=indices, zero_division=0)
        ),
        "ci_low": float(np.quantile(draws, tail)),
        "ci_high": float(np.quantile(draws, 1.0 - tail)),
        "n": int(n),
    }


def paired_macro_f1_difference(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    labels: list[str],
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260823,
) -> dict[str, float | bool]:
    """Is B's macro F1 really above A's, on the same examples?

    Resamples EXAMPLES and recomputes both models' macro F1 on the same
    resample, so the example-to-example variation cancels. Two independent
    intervals that overlap do not mean "no difference"; this is the test that
    answers the question actually being asked.
    """
    rng = np.random.default_rng(seed)
    indices = list(range(len(labels)))
    n = len(y_true)
    differences = np.empty(resamples, dtype=np.float64)

    for i in range(resamples):
        picked = rng.integers(0, n, size=n)
        truth = y_true[picked]
        differences[i] = f1_score(
            truth, y_pred_b[picked], average="macro", labels=indices, zero_division=0
        ) - f1_score(truth, y_pred_a[picked], average="macro", labels=indices, zero_division=0)

    tail = (1.0 - confidence) / 2.0
    low = float(np.quantile(differences, tail))
    high = float(np.quantile(differences, 1.0 - tail))
    observed = f1_score(
        y_true, y_pred_b, average="macro", labels=indices, zero_division=0
    ) - f1_score(y_true, y_pred_a, average="macro", labels=indices, zero_division=0)

    return {
        "difference": float(observed),
        "ci_low": low,
        "ci_high": high,
        "n": int(n),
        "significant": bool(low > 0.0 or high < 0.0),
    }
