"""Splits, metrics, calibration, batching.

Everything here runs on synthetic data in milliseconds, with no model, no GPU
and no network, which is what lets CI gate on it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from arabic_nlp.config import load_settings
from arabic_nlp.data.splits import distribution, make_splits
from arabic_nlp.evaluate import (
    evaluate,
    expected_calibration_error,
    paired_macro_f1_difference,
    worst_errors,
)

LABELS = ["EG", "LB", "LY", "MA", "SD"]


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def frame():
    """A 5-class frame with the real corpus's rough imbalance."""
    rng = np.random.default_rng(0)
    shares = {"EG": 3900, "LY": 2470, "LB": 1870, "SD": 980, "MA": 780}
    rows = [
        {"text": f"text {label} {i} {rng.integers(1000)}", "label": label}
        for label, count in shares.items()
        for i in range(count)
    ]
    return pd.DataFrame(rows)


# ---- splits --------------------------------------------------------------


def test_splits_are_stratified(frame, settings):
    settings.split.max_train_rows = 0
    splits = make_splits(frame, settings)
    whole = distribution(frame)
    for name, part in splits.items():
        for label, share in distribution(part).items():
            assert share == pytest.approx(whole[label], abs=0.01), f"{name}/{label} drifted"


def test_splits_are_disjoint(frame, settings):
    """Overlap between train and test is leakage, and it inflates everything."""
    settings.split.max_train_rows = 0
    splits = make_splits(frame, settings)
    train = set(splits["train"]["text"])
    validation = set(splits["validation"]["text"])
    test = set(splits["test"]["text"])
    assert not train & test
    assert not train & validation
    assert not validation & test


def test_splits_cover_everything(frame, settings):
    settings.split.max_train_rows = 0
    splits = make_splits(frame, settings)
    assert sum(len(p) for p in splits.values()) == len(frame)


def test_splits_are_reproducible(frame, settings):
    settings.split.max_train_rows = 0
    first = make_splits(frame, settings)["test"]["text"].tolist()
    second = make_splits(frame, settings)["test"]["text"].tolist()
    assert first == second


def test_the_train_cap_does_not_shrink_the_held_out_sets(frame, settings):
    """Capping before splitting would make the evaluation sets smaller too.

    That would turn every reported number into a measurement of the cap.
    """
    settings.split.max_train_rows = 0
    uncapped = make_splits(frame, settings)

    settings.split.max_train_rows = 1000
    capped = make_splits(frame, settings)

    assert len(capped["train"]) == 1000
    assert len(capped["validation"]) == len(uncapped["validation"])
    assert len(capped["test"]) == len(uncapped["test"])


def test_the_capped_train_set_is_still_stratified(frame, settings):
    settings.split.max_train_rows = 1000
    capped = make_splits(frame, settings)["train"]
    whole = distribution(frame)
    for label, share in distribution(capped).items():
        assert share == pytest.approx(whole[label], abs=0.02)


# ---- metrics -------------------------------------------------------------


def test_macro_f1_exposes_a_majority_class_predictor():
    """The whole reason accuracy is not reported anywhere in this repo.

    A model that always says EG is 39% accurate and useless. Macro F1 says so.
    """
    y_true = np.array([0] * 39 + [1] * 25 + [2] * 19 + [3] * 10 + [4] * 7)
    y_pred = np.zeros_like(y_true)
    probabilities = np.zeros((len(y_true), 5))
    probabilities[:, 0] = 1.0

    result = evaluate(y_true, y_pred, probabilities, LABELS)
    accuracy = float((y_pred == y_true).mean())
    assert accuracy == pytest.approx(0.39, abs=0.01)
    assert result.macro_f1 < 0.12


def test_per_class_f1_is_reported_for_every_class_including_absent_ones():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    probabilities = np.eye(5)[y_pred]
    result = evaluate(y_true, y_pred, probabilities, LABELS)
    assert {m.label for m in result.per_class} == set(LABELS)
    assert all(m.support == 0 for m in result.per_class if m.label in {"LY", "MA", "SD"})


def test_confusion_matrix_orientation_is_true_by_predicted():
    y_true = np.array([0, 0, 1])
    y_pred = np.array([0, 1, 1])
    probabilities = np.eye(5)[y_pred]
    result = evaluate(y_true, y_pred, probabilities, LABELS)
    assert result.confusion[0][0] == 1  # one EG correctly called EG
    assert result.confusion[0][1] == 1  # one EG wrongly called LB
    assert result.confusion[1][1] == 1


# ---- calibration ---------------------------------------------------------


def test_a_perfectly_calibrated_model_has_near_zero_ece():
    rng = np.random.default_rng(1)
    confidences = rng.uniform(0.5, 1.0, 20000)
    correct = (rng.uniform(size=20000) < confidences).astype(float)
    error, bins = expected_calibration_error(confidences, correct)
    assert error < 0.02
    assert len(bins) > 0


def test_an_overconfident_model_has_large_ece():
    confidences = np.full(1000, 0.99)
    correct = np.zeros(1000)
    correct[:500] = 1.0  # says 99%, right 50% of the time
    error, _ = expected_calibration_error(confidences, correct)
    assert error == pytest.approx(0.49, abs=0.01)


def test_ece_counts_confidence_of_exactly_one():
    """The top bin is upper-inclusive; otherwise 1.0 falls out of every bin."""
    confidences = np.ones(100)
    correct = np.zeros(100)
    error, bins = expected_calibration_error(confidences, correct)
    assert error == pytest.approx(1.0)
    assert sum(b["n"] for b in bins) == 100


# ---- error analysis ------------------------------------------------------


def test_worst_errors_are_the_confident_ones():
    """A low-confidence mistake is a hard example; a confident one is a wrong belief."""
    texts = ["a", "b", "c"]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([1, 0, 2])
    probabilities = np.array([[0.4, 0.6, 0, 0, 0], [0.95, 0.05, 0, 0, 0], [0, 0, 1.0, 0, 0]])
    errors = worst_errors(texts, y_true, y_pred, probabilities, LABELS)
    assert [e["text"] for e in errors] == ["b", "a"]
    assert errors[0]["confidence"] > errors[1]["confidence"]


# ---- paired comparison ---------------------------------------------------


def test_paired_difference_finds_a_consistent_gain():
    rng = np.random.default_rng(2)
    y_true = rng.integers(0, 5, 4000)
    weak = y_true.copy()
    weak[rng.choice(4000, 1200, replace=False)] = rng.integers(0, 5, 1200)
    strong = y_true.copy()
    strong[rng.choice(4000, 400, replace=False)] = rng.integers(0, 5, 400)

    paired = paired_macro_f1_difference(y_true, weak, strong, LABELS, resamples=400)
    assert paired["difference"] > 0
    assert paired["significant"]


def test_paired_difference_reports_no_difference_for_the_same_model():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, 2000)
    y_pred = y_true.copy()
    y_pred[rng.choice(2000, 500, replace=False)] = rng.integers(0, 5, 500)

    paired = paired_macro_f1_difference(y_true, y_pred, y_pred, LABELS, resamples=400)
    assert paired["difference"] == pytest.approx(0.0)
    assert not paired["significant"]
