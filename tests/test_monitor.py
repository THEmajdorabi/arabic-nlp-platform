"""Drift detection.

Two properties matter and both are tested: it fires when the distribution
really moves, and it does NOT fire on a resample of the same distribution. A
monitor that always alerts gets muted in a week, which is the same as not
having one.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from arabic_nlp.config import load_settings
from arabic_nlp.serving.monitor import compute_drift, latest_report, load_request_log


def sample(rows: int, seed: int, *, chars=(90, 20), skew=(0.5, 0.3, 0.2), confidence=(0.6, 1.0)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "chars": rng.normal(chars[0], chars[1], rows).clip(5),
            "words": rng.normal(chars[0] / 6, 3, rows).clip(1),
            "label": rng.choice(["EG", "LY", "LB"], rows, p=list(skew)),
            "confidence": rng.uniform(confidence[0], confidence[1], rows),
        }
    )


@pytest.fixture
def settings():
    return load_settings()


def test_no_alert_when_the_distribution_has_not_moved(settings):
    reference = sample(3000, seed=1)
    current = sample(600, seed=2)
    report = compute_drift(current, reference, settings)
    assert not report.alert, report.render()
    assert report.drift_share < settings.drift.alert_threshold


def test_alert_when_texts_get_much_shorter(settings):
    reference = sample(3000, seed=1)
    current = sample(600, seed=2, chars=(25, 6))
    report = compute_drift(current, reference, settings)
    assert report.per_column["chars"]["drifted"]
    assert report.alert


def test_alert_when_the_predicted_class_mix_flips(settings):
    reference = sample(3000, seed=1)
    current = sample(600, seed=2, skew=(0.05, 0.05, 0.90))
    report = compute_drift(current, reference, settings)
    assert report.per_column["label"]["drifted"]


def test_alert_when_confidence_collapses(settings):
    """The leading indicator of out-of-distribution input."""
    reference = sample(3000, seed=1)
    current = sample(600, seed=2, confidence=(0.2, 0.5))
    report = compute_drift(current, reference, settings)
    assert report.per_column["confidence"]["drifted"]


def test_every_monitored_column_is_reported(settings):
    report = compute_drift(sample(600, 2), sample(3000, 1), settings)
    assert set(report.per_column) == {"chars", "words", "label", "confidence"}


def test_the_report_round_trips_through_json(settings):
    report = compute_drift(sample(600, 2), sample(3000, 1), settings)
    assert json.loads(json.dumps(report.as_dict()))["total_columns"] == 4


def test_a_missing_report_says_not_measured_rather_than_zero(tmp_path):
    """A gauge at 0.0 because nothing ran looks exactly like 'no drift'."""
    payload = latest_report(tmp_path / "nope.json")
    assert payload["status"] == "no report"
    assert "not measured" in payload["detail"]


def test_request_log_reading_is_empty_when_there_is_no_log(tmp_path):
    frame = load_request_log(tmp_path / "missing.jsonl")
    assert len(frame) == 0
