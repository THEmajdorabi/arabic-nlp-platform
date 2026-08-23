"""The service, with a stub model.

Marked integration only where a real model is needed. Everything here uses a
fake predictor, so the routing logic, the request log and the metrics format
are all covered without a GPU.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from arabic_nlp.serving.batcher import MicroBatcher
from arabic_nlp.serving.predictor import Prediction


class StubPredictor:
    """Returns a fixed prediction, so the tests are about the service."""

    backend = "stub"
    device = "cpu"
    labels = ["EG", "LB", "LY", "MA", "SD"]

    def __init__(self, label: str = "EG", confidence: float = 0.91) -> None:
        self.label = label
        self.confidence = confidence

    def predict(self, texts: list[str]) -> list[Prediction]:
        return [
            Prediction(
                label=self.label,
                confidence=self.confidence,
                all_scores={name: 0.0 for name in self.labels} | {self.label: self.confidence},
            )
            for _ in texts
        ]

    def model_info(self) -> dict:
        return {"model_dir": "stub", "labels": self.labels, "backend": "stub"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    from arabic_nlp.serving import app as module

    predictor = StubPredictor()
    module.service.predictor = predictor
    module.service.log_path = tmp_path / "predictions.jsonl"
    module.service.requests = 0
    module.service.texts_seen = 0
    module.service.by_class = {}
    module.service.confidence_sum = 0.0

    batcher = MicroBatcher(predictor.predict, max_batch=8, max_delay_ms=5)
    batcher.start()
    module.service.batcher = batcher

    # The real lifespan would load a model from disk; these tests supply one.
    monkeypatch.setattr(module.service, "load", lambda: None)
    with TestClient(module.app) as test_client:
        yield test_client, module
    batcher.stop()


def test_predict_returns_one_result_per_text(client):
    test_client, _ = client
    response = test_client.post("/predict", json={"texts": ["نص أول", "نص ثاني", "نص ثالث"]})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3
    assert body[0]["label"] == "EG"
    assert 0 <= body[0]["confidence"] <= 1
    assert set(body[0]["all_scores"]) == set(StubPredictor.labels)


def test_predict_rejects_an_empty_list(client):
    test_client, _ = client
    assert test_client.post("/predict", json={"texts": []}).status_code == 422


def test_the_request_log_stores_a_hash_and_never_the_text(client):
    """Raw user text must not end up in a file that outlives the request."""
    test_client, module = client
    secret = "نص سري جدا لا يجب أن يظهر في السجل"
    test_client.post("/predict", json={"texts": [secret]})

    contents = module.service.log_path.read_text(encoding="utf-8")
    assert secret not in contents
    row = json.loads(contents.strip().splitlines()[-1])
    assert len(row["text_sha256"]) == 16
    assert row["chars"] == len(secret)
    assert row["label"] == "EG"


def test_route_returns_a_dialect_when_confident(client):
    test_client, _ = client
    body = test_client.post("/route", json={"question": "ايه الاخبار"}).json()
    assert body["route"] == "dialect"
    assert body["dialect"] == "EG"


def test_route_declines_below_the_confidence_floor(client):
    """A confident wrong route is worse than no route."""
    test_client, module = client
    module.service.predictor = StubPredictor(confidence=0.3)
    module.service.batcher.predict = module.service.predictor.predict

    body = test_client.post("/route", json={"question": "نص غامض", "min_confidence": 0.6}).json()
    assert body["route"] == "default"
    assert body["dialect"] is None
    assert "below" in body["reason"]


def test_healthz_reports_the_labels(client):
    test_client, _ = client
    body = test_client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["labels"] == StubPredictor.labels


def test_metrics_are_prometheus_text_with_per_class_counters(client):
    test_client, _ = client
    test_client.post("/predict", json={"texts": ["a", "b"]})
    text = test_client.get("/metrics").text

    assert "# TYPE dialect_requests_total counter" in text
    assert 'dialect_predictions_total{dialect="EG"} 2' in text
    assert "dialect_mean_confidence" in text
    assert "dialect_drift_share" in text


def test_the_drift_gauge_says_not_measured_before_any_report_exists(client):
    """0.0 because nothing ran looks identical to 0.0 because nothing is wrong."""
    test_client, _ = client
    body = test_client.get("/drift").json()
    assert body["status"] in {"no report", "ok"}
    if body["status"] == "no report":
        assert "not measured" in body["detail"]
