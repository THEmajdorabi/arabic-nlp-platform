"""The FastAPI service.

    POST /predict     {texts: [...]}  -> [{label, confidence, all_scores}]
    GET  /healthz
    GET  /model-info  version, run id, training date, validation metrics
    GET  /metrics     Prometheus
    GET  /drift       the current drift report
    POST /route       the hook repo 02 calls: dialect -> a routing decision

Every prediction is logged -- as a hash of the text, never the text itself,
plus length, predicted label and confidence -- to a rolling file the drift
monitor reads. Logging raw user text into a file that outlives the request is a
privacy decision nobody should make by default, and the drift signals worth
watching (length, OOV rate, class mix, confidence) do not need it.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from arabic_nlp.config import REPO_ROOT, load_settings
from arabic_nlp.serving.batcher import MicroBatcher
from arabic_nlp.serving.predictor import Predictor

log = structlog.get_logger(__name__)

DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "transformer" / "class_weights_seed20260823" / "model"


class PredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=128)


class PredictionOut(BaseModel):
    label: str
    confidence: float
    all_scores: dict[str, float]


class RouteRequest(BaseModel):
    """Repo 02's hook. One question, one routing decision."""

    question: str
    # Below this, the router declines to route rather than guessing. A
    # confident wrong route is worse than no route.
    min_confidence: float = 0.60


class RouteResponse(BaseModel):
    dialect: str | None
    confidence: float
    route: str
    reason: str


class Service:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.predictor: Predictor | None = None
        self.batcher: MicroBatcher | None = None
        self.started = time.time()
        self.requests = 0
        self.texts_seen = 0
        self.by_class: dict[str, int] = {}
        self.confidence_sum = 0.0
        self.log_path = REPO_ROOT / self.settings.serving.request_log

    def load(self) -> None:
        model_dir = os.environ.get("MODEL_DIR", str(DEFAULT_MODEL_DIR))
        backend = os.environ.get("BACKEND", "torch")
        self.predictor = Predictor(model_dir, self.settings, backend=backend)

        def run(texts: list[str]) -> list[Any]:
            assert self.predictor is not None
            return self.predictor.predict(texts)

        self.batcher = MicroBatcher(
            run,
            max_batch=self.settings.serving.max_batch,
            max_delay_ms=self.settings.serving.max_delay_ms,
        )
        self.batcher.start()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("service.ready", model_dir=model_dir, backend=backend)

    def record(self, text: str, prediction: Any) -> None:
        self.texts_seen += 1
        self.by_class[prediction.label] = self.by_class.get(prediction.label, 0) + 1
        self.confidence_sum += prediction.confidence

        # Hash, not text. Enough to spot duplicates and to join back to a
        # request in an incident, and nothing that identifies anyone.
        row = {
            "ts": round(time.time(), 3),
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            "chars": len(text),
            "words": len(text.split()),
            "label": prediction.label,
            "confidence": round(prediction.confidence, 4),
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.texts_seen if self.texts_seen else 0.0


service = Service()


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    service.load()
    yield
    if service.batcher:
        service.batcher.stop()


app = FastAPI(
    title="arabic-nlp-platform",
    description="Arabic dialect identification: EG, LY, LB, SD, MA.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_batcher() -> MicroBatcher:
    if service.batcher is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return service.batcher


def get_predictor() -> Predictor:
    if service.predictor is None:
        raise HTTPException(status_code=503, detail="model still loading")
    return service.predictor


@app.post("/predict", response_model=list[PredictionOut])
def predict(
    body: PredictRequest, batcher: MicroBatcher = Depends(get_batcher)
) -> list[PredictionOut]:
    service.requests += 1
    try:
        predictions = batcher.submit(body.texts)
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc

    for text, prediction in zip(body.texts, predictions, strict=True):
        service.record(text, prediction)

    return [
        PredictionOut(label=p.label, confidence=round(p.confidence, 4), all_scores=p.all_scores)
        for p in predictions
    ]


@app.post("/route", response_model=RouteResponse)
def route(body: RouteRequest, batcher: MicroBatcher = Depends(get_batcher)) -> RouteResponse:
    """The integration hook for repo 02.

    A question in Gulf colloquial and one in formal MSA want different
    retrieval treatment. This says which it looks like -- and, below the
    confidence floor, says it does not know, which is the answer that keeps the
    caller correct.
    """
    service.requests += 1
    prediction = batcher.submit([body.question])[0]
    service.record(body.question, prediction)

    if prediction.confidence < body.min_confidence:
        return RouteResponse(
            dialect=None,
            confidence=round(prediction.confidence, 4),
            route="default",
            reason=(
                f"top class {prediction.label} at {prediction.confidence:.2f} is below the "
                f"{body.min_confidence:.2f} floor; routing on a guess is worse than not routing"
            ),
        )

    return RouteResponse(
        dialect=prediction.label,
        confidence=round(prediction.confidence, 4),
        route="dialect",
        reason=f"classified as {prediction.label}",
    )


@app.get("/healthz")
def healthz(predictor: Predictor = Depends(get_predictor)) -> dict[str, Any]:
    return {
        "status": "ok",
        "uptime_s": round(time.time() - service.started),
        "backend": predictor.backend,
        "device": predictor.device,
        "labels": predictor.labels,
    }


@app.get("/model-info")
def model_info(predictor: Predictor = Depends(get_predictor)) -> dict[str, Any]:
    info = predictor.model_info()
    info["batching"] = service.batcher.stats.as_dict() if service.batcher else {}
    return info


@app.get("/drift")
def drift() -> dict[str, Any]:
    """The most recent drift report, or an explanation of why there is none."""
    from arabic_nlp.serving.monitor import latest_report

    return latest_report()


@app.get("/metrics")
def metrics() -> Any:
    from fastapi.responses import PlainTextResponse

    from arabic_nlp.serving.monitor import latest_report

    report = latest_report()
    batching = service.batcher.stats.as_dict() if service.batcher else {}
    lines = [
        "# HELP dialect_requests_total Requests served.",
        "# TYPE dialect_requests_total counter",
        f"dialect_requests_total {service.requests}",
        "# HELP dialect_texts_total Texts classified.",
        "# TYPE dialect_texts_total counter",
        f"dialect_texts_total {service.texts_seen}",
        "# HELP dialect_mean_confidence Mean confidence since start.",
        "# TYPE dialect_mean_confidence gauge",
        f"dialect_mean_confidence {service.mean_confidence:.4f}",
        "# HELP dialect_batch_size_mean Mean micro-batch size.",
        "# TYPE dialect_batch_size_mean gauge",
        f"dialect_batch_size_mean {batching.get('mean_batch_size', 0)}",
        "# HELP dialect_items_per_second Throughput of the batched forward pass.",
        "# TYPE dialect_items_per_second gauge",
        f"dialect_items_per_second {batching.get('items_per_second', 0)}",
        "# HELP dialect_drift_share Share of monitored columns showing drift.",
        "# TYPE dialect_drift_share gauge",
        f"dialect_drift_share {report.get('drift_share', 0.0)}",
        "# HELP dialect_predictions_total Predictions per class.",
        "# TYPE dialect_predictions_total counter",
    ]
    for label, count in sorted(service.by_class.items()):
        lines.append(f'dialect_predictions_total{{dialect="{label}"}} {count}')
    return PlainTextResponse("\n".join(lines) + "\n")


def instrument() -> None:
    """Standard HTTP metrics, on top of the custom ones above.

    Optional so the service still starts if the instrumentator is missing --
    losing request-rate metrics is worth less than losing the service.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app, endpoint="/http-metrics")
    except Exception as exc:  # noqa: BLE001
        log.warning("instrumentator.unavailable", error=str(exc))


instrument()


def request_log_path() -> Path:
    return service.log_path
