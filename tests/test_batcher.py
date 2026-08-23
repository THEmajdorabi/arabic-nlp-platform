"""The micro-batcher.

Concurrency bugs here are the expensive kind: results returned to the wrong
caller. Every test below is about that, not about throughput.
"""

from __future__ import annotations

import threading
import time

import pytest

from arabic_nlp.serving.batcher import MicroBatcher


@pytest.fixture
def batcher():
    """A batcher whose 'model' returns each text upper-cased.

    Identity-preserving on purpose: if a result reaches the wrong caller, the
    assertion fails rather than passing on a coincidence of shapes.
    """
    b = MicroBatcher(lambda texts: [t.upper() for t in texts], max_batch=8, max_delay_ms=20)
    b.start()
    yield b
    b.stop()


def test_a_single_request_comes_back_correct(batcher):
    assert batcher.submit(["ab", "cd"]) == ["AB", "CD"]


def test_concurrent_callers_each_get_their_own_results(batcher):
    """The bug this exists to prevent: fan-out crossing the wires."""
    results: dict[int, list[str]] = {}
    errors: list[Exception] = []

    def call(index: int) -> None:
        try:
            results[index] = batcher.submit([f"caller{index}-a", f"caller{index}-b"])
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(results) == 20
    for index, value in results.items():
        assert value == [f"CALLER{index}-A", f"CALLER{index}-B"]


def test_requests_are_actually_batched(batcher):
    """Twenty concurrent single-text calls must not become twenty forward passes."""
    threads = [threading.Thread(target=lambda i=i: batcher.submit([f"t{i}"])) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert batcher.stats.items == 20
    assert batcher.stats.batches < 20
    assert batcher.stats.largest_batch > 1


def test_the_batch_size_cap_is_respected():
    b = MicroBatcher(lambda texts: list(texts), max_batch=4, max_delay_ms=50)
    b.start()
    try:
        threads = [threading.Thread(target=lambda i=i: b.submit([f"t{i}"])) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        # The cap is checked before adding a job, so a job can push one batch
        # past it; what must not happen is unbounded growth.
        assert b.stats.largest_batch <= 4 + 1
    finally:
        b.stop()


def test_a_lone_request_is_not_delayed_beyond_the_window():
    b = MicroBatcher(lambda texts: list(texts), max_batch=32, max_delay_ms=20)
    b.start()
    try:
        started = time.perf_counter()
        b.submit(["only"])
        elapsed = (time.perf_counter() - started) * 1000
        # The window plus the collector's poll interval, with room for a slow
        # CI box. The point is that it is bounded, not that it is instant.
        assert elapsed < 500
    finally:
        b.stop()


def test_a_failing_model_raises_to_every_caller_and_does_not_kill_the_thread():
    calls = {"n": 0}

    def flaky(texts):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return list(texts)

    b = MicroBatcher(flaky, max_batch=8, max_delay_ms=10)
    b.start()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            b.submit(["a"])
        # The worker survived, so the next request still works.
        assert b.submit(["b"]) == ["b"]
    finally:
        b.stop()


def test_predictor_batches_internally(monkeypatch):
    """A method safe at 32 texts and fatal at 4,000 is a trap.

    Regression: passing the whole validation split to `probabilities` in one
    call asked ONNX Runtime for a 4.7 GB activation buffer and it refused.
    """
    import numpy as np

    from arabic_nlp.serving.predictor import Predictor

    seen: list[int] = []

    class Fake(Predictor):
        def __init__(self) -> None:  # noqa: D107 - deliberately skips model loading
            from arabic_nlp.config import load_settings

            self.settings = load_settings()
            self.backend = "fake"
            self.labels = self.settings.dataset.label_names

        def preprocess(self, texts: list[str]) -> list[str]:
            return texts

        def _logits(self, cleaned: list[str]):  # noqa: ANN202
            seen.append(len(cleaned))
            return np.zeros((len(cleaned), len(self.labels)), dtype=np.float32)

    predictor = Fake()
    out = predictor.probabilities(["t"] * 200, batch_size=32)

    assert out.shape == (200, 5)
    assert max(seen) <= 32
    assert sum(seen) == 200
    # Uniform logits must come back as a uniform distribution, not NaN.
    assert np.allclose(out.sum(axis=1), 1.0)
