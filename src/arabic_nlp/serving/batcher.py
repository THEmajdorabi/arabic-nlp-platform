"""Micro-batching.

A transformer forward pass on one text and on thirty-two texts cost almost the
same wall time -- the GPU is idle either way, waiting on kernel launches. So a
service that runs one pass per request wastes most of its throughput on light
load and collapses under heavy load.

This collects arriving requests for up to `max_delay_ms`, or until `max_batch`
items are waiting, then runs one pass and fans the results back out. The added
latency is bounded by `max_delay_ms` and nothing else, which is why that number
is small and in config.yaml rather than tuned by feel.

The design is deliberately one background thread with a queue rather than
asyncio: the model call is a blocking, GIL-releasing C extension, so a thread
is both correct and simpler to reason about than an executor inside an event
loop.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class _Job:
    """One caller's texts, and somewhere to put the answer."""

    id: str
    texts: list[str]
    done: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: Exception | None = None


@dataclass
class BatchStats:
    batches: int = 0
    items: int = 0
    forward_seconds: float = 0.0
    largest_batch: int = 0

    @property
    def mean_batch_size(self) -> float:
        return self.items / self.batches if self.batches else 0.0

    @property
    def items_per_second(self) -> float:
        return self.items / self.forward_seconds if self.forward_seconds else 0.0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "batches": self.batches,
            "items": self.items,
            "mean_batch_size": round(self.mean_batch_size, 2),
            "largest_batch": self.largest_batch,
            "items_per_second": round(self.items_per_second, 1),
        }


class MicroBatcher:
    """Collect, batch, run, fan out."""

    def __init__(
        self,
        predict: Callable[[list[str]], list[Any]],
        max_batch: int = 32,
        max_delay_ms: int = 10,
    ) -> None:
        self.predict = predict
        self.max_batch = max_batch
        self.max_delay = max_delay_ms / 1000.0
        self.queue: Queue[_Job] = Queue()
        self.stats = BatchStats()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="micro-batcher", daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info("batcher.started", max_batch=self.max_batch, max_delay_ms=self.max_delay * 1000)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def submit(self, texts: list[str], timeout: float = 30.0) -> list[Any]:
        """Blocking. Returns one result per text, in order."""
        job = _Job(id=uuid.uuid4().hex[:8], texts=texts)
        self.queue.put(job)
        if not job.done.wait(timeout=timeout):
            raise TimeoutError(f"batcher did not answer within {timeout}s")
        if job.error is not None:
            raise job.error
        return job.result

    def _collect(self) -> list[_Job]:
        """Wait for the first job, then take whatever else arrives in the window."""
        try:
            first = self.queue.get(timeout=0.1)
        except Empty:
            return []

        jobs = [first]
        pending = len(first.texts)
        deadline = time.perf_counter() + self.max_delay

        while pending < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                job = self.queue.get(timeout=remaining)
            except Empty:
                break
            jobs.append(job)
            pending += len(job.texts)
        return jobs

    def _loop(self) -> None:
        while not self._stop.is_set():
            jobs = self._collect()
            if not jobs:
                continue

            texts: list[str] = []
            spans: list[tuple[int, int]] = []
            for job in jobs:
                start = len(texts)
                texts.extend(job.texts)
                spans.append((start, len(texts)))

            started = time.perf_counter()
            try:
                results = self.predict(texts)
                elapsed = time.perf_counter() - started

                self.stats.batches += 1
                self.stats.items += len(texts)
                self.stats.forward_seconds += elapsed
                self.stats.largest_batch = max(self.stats.largest_batch, len(texts))

                for job, (start, end) in zip(jobs, spans, strict=True):
                    job.result = results[start:end]
            except Exception as exc:  # noqa: BLE001 - one bad batch must not kill the thread
                log.error("batcher.failed", error=str(exc), batch=len(texts))
                for job in jobs:
                    job.error = exc
            finally:
                for job in jobs:
                    job.done.set()
