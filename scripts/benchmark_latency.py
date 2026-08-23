"""p50/p95/p99 latency for PyTorch and ONNX, at batch 1, 8 and 32.

Both numbers on the SAME machine, in the same run, with the hardware profile
printed next to them. A latency improvement measured on a different box than
its baseline is not a measurement.

    python scripts/benchmark_latency.py --torch-dir <dir> --onnx-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from arabic_nlp.config import load_settings  # noqa: E402
from arabic_nlp.data.splits import load_split  # noqa: E402


def profile() -> dict:
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def percentile(values: list[float], fraction: float) -> float:
    return float(np.quantile(values, fraction)) if values else 0.0


def measure(predictor, texts: list[str], batch_size: int, warmup: int, iterations: int) -> dict:
    batch = texts[:batch_size]
    for _ in range(warmup):
        predictor.probabilities(batch)

    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        predictor.probabilities(batch)
        timings.append((time.perf_counter() - started) * 1000)

    return {
        "batch_size": batch_size,
        "p50_ms": round(percentile(timings, 0.50), 2),
        "p95_ms": round(percentile(timings, 0.95), 2),
        "p99_ms": round(percentile(timings, 0.99), 2),
        "mean_ms": round(float(np.mean(timings)), 2),
        "items_per_second": round(batch_size / (float(np.mean(timings)) / 1000), 1),
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-dir", default="")
    parser.add_argument("--onnx-dir", default="")
    parser.add_argument("--out", default="artifacts/latency.json")
    args = parser.parse_args()

    from arabic_nlp.serving.predictor import Predictor

    settings = load_settings()
    texts = load_split("validation")["text"].tolist()[:64]

    # Three rows, because two of them are the only fair comparison. ONNX
    # Runtime here is CPU-only, so "ONNX is slower than PyTorch" measured
    # against PyTorch on a GPU is a statement about devices, not runtimes.
    results: dict[str, list[dict]] = {}
    for name, directory, backend, device in (
        ("pytorch-cuda", args.torch_dir, "torch", "cuda"),
        ("pytorch-cpu", args.torch_dir, "torch", "cpu"),
        ("onnx-cpu", args.onnx_dir, "onnx", None),
    ):
        if not directory:
            continue
        if device == "cuda":
            import torch

            if not torch.cuda.is_available():
                continue
        predictor = Predictor(directory, settings, backend=backend, device=device)
        results[name] = [
            measure(
                predictor,
                texts,
                size,
                settings.onnx.benchmark_warmup,
                settings.onnx.benchmark_iterations,
            )
            for size in settings.onnx.benchmark_batch_sizes
        ]

    payload = {"profile": profile(), "results": results}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("LATENCY")
    for key, value in payload["profile"].items():
        print(f"  {key}: {value}")
    print()
    print("| backend | batch | p50 ms | p95 ms | p99 ms | items/s |")
    print("|---|---|---|---|---|---|")
    for backend, rows in results.items():
        for row in rows:
            print(
                f"| {backend} | {row['batch_size']} | {row['p50_ms']} | {row['p95_ms']} | "
                f"{row['p99_ms']} | {row['items_per_second']} |"
            )
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
