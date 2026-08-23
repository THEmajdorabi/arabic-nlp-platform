"""Export to ONNX, quantize, and re-check accuracy.

The step people skip is the last one. Dynamic int8 quantization is nearly free
and usually costs a fraction of a point of F1, but "usually" is not a
measurement, and a quantized model that lost three points is a silent
regression that ships. So this exports, quantizes, scores both on the same
validation split, and reports the delta. If the delta is unacceptable, keeping
fp32 is a legitimate outcome and the README says which one is deployed.

Install note that costs an hour if you get it wrong: ONNX Runtime support moved
out of bare `optimum` in 2.x. `optimum[onnxruntime]` is what makes
`optimum.onnxruntime` importable; installing `optimum` alone gives an
ImportError on that path and nothing that explains why.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from arabic_nlp.config import Settings

log = structlog.get_logger(__name__)


@dataclass
class ExportResult:
    fp32_dir: Path
    int8_dir: Path | None
    torch_macro_f1: float
    fp32_macro_f1: float
    int8_macro_f1: float | None
    delta: float | None
    fp32_size_mb: float
    int8_size_mb: float | None
    recommended: str

    @property
    def export_delta(self) -> float:
        """fp32 ONNX minus the PyTorch model it came from.

        The question nobody asks and everybody should: did exporting change
        the model? Anything but a rounding difference here means the traced
        graph is not the model that was evaluated.
        """
        return self.fp32_macro_f1 - self.torch_macro_f1

    def as_dict(self) -> dict:
        return {
            "fp32_dir": str(self.fp32_dir),
            "int8_dir": str(self.int8_dir) if self.int8_dir else None,
            "torch_macro_f1": round(self.torch_macro_f1, 4),
            "export_delta": round(self.export_delta, 4),
            "fp32_macro_f1": round(self.fp32_macro_f1, 4),
            "int8_macro_f1": round(self.int8_macro_f1, 4) if self.int8_macro_f1 else None,
            "macro_f1_delta": round(self.delta, 4) if self.delta is not None else None,
            "fp32_size_mb": round(self.fp32_size_mb, 1),
            "int8_size_mb": round(self.int8_size_mb, 1) if self.int8_size_mb else None,
            "recommended": self.recommended,
        }


def directory_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def export(model_dir: str | Path, output_dir: str | Path, settings: Settings) -> Path:
    """Export a saved PyTorch model to ONNX, tokenizer included."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    source = Path(model_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    log.info("onnx.exporting", source=str(source), target=str(target))
    model = ORTModelForSequenceClassification.from_pretrained(str(source), export=True)
    model.save_pretrained(str(target))
    # The tokenizer travels with the weights here too, for the same reason it
    # does in training: a service that fetches it separately can silently get a
    # different one.
    AutoTokenizer.from_pretrained(str(source)).save_pretrained(str(target))

    # And the training record, so `/model-info` can report which run produced
    # the graph it is serving. Without it the endpoint answers `null` for the
    # validation macro F1, which is worse than not offering the field.
    record = source / "training_record.json"
    if record.exists():
        shutil.copy2(record, target / "training_record.json")
    return target


def quantize(onnx_dir: str | Path, output_dir: str | Path) -> Path:
    """Dynamic int8 quantization of the exported graph."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    source = Path(onnx_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    quantizer = ORTQuantizer.from_pretrained(str(source))
    config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    log.info("onnx.quantizing", source=str(source), target=str(target))
    quantizer.quantize(save_dir=str(target), quantization_config=config)

    # The quantizer writes the graph but not the tokenizer or the label map,
    # and a model directory missing either is one the service refuses to load.
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "config.json",
        "training_record.json",
    ):
        candidate = source / name
        if candidate.exists() and not (target / name).exists():
            shutil.copy2(candidate, target / name)
    return target


def score(
    model_dir: str | Path,
    texts: list[str],
    y_true: np.ndarray,
    settings: Settings,
    backend: str = "onnx",
    onnx_file_name: str | None = None,
) -> float:
    """Macro F1 of a model directory on the given texts, via the serving path.

    Deliberately through `Predictor`, the same class the service uses, so this
    measures what will actually be served rather than a parallel code path that
    happens to agree today.
    """
    from sklearn.metrics import f1_score

    from arabic_nlp.serving.predictor import Predictor

    predictor = Predictor(model_dir, settings, backend=backend, onnx_file_name=onnx_file_name)
    probabilities = predictor.probabilities(texts)
    y_pred = probabilities.argmax(axis=1)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def export_and_verify(
    model_dir: str | Path,
    output_root: str | Path,
    texts: list[str],
    y_true: np.ndarray,
    settings: Settings,
    max_acceptable_drop: float = 0.01,
) -> ExportResult:
    root = Path(output_root)

    # The reference: the PyTorch model, on exactly these texts. Without it,
    # "int8 scores 0.87" is a number with nothing to be 0.87 against.
    torch_f1 = score(model_dir, texts, y_true, settings, backend="torch")
    log.info("onnx.torch_scored", macro_f1=round(torch_f1, 4))

    fp32_dir = export(model_dir, root / "fp32", settings)
    fp32_f1 = score(fp32_dir, texts, y_true, settings, onnx_file_name="model.onnx")
    log.info("onnx.fp32_scored", macro_f1=round(fp32_f1, 4), delta=round(fp32_f1 - torch_f1, 4))

    int8_dir: Path | None = None
    int8_f1: float | None = None
    delta: float | None = None
    int8_size: float | None = None

    if settings.onnx.quantize:
        try:
            int8_dir = quantize(fp32_dir, root / "int8")
            int8_f1 = score(
                int8_dir, texts, y_true, settings, onnx_file_name="model_quantized.onnx"
            )
            delta = int8_f1 - fp32_f1
            int8_size = directory_size_mb(int8_dir)
            log.info("onnx.int8_scored", macro_f1=round(int8_f1, 4), delta=round(delta, 4))
        except Exception as exc:  # noqa: BLE001
            # Quantization failing is a result, not a crash: fp32 still ships.
            log.error("onnx.quantize_failed", error=str(exc))

    if int8_f1 is None:
        recommended = "fp32 (quantization did not produce a working model)"
    elif delta is not None and delta < -max_acceptable_drop:
        recommended = (
            f"fp32 (int8 lost {abs(delta):.4f} macro F1, over the {max_acceptable_drop:.3f} budget)"
        )
    else:
        recommended = "int8"

    result = ExportResult(
        fp32_dir=fp32_dir,
        int8_dir=int8_dir,
        torch_macro_f1=torch_f1,
        fp32_macro_f1=fp32_f1,
        int8_macro_f1=int8_f1,
        delta=delta,
        fp32_size_mb=directory_size_mb(fp32_dir),
        int8_size_mb=int8_size,
        recommended=recommended,
    )
    (root / "export_report.json").write_text(
        json.dumps(result.as_dict(), indent=2), encoding="utf-8"
    )
    return result
