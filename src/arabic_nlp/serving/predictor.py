"""Loading a trained model and predicting with it, PyTorch or ONNX.

Two things are asserted at startup rather than discovered in production:

**The tokenizer and the weights come from the same directory.** A tokenizer
mismatch does not raise -- it produces token ids the model was never trained on
and the service returns confident nonsense. The check is cheap and the failure
it prevents is silent.

**The label order comes from the saved model's own `id2label`**, not from
config.yaml. If the two disagree, that is a real inconsistency and it stops the
service, because the alternative is a model that predicts the right class and a
service that reports the wrong name for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from arabic_nlp.config import Settings
from arabic_nlp.data.preprocess import clean

log = structlog.get_logger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    all_scores: dict[str, float]


class Predictor:
    def __init__(
        self,
        model_dir: str | Path,
        settings: Settings,
        backend: str = "torch",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.settings = settings
        self.backend = backend
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"no model at {self.model_dir}. Train one with scripts/train_transformer.py."
            )

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))

        if backend == "onnx":
            from optimum.onnxruntime import ORTModelForSequenceClassification

            self.model = ORTModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self.device = "cpu"
        else:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device).eval()

        self.labels = self._labels()
        self._check_tokenizer()
        log.info(
            "predictor.loaded",
            model_dir=str(self.model_dir),
            backend=backend,
            device=self.device,
            labels=self.labels,
        )

    def _labels(self) -> list[str]:
        config = self.model.config
        mapping = getattr(config, "id2label", None) or {}
        labels = [mapping[i] for i in sorted(int(k) for k in mapping)]

        expected = self.settings.dataset.label_names
        if sorted(labels) != sorted(expected):
            raise ValueError(
                f"the saved model's labels {labels} do not match config.yaml's {expected}. "
                f"One of them is stale; refusing to serve rather than mislabel predictions."
            )
        return labels

    def _check_tokenizer(self) -> None:
        """The tokenizer must be the one saved beside these weights."""
        tokenizer_config = self.model_dir / "tokenizer_config.json"
        if not tokenizer_config.exists():
            raise FileNotFoundError(
                f"no tokenizer in {self.model_dir}. The tokenizer must be saved with the "
                f"weights; loading it from the Hub separately is how a silent mismatch happens."
            )
        vocab_size = getattr(self.tokenizer, "vocab_size", None)
        model_vocab = getattr(self.model.config, "vocab_size", None)
        if vocab_size and model_vocab and vocab_size > model_vocab:
            raise ValueError(
                f"tokenizer vocab ({vocab_size}) is larger than the model's ({model_vocab}). "
                f"These are not a matched pair."
            )

    def preprocess(self, texts: list[str]) -> list[str]:
        """Exactly the cleaning used in training. Imported, never reimplemented."""
        pre = self.settings.preprocess
        return [
            clean(
                text[: self.settings.serving.max_text_chars],
                replace_mentions=pre.replace_mentions,
                replace_urls=pre.replace_urls,
                strip_repeated_chars=pre.strip_repeated_chars,
                normalize_arabic=pre.normalize_arabic,
            )
            for text in texts
        ]

    def probabilities(self, texts: list[str]) -> np.ndarray:
        cleaned = self.preprocess(texts)
        encoded = self.tokenizer(
            cleaned,
            truncation=True,
            max_length=self.settings.transformer.max_length,
            padding=True,
            return_tensors="pt" if self.backend != "onnx" else "np",
        )

        if self.backend == "onnx":
            logits = self.model(**dict(encoded)).logits
            logits = np.asarray(logits)
        else:
            import torch

            with torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
            logits = logits.cpu().numpy()

        shifted = logits - logits.max(axis=1, keepdims=True)
        exponent = np.exp(shifted)
        return exponent / exponent.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[Prediction]:
        scores = self.probabilities(texts)
        out: list[Prediction] = []
        for row in scores:
            best = int(row.argmax())
            out.append(
                Prediction(
                    label=self.labels[best],
                    confidence=float(row[best]),
                    all_scores={
                        label: round(float(value), 4)
                        for label, value in zip(self.labels, row, strict=True)
                    },
                )
            )
        return out

    def model_info(self) -> dict[str, Any]:
        record_path = self.model_dir / "training_record.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
        return {
            "model_dir": str(self.model_dir),
            "backend": self.backend,
            "device": self.device,
            "labels": self.labels,
            "base_model": record.get("base_model", self.settings.transformer.base_model),
            "seed": record.get("seed"),
            "imbalance_strategy": record.get("strategy"),
            "validation_macro_f1": record.get("eval_macro_f1"),
            "train_rows": record.get("train_rows"),
            "dataset": self.settings.dataset.id,
            "max_length": self.settings.transformer.max_length,
        }
