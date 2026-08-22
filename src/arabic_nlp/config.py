"""Settings loaded from config.yaml.

Every number that a reported result depends on lives in that file, including
the split seed and the training-set cap. A result you cannot regenerate from a
committed config is an anecdote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


class DatasetSettings(BaseModel):
    id: str
    license: str
    downloaded: str
    text_column: str
    label_column: str
    source_split: str = "train"
    labels: dict[str, str]

    @property
    def label_names(self) -> list[str]:
        """Sorted, so label ids are stable across runs and machines.

        A label-to-id map built from set iteration order is the classic way to
        make a saved model silently predict the wrong classes after a reload.
        """
        return sorted(self.labels)


class SplitSettings(BaseModel):
    seed: int
    test_size: float
    validation_size: float
    max_train_rows: int = 0


class PreprocessSettings(BaseModel):
    replace_mentions: bool = True
    replace_urls: bool = True
    strip_repeated_chars: int = 3
    min_chars: int = 5
    normalize_arabic: bool = False


class BaselineSettings(BaseModel):
    char_ngram_range: tuple[int, int]
    char_max_features: int
    word_max_features: int
    min_df: int
    sublinear_tf: bool
    C: float
    max_iter: int
    class_weight: str | None = "balanced"
    normalize_arabic: bool = True


class TransformerSettings(BaseModel):
    base_model: str
    max_length: int
    batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    fp16: bool
    imbalance: Literal["class_weights", "oversample", "none"]
    seeds: list[int]
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 2


class OnnxSettings(BaseModel):
    opset: int = 17
    quantize: bool = True
    benchmark_batch_sizes: list[int] = Field(default_factory=lambda: [1, 8, 32])
    benchmark_warmup: int = 20
    benchmark_iterations: int = 100


class ServingSettings(BaseModel):
    max_batch: int = 32
    max_delay_ms: int = 10
    max_text_chars: int = 1000
    request_log: str = "logs/predictions.jsonl"


class DriftSettings(BaseModel):
    reference_rows: int = 5000
    window_rows: int = 500
    alert_threshold: float = 0.5


class MlflowSettings(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment: str = "arabic-dialect-id"


class Settings(BaseModel):
    dataset: DatasetSettings
    split: SplitSettings
    preprocess: PreprocessSettings
    baseline: BaselineSettings
    transformer: TransformerSettings
    onnx: OnnxSettings = Field(default_factory=OnnxSettings)
    serving: ServingSettings = Field(default_factory=ServingSettings)
    drift: DriftSettings = Field(default_factory=DriftSettings)
    mlflow: MlflowSettings = Field(default_factory=MlflowSettings)

    @model_validator(mode="after")
    def check_split_sizes(self) -> Settings:
        held_out = self.split.test_size + self.split.validation_size
        if not 0 < held_out < 0.9:
            raise ValueError(
                f"test_size + validation_size = {held_out}, which leaves "
                f"{1 - held_out:.0%} for training. Check config.yaml."
            )
        return self


def load_settings(path: str | Path | None = None) -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    return Settings.model_validate(
        yaml.safe_load(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    )
