"""Stratified, seeded train/validation/test splits.

Three properties, each of which is a way results get faked when it is missing:

1. **Stratified.** The corpus runs 39% Egyptian to 7.8% Moroccan. An unstratified
   test split can hold noticeably fewer Moroccan rows than the corpus does, and
   per-class F1 on the smallest class is then measured on a sample small enough
   to move several points by luck.
2. **Seeded, from config.yaml.** A split you cannot reproduce means a number you
   cannot reproduce.
3. **Test is created once and not read until the end.** `load_test` exists and
   is called in exactly one place. The README states how many times it was run.

The training cap (`split.max_train_rows`) is applied AFTER splitting and only
to the training set, so validation and test always cover the full held-out
data. Capping before splitting would shrink the evaluation sets too, and make
the results a measurement of the cap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import structlog
from sklearn.model_selection import train_test_split

from arabic_nlp.config import REPO_ROOT, Settings

log = structlog.get_logger(__name__)

SPLIT_DIR = REPO_ROOT / "data" / "splits"
SPLIT_NAMES = ("train", "validation", "test")


def make_splits(frame: pd.DataFrame, settings: Settings) -> dict[str, pd.DataFrame]:
    seed = settings.split.seed
    test_size = settings.split.test_size
    validation_size = settings.split.validation_size

    rest, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame["label"],
    )
    # The validation fraction is of the ORIGINAL frame, so it has to be
    # rescaled against what is left after the test split is taken out.
    validation_fraction = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        rest,
        test_size=validation_fraction,
        random_state=seed,
        stratify=rest["label"],
    )

    cap = settings.split.max_train_rows
    if cap and len(train) > cap:
        # Stratified subsample: take a stratified slice rather than a head,
        # which would follow whatever order the upload happened to be in.
        train, _ = train_test_split(
            train, train_size=cap, random_state=seed, stratify=train["label"]
        )
        log.info("splits.train_capped", cap=cap)

    splits = {
        "train": train.reset_index(drop=True),
        "validation": validation.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }
    for name, part in splits.items():
        log.info("splits.built", split=name, rows=len(part))
    return splits


def save_splits(splits: dict[str, pd.DataFrame], directory: str | Path = SPLIT_DIR) -> None:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_parquet(path / f"{name}.parquet", index=False)


def load_split(name: str, directory: str | Path = SPLIT_DIR) -> pd.DataFrame:
    if name not in SPLIT_NAMES:
        raise ValueError(f"unknown split {name!r}. Expected one of {SPLIT_NAMES}.")
    path = Path(directory) / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run `python scripts/prepare_data.py`.")
    return pd.read_parquet(path)


def load_test(directory: str | Path = SPLIT_DIR) -> pd.DataFrame:
    """The test set.

    Deliberately a separate function with a name that shows up in a grep, so
    "how many times was the test set touched" is a question with an auditable
    answer rather than an assurance.
    """
    log.warning("test_set.read", note="the test set is being read; this should be rare")
    return load_split("test", directory)


def splits_exist(directory: str | Path = SPLIT_DIR) -> bool:
    return all((Path(directory) / f"{n}.parquet").exists() for n in SPLIT_NAMES)


def distribution(frame: pd.DataFrame) -> dict[str, float]:
    counts = frame["label"].value_counts(normalize=True)
    return {str(k): round(float(v), 4) for k, v in counts.items()}
