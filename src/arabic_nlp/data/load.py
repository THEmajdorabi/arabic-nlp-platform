"""Download the dataset, clean it, and report what is actually in it.

The class distribution report is not decoration. Arabic dialect data is always
imbalanced, and a macro-F1 number means nothing to a reader who does not know
that the largest class is five times the smallest. It is printed before any
model is trained and written to `data/class_distribution.json`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import structlog

from arabic_nlp.config import REPO_ROOT, Settings
from arabic_nlp.data.preprocess import clean, is_usable

log = structlog.get_logger(__name__)

DATA_DIR = REPO_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "dataset.parquet"
CLEAN_PATH = DATA_DIR / "processed" / "clean.parquet"
DISTRIBUTION_PATH = DATA_DIR / "class_distribution.json"


@dataclass
class LoadReport:
    rows_raw: int
    rows_clean: int
    dropped_short: int
    dropped_empty: int
    dropped_duplicate: int
    classes: dict[str, int]

    @property
    def imbalance_ratio(self) -> float:
        """Largest class over smallest. The number that makes accuracy a lie."""
        counts: list[int] = list(self.classes.values())
        if not counts:
            return 0.0
        smallest = min(counts)
        return max(counts) / smallest if smallest else 0.0

    def as_dict(self) -> dict:
        total = sum(self.classes.values()) or 1
        return {
            "rows_raw": self.rows_raw,
            "rows_clean": self.rows_clean,
            "dropped_short": self.dropped_short,
            "dropped_empty": self.dropped_empty,
            "dropped_duplicate": self.dropped_duplicate,
            "imbalance_ratio": round(self.imbalance_ratio, 2),
            "classes": {
                label: {"n": n, "share": round(n / total, 4)}
                for label, n in sorted(self.classes.items(), key=lambda kv: -kv[1])
            },
        }

    def render(self) -> str:
        lines = [
            "CLASS DISTRIBUTION",
            f"  rows: {self.rows_raw} raw -> {self.rows_clean} usable "
            f"({self.dropped_empty} empty, {self.dropped_short} too short, "
            f"{self.dropped_duplicate} duplicate)",
            "",
            "  | Class | n | share |",
            "  |---|---|---|",
        ]
        total = sum(self.classes.values()) or 1
        for label, n in sorted(self.classes.items(), key=lambda kv: -kv[1]):
            lines.append(f"  | {label} | {n} | {n / total:.2%} |")
        lines.append("")
        lines.append(
            f"  imbalance ratio (largest/smallest): {self.imbalance_ratio:.2f}x  "
            f"-- report macro F1 and per-class F1, never accuracy alone."
        )
        return "\n".join(lines)


def download(settings: Settings, force: bool = False) -> pd.DataFrame:
    """Fetch the dataset once and cache it as parquet.

    Cached, because a training pipeline that re-downloads on every run is one
    Hub outage away from being unreproducible.
    """
    if RAW_PATH.exists() and not force:
        log.info("dataset.cached", path=str(RAW_PATH))
        return pd.read_parquet(RAW_PATH)

    from datasets import load_dataset

    log.info("dataset.downloading", id=settings.dataset.id)
    split = load_dataset(settings.dataset.id, split=settings.dataset.source_split)
    frame = split.to_pandas()[[settings.dataset.text_column, settings.dataset.label_column]]
    frame.columns = ["text", "label"]

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(RAW_PATH, index=False)
    log.info("dataset.saved", rows=len(frame), path=str(RAW_PATH))
    return frame


def build_clean(settings: Settings, force: bool = False) -> tuple[pd.DataFrame, LoadReport]:
    """Clean, deduplicate, drop the unusable, and report what happened."""
    if CLEAN_PATH.exists() and DISTRIBUTION_PATH.exists() and not force:
        frame = pd.read_parquet(CLEAN_PATH)
        report = json.loads(DISTRIBUTION_PATH.read_text(encoding="utf-8"))
        return frame, LoadReport(
            rows_raw=report["rows_raw"],
            rows_clean=report["rows_clean"],
            dropped_short=report["dropped_short"],
            dropped_empty=report["dropped_empty"],
            dropped_duplicate=report["dropped_duplicate"],
            classes={k: v["n"] for k, v in report["classes"].items()},
        )

    raw = download(settings, force=force)
    rows_raw = len(raw)

    known = set(settings.dataset.label_names)
    unknown = set(raw["label"].unique()) - known
    if unknown:
        # Loud: a label the config does not know about would otherwise become a
        # silently dropped class, or a KeyError deep inside training.
        raise ValueError(
            f"dataset contains labels not listed in config.yaml: {sorted(unknown)}. "
            f"Known: {sorted(known)}"
        )

    settings_pre = settings.preprocess
    raw["text"] = [
        clean(
            t,
            replace_mentions=settings_pre.replace_mentions,
            replace_urls=settings_pre.replace_urls,
            strip_repeated_chars=settings_pre.strip_repeated_chars,
            normalize_arabic=settings_pre.normalize_arabic,
        )
        for t in raw["text"].astype(str)
    ]

    empty = int((raw["text"].str.len() == 0).sum())
    frame = raw[raw["text"].str.len() > 0]

    keep = frame["text"].map(lambda t: is_usable(t, settings_pre.min_chars))
    short = int((~keep).sum())
    frame = frame[keep]

    before = len(frame)
    # Retweets and copy-pastes are rife in tweet corpora, and a duplicate that
    # lands in both train and test is leakage that inflates every number.
    frame = frame.drop_duplicates(subset=["text"]).reset_index(drop=True)
    duplicate = before - len(frame)

    report = LoadReport(
        rows_raw=rows_raw,
        rows_clean=len(frame),
        dropped_short=short,
        dropped_empty=empty,
        dropped_duplicate=duplicate,
        classes=frame["label"].value_counts().to_dict(),
    )

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CLEAN_PATH, index=False)
    DISTRIBUTION_PATH.write_text(
        json.dumps(report.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("dataset.cleaned", rows=len(frame), dropped_duplicate=duplicate)
    return frame, report


def label_to_id(settings: Settings) -> dict[str, int]:
    """Stable label ids, from the sorted label list in config.yaml."""
    return {name: index for index, name in enumerate(settings.dataset.label_names)}


def id_to_label(settings: Settings) -> dict[int, str]:
    return dict(enumerate(settings.dataset.label_names))


def write_distribution_markdown(report: LoadReport, path: str | Path) -> None:
    Path(path).write_text(report.render() + "\n", encoding="utf-8")
