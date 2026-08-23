"""Drift monitoring with Evidently.

**The API here is Evidently 0.7's and only 0.7's.** 0.7 rewrote everything
around `Report` + `Dataset` + `DataDefinition`. Essentially every tutorial
online, and the archived docs, use `evidently.report.Report` with
`ColumnDriftMetric`, which does not import on this version. The version is
pinned in pyproject.toml and this docstring is the warning.

What is compared, and why these four:

  * **text length**: the cheapest early warning there is. A change in where
    the traffic comes from (a new client, a new form field, a bot) shows up
    here before it shows up anywhere else.
  * **word count**: length in characters and length in words drift apart when
    the script or the tokenisation changes.
  * **predicted class distribution**: the model's own output. If the mix
    moves and the inputs did not, the model is behaving differently.
  * **mean confidence**: the leading indicator of out-of-distribution input.
    A model asked about something it has never seen gets *less* certain before
    it gets wrong, and this is the number that catches it.

Raw text is never compared, because raw text is never stored: the request log
keeps a hash. That is a deliberate limit: an embedding-drift check would be
stronger and would require keeping the text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import structlog

from arabic_nlp.config import REPO_ROOT, Settings, load_settings

log = structlog.get_logger(__name__)

REPORT_PATH = REPO_ROOT / "artifacts" / "drift" / "latest.json"
REFERENCE_PATH = REPO_ROOT / "artifacts" / "drift" / "reference.parquet"

MONITORED_NUMERIC = ["chars", "words", "confidence"]
MONITORED_CATEGORICAL = ["label"]


@dataclass
class DriftReport:
    drifted_columns: int
    total_columns: int
    drift_share: float
    per_column: dict[str, dict[str, Any]]
    reference_rows: int
    current_rows: int
    alert: bool
    threshold: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "drifted_columns": self.drifted_columns,
            "total_columns": self.total_columns,
            "drift_share": round(self.drift_share, 4),
            "alert": self.alert,
            "threshold": self.threshold,
            "reference_rows": self.reference_rows,
            "current_rows": self.current_rows,
            "per_column": self.per_column,
        }

    def render(self) -> str:
        lines = [
            f"DRIFT  {self.drifted_columns}/{self.total_columns} columns "
            f"(share {self.drift_share:.2f}, threshold {self.threshold:.2f})"
            + ("   *** ALERT ***" if self.alert else ""),
            f"  reference {self.reference_rows} rows, current {self.current_rows} rows",
            "",
            "| Column | method | score | drifted |",
            "|---|---|---|---|",
        ]
        for column, values in self.per_column.items():
            lines.append(
                f"| {column} | {values['method']} | {values['score']:.4f} | "
                f"{'yes' if values['drifted'] else 'no'} |"
            )
        return "\n".join(lines)


def build_reference(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The training-time reference distribution.

    Built from the training split's texts and the model's own predictions on
    them, so the reference for `label` and `confidence` is what the model
    actually does, not the ground-truth distribution, which the service will
    never see again.
    """
    rows = frame.head(settings.drift.reference_rows).copy()
    rows["chars"] = rows["text"].str.len()
    rows["words"] = rows["text"].str.split().str.len()
    return rows[["chars", "words", "label", "confidence"]]


def load_request_log(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Read the rolling prediction log written by the service."""
    file = Path(path)
    if not file.exists():
        return pd.DataFrame(columns=["chars", "words", "label", "confidence"])
    records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line]
    if limit:
        records = records[-limit:]
    frame = pd.DataFrame(records)
    return frame[["chars", "words", "label", "confidence"]] if len(frame) else frame


def compute_drift(
    current: pd.DataFrame, reference: pd.DataFrame, settings: Settings | None = None
) -> DriftReport:
    """Run the Evidently drift preset and flatten it into something serveable."""
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    settings = settings or load_settings()
    threshold = settings.drift.alert_threshold

    definition = DataDefinition(
        numerical_columns=MONITORED_NUMERIC,
        categorical_columns=MONITORED_CATEGORICAL,
    )
    report = Report([DataDriftPreset(drift_share=threshold)])
    run = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )

    payload = run.dict()
    per_column: dict[str, dict[str, Any]] = {}
    drifted = 0
    total = 0
    share = 0.0

    for metric in payload.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")

        if name.startswith("DriftedColumnsCount") and isinstance(value, dict):
            drifted = int(value.get("count", 0))
            share = float(value.get("share", 0.0))
            continue

        if not name.startswith("ValueDrift"):
            continue

        config = metric.get("config", {})
        column = config.get("column")
        column_threshold = float(config.get("threshold", 0.1))
        score = float(value) if isinstance(value, int | float) else 0.0
        # Evidently's distance metrics are "higher means more different", and
        # each carries its own threshold in the config, so the comparison
        # uses that rather than a number hardcoded here.
        per_column[str(column)] = {
            "method": config.get("method", "unknown"),
            "score": score,
            "threshold": column_threshold,
            "drifted": score > column_threshold,
        }
        total += 1

    return DriftReport(
        drifted_columns=drifted,
        total_columns=total or len(per_column),
        drift_share=share,
        per_column=per_column,
        reference_rows=len(reference),
        current_rows=len(current),
        alert=share >= threshold,
        threshold=threshold,
    )


def save_report(report: DriftReport, path: str | Path = REPORT_PATH) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    log.info("drift.saved", alert=report.alert, share=report.drift_share, path=str(target))


def latest_report(path: str | Path = REPORT_PATH) -> dict[str, Any]:
    """What `/drift` and `/metrics` serve.

    Returns an explanation rather than zeros when no report exists. A drift
    gauge sitting at 0.0 because nothing has ever run looks identical to one
    sitting at 0.0 because nothing is wrong.
    """
    file = Path(path)
    if not file.exists():
        return {
            "status": "no report",
            "drift_share": 0.0,
            "alert": False,
            "detail": (
                "no drift report has been generated. Run scripts/check_drift.py. "
                "A gauge of 0.0 here means 'not measured', not 'no drift'."
            ),
        }
    payload = json.loads(file.read_text(encoding="utf-8"))
    payload["status"] = "ok"
    return payload
