"""Replacing the model in use: a champion, a challenger, criteria fixed in advance, and a recorded decision.

The honest way to replace a model that is in use is to compare it (the champion) with a candidate (the challenger) on
the same recent labelled rows, against criteria stated *before* the results are seen, and to promote only when the
challenger meets every one. Criteria chosen after seeing the numbers are a rationalisation.

A **comparison** names the two runs and carries its criteria. The criteria are a closed set of typed constraints --
data, never code, the same rule as business rules -- and they can be edited only until the comparison is evaluated;
after that an edit is refused. The evaluation, the decision log and the rollback build on this record.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

FOLDER = "promotion"
COMPARISONS_FILENAME = "comparisons.json"

Metric = Literal["accuracy", "precision", "recall", "f1", "roc_auc", "brier", "r2", "mae", "rmse"]
# Smaller is better for these; an improvement is a fall.
LOWER_IS_BETTER = {"brier", "mae", "rmse"}

_LOCK = threading.Lock()


class _Constraint(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MetricAtLeast(_Constraint):
    """The challenger's metric, on the evaluation rows at its own cut-off, is at least `value`."""

    kind: Literal["metric_at_least"]
    metric: Metric
    value: float


class MetricAtMost(_Constraint):
    """The challenger's metric is at most `value` (for a metric where smaller is better, such as the Brier score)."""

    kind: Literal["metric_at_most"]
    metric: Metric
    value: float


class NoWorseThanChampion(_Constraint):
    """The challenger is no worse than the champion on `metric`, by more than `tolerance`."""

    kind: Literal["no_worse_than_champion"]
    metric: Metric
    tolerance: float = Field(default=0.0, ge=0.0)


class NoNewRequiredColumns(_Constraint):
    """The challenger needs no input column the champion did not: the files people send keep working."""

    kind: Literal["no_new_required_columns"]


Constraint = Annotated[
    Union[MetricAtLeast, MetricAtMost, NoWorseThanChampion, NoNewRequiredColumns], Field(discriminator="kind")
]


class Criteria(BaseModel):
    """What a challenger must achieve to replace the champion: every constraint, and an improvement on the deciding
    metric of at least `min_improvement` that is also beyond the noise of the evaluation rows."""

    model_config = ConfigDict(extra="forbid")

    deciding_metric: Metric
    min_improvement: float = Field(default=0.0, ge=0.0)
    constraints: list[Constraint] = Field(default_factory=list, max_length=20)


class ComparisonError(Exception):
    """A request about a comparison that cannot be carried out, with a message for the user and a stable code."""

    def __init__(self, message: str, code: str):
        super().__init__(message)
        self.code = code


def _folder(directory: Path) -> Path:
    path = Path(directory) / FOLDER
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read(directory: Path) -> dict[str, dict]:
    path = Path(directory) / FOLDER / COMPARISONS_FILENAME
    if not path.exists():
        return {}
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
        return found if isinstance(found, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(directory: Path, comparisons: dict[str, dict]) -> None:
    (_folder(directory) / COMPARISONS_FILENAME).write_text(json.dumps(comparisons, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(directory: Path, *, champion_run_id: str, challenger_run_id: str, criteria: Criteria) -> dict:
    """A new comparison, open for its criteria to be adjusted until it is evaluated."""
    if champion_run_id == challenger_run_id:
        raise ComparisonError("The champion and the challenger are the same run.", "same_run")
    record = {
        "comparison_id": uuid.uuid4().hex[:12],
        "created_at": _now(),
        "champion_run_id": champion_run_id,
        "challenger_run_id": challenger_run_id,
        "criteria": criteria.model_dump(mode="json"),
        "status": "open",
        "evaluation": None,
    }
    with _LOCK:
        comparisons = _read(directory)
        comparisons[record["comparison_id"]] = record
        _write(directory, comparisons)
    return record


def get(directory: Path, comparison_id: str) -> dict:
    found = _read(directory).get(comparison_id)
    if found is None:
        raise ComparisonError(f"There is no comparison {comparison_id} in this session.", "comparison_not_found")
    return found


def all_comparisons(directory: Path) -> list[dict]:
    """Newest first."""
    return sorted(_read(directory).values(), key=lambda c: c.get("created_at") or "", reverse=True)


def set_criteria(directory: Path, comparison_id: str, criteria: Criteria) -> dict:
    """Replaces the criteria of an open comparison. Refused once it has been evaluated: criteria changed after the
    results are seen are not criteria."""
    with _LOCK:
        comparisons = _read(directory)
        record = comparisons.get(comparison_id)
        if record is None:
            raise ComparisonError(f"There is no comparison {comparison_id} in this session.", "comparison_not_found")
        if record["status"] != "open":
            raise ComparisonError(
                "This comparison has been evaluated, so its criteria are fixed: criteria changed after seeing the "
                "results would be chosen to fit them. Start a new comparison to use different criteria.",
                "criteria_frozen",
            )
        record["criteria"] = criteria.model_dump(mode="json")
        record["criteria_changed_at"] = _now()
        _write(directory, comparisons)
    return record


def update(directory: Path, comparison_id: str, **fields) -> dict:
    """Adds fields to a comparison record (the evaluation, its status)."""
    with _LOCK:
        comparisons = _read(directory)
        record = comparisons[comparison_id]
        record.update(fields)
        _write(directory, comparisons)
    return record
