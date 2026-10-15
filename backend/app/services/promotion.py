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


# --- Evaluating both on the same rows (#82) -------------------------------------------------------------------------

# The bootstrap's name for a metric where it differs from the criteria's.
_BOOTSTRAP_NAME = {"brier": "brier_score"}


def _scored(directory: Path, run_id: str, new, original, fallback_actions) -> dict:
    """One run's view of the new file: its own cleaning (with its own learned values), its own schema check, its own
    inputs and model, at its own cut-off. `rows` are the labels of the rows it could score."""
    import pandas as pd

    from app.services import model_store, replay, scoring, validation

    result = model_store.load_run(directory, run_id)
    pipeline, notes = model_store.load_run_pipeline(directory, run_id)
    actions = (result.get("run_metadata") or {}).get("actions")
    if actions is None:
        actions = fallback_actions or []
    steps, reproduced = replay.freeze(original, actions)
    target = result.get("target_column")
    applied = replay.apply(new, steps, target_column=target)
    schema = result.get("input_schema")
    if schema:
        report, _ = validation.validate(schema, applied.frame, target)
        if report["blocking"]:
            return {"run_id": run_id, "result": result, "refused": validation.describe_blocking(report)}
    X, _ = scoring.prepare_inputs(applied.frame, result, reference=reproduced)
    predicted = scoring.predict(pipeline, X, result) if len(X) else pd.DataFrame(index=X.index)
    return {"run_id": run_id, "result": result, "predicted": predicted, "notes": notes, "rows": set(X.index)}


def _positive_label(result: dict, truth) -> str | None:
    import pandas as pd

    labels = sorted(pd.Series(truth).astype(str).unique())
    if len(labels) != 2:
        return None
    wanted = (result.get("operating_point") or {}).get("positive_label")
    return str(wanted) if wanted is not None and str(wanted) in labels else labels[-1]


def _predictions(scored: dict, rows: list, task: str, positive: str | None) -> dict:
    predicted = scored["predicted"].loc[rows]
    target = scored["result"].get("target_column")
    called = predicted[f"predicted_{target}"].to_numpy()
    column = f"probability_{positive}"
    scores = predicted[column].to_numpy() if positive is not None and column in predicted.columns else None
    return {"called": called if task == "regression" else called.astype(str), "scores": scores}


def _criterion(constraint: dict, champion: dict, challenger: dict, champion_inputs: set, challenger_inputs: set) -> dict:
    kind = constraint["kind"]
    if kind == "no_new_required_columns":
        added = sorted(challenger_inputs - champion_inputs)
        return {**constraint, "passed": not added,
                "evidence": "needs no column the champion did not" if not added else f"also needs {', '.join(added)}"}
    metric = constraint["metric"]
    have, before = challenger.get(metric), champion.get(metric)
    if have is None:
        return {**constraint, "passed": False, "evidence": f"{metric} could not be computed for the challenger"}
    if kind == "metric_at_least":
        return {**constraint, "passed": have >= constraint["value"], "evidence": f"{metric} {have:.4f}"}
    if kind == "metric_at_most":
        return {**constraint, "passed": have <= constraint["value"], "evidence": f"{metric} {have:.4f}"}
    if before is None:
        return {**constraint, "passed": False, "evidence": f"{metric} could not be computed for the champion"}
    tolerance = constraint.get("tolerance", 0.0)
    worse_by = (have - before) if metric in LOWER_IS_BETTER else (before - have)
    return {**constraint, "passed": worse_by <= tolerance,
            "evidence": f"{metric} {have:.4f} against {before:.4f} for the champion"}


def _inputs(result: dict) -> set:
    columns = {c["name"] for c in (result.get("input_schema") or {}).get("columns") or []}
    if columns:
        return columns
    feats = result.get("features") or {}
    return set(feats.get("numeric", []) + feats.get("categorical", []) + feats.get("frequent", []))


def evaluate(directory: Path, comparison_id: str, new, *, file_fingerprint: str, fallback_actions=None) -> dict:
    """Both runs scored on the same rows of a labelled file, the criteria checked, the comparison frozen.

    Each run applies its own cleaning and inputs. Rows either cannot score, rows without an answer, and rows that
    were in the file the models were built from (scoring a model on its training rows is not new evidence) are left
    out and counted. The deciding metric's difference is paired on the shared rows, so it is far narrower than two
    separate ranges.
    """
    import pandas as pd

    from app.api.routes_apply import _rows_seen_at_build
    from app.services import uncertainty

    record = get(directory, comparison_id)
    if record["status"] != "open":
        raise ComparisonError(
            "This comparison has already been evaluated, and its result stands. Start a new comparison to evaluate "
            "on another file.", "already_evaluated",
        )
    criteria = Criteria.model_validate(record["criteria"])
    original = pd.read_csv(Path(directory) / "original.csv")
    champion = _scored(directory, record["champion_run_id"], new, original, fallback_actions)
    challenger = _scored(directory, record["challenger_run_id"], new, original, fallback_actions)
    target = champion["result"].get("target_column")
    task = champion["result"].get("task") or "classification"
    if target not in new.columns:
        raise ComparisonError(
            f"The evaluation file has no '{target}' column, so there are no answers to compare the models on.",
            "no_answers",
        )
    refused = {role: s["refused"] for role, s in (("champion", champion), ("challenger", challenger)) if "refused" in s}
    if refused:
        raise ComparisonError(" ".join(f"The {role}: {why}" for role, why in refused.items()), "cannot_score")

    labelled = set(new.index[new[target].notna()])
    seen = _rows_seen_at_build(new, original, target)
    both = champion["rows"] & challenger["rows"]
    rows = sorted((both & labelled) - seen)
    excluded = {
        "only_one_model_could_score": len((champion["rows"] | challenger["rows"]) - both),
        "no_answer": len(both - labelled),
        "in_the_training_file": len((both & labelled) & seen),
    }
    if len(rows) < 2:
        raise ComparisonError(
            f"Only {len(rows)} labelled row{'s' if len(rows) != 1 else ''} could be scored by both models and were "
            f"not in the training file, which is too few to compare anything.", "too_few_rows",
        )
    truth = new.loc[rows, target]
    truth = truth.astype(float).to_numpy() if task == "regression" else truth.astype(str).to_numpy()
    positive = _positive_label(challenger["result"], truth) if task != "regression" else None
    a = _predictions(challenger, rows, task, positive)
    b = _predictions(champion, rows, task, positive)
    metrics = {
        "champion": uncertainty.point_estimates(truth, b["called"], b["scores"], task, positive),
        "challenger": uncertainty.point_estimates(truth, a["called"], a["scores"], task, positive),
    }
    for side in metrics.values():
        if "brier_score" in side:
            side["brier"] = side["brier_score"]

    deciding = criteria.deciding_metric
    paired = uncertainty.paired_difference(truth, a, b, task, _BOOTSTRAP_NAME.get(deciding, deciding),
                                           positive_label=positive)
    sign = -1.0 if deciding in LOWER_IS_BETTER else 1.0
    if paired is None:
        deciding_result = {"metric": deciding, "passed": False,
                           "evidence": f"{deciding} could not be computed for both models on these rows"}
    else:
        improvement = sign * paired["difference"]
        # The improvement's own range: for a metric where smaller is better, a fall is a gain, so the bounds swap.
        low, high = sorted((sign * paired["low"], sign * paired["high"]))
        deciding_result = {
            "metric": deciding, "min_improvement": criteria.min_improvement,
            "difference": round(paired["difference"], 6), "improvement": round(improvement, 6),
            "improvement_low": round(low, 6), "improvement_high": round(high, 6),
            "passed": bool(improvement >= criteria.min_improvement and low > 0),
            "evidence": (
                f"{deciding} {metrics['challenger'].get(deciding, float('nan')):.4f} against "
                f"{metrics['champion'].get(deciding, float('nan')):.4f} for the champion: an improvement of "
                f"{improvement:+.4f} (95% range {low:+.4f} to {high:+.4f}), needing at least "
                f"{criteria.min_improvement:g} and to be beyond the noise"
            ),
        }
    constraints = [
        _criterion(c.model_dump(mode="json"), metrics["champion"], metrics["challenger"],
                   _inputs(champion["result"]), _inputs(challenger["result"]))
        for c in criteria.constraints
    ]
    evaluation = {
        "evaluated_at": _now(),
        "file_fingerprint": file_fingerprint,
        "rows_compared": len(rows),
        "rows_excluded": excluded,
        "metrics": metrics,
        "deciding": deciding_result,
        "constraints": constraints,
        "all_passed": bool(deciding_result["passed"] and all(c["passed"] for c in constraints)),
    }
    return update(directory, comparison_id, status="evaluated", evaluation=evaluation)
