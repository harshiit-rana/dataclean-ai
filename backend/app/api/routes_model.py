"""Model building: plan, train, retrieve. PRD-adjacent -- this extends the tool past cleaning.

Three endpoints, in the order a person actually works:

- `GET  /api/model/{id}/plan`     what would be built, and what the app proposes leaving out
- `POST /api/model/{id}/train`    build it, with the user's adjustments
- `POST /api/model/{id}/improve`  keep searching for a better one, and say when to stop
- `GET  /api/model/{id}/result`   the last evaluation, so a refresh doesn't lose it

The plan step exists because the interesting decisions in a model build are made before any
fitting happens -- which columns count as inputs, how the data is split, what is being
optimized. Hiding those behind a "train" button makes them defaults nobody chose. This is the
same human-in-the-loop shape the cleaning side uses: the app proposes with reasons, the
person decides.

Training runs in a worker thread. It is CPU-bound and would otherwise block the event loop
for every other request in the process, including the SSE streams.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from collections.abc import AsyncIterator

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, PlainTextResponse

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import ImproveRequest, ModelPlanResponse, TrainRequest
from app.models.graph_state import GoalContext
from app.services import (
    action_log as action_log_service,
    clipping,
    diagnostics,
    evaluation_ledger,
    features as feature_recipes,
    improvement,
    ml_readiness,
    model_card,
    model_export,
    model_store,
    modeling,
    operating_point,
)
from app.api.routes_rules import stored_rules
from app.dependencies import get_graph_if_any
from app.api.sse import sse_response
from app.services.session_store import (
    current_dataset_version_in,
    read_session_meta,
    session_dir,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class NoTargetColumnError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "No prediction target",
            "This session didn't name a column to predict, so there's nothing to build a "
            "model for. Start a new session and choose a target on the upload step.",
        )


class TrainingFailedError(AppError):
    status_code = 422  # the request was well-formed; the data can't support it

    def __init__(self, detail: str):
        super().__init__("Model could not be trained", detail)


class NoTrainedModelError(AppError):
    status_code = 404

    def __init__(self):
        super().__init__("No model yet", "Nothing has been trained for this session.")


class BadOperatingPointError(AppError):
    status_code = 422

    def __init__(self, detail: str):
        super().__init__("Cut-off trade-off not understood", detail)


class NoSealedHoldoutError(AppError):
    status_code = 409

    def __init__(self, reason: str | None = None):
        super().__init__(
            "There is no sealed holdout to open",
            "This model was trained without a sealed final holdout"
            + (f" ({reason})" if reason else "")
            + ". Tick \"keep a sealed final holdout\" in the plan and train again; it has to "
            "be taken before the first look to be worth anything.",
        )


class SealedRowsChangedError(AppError):
    status_code = 409

    def __init__(self):
        super().__init__(
            "The sealed rows are not what was sealed",
            "The data no longer produces the rows that were set aside, so opening them would "
            "not be the evaluation that was promised. Train again to seal a new holdout.",
        )


def _operating_point(request: TrainRequest) -> operating_point.OperatingPoint:
    try:
        return operating_point.parse(
            request.operating_point.model_dump() if request.operating_point else None
        )
    except ValueError as exc:
        raise BadOperatingPointError(str(exc)) from exc


def _load_session(
    settings: Settings, session_id: str
) -> tuple[pd.DataFrame, GoalContext, object, list]:
    """Reads the cleaned file if cleaning has run, the upload otherwise, the goal, and the
    business rules in force.

    Rules come back from here rather than being re-read at each call site: every caller
    that assesses readiness needs them, and reading the meta sidecar twice per request is
    how the two copies start disagreeing."""
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    cleaned = directory / "cleaned.csv"
    path = cleaned if cleaned.exists() else directory / "original.csv"

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise UnparseableCSVError() from exc

    raw_goal = meta.get("goal_context")
    goal = GoalContext.model_validate(raw_goal) if raw_goal else None
    if goal is None or not (goal.target_column or "").strip():
        raise NoTargetColumnError()

    return df, goal, directory, stored_rules(meta)


def _leakage_rows(df: pd.DataFrame, target: str, task: str) -> tuple[pd.DataFrame, str]:
    """The rows leakage may be measured on, and a sentence saying which they are.

    Whether a column predicts the answer too well to be legitimate is read from the answers,
    and the answers on the held-out rows are exactly what no decision may be informed by. So
    the measurement is made on the training rows of the split a run would make, not on every
    labelled row. The effect is small -- only near-perfect predictors are flagged, and they
    are near-perfect on any sample -- but "small" is a judgement about the consequence, and
    the rule is about the inputs.

    A file too small to split falls back to every row, and says so. Failing the plan over it
    would stop the user reading why the file cannot be trained on, which `blockers` already
    explains.
    """
    try:
        X_train, _, _, _ = modeling.split_data(df, target, task)
    except modeling.ModelingError:
        return df, (
            "Computed on every row, because the file is too small to set a test set aside."
        )
    return df.loc[X_train.index], (
        f"Leakage checked on the {len(X_train):,} training rows of the default random "
        "split; the held-out rows are not read. A different split would hold out different "
        "rows, so these suggestions are for the default."
    )


def _suggested_exclusions(
    df: pd.DataFrame,
    goal: GoalContext,
    task: str,
    leakage_rows: pd.DataFrame | None = None,
) -> list[dict]:
    """Columns the app thinks shouldn't be model inputs, each with the reason.

    Suggestions, not decisions -- the plan hands them to the user pre-ticked and they can
    untick any of them. A leaky column is sometimes legitimately available at prediction
    time, and only the person who owns the data knows that.

    `leakage_rows` is what the leakage check reads; the name and identifier checks read only
    the columns themselves and use the whole frame.
    """
    target = goal.target_column or ""
    suggestions: list[dict] = []

    for issue in ml_readiness.detect_leakage(
        leakage_rows if leakage_rows is not None else df, target, task
    ):
        if issue.column_name in {s["column"] for s in suggestions}:
            continue
        suggestions.append(
            {
                "column": issue.column_name,
                "reason": issue.description,
                "kind": "leakage",
                "severity": issue.severity,
            }
        )

    for column in df.columns:
        if column == target or column in {s["column"] for s in suggestions}:
            continue
        if _looks_like_personal_details(str(column), df[column]):
            suggestions.append(
                {
                    "column": column,
                    "reason": "This is a person's name or contact details. It says who someone "
                    "is, not anything about what they do, so whatever a model learns from it "
                    "is either chance or something the name stands in for -- such as gender "
                    "or ethnicity -- which you almost certainly don't want it deciding on.",
                    "kind": "personal",
                    "severity": "high",
                }
            )

    rows = len(df)
    for column in df.columns:
        if column == target or column in {s["column"] for s in suggestions}:
            continue
        if _looks_like_an_identifier(df[column], rows):
            suggestions.append(
                {
                    "column": column,
                    "reason": "Nearly every row has a different value, so this identifies "
                    "rows rather than describing them. A model can't learn anything from it "
                    "that would apply to a new row.",
                    "kind": "identifier",
                    "severity": "medium",
                }
            )

    return suggestions


# Column names that hold a person's name or contact details, compared after lower-casing and
# dropping everything but letters. A closed list of known words rather than a guess at what
# sounds personal, for the same reason the business-rule proposals use one: "account_name"
# and "product_name" are not people.
PERSONAL_COLUMN_NAMES = frozenset(
    {
        "name", "fullname", "firstname", "lastname", "surname", "forename", "givenname",
        "familyname", "middlename", "customername", "clientname", "employeename",
        "patientname", "contactname", "username",
        "email", "emailaddress", "mail", "phone", "phonenumber", "telephone", "mobile",
        "mobilenumber", "cellphone",
    }
)


def _looks_like_personal_details(column: str, series: pd.Series) -> bool:
    """Names and contact details, found by the column's name.

    Found on the churn sample: `last_name` was the model's second most important input. With
    about a hundred rows per surname, a churn rate spread of 0.21 to 0.35 across twelve of
    them is what chance produces -- and the importance chart, which sums the coefficients of
    every category, made that chance look like a finding. The data alone cannot show that a
    name is meaningless; the column's name can. When a name *does* carry signal it is
    usually worse, because what it carries is whatever the name stands in for.

    Suggested for exclusion, never enforced -- a `username` can be a legitimate key into
    something the owner of the data knows about.
    """
    if pd.api.types.is_numeric_dtype(series):
        return False
    normalised = "".join(ch for ch in column.lower() if ch.isalpha())
    return normalised in PERSONAL_COLUMN_NAMES


IDENTIFIER_UNIQUENESS = 0.95
MIN_ROWS_TO_JUDGE_UNIQUENESS = 20


def _looks_like_an_identifier(series: pd.Series, rows: int) -> bool:
    """Row-unique columns are usually identifiers: they can't generalize, because every future
    row carries a value the model has never seen, yet they look highly predictive in training.

    Uniqueness alone isn't the test, though, and getting that wrong is expensive in both
    directions.

    A continuous measurement is *naturally* near-unique -- 809 employees have 809 different
    salaries -- and calling salary an identifier would strip out the most informative column
    in the file. So a float is never an identifier on these grounds, and an integer has to
    look like a key as well as be unique: a dense run of values, the shape an auto-increment
    column has.

    Dates and free text are near-unique too, and they are the other half of the mistake. The
    RSNA run dropped a column of radiology reports because "nearly every row has a different
    value, so this identifies rows rather than describing them" -- the right action for the
    wrong reason, and the wrong reason is the defect. Both now have recipes that turn them
    into usable inputs (features.py), so calling them identifiers would hide the offer as
    well as misdescribe the column. `classify_features` names them for what they are.
    """
    if rows < MIN_ROWS_TO_JUDGE_UNIQUENESS:
        return False

    distinct = int(series.nunique(dropna=True))
    if distinct / rows <= IDENTIFIER_UNIQUENESS:
        return False

    if pd.api.types.is_float_dtype(series):
        return False

    if feature_recipes.looks_like_dates(series) or feature_recipes.looks_like_free_text(series):
        return False

    if pd.api.types.is_integer_dtype(series):
        values = series.dropna()
        if values.empty:
            return False
        span = int(values.max()) - int(values.min()) + 1
        # A key runs 1001, 1002, 1003...; a measurement that happens to be integral is
        # scattered across a range far wider than the number of rows.
        return span <= len(values) * 1.05

    return True


def _group_columns(df: pd.DataFrame, target: str) -> list[str]:
    """Columns that look like an entity repeating across rows, most repeated first.

    Shares its shape test with `readiness_checks.check_repeated_entities`, which is what
    detects the problem in the first place -- offering a different set of columns here than
    the panel warns about would be two answers to one question.
    """
    from app.services import readiness_checks

    found: list[tuple[str, float]] = []
    rows = len(df)
    for column in df.columns:
        if column == target:
            continue
        series = df[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        distinct = int(series.nunique())
        if distinct < readiness_checks.MIN_DISTINCT_FOR_ENTITY or distinct > rows / 2:
            continue
        repetition = rows / distinct
        if repetition < readiness_checks.MIN_REPETITION_FOR_ENTITY:
            continue
        found.append((str(column), repetition))

    found.sort(key=lambda pair: pair[1], reverse=True)
    return [column for column, _ in found]


def _date_columns(df: pd.DataFrame) -> list[str]:
    """Columns usable to order a time-based split."""
    found = []
    for column in df.columns:
        series = df[column].dropna()
        if series.empty or pd.api.types.is_numeric_dtype(series):
            continue
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (ValueError, TypeError):
            continue
        if parsed.notna().mean() >= 0.9 and parsed.dt.normalize().nunique() > 1:
            found.append(str(column))
    return found


@router.get("/api/model/{session_id}/plan", response_model=ModelPlanResponse)
async def get_model_plan(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> ModelPlanResponse:
    df, goal, directory, rules = _load_session(settings, session_id)
    target = (goal.target_column or "").strip()

    task = ml_readiness.infer_task_type(df[target], goal.task_type) if target in df.columns else None
    if task is None:
        raise TrainingFailedError(
            f"'{target}' has no usable values left, so there is nothing to predict."
        )

    readiness = ml_readiness.assess_readiness(df, goal, rules=rules)
    leakage_rows, exclusions_basis = _leakage_rows(df, target, task)
    suggestions = _suggested_exclusions(df, goal, task, leakage_rows)

    stored = model_store.load_result(directory)
    previously_excluded = (
        stored.get("features", {}).get("excluded", []) if stored else [s["column"] for s in suggestions]
    )

    # Proposed against the columns that will actually be inputs: a missing flag on a column
    # withheld for leaking the answer would leak it straight back in.
    proposed_features = feature_recipes.propose(df, target, previously_excluded)
    chosen_features = (
        [r["id"] for r in stored.get("features", {}).get("derived", [])]
        if stored
        # Offered, not ticked. They were ticked by default until the churn sample: its
        # signup_date parts duplicate a tenure_months column already in the file, and the
        # default train got worse with them (AUC 0.825 to 0.799). An exclusion is pre-ticked
        # because it guards against harm; an extra input is a change, and a change is the
        # user's to opt into -- the same reason a proposed business rule does nothing until
        # approved. The improvement loop's fourth round tries them either way and says
        # whether they helped.
        else []
    )

    previously_clipped = stored.get("features", {}).get("clipped", []) if stored else []
    classified = modeling.classify_features(
        df,
        target,
        previously_excluded,
        [r for r in proposed_features if r.id in set(chosen_features)],
        previously_clipped,
    )

    identical_rows = _identical_rows(
        df, target, classified["numeric"] + classified["categorical"] + classified["frequent"]
    )

    return ModelPlanResponse(
        session_id=session_id,
        target_column=target,
        task_type=task,
        row_count=int(len(df)),
        # Rows carrying an answer, which is what will actually be fitted on. The form prints
        # this next to the Train button, and printing len(df) there repeated the readiness
        # bug on a second surface: "4,407 rows" under a button that would train on 58.
        trainable_row_count=int(df[target].notna().sum()) if target in df.columns else 0,
        features=classified,
        suggested_clipping=_clipping_offers(df, classified["numeric"]),
        identical_rows=identical_rows,
        keep_identical_rows_together=(
            bool(stored["split"].get("keep_duplicates_together")) if stored else bool(identical_rows)
        ),
        sealed_holdout_offer=_sealed_offer(df, target, task),
        sealed_holdout=bool(stored["split"].get("sealed")) if stored else False,
        suggested_exclusions=suggestions,
        exclusions_basis=exclusions_basis,
        excluded_columns=previously_excluded,
        candidate_models=[
            {
                "name": spec.name,
                "label": spec.label,
                "description": spec.description,
                "explains_features": spec.explains_features,
            }
            for spec in modeling.models_for_task(task)
        ],
        split_options=[
            {
                "value": "random",
                "label": "Random split",
                "description": "Standard. Holds back a random fifth of the rows for testing.",
            },
            {
                "value": "time",
                "label": "Split by date",
                "description": "Trains on older rows and tests on the most recent ones; rows "
                "sharing a date stay on the same side. Use this if predictions will be made "
                "about the future -- a random split lets the model learn from the future to "
                "predict the past, which flatters it.",
            },
            {
                "value": "group",
                "label": "Keep related rows together",
                "description": "Puts every row about the same customer, patient or machine "
                "on one side of the split. Use this when one thing appears in many rows -- "
                "otherwise the model can score well by recognising it rather than learning "
                "the pattern, and the score won't survive meeting a new one.",
            },
        ],
        date_columns=_date_columns(df),
        group_columns=_group_columns(df, target),
        blockers=readiness.get("blockers", []),
        warnings=readiness.get("warnings", []),
        trained=stored is not None,
        class_labels=(
            sorted(df[target].dropna().astype(str).unique().tolist())
            if task == "classification"
            else []
        ),
        suggested_features=[r.as_dict() for r in proposed_features],
        feature_recipes=chosen_features,
    )


def _sealed_offer(df: pd.DataFrame, target: str, task: str) -> dict:
    """Whether a sealed final holdout can be taken from this file, and the part sizes or why not.

    Checked on the default random split by a dry run. It is only worth offering when both
    held-out parts are big enough for a score on them to mean anything -- the app's own bar --
    because a sealed holdout that cannot support a verdict spends data and learns nothing. The
    train request re-checks it for the split actually chosen.
    """
    try:
        X_train, X_dev, X_final, y_train, y_dev, y_final = modeling.split_with_sealed_holdout(
            df, target, task
        )
    except modeling.ModelingError as exc:
        return {"available": False, "reason": str(exc)}
    except Exception:  # noqa: BLE001 -- an offer is a nicety; it must never fail the plan
        logger.warning("Could not work out whether a sealed holdout is possible", exc_info=True)
        return {"available": False, "reason": "It could not be worked out for this file."}
    reason = modeling.sealed_unavailable_reason(y_dev, y_final, task)
    return {
        "available": reason is None,
        "reason": reason,
        "train_rows": int(len(X_train)),
        "development_rows": int(len(X_dev)),
        "final_rows": int(len(X_final)),
    }


def _identical_rows(df: pd.DataFrame, target: str, columns: list[str]) -> dict | None:
    """Rows that are exact copies of another row -- the model's inputs and the answer -- or None.

    Counted on the inputs the plan starts with; the train request recomputes it on whatever the
    user finally chooses. A copy that lands in the test set is a question the model has already
    been shown the answer to, and flatters the score.
    """
    labelled = df[df[target].notna()]
    if labelled.empty or not columns:
        return None
    ids = modeling.identical_row_ids(labelled, columns, target)
    copies = int(len(ids) - ids.nunique())
    if copies == 0:
        return None
    return {
        "copies": copies,
        "share": round(copies / len(ids), 4),
        "rows": int(len(ids)),
    }


def _clipping_offers(df: pd.DataFrame, numeric_inputs: list[str]) -> list[dict]:
    """Numeric inputs that have values far outside the rest, each with what was found.

    Read from the column alone, never the answers. Described by the values in the column --
    how many, the largest, where the ordinary ones sit -- and never by the fence that found
    them, for the reason `profiling._describe_outliers` gives.
    """
    offers = []
    for column in numeric_inputs:
        found = clipping.outliers_in(df[column])
        if found is None:
            continue
        parts = []
        if found["high"]:
            parts.append(
                f"{found['high']} unusually high ({'largest ' + format(found['highest'], 'g')})"
            )
        if found["low"]:
            parts.append(
                f"{found['low']} unusually low ({'smallest ' + format(found['lowest'], 'g')})"
            )
        offers.append(
            {
                "column": column,
                "reason": f"{' and '.join(parts)}, while the rest run from "
                f"{format(found['typical_min'], 'g')} to {format(found['typical_max'], 'g')}. "
                "Clipping pulls those back to the edge of the usual range, which the model "
                "learns from the training rows only. No row is lost.",
                "evidence": found,
            }
        )
    return offers


def _refuse_if_the_data_cannot_support_a_model(df, goal, rules=None) -> str:
    """The checks every route that produces a model has to clear first.

    Shared rather than repeated because the first version of this guard lived inline in
    `train_model` only, which left `/improve` -- a route that ends by calling
    `train_candidates` itself -- as an unguarded way to reach the same result. A guard that
    one of two doors enforces is not a guard. Any future route that fits a model calls this.

    The readiness verdict has to bind or it is decoration. It was advisory until the RSNA knee
    dataset went through: readiness returned "blocked" because 4,349 of 4,407 rows had no
    label, no route asked, and the app trained on the 58 that remained and reported the result
    as useful. A tool whose whole claim is that it will say no cannot say no and then proceed.
    """
    target = (goal.target_column or "").strip()

    if target not in df.columns:
        raise TrainingFailedError(
            f"'{target}' is no longer in the data -- it was renamed or removed during cleaning."
        )

    task = ml_readiness.infer_task_type(df[target], goal.task_type)
    if task is None:
        raise TrainingFailedError(f"'{target}' has no usable values to learn from.")

    readiness = ml_readiness.assess_readiness(df, goal, rules=rules)
    if readiness.get("verdict") == "blocked":
        blockers = readiness.get("blockers") or ["This data cannot support a model."]
        raise TrainingFailedError(
            "This data cannot support a model that would mean anything, so nothing was "
            "trained. " + " ".join(blockers)
        )
    return task


def _save_and_record(directory, result: dict, df, *, route: str, action_log=None) -> dict:
    """Saves a finished run and records the look at the held-out rows that it was.

    Every route that fits a model and returns its held-out score ends here, so there is one
    place that decides what counts: a *successful* run, once. A refused or failed request never
    reaches this function, and a stored result being read again does not either (decision
    0001, section 1). The look is recorded after the result is saved -- a record of a score
    nobody can see would be worse than none -- and recording never raises, so a ledger problem
    cannot lose a finished run.

    The run id is made here so the saved result, the run history and the ledger all name the
    same run.
    """
    result["run_id"] = uuid.uuid4().hex[:12]
    # Worked out from the ledger *before* this look is written, so the first run of a split
    # reads independent, and carried on the result so the page, the card and the history all
    # say the same thing. The headline is qualified here rather than where the verdict is made:
    # the verdict knows about the numbers, not about how often the rows have been looked at.
    evaluation = evaluation_ledger.evaluation_status(directory, result.get("split"))
    # Fixes approved earlier that worked a value out over every row -- the held-out ones too --
    # on columns this model uses. The pipeline's own training-only imputer has nothing left to
    # do once they are applied, so the run carries a small leak and should say so.
    inputs = [
        c
        for key in ("numeric", "categorical", "frequent")
        for c in (result.get("features") or {}).get(key, [])
    ]
    evaluation["whole_file_statistics"] = action_log_service.whole_file_statistics(
        action_log, inputs
    )
    result["evaluation"] = evaluation
    result["assessment"] = evaluation_ledger.qualify(result.get("assessment"), evaluation)
    version = current_dataset_version_in(directory, ensure=True)
    if (result.get("split") or {}).get("sealed"):
        result["split"]["sealed_after_looks"] = _record_the_seal(directory, result["split"], version)
    payload = model_store.save_result(
        directory, result, fingerprint=model_store.fingerprint_frame(df)
    )
    best = next(
        (r for r in payload.get("results", []) if r.get("model") == payload.get("best_model")),
        {},
    )
    evaluation_ledger.record(
        directory,
        event=evaluation_ledger.LOOK,
        split=payload.get("split"),
        route=route,
        run_id=payload["run_id"],
        version=version,
        metrics=best.get("metrics"),
        extra={"primary_metric": payload.get("primary_metric"), "model": payload.get("best_model")},
    )
    return payload


def _record_the_seal(directory, split: dict, version: dict | None) -> int:
    """Writes down which rows were sealed and how many looks had already been taken.

    Returns that number, which the result shows. A holdout sealed *after* looks have been taken
    is still unopened, but the rows were training or development rows until now and informed the
    decisions made on them, so the page says "sealed after N looks" rather than presenting it as
    pristine (decision 0001, section 4). The first record for a (data, split rule) wins: later
    runs on the same data cannot make the seal look older than it is.
    """
    import json

    path = directory / model_store.SEALED_FILENAME
    key = evaluation_ledger.split_key(split)
    fingerprint = (version or {}).get("fingerprint")
    try:
        records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        records = []
    for record in records:
        if record.get("split_key") == key and record.get("dataset_fingerprint") == fingerprint:
            return int(record.get("sealed_after_looks", 0))

    looks_so_far = sum(
        1 for e in evaluation_ledger.looks(directory) if e.get("event") == evaluation_ledger.LOOK
    )
    records.append(
        {
            "split_key": key,
            "dataset_version": (version or {}).get("version"),
            "dataset_fingerprint": fingerprint,
            "rows": split.get("final_rows"),
            "rows_hash": split.get("sealed_rows_hash"),
            "sealed_after_looks": looks_so_far,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    try:
        path.write_text(json.dumps(records), encoding="utf-8")
    except OSError:
        logger.warning("Could not record the sealed holdout", exc_info=True)
    return looks_so_far


@router.post("/api/model/{session_id}/train")
async def train_model(
    session_id: str,
    request: TrainRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
) -> dict:
    df, goal, directory, rules = _load_session(settings, session_id)
    log = await action_log_service.read(graph, session_id)
    target = (goal.target_column or "").strip()
    task = _refuse_if_the_data_cannot_support_a_model(df, goal, rules)

    model_names = request.model_names or [spec.name for spec in modeling.models_for_task(task)]
    point = _operating_point(request)

    def run() -> dict:
        return modeling.train_candidates(
            df,
            target_column=target,
            task=task,
            model_names=model_names,
            excluded_columns=request.excluded_columns,
            split_strategy=request.split_strategy,
            time_column=request.time_column,
            group_column=request.group_column,
            include_deep_diagnostics=request.include_deep_diagnostics,
            point=point,
            feature_recipes_wanted=request.feature_recipes,
            clip_outliers=request.clip_outliers,
            keep_identical_rows_together=request.keep_identical_rows_together,
            sealed_holdout=request.sealed_holdout,
        )

    try:
        # Off the event loop: fitting is CPU-bound and would otherwise stall every other
        # request in this process, including in-flight SSE streams.
        result = await asyncio.to_thread(run)
    except modeling.ModelingError as exc:
        raise TrainingFailedError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- surface the reason rather than a bare 500
        logger.exception("Training failed for session %s", session_id)
        raise TrainingFailedError(f"Training stopped unexpectedly: {exc}") from exc

    # Recorded so the plan can show what was excluded last time and the model card can say
    # which columns were deliberately withheld.
    result["features"]["excluded"] = list(request.excluded_columns)
    return _save_and_record(directory, result, df, route="train", action_log=log)


def _improve_and_save(
    df,
    *,
    target: str,
    task: str,
    request: ImproveRequest,
    point,
    directory,
    route: str = "improve",
    action_log=None,
    on_progress=None,
    on_stage=None,
) -> dict:
    """Search, rebuild the winner the normal way, and save it. Runs in a worker thread.

    Two passes on purpose. The loop searches on cross-validated training folds and never
    looks at the test set, which is what makes its answer trustworthy but also means it has
    only a score, not a full evaluation. So whatever it settles on is handed back to
    `train_candidates`, which produces exactly the shape every other result has -- the charts,
    the model card, the joblib and the exported script all work on it unchanged, and the
    headline number is measured the same way it always was.

    Saving happens here, in the worker, rather than in whichever route called it. A streamed
    search whose browser went away mid-run still finishes and is still kept, so a refresh
    finds it -- a minute of fitting thrown away because a tab was closed would be the worst
    outcome available.

    The trail of rounds rides along on the result, including the rounds that were rejected.
    "We tried tuning it and tuning did not help" is a finding about the data.
    """
    found = improvement.improve(
        df,
        target_column=target,
        task=task,
        model_names=request.model_names or None,
        excluded_columns=request.excluded_columns,
        split_strategy=request.split_strategy,
        time_column=request.time_column,
        group_column=request.group_column,
        goal=request.goal,
        budget=request.budget,
        feature_recipes_wanted=request.feature_recipes,
        on_progress=on_progress,
        on_stage=on_stage,
        clip_outliers=request.clip_outliers,
        keep_identical_rows_together=request.keep_identical_rows_together,
        sealed_holdout=request.sealed_holdout,
    )
    best = found["best"]
    # Plain constructor names, not the `model__`-prefixed pipeline paths the search uses.
    # train_candidates hands these to ModelSpec.build, which passes them straight to the
    # estimator's constructor -- and to constructor_source, which writes them into the
    # exported script. The prefix used to be re-added here, so the first time a search
    # actually kept a setting, both broke: the rebuild raised "unexpected keyword argument
    # 'model__class_weight'" and "Improve it" failed outright. Found on the churn sample;
    # the tests passed because on their data the search never kept anything.
    params = dict(best["params"])
    if on_stage:
        on_stage(
            "rebuilding",
            f"Refitting {best['label'].lower()} on the training rows and scoring it once on "
            f"the held-out test set.",
        )
    rebuilt = modeling.train_candidates(
        df,
        target_column=target,
        task=task,
        model_names=[best["model"]],
        excluded_columns=request.excluded_columns,
        split_strategy=request.split_strategy,
        time_column=request.time_column,
        group_column=request.group_column,
        include_deep_diagnostics=request.include_deep_diagnostics,
        model_params={best["model"]: params},
        point=point,
        # Whatever the fourth round settled on, which may not be what was sent in -- the
        # loop is allowed to add the derived inputs or drop them, and the rebuild has to
        # be the configuration that won rather than the one that started.
        feature_recipes_wanted=best.get("feature_recipes", []),
        clip_outliers=request.clip_outliers,
        keep_identical_rows_together=request.keep_identical_rows_together,
        sealed_holdout=request.sealed_holdout,
    )
    rebuilt["improvement"] = {
        "rounds": found["rounds"],
        "metric": found["metric"],
        "goal": found["goal"],
        "stopped_because": found["stopped_because"],
        "cv_score": best["cv_score"],
        "cv_std": best["cv_std"],
        "settings": best["params"],
    }
    rebuilt["features"]["excluded"] = list(request.excluded_columns)
    # One look, however many rounds the search ran: the rounds are scored on training folds and
    # never open the held-out rows. Only this final rebuild does, once.
    return _save_and_record(directory, rebuilt, df, route=route, action_log=action_log)


@router.post("/api/model/{session_id}/improve")
async def improve_model(
    session_id: str,
    request: ImproveRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
) -> dict:
    """Search for a better model, then rebuild the result the normal way.

    The blocking form: one response when it is done. The page uses `/improve/stream` instead;
    this stays for anything calling the API directly, and for the tests.
    """
    df, goal_context, directory, rules = _load_session(settings, session_id)
    target = (goal_context.target_column or "").strip()
    task = _refuse_if_the_data_cannot_support_a_model(df, goal_context, rules)
    point = _operating_point(request)
    log = await action_log_service.read(graph, session_id)

    try:
        return await asyncio.to_thread(
            _improve_and_save,
            df,
            target=target,
            task=task,
            request=request,
            point=point,
            directory=directory,
            action_log=log,
        )
    except modeling.ModelingError as exc:
        raise TrainingFailedError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- surface the reason rather than a bare 500
        logger.exception("Improvement failed for session %s", session_id)
        raise TrainingFailedError(f"Improving stopped unexpectedly: {exc}") from exc


# Seconds of silence before the stream sends a comment line. A settings search can run for
# most of a minute without finishing a round, and proxies close connections that look idle
# for far less than that.
KEEPALIVE_SECONDS = 15.0


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"


@router.post("/api/model/{session_id}/improve/stream")
async def improve_model_stream(
    session_id: str,
    request: ImproveRequest,
    settings: Settings = Depends(get_settings),
    graph=Depends(get_graph_if_any),
):
    """The same search as `/improve`, reported as it happens.

    A search takes tens of seconds to minutes, and the page used to show a paragraph of
    explanation next to a spinner for all of it. The loop already reported each round as it
    finished; this sends those reports on, plus a line as each round *starts*, because the
    settings search alone can outlast any patience with a frozen screen.

    Events, one JSON object per `data:` line:

    - `{"type": "stage", "name", "description"}` -- a round, or the final rebuild, starting.
    - `{"type": "round", "round": {...}}` -- a round finished, kept or not. Same shape as the
      entries in the result's `improvement.rounds`.
    - `{"type": "result", "result": {...}}` -- done; the same payload `/improve` returns.
    - `{"type": "error", "detail"}` -- it failed; the detail is written for the user.

    Everything that can be refused is refused *before* the stream opens -- a blocked verdict,
    an unreadable trade-off -- so it arrives as an ordinary JSON error with a status code
    rather than as an event inside a 200.
    """
    df, goal_context, directory, rules = _load_session(settings, session_id)
    target = (goal_context.target_column or "").strip()
    task = _refuse_if_the_data_cannot_support_a_model(df, goal_context, rules)
    point = _operating_point(request)
    log = await action_log_service.read(graph, session_id)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def push(event: dict) -> None:
        # Called from the worker thread; the queue belongs to the event loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def on_progress(entry) -> None:
        push({"type": "round", "round": entry.as_dict()})

    def on_stage(name: str, description: str) -> None:
        push({"type": "stage", "name": name, "description": description})

    async def work() -> None:
        try:
            result = await asyncio.to_thread(
                _improve_and_save,
                df,
                target=target,
                task=task,
                request=request,
                point=point,
                directory=directory,
                route="improve_stream",
                action_log=log,
                on_progress=on_progress,
                on_stage=on_stage,
            )
            await queue.put({"type": "result", "result": result})
        except modeling.ModelingError as exc:
            await queue.put({"type": "error", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001 -- the reason goes to the page, not a 500
            logger.exception("Streamed improvement failed for session %s", session_id)
            await queue.put(
                {"type": "error", "detail": f"Improving stopped unexpectedly: {exc}"}
            )

    async def events() -> AsyncIterator[str]:
        # Held in a variable so the task is not garbage-collected mid-run. It is deliberately
        # not cancelled when the client goes away: the worker thread cannot be interrupted
        # anyway, and letting it finish is what gets the result saved.
        task_handle = asyncio.create_task(work())
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
            except asyncio.TimeoutError:
                yield ": still searching\n\n"
                continue
            yield _frame(event)
            if event["type"] in ("result", "error"):
                break
        await task_handle

    return sse_response(events())


@router.get("/api/model/{session_id}/result")
async def get_model_result(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    stored = model_store.load_result(directory)
    if stored is None:
        raise NoTrainedModelError()
    # Looked up at read time, not stored with the run: the model is checked against new files
    # after it is built, and a stored copy would be stale the moment one arrived.
    if isinstance(stored.get("evaluation"), dict):
        stored["evaluation"]["new_data"] = evaluation_ledger.new_data_summary(
            directory, stored.get("run_id")
        )
        stored["evaluation"]["final"] = evaluation_ledger.final_summary(
            directory, stored.get("run_id")
        )
    return stored


def _open_the_sealed_holdout(df, stored: dict, pipeline) -> dict:
    """Scores the fitted model, once, on the sealed rows. Runs in a worker thread.

    The partitions are rebuilt from the stored settings by the same rule that made them, and
    the sealed rows are checked against the hash taken when they were set aside -- if the data
    no longer produces those rows, the evaluation would not be the one promised, and it is
    refused rather than quietly done on different rows.
    """
    split = stored["split"]
    features = stored["features"]
    task = stored["task"]
    target = stored["target_column"]
    used = features["numeric"] + features["categorical"] + features.get("frequent", [])
    recipes = [feature_recipes.Recipe(**r) for r in features.get("derived", []) or []]

    X_train, X_dev, X_final, y_train, y_dev, y_final = modeling.split_with_sealed_holdout(
        feature_recipes.apply(df, recipes),
        target,
        task,
        strategy=split["strategy"],
        time_column=split.get("time_column"),
        group_column=split.get("group_column"),
        duplicate_columns=used if split.get("keep_duplicates_together") else None,
        copy_columns=split.get("seal_key_columns"),
    )
    if modeling.sealed_rows_hash(X_final.index) != split.get("sealed_rows_hash") or len(
        X_train
    ) != split.get("train_rows"):
        raise SealedRowsChangedError()

    best = next((r for r in stored["results"] if r["model"] == stored["best_model"]), {})
    positive = (stored.get("operating_point") or {}).get("positive_label") or (
        best.get("metrics") or {}
    ).get("positive_label")
    metrics = modeling.evaluate(pipeline, X_final[used], y_final, task, positive)
    baseline = modeling.baseline_score(y_train, y_final, task)
    assessment = modeling.assess_result(
        metrics,
        baseline,
        task,
        test_rows=int(len(y_final)),
        class_counts=(
            pd.Series(y_final).astype(str).value_counts().tolist()
            if task == "classification"
            else None
        ),
        model_count=len(stored["results"]),
    )
    return {
        "rows": int(len(y_final)),
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "baseline": baseline,
        "assessment": assessment,
        "primary_metric": stored.get("primary_metric"),
        "rows_seen_in_training": modeling.test_rows_seen_in_training(
            X_train[used], X_final[used], y_train, y_final
        ),
    }


@router.post("/api/model/{session_id}/finalise")
async def finalise_model(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Opens the sealed final holdout, once, and scores the current model on it.

    The one route that evaluates on the sealed rows; every other route works on the development
    partition, so nothing done while iterating can have been informed by them. It can be called
    once per (data version, split rule): a second call returns the stored result with a note
    rather than evaluating again, because a second model scored on the same rows -- chosen,
    perhaps, because of the first result -- is not an independent evaluation.
    """
    df, _goal, directory, _rules = _load_session(settings, session_id)
    stored = model_store.load_result(directory)
    if stored is None:
        raise NoTrainedModelError()
    split = stored.get("split") or {}
    if not split.get("sealed"):
        raise NoSealedHoldoutError(split.get("sealed_unavailable_reason"))

    version = current_dataset_version_in(directory, ensure=True)
    earlier = evaluation_ledger.find_final(directory, version, split)
    if earlier is not None:
        extra = earlier.get("extra") or {}
        same_model = earlier.get("run_id") == stored.get("run_id")
        return {
            "already_opened": True,
            "for_this_model": same_model,
            "opened_at": earlier.get("at"),
            "run_id": earlier.get("run_id"),
            "rows": extra.get("rows"),
            "metrics": earlier.get("metrics"),
            "baseline": extra.get("baseline"),
            "assessment": extra.get("assessment"),
            "primary_metric": extra.get("primary_metric"),
            "rows_seen_in_training": extra.get("rows_seen_in_training", 0),
            "note": (
                "The sealed rows were already opened for this model; this is the stored "
                "result, not a new evaluation."
                if same_model
                else "The sealed rows were already opened, for an earlier model. Scoring this "
                "one on them as well would not be independent, since the first result may "
                "have informed it. This is the stored result for that model."
            ),
        }

    pipeline_file = model_store.pipeline_path(directory)
    if pipeline_file is None:
        raise NoTrainedModelError()

    import joblib

    try:
        pipeline = await asyncio.to_thread(joblib.load, pipeline_file)
        opened = await asyncio.to_thread(_open_the_sealed_holdout, df, stored, pipeline)
    except (SealedRowsChangedError, modeling.ModelingError):
        raise
    except Exception as exc:  # noqa: BLE001 -- surface the reason rather than a bare 500
        logger.exception("Opening the sealed holdout failed for session %s", session_id)
        raise TrainingFailedError(f"The sealed holdout could not be opened: {exc}") from exc

    entry = evaluation_ledger.record(
        directory,
        event=evaluation_ledger.FINAL,
        split=split,
        route="finalise",
        run_id=stored.get("run_id"),
        version=version,
        metrics=opened["metrics"],
        extra={k: opened[k] for k in ("rows", "baseline", "assessment", "primary_metric", "rows_seen_in_training")},
    )
    return {
        "already_opened": False,
        "for_this_model": True,
        "opened_at": (entry or {}).get("at"),
        "run_id": stored.get("run_id"),
        "earlier_openings_on_other_versions": evaluation_ledger.final_openings(directory, split) - 1,
        **opened,
    }


@router.get("/api/model/{session_id}/history")
async def get_model_history(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Every run in this session, newest first.

    `save_result` used to overwrite a single file, so retraining silently discarded the
    previous run -- which made the one question people actually ask unanswerable: did the
    thing I just changed help? The verdict on a single run says whether a model is worth
    anything; only the sequence says whether a decision was.

    Each entry carries the fingerprint of the data it was trained on. `current_fingerprint`
    lets the client mark the runs that predate a cleaning change, because comparing across
    one is often the most interesting comparison available -- it just cannot be silent.
    """
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    cleaned = directory / "cleaned.csv"
    path = cleaned if cleaned.exists() else directory / "original.csv"

    current = ""
    try:
        current = model_store.fingerprint_frame(pd.read_csv(path))
    except Exception:  # noqa: BLE001 -- history is still worth serving without it
        logger.warning("Could not fingerprint the current data for session %s", session_id)

    return {
        "session_id": session_id,
        "runs": model_store.load_history(directory),
        "current_fingerprint": current,
    }


@router.get("/api/model/{session_id}/card")
async def get_model_card(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """The model card as markdown. Generated on first request rather than at training time:
    it costs an LLM call, and a user who trains three times while adjusting columns should
    pay for it once, when they ask to see the write-up."""
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    directory = session_dir(settings.upload_dir, session_id)
    result = model_store.load_result(directory)
    if result is None:
        raise NoTrainedModelError()

    existing = directory / model_card.CARD_FILENAME
    if existing.exists():
        return {"session_id": session_id, "markdown": existing.read_text(encoding="utf-8")}

    raw_goal = meta.get("goal_context")
    markdown = await model_card.generate_model_card(
        session_id=session_id,
        file_name=meta.get("file_name", "your data"),
        result=result,
        goal_context=GoalContext.model_validate(raw_goal) if raw_goal else None,
        directory=directory,
    )
    return {"session_id": session_id, "markdown": markdown}


@router.get("/api/model/{session_id}/script", response_class=PlainTextResponse)
async def get_training_script(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> PlainTextResponse:
    """A runnable script that rebuilds the model from scratch. The actual deliverable -- a
    score in a web page has to be trusted, a script can be read and re-run."""
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    result = model_store.load_result(session_dir(settings.upload_dir, session_id))
    if result is None:
        raise NoTrainedModelError()

    try:
        script = model_export.build_training_script(result, file_name="cleaned.csv")
    except ValueError as exc:
        raise TrainingFailedError(str(exc)) from exc

    return PlainTextResponse(
        script,
        headers={"Content-Disposition": 'attachment; filename="train_model.py"'},
    )


@router.get("/api/model/{session_id}/download")
async def download_fitted_model(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """The fitted pipeline itself, as a joblib file."""
    meta = read_session_meta(settings.upload_dir, session_id)
    if meta is None:
        raise SessionNotFoundError(session_id)

    path = model_store.pipeline_path(session_dir(settings.upload_dir, session_id))
    if path is None:
        raise NoTrainedModelError()
    return FileResponse(path, media_type="application/octet-stream", filename="model.joblib")


@router.get("/api/model/{session_id}/correlations")
async def get_feature_correlations(
    session_id: str,
    settings: Settings = Depends(get_settings),
) -> dict:
    """The feature correlation heatmap, available before anything is trained -- it describes
    the data, not a model, and it's worth seeing while deciding what to exclude."""
    df, goal, _directory, _rules = _load_session(settings, session_id)
    target = (goal.target_column or "").strip()
    numeric = [
        c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    return diagnostics.feature_correlations(df, numeric)
