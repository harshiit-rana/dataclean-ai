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
from collections.abc import AsyncIterator

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, PlainTextResponse

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import ImproveRequest, ModelPlanResponse, TrainRequest
from app.models.graph_state import GoalContext
from app.services import (
    diagnostics,
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
from app.api.sse import sse_response
from app.services.session_store import read_session_meta, session_dir

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

    return ModelPlanResponse(
        session_id=session_id,
        target_column=target,
        task_type=task,
        row_count=int(len(df)),
        # Rows carrying an answer, which is what will actually be fitted on. The form prints
        # this next to the Train button, and printing len(df) there repeated the readiness
        # bug on a second surface: "4,407 rows" under a button that would train on 58.
        trainable_row_count=int(df[target].notna().sum()) if target in df.columns else 0,
        features=modeling.classify_features(
            df,
            target,
            previously_excluded,
            [r for r in proposed_features if r.id in set(chosen_features)],
        ),
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
                "description": "Trains on older rows and tests on the most recent ones. Use "
                "this if predictions will be made about the future -- a random split lets "
                "the model learn from the future to predict the past, which flatters it.",
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


@router.post("/api/model/{session_id}/train")
async def train_model(
    session_id: str,
    request: TrainRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    df, goal, directory, rules = _load_session(settings, session_id)
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
    return model_store.save_result(
        directory, result, fingerprint=model_store.fingerprint_frame(df)
    )


def _improve_and_save(
    df,
    *,
    target: str,
    task: str,
    request: ImproveRequest,
    point,
    directory,
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
    return model_store.save_result(
        directory, rebuilt, fingerprint=model_store.fingerprint_frame(df)
    )


@router.post("/api/model/{session_id}/improve")
async def improve_model(
    session_id: str,
    request: ImproveRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Search for a better model, then rebuild the result the normal way.

    The blocking form: one response when it is done. The page uses `/improve/stream` instead;
    this stays for anything calling the API directly, and for the tests.
    """
    df, goal_context, directory, rules = _load_session(settings, session_id)
    target = (goal_context.target_column or "").strip()
    task = _refuse_if_the_data_cannot_support_a_model(df, goal_context, rules)
    point = _operating_point(request)

    try:
        return await asyncio.to_thread(
            _improve_and_save,
            df,
            target=target,
            task=task,
            request=request,
            point=point,
            directory=directory,
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

    stored = model_store.load_result(session_dir(settings.upload_dir, session_id))
    if stored is None:
        raise NoTrainedModelError()
    return stored


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
