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
import logging

import pandas as pd
from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, PlainTextResponse

from app.config import Settings, get_settings
from app.core.errors import AppError, SessionNotFoundError, UnparseableCSVError
from app.models.api_schemas import ImproveRequest, ModelPlanResponse, TrainRequest
from app.models.graph_state import GoalContext
from app.services import (
    diagnostics,
    improvement,
    ml_readiness,
    model_card,
    model_export,
    model_store,
    modeling,
)
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


def _load_session(settings: Settings, session_id: str) -> tuple[pd.DataFrame, GoalContext, object]:
    """Reads the cleaned file if cleaning has run, the upload otherwise, plus the goal."""
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

    return df, goal, directory


def _suggested_exclusions(df: pd.DataFrame, goal: GoalContext, task: str) -> list[dict]:
    """Columns the app thinks shouldn't be model inputs, each with the reason.

    Suggestions, not decisions -- the plan hands them to the user pre-ticked and they can
    untick any of them. A leaky column is sometimes legitimately available at prediction
    time, and only the person who owns the data knows that.
    """
    target = goal.target_column or ""
    suggestions: list[dict] = []

    for issue in ml_readiness.detect_leakage(df, target, task):
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


IDENTIFIER_UNIQUENESS = 0.95
MIN_ROWS_TO_JUDGE_UNIQUENESS = 20


def _looks_like_an_identifier(series: pd.Series, rows: int) -> bool:
    """Row-unique columns are usually identifiers: they can't generalize, because every future
    row carries a value the model has never seen, yet they look highly predictive in training.

    Uniqueness alone isn't the test, though. A continuous measurement is *naturally* near-
    unique -- 809 employees have 809 different salaries -- and calling salary an identifier
    would strip out the most informative column in the file. So a float is never an
    identifier on these grounds, and an integer has to look like a key as well as be unique:
    a dense run of values, the shape an auto-increment column has. That leaves the case this
    is really for -- text ids, order numbers, emails -- caught as before.
    """
    if rows < MIN_ROWS_TO_JUDGE_UNIQUENESS:
        return False

    distinct = int(series.nunique(dropna=True))
    if distinct / rows <= IDENTIFIER_UNIQUENESS:
        return False

    if pd.api.types.is_float_dtype(series):
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
    df, goal, directory = _load_session(settings, session_id)
    target = (goal.target_column or "").strip()

    task = ml_readiness.infer_task_type(df[target], goal.task_type) if target in df.columns else None
    if task is None:
        raise TrainingFailedError(
            f"'{target}' has no usable values left, so there is nothing to predict."
        )

    readiness = ml_readiness.assess_readiness(df, goal)
    suggestions = _suggested_exclusions(df, goal, task)

    stored = model_store.load_result(directory)
    previously_excluded = (
        stored.get("features", {}).get("excluded", []) if stored else [s["column"] for s in suggestions]
    )

    return ModelPlanResponse(
        session_id=session_id,
        target_column=target,
        task_type=task,
        row_count=int(len(df)),
        features=modeling.classify_features(df, target, previously_excluded),
        suggested_exclusions=suggestions,
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
        ],
        date_columns=_date_columns(df),
        blockers=readiness.get("blockers", []),
        warnings=readiness.get("warnings", []),
        trained=stored is not None,
    )


@router.post("/api/model/{session_id}/train")
async def train_model(
    session_id: str,
    request: TrainRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    df, goal, directory = _load_session(settings, session_id)
    target = (goal.target_column or "").strip()

    if target not in df.columns:
        raise TrainingFailedError(
            f"'{target}' is no longer in the data -- it was renamed or removed during cleaning."
        )

    task = ml_readiness.infer_task_type(df[target], goal.task_type)
    if task is None:
        raise TrainingFailedError(f"'{target}' has no usable values to learn from.")

    model_names = request.model_names or [spec.name for spec in modeling.models_for_task(task)]

    def run() -> dict:
        return modeling.train_candidates(
            df,
            target_column=target,
            task=task,
            model_names=model_names,
            excluded_columns=request.excluded_columns,
            split_strategy=request.split_strategy,
            time_column=request.time_column,
            include_deep_diagnostics=request.include_deep_diagnostics,
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
    return model_store.save_result(directory, result)


@router.post("/api/model/{session_id}/improve")
async def improve_model(
    session_id: str,
    request: ImproveRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    """Search for a better model, then rebuild the result the normal way.

    Two passes on purpose. The loop searches on cross-validated training folds and never
    looks at the test set, which is what makes its answer trustworthy but also means it has
    only a score, not a full evaluation. So whatever it settles on is handed back to
    `train_candidates`, which produces exactly the shape every other result has -- the charts,
    the model card, the joblib and the exported script all work on it unchanged, and the
    headline number is measured the same way it always was.

    The trail of rounds rides along on the result, including the rounds that were rejected.
    "We tried tuning it and tuning did not help" is a finding about the data.
    """
    df, goal_context, directory = _load_session(settings, session_id)
    target = (goal_context.target_column or "").strip()

    if target not in df.columns:
        raise TrainingFailedError(
            f"'{target}' is no longer in the data -- it was renamed or removed during cleaning."
        )

    task = ml_readiness.infer_task_type(df[target], goal_context.task_type)
    if task is None:
        raise TrainingFailedError(f"'{target}' has no usable values to learn from.")

    def run() -> dict:
        found = improvement.improve(
            df,
            target_column=target,
            task=task,
            model_names=request.model_names or None,
            excluded_columns=request.excluded_columns,
            split_strategy=request.split_strategy,
            time_column=request.time_column,
            goal=request.goal,
            budget=request.budget,
        )
        best = found["best"]
        params = {f"model__{k}": v for k, v in best["params"].items()}
        rebuilt = modeling.train_candidates(
            df,
            target_column=target,
            task=task,
            model_names=[best["model"]],
            excluded_columns=request.excluded_columns,
            split_strategy=request.split_strategy,
            time_column=request.time_column,
            include_deep_diagnostics=request.include_deep_diagnostics,
            model_params={best["model"]: params},
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
        return rebuilt

    try:
        result = await asyncio.to_thread(run)
    except modeling.ModelingError as exc:
        raise TrainingFailedError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 -- surface the reason rather than a bare 500
        logger.exception("Improvement failed for session %s", session_id)
        raise TrainingFailedError(f"Improving stopped unexpectedly: {exc}") from exc

    result["features"]["excluded"] = list(request.excluded_columns)
    return model_store.save_result(directory, result)


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
    df, goal, _directory = _load_session(settings, session_id)
    target = (goal.target_column or "").strip()
    numeric = [
        c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    return diagnostics.feature_correlations(df, numeric)
