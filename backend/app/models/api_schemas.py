"""Request/response models for the API surface. PRD §10.1, error shape §10.3."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.models.graph_state import CleaningAggression, TaskType, UseCase


class ErrorResponse(BaseModel):
    """PRD §10.3. All error responses follow this shape."""

    error: str
    detail: str


class UploadResponse(BaseModel):
    session_id: str
    # Proof that this browser is the one that uploaded the file. Sent back on every request
    # scoped to the session -- see services/session_access.py.
    session_token: str = ""
    file_name: str
    row_count: int
    column_count: int
    # Returned so the goal form can offer a real target-column picker instead of asking the
    # user to type a column name and hope they spell it the same way.
    columns: list[str] = []


class GoalContextRequest(BaseModel):
    use_case: UseCase
    expected_output: str
    critical_columns: list[str] = []
    cleaning_aggression: CleaningAggression = "moderate"
    target_column: Optional[str] = None
    task_type: TaskType = "auto"


class GoalContextResponse(BaseModel):
    session_id: str
    accepted: bool


class PreviewColumn(BaseModel):
    name: str
    dtype: str
    null_count: int
    unique_count: int
    is_critical: bool
    is_target: bool = False  # the column the user wants to predict, if any


class PreviewResponse(BaseModel):
    """Actual CSV contents for the Review page's data table (see routes_preview.py)."""

    session_id: str
    columns: list[PreviewColumn]
    rows: list[dict]
    total_rows: int
    total_columns: int
    preview_row_count: int


# --- Stub-endpoint request/response shapes -------------------------------------------
# These endpoints are not implemented yet (see routes_resume/chat/undo/export). The
# schemas below exist now so the route signatures and the frontend API client can be
# written against a stable contract ahead of the real implementation.


class ResumeRequest(BaseModel):
    """PRD §13.2 resume payload."""

    approved_action_ids: list[str]
    skipped_action_ids: list[str] = []
    skip_reasons: dict[str, str] = {}


class ChatRequest(BaseModel):
    """PRD FR-37: plain-English cleaning instruction."""

    message: str


class UndoRequest(BaseModel):
    """PRD US-10. If action_id is omitted, undo the most recent action."""

    action_id: Optional[str] = None


class ReadinessResponse(BaseModel):
    """GET /api/readiness/{session_id}. Whether the data as it currently stands can support
    the model the user said they wanted -- computed on demand from the file on disk, so it
    stays correct after an undo or a conversational edit."""

    session_id: str
    target_column: Optional[str]
    task_type: Optional[str]
    row_count: int  # rows in the file
    trainable_row_count: int  # rows carrying an answer -- what would actually be modelled
    column_count: int
    verdict: str  # ready | warnings | blocked | not_applicable
    blockers: list[str]
    warnings: list[str]
    notes: list[str]
    target_summary: Optional[dict]
    remaining_issues: dict
    # The working behind the verdict: every check with its status, evidence and what would
    # change it -- including the ones that could not run and why. `blockers` and `warnings`
    # above are derived views over this, not a separate record.
    checks: list[dict] = []
    check_counts: dict = {}


class ModelPlanResponse(BaseModel):
    """What the app proposes to build, before anything is trained. The user reviews and
    adjusts this -- which columns to use, how to split, which models to try -- so the setup
    is a decision they made rather than a default they never saw."""

    session_id: str
    target_column: str
    task_type: str
    row_count: int  # rows in the file
    trainable_row_count: int  # rows carrying an answer -- what will actually be fitted on
    features: dict  # numeric / categorical / dropped-with-reasons
    suggested_exclusions: list[dict]  # column + why the app thinks it should be left out
    # Which rows the leakage suggestions were measured on. Always the training rows of the
    # default split when the file can be split -- never the held-out ones.
    exclusions_basis: str = ""
    excluded_columns: list[str]  # what was excluded on the last run, if any
    candidate_models: list[dict]
    split_options: list[dict]
    date_columns: list[str]  # candidates for a time-based split
    group_columns: list[str]  # candidates for a grouped split, most repeated first
    blockers: list[str]
    warnings: list[str]
    trained: bool
    # The distinct answers, for a classification target. The form uses them to let the user
    # say which one counts as "a case" when stating a cut-off trade-off.
    class_labels: list[str] = []
    # Derived inputs this file could support, each with the evidence behind it. Proposals --
    # nothing is applied until the user sends the ids back with a train request.
    suggested_features: list[dict] = []
    # The recipe ids the last run used, so the form comes back showing what was chosen.
    feature_recipes: list[str] = []
    # Numeric inputs with values far outside the rest, each with the evidence. Offered, never
    # ticked: clipping changes what the model sees, and a change is the user's to opt into.
    # What the last run clipped is in `features.clipped`.
    suggested_clipping: list[dict] = []
    # Exact copies found among the default inputs plus the answer, or None when there are none.
    # Offered with the count as evidence and pre-selected, because a copy in the test set is a
    # question the model has already been shown the answer to.
    identical_rows: Optional[dict] = None
    keep_identical_rows_together: bool = False
    # Whether a sealed final holdout can be taken from this file (each held-out part big enough
    # to judge), with the part sizes or the reason it cannot. Computed for the default random
    # split; the train request re-checks it for whichever split is chosen.
    sealed_holdout_offer: Optional[dict] = None
    sealed_holdout: bool = False
    # How much to hold back for testing, and why: from the number of labelled rows and, for two
    # answers, how many carry the rarer one. The form pre-fills it and the user can change it.
    recommended_test_size: Optional[dict] = None
    # How to divide the rows, with the evidence: {strategy, column, reasons[], confidence,
    # alternatives[]}. The form pre-selects it; the user can choose differently, and the run
    # records that they did.
    recommended_split: Optional[dict] = None


class SetRulesRequest(BaseModel):
    """The complete set of rules the user has approved.

    A whole-list replace, not a patch: the client always holds the full set, and partial
    updates would let the client's idea of the rules and the server's drift apart.
    """

    rules: list[dict] = []


class RulesResponse(BaseModel):
    """What is in force, what is suggested, and what the approved rules currently find.

    `proposed` are guesses and have no effect until they appear in `rules`. `checks` are the
    same Check shape the readiness panel already renders, so rule findings need no display of
    their own.
    """

    session_id: str
    rules: list[dict]
    proposed: list[dict]
    checks: list[dict]


class OperatingPointRequest(BaseModel):
    """Where the cut-off should sit, stated as a trade-off rather than a number.

    The number is chosen by the app on the training folds (operating_point.py). Letting the
    client send a threshold directly would invite picking one off the test-set sweep, which
    is selecting on the test set in miniature.
    """

    objective: str = "default"  # "default" | "cost" | "min_recall" | "min_precision" | "f1"
    cost_of_missing: float = Field(default=1.0, ge=0.0)
    cost_of_false_alarm: float = Field(default=1.0, ge=0.0)
    target_rate: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    positive_label: Optional[str] = None


class TrainRequest(BaseModel):
    excluded_columns: list[str] = []
    model_names: list[str] = []  # empty means "every model that fits the task"
    split_strategy: str = "random"  # "random" | "time" | "group"
    time_column: Optional[str] = None
    # The column identifying which rows belong to the same entity, for a grouped split.
    group_column: Optional[str] = None
    include_deep_diagnostics: bool = True
    # Two-class targets only; ignored otherwise.
    operating_point: Optional[OperatingPointRequest] = None
    # Ids of the derived inputs to use, from the plan's suggested_features. Re-proposed from
    # the data server-side rather than trusted, so an id for a column that no longer supports
    # the recipe is dropped instead of failing the run.
    feature_recipes: list[str] = []
    # Numeric inputs whose extreme values are clipped to limits learned from the training rows,
    # inside the pipeline. Off unless ticked; names that are not numeric inputs are ignored.
    clip_outliers: list[str] = []
    # Put rows that are exact copies of each other (the model's inputs and the answer) on one
    # side of a random split. Off unless asked for; the plan pre-selects it when copies exist.
    # Ignored by the other split strategies, which keep their own units whole.
    keep_identical_rows_together: bool = False
    # Take a final partition out before anything else happens to the rows, for the user to open
    # once with `finalise`. Needs a file big enough for both held-out parts to be judged; when
    # it is not, the ordinary two-way split is used and the result says why.
    sealed_holdout: bool = False
    # Correct the chosen model's probabilities, fitted on folds of the training rows. Two-class
    # targets only; when the data cannot support the method the model is left as it was and the
    # result says why. Isotonic needs a thousand rows and a hundred of the rarer answer.
    calibration: Literal["none", "sigmoid", "isotonic"] = "none"
    # The fraction of labelled rows to hold back for testing. None means the default; the plan
    # recommends one from the file's size and class balance. Ignored with a sealed holdout, whose
    # three parts are fixed.
    test_size: Optional[float] = Field(default=None, ge=0.05, le=0.5)


class ImproveRequest(TrainRequest):
    """Everything a train needs, plus where to stop.

    `goal` is on the same metric the loop reports -- AUC for a two-class target, accuracy for
    more than two, R² for regression. None means "as good as it can get", which stops when
    the rounds stop earning their place rather than when a number is hit.
    """

    goal: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    # Fits per search. Higher explores more settings and takes proportionally longer; the
    # default is sized so a few thousand rows finish in well under a minute.
    budget: int = Field(default=24, ge=4, le=120)


class SessionStatusResponse(BaseModel):
    session_id: str
    current_agent: Optional[str]
    error_message: Optional[str]
    issue_count: int
    approved_count: int
    rejected_count: int
    # Every state the session's working data has been in, oldest first: version, full content
    # fingerprint, row and column counts, when, and what produced it (upload, cleaning_run,
    # undo:<action>, chat:<action>). A model or a test-set look names one of these.
    dataset_versions: list[dict] = []
