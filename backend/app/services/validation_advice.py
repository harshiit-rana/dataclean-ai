"""How to validate a model on this file: how much to hold back, and how to split it.

Both used to be fixed. A fifth of the rows is held out for every file, which on 150 labelled rows
leaves 30 test rows (exactly the least a verdict can be given on) and, at a 10% positive rate, three
positives, so "not enough test data" is the answer by construction. On 200,000 rows it holds out
forty thousand, far more than anything needs, and takes them from training.

These are pure functions of the file's shape, so they can be tested without the API and called from
the plan, the readiness check and the train route alike. They say *why* in a sentence, because the
point is a decision the user can see and overrule, not a default they never saw.
"""

from __future__ import annotations

import math

# Never hold back less than this, however large the file: below it a rare answer in the test rows
# is a handful of cases, and above a few thousand test rows nothing is learned by adding more.
MIN_FRACTION = 0.10
# Never more than this: the model needs the rest to learn from.
MAX_FRACTION = 0.30
# The usual fraction for a file of ordinary size, and the lower ones for larger files.
USUAL_FRACTION = 0.20
LARGE_FILE_ROWS = 50_000
LARGE_FILE_FRACTION = MIN_FRACTION
MIDSIZE_FILE_ROWS = 10_000
MIDSIZE_FILE_FRACTION = 0.15
# A little over what judging strictly needs, so a split that lands a row or two short of its target
# does not tip a file into "not enough test data".
HEADROOM = 1.1
# The training side needs enough of the rarer answer to be folded and fitted at all.
MIN_TRAIN_RARE_ROWS = 10


def recommend_for(answers, task: str | None) -> dict | None:
    """`recommend_test_size` for a target column: counts the labelled rows and, for exactly two
    answers, the rarer one. None when there is nothing labelled."""
    import pandas as pd

    labelled = pd.Series(answers).dropna()
    if labelled.empty:
        return None
    counts = labelled.astype(str).value_counts()
    rarest = int(counts.min()) if task == "classification" and len(counts) == 2 else None
    return recommend_test_size(int(len(labelled)), rarest)


def _up_to_five_percent(fraction: float) -> float:
    return math.ceil(round(fraction * 20, 6)) / 20


def recommend_test_size(rows: int, rarest: int | None = None) -> dict:
    """The fraction to hold back, and why.

    `rows` is the number of labelled rows; `rarest` is how many of them carry the rarer answer of a
    two-class target (None for anything else). The fraction is chosen so the test rows number at
    least what a verdict needs -- `MIN_TEST_ROWS_TO_JUDGE` of them, and for two answers
    `MIN_TEST_ROWS_PER_CLASS_TO_JUDGE` of the rarer one -- inside 10-30%, and lower for larger
    files, where the rest is better spent on training. When no fraction inside the range leaves
    both enough to judge on and enough to train on, `feasible` is false and the reason says so; the
    fraction returned is then the largest allowed, the best chance of judging at all.
    """
    from app.services import modeling

    need_rows = modeling.MIN_TEST_ROWS_TO_JUDGE
    need_rare = modeling.MIN_TEST_ROWS_PER_CLASS_TO_JUDGE if rarest is not None else 0
    rows = max(int(rows), 0)
    if rows == 0:
        return _answer(USUAL_FRACTION, rows, rarest, False, "There are no labelled rows to hold back.")

    required = max(need_rows / rows, need_rare / rarest if rarest else 0.0)
    usual = (
        LARGE_FILE_FRACTION if rows >= LARGE_FILE_ROWS
        else MIDSIZE_FILE_FRACTION if rows >= MIDSIZE_FILE_ROWS
        else USUAL_FRACTION
    )
    wanted = _up_to_five_percent(required * HEADROOM) if required * HEADROOM > MIN_FRACTION else MIN_FRACTION
    fraction = min(MAX_FRACTION, max(usual, wanted))

    test_rows = rows * fraction
    train_rows = rows - test_rows
    test_rare = None if rarest is None else rarest * fraction
    train_rare = None if rarest is None else rarest - test_rare
    can_judge = test_rows >= need_rows and (rarest is None or test_rare >= need_rare)
    can_train = train_rows >= modeling.MIN_ROWS_TO_TRAIN and (rarest is None or train_rare >= MIN_TRAIN_RARE_ROWS)

    if can_judge and can_train:
        return _answer(fraction, rows, rarest, True, _why(fraction, USUAL_FRACTION, rows, rarest, need_rows, need_rare))

    problems = []
    if not can_judge:
        problems.append(
            f"holding back the most this app allows ({MAX_FRACTION:.0%}) leaves {round(test_rows):,} test rows"
            + (f" and about {round(test_rare):,} of the rarer answer" if rarest is not None else "")
            + f", and a score needs at least {need_rows}"
            + (f" and {need_rare} of the rarer answer" if rarest is not None else "")
            + " before it means anything"
        )
    if not can_train:
        problems.append("holding back that much also leaves too little to train on")
    return _answer(
        MAX_FRACTION if not can_judge else fraction,
        rows,
        rarest,
        False,
        f"No fraction works for this file: with {rows:,} labelled rows"
        + (f", only {rarest:,} of them the rarer answer" if rarest is not None else "")
        + f", {'; '.join(problems)}. Whatever it scores will not be a verdict to lean on; more "
        f"labelled rows are the only fix.",
    )


def _why(fraction: float, usual: float, rows: int, rarest: int | None, need_rows: int, need_rare: int) -> str:
    test_rows = round(rows * fraction)
    leaves = f"{fraction:.0%} leaves {test_rows:,} test rows" + (
        f" ({round(rarest * fraction):,} of the rarer answer)" if rarest is not None else ""
    )
    if fraction > usual:
        at_usual_rows = round(rows * usual)
        if rarest is not None and need_rare / rarest >= need_rows / rows:
            problem = (
                f"{usual:.0%} would leave about {round(rarest * usual):,} of the rarer answer to test on, "
                f"and at least {need_rare} are needed before a score means anything."
            )
        else:
            problem = (
                f"{usual:.0%} would leave {at_usual_rows:,} test rows, too close to the {need_rows} "
                f"needed before a score means anything."
            )
        return f"Raised to {fraction:.0%}: {problem} {leaves} -- enough to judge a model."
    if fraction < usual:
        return (
            f"Lowered to {fraction:.0%}: {leaves}, far more than the {need_rows} needed to judge a "
            f"model, and the rest is better spent on training."
        )
    return f"{leaves} -- enough to judge a model."


def _answer(fraction: float, rows: int, rarest: int | None, feasible: bool, reason: str) -> dict:
    test_rows = round(rows * fraction)
    return {
        "fraction": round(float(fraction), 2),
        "reason": reason,
        "feasible": feasible,
        "labelled_rows": rows,
        "test_rows": test_rows,
        "train_rows": rows - test_rows,
        "test_rare_rows": None if rarest is None else round(rarest * fraction),
    }


# ------------------------------------------------------------------------------------ the split
#
# Which way to divide the rows. The rules, in order:
#
#   1. A column that looks like one entity repeating across rows (the readiness panel's own test,
#      not re-tuned here) -> group on the most repeated one: a random split would put the same
#      customer on both sides and the score would partly measure recognising them.
#   2. A date column, and either the use case talks about the future or the answer drifts over time
#      -> split by date: a random split lets the model learn from the future to predict the past.
#   3. Otherwise random, stratified for a classification.
#
# Every rule is stated with its evidence, and every one can be overruled on the plan.

import re

import numpy as np
import pandas as pd

# Phrases that say the model will be used on rows from after the ones it learns from.
_FORECAST = re.compile(
    r"\b(forecast\w*|future|upcoming|going forward|time[- ]series|over time|projection\w*|"
    r"next (?:day|week|month|quarter|year|season)|in the coming|tomorrow|seasonal\w*|trend\w*)\b",
    re.IGNORECASE,
)
# A date that is a property of a person or a deadline, not a moment something happened: ordering
# rows by it says nothing about before and after.
_NOT_EVENT_TIME = re.compile(r"birth|dob|born|expir|deadline|due_?date|renewal", re.IGNORECASE)
# Names that usually mean "when this row happened".
_EVENT_TIME = re.compile(
    r"creat|date|time|timestamp|stamp|day|month|week|year|when|_at$|_on$|period", re.IGNORECASE
)
# Names that usually mean "which one": a customer, a patient, a machine.
_IDENTIFIER = re.compile(
    r"(^|_)(id|ids|key|uuid|guid|number|no|code)($|_)|customer|patient|user|client|account|member|"
    r"machine|device|subject|person|employee|session|household",
    re.IGNORECASE,
)

# Drift over time is only worth building a split on when it is both unlikely to be chance and big
# enough to matter. Chosen to be conservative: a time split costs the model its random sample.
DRIFT_P_VALUE = 0.001
DRIFT_MIN_CRAMERS_V = 0.15
DRIFT_MIN_COHENS_D = 0.3
MIN_ROWS_TO_SEE_DRIFT = 60
MIN_DATES_TO_SEE_DRIFT = 10


def _mentions_the_future(text: str) -> str | None:
    found = _FORECAST.search(text or "")
    return found.group(0) if found else None


def entity_evidence(df: pd.DataFrame, column: str) -> dict:
    """How much of the file `column` repeats across: the rows whose value occurs more than once."""
    counts = df[column].dropna().astype(str).value_counts()
    rows = int(counts.sum())
    repeated_rows = int(counts[counts > 1].sum())
    distinct = int(len(counts))
    return {
        "column": column,
        "rows": rows,
        "repeated_rows": repeated_rows,
        "distinct": distinct,
        "rows_each": round(rows / distinct, 1) if distinct else 0.0,
    }


def drift_over_time(frame: pd.DataFrame, column: str, target: str, task: str | None) -> dict | None:
    """Whether the answer changes between the earliest and latest third of the rows, by `column`.

    Read off `frame` -- the caller passes the training rows of the default split, never the rows
    that will be held back. Classification: a chi-square test of the answers' mix, with Cramer's V
    as the size; regression: Welch's t-test, with Cohen's d. Returns None when there is too little
    to see anything, and `drifts: False` when there is nothing to see.
    """
    from scipy import stats

    dates = pd.to_datetime(frame[column], errors="coerce", format="mixed")
    usable = dates.notna() & frame[target].notna()
    if (
        int(usable.sum()) < MIN_ROWS_TO_SEE_DRIFT
        or dates[usable].dt.normalize().nunique() < MIN_DATES_TO_SEE_DRIFT
    ):
        return None
    order = np.argsort(dates[usable].to_numpy(), kind="stable")
    answers = frame.loc[usable, target].iloc[order].reset_index(drop=True)
    third = len(answers) // 3
    early, late = answers.iloc[:third], answers.iloc[-third:]

    if task == "regression":
        early = pd.to_numeric(early, errors="coerce").dropna()
        late = pd.to_numeric(late, errors="coerce").dropna()
        if len(early) < 10 or len(late) < 10:
            return None
        pooled = float(np.sqrt((early.var(ddof=1) + late.var(ddof=1)) / 2))
        d = (float(late.mean()) - float(early.mean())) / pooled if pooled > 0 else 0.0
        p = float(stats.ttest_ind(early, late, equal_var=False).pvalue)
        return {
            "drifts": bool(p < DRIFT_P_VALUE and abs(d) >= DRIFT_MIN_COHENS_D),
            "p": p,
            "effect": abs(d),
            "text": f"the average answer is {early.mean():,.4g} in the earliest third of the rows and "
            f"{late.mean():,.4g} in the latest",
        }

    early, late = early.astype(str), late.astype(str)
    classes = sorted(set(early) | set(late))
    table = np.array([[int((early == c).sum()), int((late == c).sum())] for c in classes])
    if len(classes) < 2 or (table.sum(axis=1) == 0).any():
        return {"drifts": False, "p": 1.0, "effect": 0.0, "text": ""}
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    v = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
    # The answer whose share moved most; ties (always the case with two answers) go to the later
    # label, the one the app treats as "a case" by default.
    moved = max(classes, key=lambda c: (round(abs(float((early == c).mean()) - float((late == c).mean())), 6), c))
    return {
        "drifts": bool(p < DRIFT_P_VALUE and v >= DRIFT_MIN_CRAMERS_V),
        "p": float(p),
        "effect": v,
        "text": f"{moved!r} is {float((early == moved).mean()):.0%} of the answers in the earliest third of "
        f"the rows and {float((late == moved).mean()):.0%} in the latest",
    }


def recommend_split(
    df: pd.DataFrame,
    target: str,
    task: str | None,
    *,
    group_columns: list[str],
    date_columns: list[str],
    goal_text: str = "",
    training_rows: pd.DataFrame | None = None,
) -> dict:
    """How to divide the rows into training and test, with the evidence and a confidence.

    `group_columns` and `date_columns` are the plan's candidates (the group candidates share their
    test with the readiness panel). `training_rows` is the training side of the default split
    with the target attached -- the only rows the drift test is allowed to read, for the same
    reason the leakage suggestions read only those. Returns
    `{"strategy", "column", "reasons", "confidence", "alternatives"}`.
    """
    reasons: list[str] = []
    alternatives: list[dict] = []

    # --- repeated entities ------------------------------------------------------------------
    if group_columns:
        entity = entity_evidence(df, group_columns[0])
        column = entity["column"]
        id_like = bool(_IDENTIFIER.search(column))
        reasons.append(
            f"'{column}' repeats in {entity['repeated_rows']:,} of {entity['rows']:,} rows "
            f"({entity['distinct']:,} distinct values, about {entity['rows_each']} rows each): a random "
            f"split would put the same {column} on both sides, and the score would partly measure "
            f"recognising it rather than learning the pattern."
        )
        if not id_like:
            reasons.append(
                f"Check that '{column}' really is one thing appearing many times. If it is a category "
                f"such as a region or a plan, which will also appear when the model is used, group on "
                f"something else or split at random."
            )
        if date_columns:
            reasons.append(
                "The file also has dates, but a split can divide on only one thing at a time; "
                "grouping protects against the larger leak here."
            )
        alternatives = [{"strategy": "group", "column": other} for other in group_columns[1:3]]
        return {
            "strategy": "group",
            "column": column,
            "reasons": reasons,
            "confidence": "high" if id_like else "medium",
            "alternatives": alternatives,
        }

    # --- time order --------------------------------------------------------------------------
    usable_dates = [c for c in date_columns if not _NOT_EVENT_TIME.search(str(c))]
    skipped = [c for c in date_columns if c not in usable_dates]
    phrase = _mentions_the_future(goal_text)
    best: tuple[str, dict | None] | None = None
    if usable_dates:
        scored = []
        for column in usable_dates:
            drift = (
                drift_over_time(training_rows, column, target, task)
                if training_rows is not None and target in training_rows.columns
                else None
            )
            scored.append((column, drift, (bool(drift and drift["drifts"]), bool(_EVENT_TIME.search(str(column))))))
        scored.sort(key=lambda item: item[2], reverse=True)
        best = (scored[0][0], scored[0][1])

    if best is not None:
        column, drift = best
        drifting = bool(drift and drift["drifts"])
        if phrase or drifting:
            if phrase:
                reasons.append(
                    f"You described the goal in terms of the future ('{phrase}'), and the file has a "
                    f"date column, '{column}': a model that will be used on later rows should be tested "
                    f"on later rows, and a random split lets it learn from the future to predict the past."
                )
            if drifting:
                reasons.append(
                    f"The answer changes over time in '{column}': {drift['text']}. A random split would "
                    f"test the model on the same period it trained on and flatter it."
                )
            return {
                "strategy": "time",
                "column": column,
                "reasons": reasons,
                "confidence": "high" if (phrase and drifting) else "medium",
                "alternatives": [{"strategy": "random", "column": None}],
            }
        reasons.append(
            f"There is a date column, '{column}', but nothing says the order matters here: the use case "
            f"does not talk about the future and the answer does not drift over time in it."
            if drift is not None
            else f"There is a date column, '{column}', but nothing says the order matters here."
        )
        alternatives.append({"strategy": "time", "column": column})
    if skipped:
        reasons.append(
            f"{', '.join(repr(c) for c in skipped)} looks like a date of birth or a deadline, which says "
            f"nothing about before and after, so it was not used to order the rows."
        )

    # --- otherwise random --------------------------------------------------------------------
    reasons.insert(0, "No repeated entities or time order found.")
    reasons.append(
        "A random split, stratified so the rarer answer reaches both sides."
        if task == "classification"
        else "A random split."
    )
    return {
        "strategy": "random",
        "column": None,
        "reasons": reasons,
        "confidence": "medium",
        "alternatives": alternatives,
    }
