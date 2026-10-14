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
