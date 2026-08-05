"""Detecting suggestion cards that undo or invalidate each other.

The profiler reports issues per column, independently, and the Explanation Agent writes one
card per issue. So a single bad column can produce several cards at once -- a leaky
`exit_date` will be flagged for leakage, for missing values, and for being stored as text.
Approve all three and the first removes the column, after which the other two have nothing to
act on and fail with "Column 'exit_date' is not in the data".

Nothing is broken by that -- the failures are caught and the run continues -- but the user is
told a fix "could not be applied" when the truth is that it was no longer needed. Worse, the
run reports "2 skipped" as though something went wrong.

Two halves, in the two places they belong:

- `describe_conflicts` runs before approval, so the review page can say which cards will
  become unnecessary if another is approved. Choosing between them is the user's call.
- `superseded_reason` runs during execution, so a card whose column has already been removed
  is recorded as superseded rather than failed.
"""

from __future__ import annotations

import pandas as pd

from app.services.operations import OPERATIONS

# Operations that remove the column entirely. Anything else targeting that column afterwards
# has nothing left to work on.
COLUMN_REMOVING_OPERATIONS = frozenset({"drop_column"})

# Operations that remove rows. They don't invalidate other cards -- a later fix still has a
# column to act on -- so they are deliberately not treated as conflicts.
ROW_REMOVING_OPERATIONS = frozenset({"drop_rows_with_nulls", "drop_outlier_rows", "filter_rows"})


def describe_conflicts(cards: list[dict]) -> dict[str, str]:
    """Returns {card_id: explanation} for cards made unnecessary by another card.

    Only reports the dependent card, never the one doing the removing: the removal is
    generally the more important fix (it is what leakage detection prescribes), and warning on
    both would just be noise.
    """
    removers: dict[str, str] = {}
    for card in cards:
        if card.get("operation") in COLUMN_REMOVING_OPERATIONS:
            column = str(card.get("operation_params", {}).get("column") or card.get("column_name") or "")
            if column:
                removers.setdefault(column, str(card.get("card_id")))

    conflicts: dict[str, str] = {}
    for card in cards:
        card_id = str(card.get("card_id"))
        column = str(card.get("column_name") or "")
        remover = removers.get(column)
        if not remover or remover == card_id:
            continue
        if not card.get("operation"):
            continue
        conflicts[card_id] = (
            f"Another suggestion removes the '{column}' column entirely. If you approve that "
            f"one, this fix won't be needed — approve it only if you'd rather keep the column "
            f"and clean it instead."
        )
    return conflicts


def superseded_reason(df: pd.DataFrame, operation: str, params: dict, column: str) -> str | None:
    """Explains why a fix is no longer needed, or None if it should still run.

    Called just before execution, when the earlier approved fixes have already been applied,
    so this reflects what the data actually looks like now rather than what was predicted.
    """
    # A dataset-level fix has no column to lose. Deduplication carries the sentinel
    # column_name "DATASET", which is never a real column, so the check below declared it
    # superseded every single time and the duplicates were never removed -- an approved fix
    # silently not running, reported as "no longer needed" when it was very much needed.
    spec = OPERATIONS.get(operation)
    if spec is not None and not spec.needs_column:
        return None

    target = str(params.get("column") or column or "")
    if not target or target in df.columns:
        return None

    return (
        f"No longer needed: the '{target}' column was removed by a fix you approved earlier, "
        f"so there was nothing left for this one to change."
    )
