"""Suggestion cards that make each other unnecessary.

Observed on a real run: a leaky `Exit_Date` column produced three cards at once -- one for
the leakage, one for its nulls, one for its type. Approving all three dropped the column with
the first, after which the other two reported "Could not be applied: Column 'Exit_Date' is
not in the data" and the run announced "2 skipped", as though something had gone wrong. It
hadn't. The app did exactly the right thing and then blamed itself for it.
"""

from __future__ import annotations

import pandas as pd

from app.services import conflicts


def _card(card_id: str, column: str, operation: str | None, **params) -> dict:
    return {
        "card_id": card_id,
        "column_name": column,
        "operation": operation,
        "operation_params": {"column": column, **params} if operation else {},
    }


class TestDescribeConflicts:
    def test_flags_cards_made_pointless_by_a_column_removal(self):
        cards = [
            _card("c1", "exit_date", "drop_column"),
            _card("c2", "exit_date", "drop_rows_with_nulls"),
            _card("c3", "exit_date", "cast_to_datetime"),
        ]
        found = conflicts.describe_conflicts(cards)
        assert set(found) == {"c2", "c3"}
        assert "removes the 'exit_date' column entirely" in found["c2"]

    def test_never_warns_on_the_card_doing_the_removing(self):
        """The removal is usually the more important fix -- it is what leakage detection
        prescribes. Warning on both sides would just be noise."""
        cards = [_card("c1", "exit_date", "drop_column"), _card("c2", "exit_date", "fill_mode")]
        assert "c1" not in conflicts.describe_conflicts(cards)

    def test_leaves_other_columns_alone(self):
        cards = [
            _card("c1", "exit_date", "drop_column"),
            _card("c2", "salary", "fill_median"),
        ]
        assert conflicts.describe_conflicts(cards) == {}

    def test_row_removal_is_not_a_conflict(self):
        """Dropping rows leaves the column in place, so a later fix still has something to
        act on. Treating it as a conflict would warn on most ordinary sessions."""
        cards = [
            _card("c1", "salary", "drop_rows_with_nulls"),
            _card("c2", "salary", "cap_outliers"),
        ]
        assert conflicts.describe_conflicts(cards) == {}

    def test_informational_cards_are_not_flagged(self):
        """A card with no executable fix can't conflict with anything."""
        cards = [_card("c1", "exit_date", "drop_column"), _card("c2", "exit_date", None)]
        assert "c2" not in conflicts.describe_conflicts(cards)

    def test_no_conflicts_in_an_ordinary_session(self):
        cards = [
            _card("c1", "salary", "fill_median"),
            _card("c2", "department", "standardize_case", case="lower"),
            _card("c3", "hire_date", "standardize_dates"),
        ]
        assert conflicts.describe_conflicts(cards) == {}


class TestSupersededReason:
    def test_reports_a_column_removed_by_an_earlier_fix(self):
        df = pd.DataFrame({"salary": [1, 2, 3]})
        reason = conflicts.superseded_reason(
            df, "cast_to_datetime", {"column": "exit_date"}, "exit_date"
        )
        assert reason is not None
        assert "no longer needed" in reason.lower()
        assert "exit_date" in reason
        # The wording must not read as a failure -- that was the whole problem.
        assert "could not" not in reason.lower()
        assert "error" not in reason.lower()

    def test_returns_none_when_the_column_is_still_there(self):
        df = pd.DataFrame({"salary": [1, 2, 3]})
        assert conflicts.superseded_reason(df, "fill_median", {"column": "salary"}, "salary") is None

    def test_falls_back_to_the_cards_own_column(self):
        df = pd.DataFrame({"salary": [1, 2, 3]})
        assert conflicts.superseded_reason(df, "fill_median", {}, "gone") is not None

    def test_a_dataset_level_operation_is_never_superseded(self):
        """drop_duplicate_rows has no column, so nothing can remove what it acts on."""
        df = pd.DataFrame({"salary": [1, 2, 3]})
        assert conflicts.superseded_reason(df, "drop_duplicate_rows", {}, "") is None

    def test_the_dataset_sentinel_is_not_mistaken_for_a_missing_column(self):
        """Found live, and it silently disabled deduplication entirely.

        Dataset-level issues carry column_name "DATASET" (see GraphState), which is never a
        real column -- so "is this column still in the dataframe?" answered no every single
        time, and every approved de-duplication was recorded as "no longer needed" and never
        ran. The duplicates stayed in the file and the user was told the fix wasn't needed."""
        df = pd.DataFrame({"salary": [1, 2, 3]})
        assert conflicts.superseded_reason(df, "drop_duplicate_rows", {}, "DATASET") is None
