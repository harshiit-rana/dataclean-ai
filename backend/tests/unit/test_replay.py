"""Cleaning a new file the way a session's file was cleaned, and scoring it.

The property the whole feature rests on is `test_a_new_file_is_filled_with_the_sessions_median`.
If replay filled February's gaps with February's own median, the new rows would be cleaned
differently from the rows the model learned from, and every prediction would sit on inputs the
model was never shown. The rest pin that no row vanishes without a reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import drift, operations, replay, scoring


def _log(*steps: tuple[str, str, dict]) -> list[dict]:
    return [
        {"action_id": f"a{i}", "operation": op, "column_name": col, "operation_params": params,
         "status": "applied"}
        for i, (op, col, params) in enumerate(steps)
    ]


class TestLearnedValues:
    def test_a_new_file_is_filled_with_the_sessions_median(self):
        original = pd.DataFrame({"spend": [10.0, 20.0, None, 30.0]})  # median 20
        new = pd.DataFrame({"spend": [1000.0, None, 3000.0]})  # its own median would be 2000
        steps, _ = replay.freeze(original, _log(("fill_median", "spend", {})))
        assert steps[0].learned == {"fill_value": 20.0}

        applied = replay.apply(new, steps)
        assert applied.frame["spend"].tolist() == [1000.0, 20.0, 3000.0]

    def test_the_mode_is_frozen_too(self):
        original = pd.DataFrame({"plan": ["basic", "basic", "pro", None]})
        new = pd.DataFrame({"plan": ["pro", "pro", None]})
        steps, _ = replay.freeze(original, _log(("fill_mode", "plan", {})))
        assert replay.apply(new, steps).frame["plan"].tolist() == ["pro", "pro", "basic"]

    def test_outlier_limits_come_from_the_session_not_the_new_file(self):
        original = pd.DataFrame({"x": [10.0, 11, 12, 13, 14, 15, 16, 17, 18, 500]})
        steps, _ = replay.freeze(original, _log(("drop_outlier_rows", "x", {})))
        lower, upper = steps[0].learned["lower"], steps[0].learned["upper"]
        # A new file made entirely of values the session would call outliers. Its own IQR
        # would call none of them outliers; the session's limits drop every one.
        new = pd.DataFrame({"x": [upper + 100, upper + 200, upper + 300, upper + 400, upper + 500]})
        applied = replay.apply(new, steps)
        assert applied.frame.empty
        assert applied.dropped[0].startswith("x was outside the range")
        assert lower < 10

    def test_learned_steps_are_learned_at_the_point_they_ran(self):
        """A median taken after an earlier step capped the outliers is not the median of
        the raw column. Freezing replays in order, so each value reflects what came before."""
        original = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10_000, None]})
        steps, _ = replay.freeze(
            original, _log(("cap_outliers", "x", {}), ("fill_mean", "x", {}))
        )
        capped_mean = original["x"].clip(upper=steps[0].learned["upper"]).mean()
        assert steps[1].learned["fill_value"] == pytest.approx(capped_mean)

    def test_the_replay_reproduces_the_sessions_own_cleaning(self):
        original = pd.DataFrame(
            {"spend": [10.0, None, 30.0, 30.0], "city": ["Paris", "paris", "Rome", "Rome"]}
        )
        log = _log(
            ("fill_median", "spend", {}),
            ("standardize_case", "city", {"case": "lower"}),
            ("drop_duplicate_rows", "", {}),
        )
        _steps, reproduced = replay.freeze(original, log)
        expected = original
        for entry in log:
            params = dict(entry["operation_params"])
            if operations.OPERATIONS[entry["operation"]].needs_column:
                params["column"] = entry["column_name"]
            expected, _b, _a = operations.apply_operation(expected, entry["operation"], params)
        pd.testing.assert_frame_equal(reproduced, expected)

    def test_a_learned_value_cannot_arrive_as_a_parameter(self):
        """Parameters come from the LLM and from chat instructions. A `fill_value` sneaked in
        that way must not turn a median fill into a constant fill."""
        df = pd.DataFrame({"x": [1.0, None, 3.0]})
        name, params = operations.parse_operation('fill_median(column="x", fill_value="999")')
        out, _b, _a = operations.apply_operation(df, name, params)
        assert out["x"].tolist() == [1.0, 2.0, 3.0]

    def test_undone_and_rejected_steps_are_not_replayed(self):
        original = pd.DataFrame({"x": [1.0, None]})
        log = _log(("fill_median", "x", {}), ("drop_column", "x", {}))
        log[1]["status"] = "undone"
        steps, _ = replay.freeze(original, log)
        assert [s.operation for s in steps] == ["fill_median"]


class TestNothingVanishes:
    def test_every_dropped_row_is_reported_with_its_reason(self):
        original = pd.DataFrame({"spend": [1.0, 2.0, 3.0]})
        steps, _ = replay.freeze(original, _log(("drop_rows_with_nulls", "spend", {})))
        new = pd.DataFrame({"spend": [5.0, None, 7.0, None]})
        applied = replay.apply(new, steps)
        assert list(applied.frame.index) == [0, 2]  # input row numbers survive the step
        assert applied.dropped == {1: "spend was empty", 3: "spend was empty"}
        assert applied.steps[0].as_dict()["dropped_count"] == 2

    def test_duplicates_are_reported_as_duplicates(self):
        steps, _ = replay.freeze(pd.DataFrame({"a": [1, 2]}), _log(("drop_duplicate_rows", "", {})))
        applied = replay.apply(pd.DataFrame({"a": [1, 1, 2]}), steps)
        assert applied.dropped == {1: "it duplicated an earlier row"}


class TestColumnsTheNewFileLacks:
    def test_a_missing_target_is_expected_not_an_error(self):
        original = pd.DataFrame({"churned": ["yes", None, "no"]})
        steps, _ = replay.freeze(original, _log(("drop_rows_with_nulls", "churned", {})))
        applied = replay.apply(pd.DataFrame({"spend": [1, 2]}), steps, target_column="churned")
        assert applied.steps[0].status == "skipped"
        assert "expected" in applied.steps[0].note
        assert len(applied.frame) == 2

    def test_a_missing_input_is_said_and_the_rest_still_runs(self):
        original = pd.DataFrame({"a": [1.0, None], "b": ["X", "x"]})
        steps, _ = replay.freeze(
            original, _log(("fill_median", "a", {}), ("standardize_case", "b", {"case": "lower"}))
        )
        applied = replay.apply(pd.DataFrame({"b": ["Y", "y"]}), steps)
        assert [s.status for s in applied.steps] == ["skipped", "applied"]
        assert "the original did" in applied.steps[0].note
        assert applied.frame["b"].tolist() == ["y", "y"]

    def test_a_step_that_cannot_run_fails_alone(self):
        original = pd.DataFrame({"a": [1.0, None], "b": ["X", "x"]})
        steps, _ = replay.freeze(
            original, _log(("cast_to_numeric", "a", {}), ("standardize_case", "b", {"case": "lower"}))
        )
        steps[0] = replay.Step("a0", "fill_median", {"column": "a"}, "a", None)  # no learned value
        applied = replay.apply(pd.DataFrame({"a": ["x", "y"], "b": ["Q", "q"]}), steps)
        assert applied.steps[0].status == "failed"
        assert applied.steps[1].status == "applied"


class TestDrift:
    rng = np.random.default_rng(3)

    def _frames(self, shift=0.0, unseen=0.0, n=400):
        ref = pd.DataFrame({"x": self.rng.normal(50, 10, n), "c": self.rng.choice(["a", "b", "c"], n)})
        new_c = np.where(self.rng.random(n) < unseen, "zzz", self.rng.choice(["a", "b", "c"], n))
        new = pd.DataFrame({"x": self.rng.normal(50 + shift, 10, n), "c": new_c})
        return ref, new

    def test_the_same_population_is_similar(self):
        ref, new = self._frames()
        assert drift.compare(ref, new, ["x"], ["c"])["verdict"] == "similar"

    def test_a_moved_number_is_called_different(self):
        ref, new = self._frames(shift=25)
        report = drift.compare(ref, new, ["x"], ["c"])
        assert report["verdict"] == "different"
        assert report["columns"][0]["column"] == "x"  # most changed first
        assert "'x'" in report["headline"]

    def test_categories_the_model_never_saw_are_counted(self):
        ref, new = self._frames(unseen=0.4)
        column = next(c for c in drift.compare(ref, new, ["x"], ["c"])["columns"] if c["column"] == "c")
        assert column["status"] == "different"
        assert column["unseen_examples"] == ["zzz"]
        assert 0.3 < column["unseen_share"] < 0.5

    def test_a_few_values_past_the_training_extremes_are_ordinary(self):
        """A training maximum is an extreme by definition; nearly any new file has some values
        beyond it. Flagging that would fire on every column of every file."""
        ref, new = self._frames()
        new.loc[:3, "x"] = ref["x"].max() + 1
        column = next(c for c in drift.compare(ref, new, ["x"], ["c"])["columns"] if c["column"] == "x")
        assert column["status"] == "stable"

    def test_too_few_rows_is_not_a_verdict(self):
        ref, new = self._frames()
        report = drift.compare(ref, new.head(10), ["x"], ["c"])
        assert report["verdict"] == "not_enough_rows"
        assert "10 rows" in report["headline"]


class TestScoring:
    RESULT = {
        "task": "classification",
        "target_column": "churned",
        "features": {"numeric": ["spend"], "categorical": ["plan"], "frequent": [], "derived": []},
        "operating_point": {"objective": "cost", "threshold": 0.2, "positive_label": "yes"},
    }

    def test_every_missing_input_is_named_at_once(self):
        with pytest.raises(scoring.ScoringError, match="missing 2 columns the model needs: spend, plan"):
            scoring.prepare_inputs(pd.DataFrame({"other": [1]}), self.RESULT)

    def test_text_in_a_number_column_is_blanked_and_said(self):
        X, notes = scoring.prepare_inputs(
            pd.DataFrame({"spend": ["12", "n/a", "30"], "plan": ["a", "b", "a"]}), self.RESULT
        )
        assert X["spend"].isna().tolist() == [False, True, False]
        assert "1 value in 'spend'" in notes[0]

    def test_a_category_read_as_numbers_is_matched_to_the_training_kind(self):
        reference = pd.DataFrame({"spend": [1.0], "plan": ["101"]})
        X, _ = scoring.prepare_inputs(
            pd.DataFrame({"spend": [1.0, 2.0], "plan": [101, 102]}), self.RESULT, reference=reference
        )
        assert X["plan"].tolist() == ["101", "102"]

    def test_answers_are_given_at_the_chosen_cut_off_not_at_one_half(self):
        class Fixed:
            classes_ = np.array(["no", "yes"])

            def predict_proba(self, X):
                return np.array([[0.7, 0.3]] * len(X))  # P(yes) = 0.3

        out = scoring.predict(Fixed(), pd.DataFrame({"spend": [1.0, 2.0]}), self.RESULT)
        # 0.3 is below one half but above the chosen 0.2: the reported model says "yes".
        assert out["predicted_churned"].tolist() == ["yes", "yes"]
        assert out["probability_yes"].tolist() == [0.3, 0.3]
