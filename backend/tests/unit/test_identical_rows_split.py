"""Exact copies of a row stay on one side of the split.

The app counts test rows that are copies of training rows, answer included -- questions the model
has already been shown the answer to. Counting is honest; preventing it is better. When the
copies are duplicated records rather than genuine repeats, a random split puts some in training
and some in test, and every score is flattered by it.
"""

from __future__ import annotations

import io
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import improvement, model_export, modeling
from app.services.session_store import session_dir

client = TestClient(app)
INPUTS = ["x", "region"]


def with_copies(copies: int = 60, n: int = 150, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.DataFrame({"x": rng.integers(0, 6, n), "region": rng.choice(["a", "b"], n)})
    base["y"] = np.where(base["x"] + rng.normal(0, 1.5, n) > 3, "yes", "no")
    return pd.concat([base, base.sample(copies, random_state=2)], ignore_index=True)


def continuous_with_copies(n: int = 220, copies: int = 60) -> pd.DataFrame:
    """Inputs that are all different except where a row was deliberately copied, so the only
    repeats are exact copies (not contradictory labels, which the readiness check blocks)."""
    rng = np.random.default_rng(12)
    base = pd.DataFrame({"x": rng.normal(0, 1, n).round(4), "z": rng.normal(0, 1, n).round(4)})
    base["y"] = np.where(base["x"] + rng.normal(0, 0.8, n) > 0, "yes", "no")
    return pd.concat([base, base.sample(copies, random_state=3)], ignore_index=True)


def every_row_twice(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    base = pd.DataFrame({"x": rng.normal(0, 1, n).round(3), "z": rng.normal(0, 1, n).round(3)})
    base["y"] = np.where(base["x"] + rng.normal(0, 0.8, n) > 0, "yes", "no")
    return pd.concat([base, base], ignore_index=True)


def keys(frame: pd.DataFrame, y: pd.Series) -> set:
    return set(frame.assign(__y__=y.to_numpy()).astype(str).itertuples(index=False, name=None))


class TestTheSplit:
    def test_no_copy_is_on_both_sides(self):
        df = every_row_twice()
        X_train, X_test, y_train, y_test = modeling.split_data(
            df, "y", "classification", duplicate_columns=["x", "z"]
        )
        assert not keys(X_train[["x", "z"]], y_train) & keys(X_test[["x", "z"]], y_test)

    def test_the_count_of_copies_in_the_test_set_is_zero(self):
        """The issue's own case: a frame where every row appears twice."""
        df = every_row_twice()
        X_train, X_test, y_train, y_test = modeling.split_data(
            df, "y", "classification", duplicate_columns=["x", "z"]
        )
        assert modeling.test_rows_seen_in_training(X_train[["x", "z"]], X_test[["x", "z"]], y_train, y_test) == 0

    def test_without_the_option_the_same_frame_has_many_copies_across_the_split(self):
        """The fixture has to be able to fail."""
        df = every_row_twice()
        X_train, X_test, y_train, y_test = modeling.split_data(df, "y", "classification")
        assert modeling.test_rows_seen_in_training(X_train[["x", "z"]], X_test[["x", "z"]], y_train, y_test) > 10

    def test_both_copies_of_a_row_land_together(self):
        df = every_row_twice(80)
        X_train, X_test, _, _ = modeling.split_data(df, "y", "classification", duplicate_columns=["x", "z"])
        for i in range(80):
            assert (i in X_train.index) == (i + 80 in X_train.index)

    def test_the_test_fraction_is_still_about_a_fifth(self):
        df = every_row_twice(200)
        _, X_test, _, _ = modeling.split_data(df, "y", "classification", duplicate_columns=["x", "z"])
        assert len(X_test) / len(df) == pytest.approx(0.2, abs=0.05)

    def test_off_is_exactly_the_ordinary_split(self):
        df = with_copies()
        a = modeling.split_data(df, "y", "classification")
        b = modeling.split_data(df, "y", "classification", duplicate_columns=None)
        assert list(a[1].index) == list(b[1].index) and list(a[0].index) == list(b[0].index)

    def test_asking_for_it_when_there_are_no_copies_changes_nothing(self):
        df = every_row_twice().drop_duplicates(ignore_index=True)
        a = modeling.split_data(df, "y", "classification")
        b = modeling.split_data(df, "y", "classification", duplicate_columns=["x", "z"])
        assert list(a[1].index) == list(b[1].index)

    def test_the_rarer_class_still_reaches_both_sides(self):
        rng = np.random.default_rng(6)
        base = pd.DataFrame({"x": rng.normal(0, 1, 300).round(3)})
        base["y"] = ["no"] * 270 + ["yes"] * 30
        df = pd.concat([base, base.iloc[:100]], ignore_index=True)  # copies of some, rare class too
        _, _, y_train, y_test = modeling.split_data(df, "y", "classification", duplicate_columns=["x"])
        assert set(y_train) == {"yes", "no"} and set(y_test) == {"yes", "no"}

    def test_a_numeric_target_is_split_too(self):
        rng = np.random.default_rng(7)
        base = pd.DataFrame({"x": rng.normal(0, 1, 120).round(3)})
        base["price"] = (base["x"] * 3).round(2)
        df = pd.concat([base, base], ignore_index=True)
        X_train, X_test, y_train, y_test = modeling.split_data(df, "price", "regression", duplicate_columns=["x"])
        assert not keys(X_train[["x"]], y_train) & keys(X_test[["x"]], y_test)

    def test_columns_the_model_never_sees_do_not_hide_a_copy(self):
        """Two rows that differ only in a customer number are, to the model, the same
        question with the same answer."""
        df = every_row_twice(60)
        df["row_id"] = range(len(df))  # unique, and not an input
        X_train, X_test, y_train, y_test = modeling.split_data(
            df, "y", "classification", duplicate_columns=["x", "z"]
        )
        assert not keys(X_train[["x", "z"]], y_train) & keys(X_test[["x", "z"]], y_test)

    def test_the_answer_is_part_of_the_identity(self):
        """Same inputs with different answers are not copies; they are the contradictory
        labels check's business, and are free to land on either side."""
        df = pd.DataFrame({"x": [1.0] * 40 + list(range(2, 62)), "y": ["yes", "no"] * 20 + ["yes", "no"] * 30})
        ids = modeling.identical_row_ids(df, ["x"], "y")
        assert ids.nunique() > 61  # the 40 rows of x=1.0 split into two identities, not one

    def test_other_strategies_ignore_it(self):
        rng = np.random.default_rng(8)
        n = 200
        df = pd.DataFrame(
            {
                "when": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
                "x": rng.integers(0, 3, n),
                "y": rng.choice(["yes", "no"], n),
            }
        )
        a = modeling.split_data(df, "y", "classification", strategy="time", time_column="when")
        b = modeling.split_data(df, "y", "classification", strategy="time", time_column="when", duplicate_columns=["x"])
        assert list(a[1].index) == list(b[1].index)


class TestInTraining:
    def _train(self, df, **kwargs):
        return modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, **kwargs,
        )

    def test_the_result_has_no_copies_across_the_split(self):
        result = self._train(with_copies(), keep_identical_rows_together=True)
        assert result["split"]["test_rows_seen_in_training"] == 0
        assert result["split"]["keep_duplicates_together"] is True

    def test_without_it_the_copies_are_still_counted(self):
        result = self._train(with_copies())
        assert result["split"]["test_rows_seen_in_training"] > 0
        assert result["split"]["keep_duplicates_together"] is False

    def test_asking_on_a_file_with_no_copies_is_recorded_as_the_ordinary_split(self):
        """The same rows are held out either way, so the ledger must key them alike."""
        df = every_row_twice().drop_duplicates(ignore_index=True)
        asked = self._train(df, keep_identical_rows_together=True)
        plain = self._train(df)
        assert asked["split"]["keep_duplicates_together"] is False
        assert ledger.split_key(asked["split"]) == ledger.split_key(plain["split"])

    def test_a_split_that_kept_copies_together_is_a_different_rule_to_the_ledger(self):
        kept = self._train(with_copies(), keep_identical_rows_together=True)
        plain = self._train(with_copies())
        assert ledger.split_key(kept["split"]) != ledger.split_key(plain["split"])

    def test_the_improvement_loop_splits_the_same_way(self, monkeypatch):
        seen = {}
        real = modeling.split_data

        def spy(*args, **kwargs):
            seen["duplicate_columns"] = kwargs.get("duplicate_columns")
            return real(*args, **kwargs)

        monkeypatch.setattr(modeling, "split_data", spy)
        improvement.improve(
            continuous_with_copies(), target_column="y", task="classification",
            model_names=["logistic_regression"], budget=4, keep_identical_rows_together=True,
        )
        assert seen["duplicate_columns"] == ["x", "z"]


class TestTheExport:
    def _result(self, df):
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, keep_identical_rows_together=True,
        )
        result["features"]["excluded"] = []
        return result

    def test_the_script_cuts_where_the_app_cut(self):
        df = with_copies()
        result = self._result(df)
        namespace = {
            "pd": pd,
            "labelled": df[df["y"].notna()],
            "NUMERIC_FEATURES": result["features"]["numeric"],
            "CATEGORICAL_FEATURES": result["features"]["categorical"],
            "FREQUENT_FEATURES": result["features"]["frequent"],
            "TARGET": "y",
        }
        namespace["features"] = namespace["labelled"].drop(columns=["y"])
        namespace["labels"] = namespace["labelled"]["y"]
        namespace["train_test_split"] = __import__(
            "sklearn.model_selection", fromlist=["train_test_split"]
        ).train_test_split
        exec(model_export._split_block(result["split"], "classification"), namespace)  # noqa: S102

        _, X_test, _, _ = modeling.split_data(df, "y", "classification", duplicate_columns=INPUTS)
        assert list(namespace["X_test"].index) == list(X_test.index)

    def test_the_script_runs_and_reproduces_the_metrics(self, tmp_path):
        df = with_copies(copies=80, n=250)
        result = self._result(df)
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), "utf-8")
        done = subprocess.run(
            [sys.executable, str(script), str(csv)], capture_output=True, text=True, cwd=tmp_path, timeout=600
        )
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        import re

        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match, done.stdout
        assert float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["accuracy"], abs=1e-4
        )

    def test_the_script_carries_the_apps_function_not_a_copy(self):
        script = model_export.build_training_script(self._result(with_copies()))
        assert model_export._function_source(modeling.split_keeping_identical_rows_together) in script

    def test_a_script_without_the_option_is_untouched(self):
        result = modeling.train_candidates(
            with_copies(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        assert "split_keeping_identical_rows_together" not in script
        assert "train_test_split(" in script


# ------------------------------------------------------------------ over the API


def _session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverTheApi:
    FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}

    def test_the_plan_offers_it_with_the_count_when_copies_exist(self):
        plan = client.get(f"/api/model/{_session(continuous_with_copies())}/plan").json()
        assert plan["identical_rows"]["copies"] > 0
        assert plan["identical_rows"]["share"] > 0
        assert plan["keep_identical_rows_together"] is True  # pre-selected: it guards against harm

    def test_the_plan_offers_nothing_when_there_are_no_copies(self):
        clean = continuous_with_copies().drop_duplicates(ignore_index=True)
        plan = client.get(f"/api/model/{_session(clean)}/plan").json()
        assert plan["identical_rows"] is None
        assert plan["keep_identical_rows_together"] is False

    def test_training_with_it_leaves_no_copies_in_the_test_set(self):
        session_id = _session(continuous_with_copies())
        body = client.post(
            f"/api/model/{session_id}/train", json={**self.FAST, "keep_identical_rows_together": True}
        ).json()
        assert body["split"]["test_rows_seen_in_training"] == 0
        assert body["split"]["keep_duplicates_together"] is True

    def test_not_asking_leaves_the_server_default_alone(self):
        """The API default is off: only the form pre-selects it."""
        session_id = _session(continuous_with_copies())
        body = client.post(f"/api/model/{session_id}/train", json=self.FAST).json()
        assert body["split"]["keep_duplicates_together"] is False
        assert body["split"]["test_rows_seen_in_training"] > 0

    def test_the_choice_is_counted_as_its_own_split_rule(self):
        session_id = _session(continuous_with_copies())
        directory = session_dir(get_settings().upload_dir, session_id)
        client.post(f"/api/model/{session_id}/train", json=self.FAST)
        client.post(f"/api/model/{session_id}/train", json={**self.FAST, "keep_identical_rows_together": True})
        keys_seen = {e["split_key"] for e in ledger.looks(directory)}
        assert len(keys_seen) == 2

    def test_the_plan_comes_back_showing_what_the_last_run_used(self):
        session_id = _session(continuous_with_copies())
        client.post(f"/api/model/{session_id}/train", json=self.FAST)  # without
        plan = client.get(f"/api/model/{session_id}/plan").json()
        assert plan["keep_identical_rows_together"] is False
        client.post(f"/api/model/{session_id}/train", json={**self.FAST, "keep_identical_rows_together": True})
        assert client.get(f"/api/model/{session_id}/plan").json()["keep_identical_rows_together"] is True
