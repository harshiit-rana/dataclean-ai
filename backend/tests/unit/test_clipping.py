"""Clipping extreme values inside the pipeline, with limits learned from training rows.

Capping outliers while cleaning reads its limits from every row, test rows included, before any
split. This is the same job where it cannot leak. The properties that matter: the limits come
from what `fit` is handed, it is off unless asked for, and the exported script rebuilds exactly
the same model -- a recipe the export cannot rebuild is a model trapped in the app.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import clipping, model_card, model_export, modeling
from app.services.clipping import IQRClipper

client = TestClient(app)


# ------------------------------------------------------------------ the transformer


class TestTheTransformer:
    def test_it_learns_its_limits_from_what_fit_is_given(self):
        rng = np.random.default_rng(0)
        train = rng.normal(0, 1, (300, 1))
        clipper = IQRClipper().fit(train)

        q1, q3 = np.percentile(train[:, 0], [25, 75])
        assert clipper.lower_[0] == pytest.approx(q1 - 1.5 * (q3 - q1))
        assert clipper.upper_[0] == pytest.approx(q3 + 1.5 * (q3 - q1))

    def test_it_applies_the_learned_limits_to_other_data_not_its_own(self):
        """transform never re-learns: a wildly different batch is held to the training limits."""
        clipper = IQRClipper().fit(np.random.default_rng(0).normal(0, 1, (300, 1)))
        out = clipper.transform(np.array([[500.0], [-500.0], [0.1]]))
        assert out[0, 0] == pytest.approx(clipper.upper_[0])
        assert out[1, 0] == pytest.approx(clipper.lower_[0])
        assert out[2, 0] == pytest.approx(0.1)
        assert clipper.upper_[0] < 10

    def test_blanks_stay_blank_for_the_imputer_after_it(self):
        clipper = IQRClipper().fit(np.random.default_rng(0).normal(0, 1, (100, 1)))
        out = clipper.transform(np.array([[np.nan], [1000.0]]))
        assert np.isnan(out[0, 0]) and np.isfinite(out[1, 0])

    def test_blanks_do_not_count_towards_the_limits(self):
        values = np.r_[np.random.default_rng(1).normal(0, 1, 200), np.full(100, np.nan)]
        clipper = IQRClipper().fit(values.reshape(-1, 1))
        assert np.isfinite(clipper.lower_[0]) and np.isfinite(clipper.upper_[0])

    def test_a_column_with_no_spread_passes_through_untouched(self):
        """Limits of a single point would clip every different value in to it."""
        clipper = IQRClipper().fit(np.full((50, 1), 7.0))
        out = clipper.transform(np.array([[7.0], [900.0]]))
        assert out[:, 0].tolist() == [7.0, 900.0]

    def test_a_column_with_too_few_values_passes_through(self):
        clipper = IQRClipper().fit(np.array([[1.0], [2.0], [np.nan]]))
        assert clipper.transform(np.array([[99.0]]))[0, 0] == 99.0

    def test_each_column_gets_its_own_limits(self):
        rng = np.random.default_rng(2)
        X = np.c_[rng.normal(0, 1, 200), rng.normal(1000, 1, 200)]
        clipper = IQRClipper().fit(X)
        assert clipper.upper_[0] < 10 and clipper.lower_[1] > 900

    def test_it_keeps_the_column_names_for_the_importance_chart(self):
        frame = pd.DataFrame({"a": np.arange(40.0), "b": np.arange(40.0) % 7})
        clipper = IQRClipper().fit(frame)
        assert list(clipper.get_feature_names_out()) == ["a", "b"]
        assert list(clipper.get_feature_names_out(["p", "q"])) == ["p", "q"]

    def test_it_is_a_valid_estimator_for_cloning(self):
        from sklearn.base import clone

        assert clone(IQRClipper(whisker=3.0)).whisker == 3.0

    def test_a_clipped_value_is_pulled_back_not_removed(self):
        values = np.r_[np.random.default_rng(3).normal(0, 1, 99), 1000.0].reshape(-1, 1)
        out = IQRClipper().fit_transform(values)
        assert out.shape == values.shape  # every row kept


class TestWhenItIsOffered:
    """A bell-shaped column has a few values past 1.5 x IQR by chance. Offering clipping for
    every column that does would pad the plan with offers that catch nothing, and teach people
    to tick without reading."""

    def test_ordinary_bell_shaped_columns_are_not_offered(self):
        rng = np.random.default_rng(0)
        offered = [
            clipping.outliers_in(pd.Series(rng.normal(0, 1, 2000))) for _ in range(20)
        ]
        assert not any(offered)

    def test_a_skewed_column_is_offered(self):
        rng = np.random.default_rng(1)
        assert clipping.outliers_in(pd.Series(rng.lognormal(3, 1.4, 1000))) is not None

    def test_one_wild_value_is_enough(self):
        values = np.r_[np.random.default_rng(2).normal(50, 5, 300), 9000.0]
        found = clipping.outliers_in(pd.Series(values))
        assert found["far_out"] == 1 and found["highest"] == 9000.0

    def test_a_constant_or_tiny_column_is_never_offered(self):
        assert clipping.outliers_in(pd.Series([5.0] * 50)) is None
        assert clipping.outliers_in(pd.Series([1.0, 2.0, 900.0])) is None

    def test_blanks_are_ignored(self):
        values = pd.Series([np.nan] * 40 + list(np.random.default_rng(3).normal(0, 1, 200)))
        assert clipping.outliers_in(values) is None

    def test_it_does_not_look_at_anything_but_the_column(self):
        """No target argument exists to read, and the result is a function of the values."""
        series = pd.Series(np.r_[np.random.default_rng(4).normal(0, 1, 100), 80.0])
        assert clipping.outliers_in(series) == clipping.outliers_in(series.sample(frac=1, random_state=1))


# ------------------------------------------------------------------ in the pipeline


@pytest.fixture
def dated_with_late_extremes() -> pd.DataFrame:
    """Under a date split the last fifth is the test set. The extremes live only there, so a
    limit computed over every row sees them and a limit learned on the training rows cannot."""
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(0, 1, n)
    x[-100:][rng.random(100) < 0.4] = 500.0
    z = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "when": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
            "x": x,
            "z": z,
            "region": rng.choice(["a", "b"], n),
            "y": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * z)), "yes", "no"),
        }
    )


def _train(df: pd.DataFrame, **kwargs) -> dict:
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False, **kwargs,
    )


def _clip_step(result: dict):
    pipeline = result["results"][0]["pipeline"]
    return pipeline.named_steps["prepare"].named_transformers_["numeric_clipped"].named_steps["clip"]


class TestInThePipeline:
    def test_the_limits_come_from_the_training_rows_only(self, dated_with_late_extremes):
        df = dated_with_late_extremes
        result = _train(df, clip_outliers=["x"], split_strategy="time", time_column="when")

        X_train, X_test, _, _ = modeling.split_data(
            df, "y", "classification", strategy="time", time_column="when"
        )
        q1, q3 = np.percentile(X_train["x"], [25, 75])
        expected_upper = q3 + 1.5 * (q3 - q1)
        assert _clip_step(result).upper_[0] == pytest.approx(expected_upper)

        # The fixture has to be able to fail: over every row the limit is something else.
        q1_all, q3_all = np.percentile(df["x"], [25, 75])
        assert q3_all + 1.5 * (q3_all - q1_all) != pytest.approx(expected_upper, rel=0.05)
        assert _clip_step(result).upper_[0] < 10

    def test_flipping_the_extremes_in_the_test_rows_does_not_move_the_limits(
        self, dated_with_late_extremes
    ):
        df = dated_with_late_extremes
        calm = df.copy()
        calm.loc[calm.index[-100:], "x"] = 0.0
        a = _clip_step(_train(df, clip_outliers=["x"], split_strategy="time", time_column="when"))
        b = _clip_step(_train(calm, clip_outliers=["x"], split_strategy="time", time_column="when"))
        assert a.upper_.tolist() == b.upper_.tolist()
        assert a.lower_.tolist() == b.lower_.tolist()

    def test_it_is_off_unless_asked_for(self, dated_with_late_extremes):
        result = _train(dated_with_late_extremes)
        assert result["features"]["clipped"] == []
        steps = result["results"][0]["pipeline"].named_steps["prepare"].named_transformers_
        assert "numeric_clipped" not in steps

    def test_a_name_that_is_not_a_numeric_input_is_ignored_not_fatal(
        self, dated_with_late_extremes
    ):
        result = _train(dated_with_late_extremes, clip_outliers=["region", "nonsense", "y"])
        assert result["features"]["clipped"] == []

    def test_unclipped_numeric_columns_keep_the_plain_branch(self, dated_with_late_extremes):
        result = _train(dated_with_late_extremes, clip_outliers=["x"])
        steps = result["results"][0]["pipeline"].named_steps["prepare"].named_transformers_
        assert steps["numeric"].feature_names_in_.tolist() == ["z"]
        assert steps["numeric_clipped"].feature_names_in_.tolist() == ["x"]

    def test_importance_still_resolves_to_the_source_column(self, dated_with_late_extremes):
        result = _train(dated_with_late_extremes, clip_outliers=["x"])
        columns = {entry["column"] for entry in result["results"][0]["column_importance"]}
        assert {"x", "z"} <= columns

    def test_clipping_a_column_with_extremes_changes_the_fit(self, dated_with_late_extremes):
        plain = _train(dated_with_late_extremes)["results"][0]["pipeline"]
        clipped = _train(dated_with_late_extremes, clip_outliers=["x"])["results"][0]["pipeline"]
        assert plain.named_steps["model"].coef_.tolist() != clipped.named_steps["model"].coef_.tolist()


class TestTheCardQuotesTheLimits:
    def test_the_result_records_the_limits_the_step_learned(self, dated_with_late_extremes):
        result = _train(dated_with_late_extremes, clip_outliers=["x"])
        step = _clip_step(result)
        limits = result["features"]["clip_limits"]["x"]
        assert limits["lower"] == pytest.approx(step.lower_[0], abs=1e-5)
        assert limits["upper"] == pytest.approx(step.upper_[0], abs=1e-5)

    def test_no_limits_are_recorded_when_nothing_was_clipped(self, dated_with_late_extremes):
        assert "clip_limits" not in _train(dated_with_late_extremes)["features"]

    def test_the_card_names_the_column_and_the_numbers(self, dated_with_late_extremes):
        result = _train(dated_with_late_extremes, clip_outliers=["x"])
        card = model_card.build_model_card(session_id="s", file_name="f.csv", result=result)
        assert "## Extreme values clipped" in card
        assert "training rows only" in card
        upper = result["features"]["clip_limits"]["x"]["upper"]
        assert f"{upper:g}" in card

    def test_the_card_says_nothing_when_nothing_was_clipped(self, dated_with_late_extremes):
        card = model_card.build_model_card(session_id="s", file_name="f.csv", result=_train(dated_with_late_extremes))
        assert "Extreme values clipped" not in card

    def test_a_column_with_no_spread_is_reported_as_unlimited_not_infinite(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "flat": np.r_[np.full(290, 3.0), np.full(10, 3.0)],
                "z": rng.normal(0, 1, 300),
                "y": rng.choice(["yes", "no"], 300),
            }
        )
        # "flat" is dropped as constant, so clip a column that survives with no spread:
        df["near_flat"] = np.r_[np.full(295, 1.0), np.full(5, 2.0)]
        result = _train(df, clip_outliers=["near_flat"])
        entry = result["features"]["clip_limits"]["near_flat"]
        assert entry == {"lower": None, "upper": None}
        assert "no limit" in model_card.build_model_card(session_id="s", file_name="f.csv", result=result)


# ------------------------------------------------------------------ the export


class TestTheExportRebuildsIt:
    @staticmethod
    def _data() -> pd.DataFrame:
        rng = np.random.default_rng(8)
        n = 500
        x = rng.normal(0, 1, n)
        x[rng.random(n) < 0.03] = 300.0
        z = rng.normal(0, 1, n)
        return pd.DataFrame(
            {
                "x": x.round(4),
                "z": z.round(4),
                "region": rng.choice(["a", "b"], n),
                "y": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * z - 0.01 * np.clip(x, -5, 5))), "yes", "no"),
            }
        )

    def _run(self, tmp_path, df, result):
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), "utf-8")
        completed = subprocess.run(
            [sys.executable, str(script), str(csv)],
            capture_output=True, text=True, cwd=tmp_path, timeout=600,
        )
        assert completed.returncode == 0, f"{completed.stdout}\n{completed.stderr}"
        return completed.stdout

    def test_the_script_reproduces_the_reported_metrics_with_clipping_on(self, tmp_path):
        df = self._data()
        result = _train(df, clip_outliers=["x"])
        result["features"]["excluded"] = []
        output = self._run(tmp_path, df, result)

        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, output
        assert float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["accuracy"], abs=1e-4
        )

    def test_the_script_carries_the_apps_class_not_a_second_copy(self):
        result = _train(self._data(), clip_outliers=["x"])
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        assert clipping.source().strip() in script
        assert "CLIPPED_FEATURES = ['x']" in script
        assert '"numeric_clipped"' in script

    def test_a_script_with_clipping_off_has_none_of_it(self):
        result = _train(self._data())
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        assert "IQRClipper" not in script
        # No numpy import: the clipping step is what needs it. (The versions header names numpy among the
        # libraries that produced the numbers, which is not using it.)
        assert "import numpy" not in script
        assert "CLIPPED_FEATURES = []" in script

    def test_the_prediction_snippet_warns_about_the_extra_step(self):
        result = _train(self._data(), clip_outliers=["x"])
        assert "IQRClipper" in model_export.build_prediction_snippet(result)
        plain = _train(self._data())
        assert "IQRClipper" not in model_export.build_prediction_snippet(plain)

    def test_the_saved_model_loads_in_a_fresh_process_and_predicts(self, tmp_path):
        df = self._data()
        result = _train(df, clip_outliers=["x"])
        path = tmp_path / "model.joblib"
        joblib.dump(result["results"][0]["pipeline"], path)
        sample = tmp_path / "sample.csv"
        df.head(5).drop(columns=["y"]).to_csv(sample, index=False)

        backend = __import__("pathlib").Path(modeling.__file__).resolve().parents[2]
        code = (
            "import sys, joblib, pandas as pd; p = joblib.load(sys.argv[1]); "
            "print(list(p.predict(pd.read_csv(sys.argv[2]))))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code, str(path), str(sample)],
            cwd=backend, capture_output=True, text=True, check=True,
        ).stdout
        assert out.strip().startswith("[")


# ------------------------------------------------------------------ over the API


def _csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(df: pd.DataFrame) -> str:
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(_csv(df)), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "predict y", "target_column": "y"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


def _frame(with_extremes: bool) -> pd.DataFrame:
    rng = np.random.default_rng(14)
    n = 300
    x = rng.normal(50, 5, n).round(2)
    if with_extremes:
        x[:4] = [400.0, 350.0, -200.0, 500.0]
    z = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "z": z.round(4),
            "y": np.where(rng.random(n) < 1 / (1 + np.exp(-2 * z)), "yes", "no"),
        }
    )


class TestOverTheApi:
    def test_the_plan_offers_columns_with_extremes_and_ticks_none(self):
        plan = client.get(f"/api/model/{_session(_frame(True))}/plan").json()
        offered = {o["column"]: o for o in plan["suggested_clipping"]}
        assert "x" in offered and "z" not in offered
        assert offered["x"]["evidence"]["far_out"] == 4
        assert "largest 500" in offered["x"]["reason"]
        assert plan["features"]["clipped"] == []

    def test_the_offer_names_values_from_the_column_not_a_threshold(self):
        plan = client.get(f"/api/model/{_session(_frame(True))}/plan").json()
        reason = next(o["reason"] for o in plan["suggested_clipping"] if o["column"] == "x")
        assert "fence" not in reason.lower() and "IQR" not in reason

    def test_clean_data_offers_nothing(self):
        plan = client.get(f"/api/model/{_session(_frame(False))}/plan").json()
        assert plan["suggested_clipping"] == []

    def test_training_with_it_records_it_and_the_plan_remembers(self):
        session_id = _session(_frame(True))
        response = client.post(
            f"/api/model/{session_id}/train",
            json={
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
                "clip_outliers": ["x"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["features"]["clipped"] == ["x"]

        history = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert history[-1]["clipped_columns"] == ["x"]
        assert client.get(f"/api/model/{session_id}/plan").json()["features"]["clipped"] == ["x"]

    def test_training_without_it_clips_nothing(self):
        session_id = _session(_frame(True))
        body = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        ).json()
        assert body["features"]["clipped"] == []

    def test_the_improvement_loop_keeps_it(self):
        session_id = _session(_frame(True))
        response = client.post(
            f"/api/model/{session_id}/improve",
            json={
                "model_names": ["logistic_regression"],
                "include_deep_diagnostics": False,
                "clip_outliers": ["x"],
                "budget": 4,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["features"]["clipped"] == ["x"]
