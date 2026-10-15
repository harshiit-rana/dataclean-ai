"""An embargo between the training and test periods of a date split.

When the answer is defined over a future window -- "churned within 30 days" -- a row just before the
cut has an answer partly decided by what happens in the test period, so a model trained on it has been
shown something about the test period. Leaving a gap removes that overlap: training stops at least that
many days before the first test row, at the train/test cut and inside every forward-in-time fold.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import evaluation_ledger, model_export, modeling, operating_point, tuning
from app.services.operating_point import OperatingPoint
from app.services.time_folds import ForwardInTimeFolds

client = TestClient(app)
DAY = pd.Timedelta(days=1)


def daily(n: int = 700, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"day": pd.date_range("2021-01-01", periods=n).astype(str), "x": x,
         "y": np.where(x + rng.normal(0, 1, n) > 0.3, "yes", "no")}
    )


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False, split_strategy="time", time_column="day", **kwargs,
    )


def span(X_train, X_test):
    return pd.to_datetime(X_test["day"]).min() - pd.to_datetime(X_train["day"]).max()


class TestTheCut:
    def test_with_a_thirty_day_gap_training_ends_at_least_thirty_days_before_the_test_period(self):
        X_train, X_test, _, _ = modeling.split_data(daily(), "y", "classification", strategy="time",
                                                    time_column="day", time_gap_days=30)
        assert span(X_train, X_test) >= 30 * DAY

    def test_without_one_the_two_periods_touch(self):
        X_train, X_test, _, _ = modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day")
        assert span(X_train, X_test) == DAY

    def test_the_gap_costs_training_rows_and_nothing_else(self):
        plain = modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day")
        gapped = modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day", time_gap_days=30)
        assert list(gapped[1].index) == list(plain[1].index)  # the test period does not move
        assert len(gapped[0]) == len(plain[0]) - 29
        assert set(gapped[0].index) < set(plain[0].index)

    def test_a_zero_gap_is_no_gap(self):
        a = modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day")
        b = modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day", time_gap_days=0)
        assert all(list(x.index) == list(y.index) for x, y in zip(a, b))

    def test_a_gap_that_leaves_too_little_to_train_on_is_refused_with_the_numbers(self):
        with pytest.raises(modeling.ModelingError) as caught:
            modeling.split_data(daily(), "y", "classification", strategy="time", time_column="day", time_gap_days=900)
        message = str(caught.value)
        assert "A gap of 900 days" in message and "leaves only" in message and "spans 699 days" in message

    def test_it_applies_to_a_date_split_only(self):
        result = modeling.train_candidates(
            daily(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, time_gap_days=30,
        )
        assert result["split"]["gap_days"] == 0 and result["split"]["gap_rows_dropped"] == 0

    def test_a_sealed_holdout_keeps_the_gap_between_training_and_development(self):
        parts = modeling.split_with_sealed_holdout(daily(1000), "y", "classification", strategy="time",
                                                   time_column="day", time_gap_days=30)
        X_train, X_dev = parts[0], parts[1]
        assert span(X_train, X_dev) >= 30 * DAY


class TestTheFolds:
    def folds_of(self, monkeypatch, **kwargs):
        made = []
        real = tuning.cv_splitter

        def spy(*a, **k):
            splitter = real(*a, **k)
            made.append(splitter)
            return splitter

        monkeypatch.setattr(tuning, "cv_splitter", spy)
        monkeypatch.setattr(operating_point, "cv_splitter", spy)
        result = modeling.train_candidates(
            daily(), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"],
            include_deep_diagnostics=False, split_strategy="time", time_column="day", **kwargs,
        )
        return result, made

    def test_every_fold_of_model_choice_respects_the_gap(self, monkeypatch):
        _, made = self.folds_of(monkeypatch, time_gap_days=30)
        df = daily()
        X_train, _, _, _ = modeling.split_data(df, "y", "classification", strategy="time",
                                               time_column="day", time_gap_days=30)
        stamps = pd.to_datetime(df["day"])
        assert made and all(isinstance(s, ForwardInTimeFolds) and s.gap_days == 30 for s in made)
        for splitter in made:
            for train_idx, test_idx in splitter.split(X_train):
                before = stamps.loc[X_train.index[train_idx]].max()
                after = stamps.loc[X_train.index[test_idx]].min()
                assert after - before >= 30 * DAY

    def test_the_cut_off_and_the_calibrator_use_the_same_gap(self, monkeypatch):
        _, made = self.folds_of(monkeypatch, time_gap_days=14, point=OperatingPoint("f1"), calibration="sigmoid")
        assert len(made) > 5 and all(s.gap_days == 14 for s in made)

    def test_the_cross_validated_spread_and_the_learning_curve_use_it(self, monkeypatch):
        import sklearn.model_selection as selection

        seen = []
        for name in ("cross_val_score", "learning_curve"):
            real = getattr(selection, name)
            monkeypatch.setattr(
                selection, name, lambda *a, _real=real, **k: (seen.append(k.get("cv")), _real(*a, **k))[1]
            )
        modeling.train_candidates(
            daily(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=True, split_strategy="time", time_column="day", time_gap_days=21,
        )
        deep = [cv for cv in seen if isinstance(cv, ForwardInTimeFolds)]
        assert deep and all(cv.gap_days == 21 for cv in deep)

    def test_a_run_without_a_gap_has_none_in_its_folds(self, monkeypatch):
        _, made = self.folds_of(monkeypatch)
        assert made and all(s.gap_days == 0 for s in made)

    def test_the_improvement_loop_searches_with_the_gap(self, monkeypatch):
        from app.services import improvement

        made = []
        real = tuning.cv_splitter
        monkeypatch.setattr(tuning, "cv_splitter", lambda *a, **k: (made.append(real(*a, **k)), made[-1])[1])
        improvement.improve(
            daily(), target_column="y", task="classification", model_names=["logistic_regression"],
            budget=4, split_strategy="time", time_column="day", time_gap_days=30,
        )
        assert made and all(isinstance(s, ForwardInTimeFolds) and s.gap_days == 30 for s in made)


class TestWhatIsReported:
    def test_the_rows_the_gap_dropped_are_counted(self):
        result = train(daily(), time_gap_days=30)
        split = result["split"]
        assert split["gap_days"] == 30 and split["gap_rows_dropped"] == 29
        assert split["train_rows"] + split["test_rows"] + split["gap_rows_dropped"] == 700

    def test_no_gap_reports_none_dropped(self):
        split = train(daily())["split"]
        assert split["gap_days"] == 0 and split["gap_rows_dropped"] == 0

    def test_the_gap_is_part_of_the_ledger_key_only_when_there_is_one(self):
        a, b, plain = train(daily(), time_gap_days=30), train(daily(), time_gap_days=14), train(daily())
        assert len({evaluation_ledger.split_key(r["split"]) for r in (a, b, plain)}) == 3

    def test_a_run_without_a_gap_keeps_the_key_it_always_had(self):
        split = train(daily())["split"]
        assert evaluation_ledger.split_key(split) == evaluation_ledger.split_key({**split, "gap_days": 0})
        without_field = {k: v for k, v in split.items() if k not in ("gap_days", "gap_rows_dropped")}
        assert evaluation_ledger.split_key(split) == evaluation_ledger.split_key(without_field)

    def test_the_card_says_so(self):
        from app.services.model_card import build_model_card

        card = build_model_card(session_id="s", file_name="f.csv", result=train(daily(), time_gap_days=30),
                                summary="A.", interpretation="B.")
        assert "leaving a gap of 30 days between training and test (29 rows inside it were dropped)" in card

    def test_the_history_records_it(self):
        from app.services import model_store

        assert model_store.summarise({**train(daily(), time_gap_days=30), "run_id": "r"})["split"]["gap_days"] == 30


def session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


BODY = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False,
        "split_strategy": "time", "time_column": "day"}


class TestOverTheApi:
    def test_train_accepts_it_and_reports_it(self):
        sid = session(daily())
        body = client.post(f"/api/model/{sid}/train", json={**BODY, "time_gap_days": 30}).json()
        assert body["split"]["gap_days"] == 30 and body["split"]["gap_rows_dropped"] == 29

    def test_a_gap_larger_than_the_data_is_refused_with_a_clear_message(self):
        sid = session(daily())
        response = client.post(f"/api/model/{sid}/train", json={**BODY, "time_gap_days": 900})
        assert response.status_code >= 400
        assert "A gap of 900 days" in response.text and "Use a shorter gap" in response.text

    def test_a_negative_or_absurd_gap_is_a_validation_error(self):
        sid = session(daily())
        assert client.post(f"/api/model/{sid}/train", json={**BODY, "time_gap_days": -1}).status_code == 422
        assert client.post(f"/api/model/{sid}/train", json={**BODY, "time_gap_days": 99999}).status_code == 422

    def test_the_improvement_route_holds_it(self):
        sid = session(daily())
        body = client.post(f"/api/model/{sid}/improve", json={**BODY, "budget": 4, "time_gap_days": 30}).json()
        assert body["split"]["gap_days"] == 30

    def test_a_sealed_holdout_can_still_be_opened_after_a_gapped_run(self):
        sid = session(daily(1200))
        trained = client.post(f"/api/model/{sid}/train", json={**BODY, "time_gap_days": 20, "sealed_holdout": True})
        assert trained.status_code == 200 and trained.json()["split"]["sealed"] is True
        opened = client.post(f"/api/model/{sid}/finalise")
        assert opened.status_code == 200, opened.text


class TestTheExportedScript:
    def run(self, tmp_path, df, result):
        (tmp_path / "data.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8")
        done = subprocess.run([sys.executable, str(script), str(tmp_path / "data.csv")],
                              capture_output=True, text=True, cwd=tmp_path, timeout=900)
        assert done.returncode == 0, done.stderr
        return done.stdout

    def prepared(self, **kwargs):
        result = train(daily(), **kwargs)
        result["features"]["excluded"] = []
        return result

    def test_the_script_cuts_where_the_app_cut_with_a_gap(self):
        df = daily()
        result = self.prepared(time_gap_days=30)
        block = model_export._split_block(result["split"], "classification")
        labelled = df[df["y"].notna()]
        namespace = {"pd": pd, "labelled": labelled, "features": labelled.drop(columns=["y"]), "labels": labelled["y"]}
        exec(block, namespace)  # noqa: S102 -- the block is generated text under test
        X_train, X_test, _, _ = modeling.split_data(df, "y", "classification", strategy="time",
                                                    time_column="day", time_gap_days=30)
        assert list(namespace["X_train"].index) == list(X_train.index)
        assert list(namespace["X_test"].index) == list(X_test.index)

    def test_the_script_without_a_gap_is_unchanged(self):
        assert "GAP" not in model_export.build_training_script(self.prepared())

    def test_the_script_declares_the_embargo_and_says_why(self):
        script = model_export.build_training_script(self.prepared(time_gap_days=30))
        assert "GAP = pd.Timedelta(days=30)" in script and "An embargo of 30 days" in script

    def test_it_reproduces_the_reported_accuracy(self, tmp_path):
        df, result = daily(), self.prepared(time_gap_days=30)
        output = self.run(tmp_path, df, result)
        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, output
        assert float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)

    def test_a_calibrated_model_reproduces_it_with_the_gap_in_its_folds(self, tmp_path):
        result = train(daily(900), time_gap_days=30, calibration="sigmoid")
        result["features"]["excluded"] = []
        assert result["calibration"]["method"] == "sigmoid"
        script = model_export.build_training_script(result, file_name="data.csv")
        assert "n_splits=5, gap_days=30)" in script
        output = self.run(tmp_path, daily(900), result)
        brier = re.search(r"Brier score:\s+([0-9.]+)", output)
        assert brier and float(brier.group(1)) == pytest.approx(result["results"][0]["metrics"]["brier_score"], abs=1e-4)
