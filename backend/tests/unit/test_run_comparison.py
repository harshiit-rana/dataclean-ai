"""Telling a real change between two runs from noise.

The history coloured any difference above 0.0005 green or red, so a change well inside what the
test rows could produce by chance read as "this helped" -- the one question the table exists to
answer honestly. A paired comparison scores both runs on the same resampled rows, which cancels
what is shared between them and is far more sensitive than comparing two independent intervals.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import model_store, uncertainty
from app.services.session_store import session_dir

client = TestClient(app)


class TestThePairedDifference:
    @staticmethod
    def _two_models(rows: int = 400, fixed: int = 12, seed: int = 1):
        """B is A with `fixed` of its mistakes corrected: strongly correlated, and a small but
        unambiguous improvement on every row that changed."""
        rng = np.random.default_rng(seed)
        truth = np.where(rng.random(rows) < 0.5, "yes", "no")
        a = truth.copy()
        wrong = rng.choice(rows, 60, replace=False)
        a[wrong] = np.where(truth[wrong] == "yes", "no", "yes")
        b = a.copy()
        b[wrong[:fixed]] = truth[wrong[:fixed]]
        return truth, a, b

    def test_identical_predictions_differ_by_exactly_nothing(self):
        truth, a, _ = self._two_models()
        found = uncertainty.paired_difference(
            truth, {"called": a, "scores": None}, {"called": a, "scores": None}, "classification", "accuracy"
        )
        assert found == {"difference": 0.0, "low": 0.0, "high": 0.0}

    def test_a_small_consistent_improvement_is_seen_by_the_paired_interval(self):
        truth, a, b = self._two_models()
        found = uncertainty.paired_difference(
            truth, {"called": b, "scores": None}, {"called": a, "scores": None}, "classification", "accuracy"
        )
        assert found["difference"] == pytest.approx(12 / 400)
        assert found["low"] > 0

    def test_two_independent_intervals_on_the_same_data_would_have_missed_it(self):
        """The reason to pair: the same improvement, and the ranges still overlap."""
        truth, a, b = self._two_models()
        first = uncertainty.metric_intervals(truth, a, None, "classification", "yes")["metrics"]["accuracy"]
        second = uncertainty.metric_intervals(truth, b, None, "classification", "yes")["metrics"]["accuracy"]
        assert first["high"] > second["low"]  # they overlap, so intervals alone say "noise"

    def test_a_change_that_helps_some_rows_and_hurts_as_many_is_within_the_noise(self):
        rng = np.random.default_rng(2)
        truth = np.where(rng.random(400) < 0.5, "yes", "no")
        a = np.where(rng.random(400) < 0.8, truth, np.where(truth == "yes", "no", "yes"))
        b = np.where(rng.random(400) < 0.8, truth, np.where(truth == "yes", "no", "yes"))
        found = uncertainty.paired_difference(
            truth, {"called": b, "scores": None}, {"called": a, "scores": None}, "classification", "accuracy"
        )
        assert found["low"] < 0 < found["high"]

    def test_it_pairs_roc_auc_on_scores(self):
        rng = np.random.default_rng(3)
        truth = np.where(rng.random(500) < 0.4, "yes", "no")
        noise = rng.normal(0, 1, 500)
        weak = noise + (truth == "yes") * 0.3
        strong = noise + (truth == "yes") * 1.2
        found = uncertainty.paired_difference(
            truth,
            {"called": np.where(strong > 0.5, "yes", "no"), "scores": strong},
            {"called": np.where(weak > 0.5, "yes", "no"), "scores": weak},
            "classification", "roc_auc", positive_label="yes",
        )
        assert found["low"] > 0

    def test_it_pairs_regression_metrics(self):
        rng = np.random.default_rng(4)
        truth = rng.normal(0, 1, 300)
        good = truth + rng.normal(0, 0.3, 300)
        bad = truth + rng.normal(0, 1.0, 300)
        found = uncertainty.paired_difference(
            truth, {"called": good, "scores": None}, {"called": bad, "scores": None}, "regression", "r2"
        )
        assert found["low"] > 0

    def test_it_is_deterministic(self):
        truth, a, b = self._two_models()
        args = (truth, {"called": b, "scores": None}, {"called": a, "scores": None}, "classification", "accuracy")
        assert uncertainty.paired_difference(*args) == uncertainty.paired_difference(*args)

    def test_a_metric_neither_run_has_is_none(self):
        truth, a, b = self._two_models()
        assert uncertainty.paired_difference(
            truth, {"called": b, "scores": None}, {"called": a, "scores": None}, "classification", "roc_auc"
        ) is None


def run_entry(run_id, *, interval=None, split_key="k", fingerprint="f", metric="roc_auc", task="classification"):
    return {
        "run_id": run_id, "primary_metric": metric, "task": task, "split_key": split_key,
        "data_fingerprint": fingerprint,
        "metric_interval": ({"metric": metric, **interval} if interval else None),
    }


def keep(directory, run_id, truth, called, scores, row_ids=None):
    model_store.save_run_predictions(
        directory, run_id,
        {
            "row_ids": np.arange(len(truth)) if row_ids is None else row_ids,
            "truth": truth, "called": called, "scores": scores, "groups": None,
            "positive_label": "yes",
        },
    )


class TestComparingTwoRuns:
    @pytest.fixture
    def rows(self):
        rng = np.random.default_rng(5)
        truth = np.where(rng.random(400) < 0.4, "yes", "no")
        noise = rng.normal(0, 1, 400)
        return truth, noise

    def test_two_runs_on_the_same_rows_are_compared_row_for_row(self, tmp_path, rows):
        truth, noise = rows
        weak, strong = noise + (truth == "yes") * 0.3, noise + (truth == "yes") * 1.2
        keep(tmp_path, "old", truth, np.where(weak > 0.5, "yes", "no"), weak)
        keep(tmp_path, "new", truth, np.where(strong > 0.5, "yes", "no"), strong)
        result = model_store.compare_runs(tmp_path, run_entry("new"), run_entry("old"))
        assert result["method"] == "paired" and result["verdict"] == "better"
        assert result["low"] > 0 and result["difference"] > 0

    def test_a_worse_run_is_called_worse(self, tmp_path, rows):
        truth, noise = rows
        weak, strong = noise + (truth == "yes") * 0.3, noise + (truth == "yes") * 1.2
        keep(tmp_path, "old", truth, np.where(strong > 0.5, "yes", "no"), strong)
        keep(tmp_path, "new", truth, np.where(weak > 0.5, "yes", "no"), weak)
        assert model_store.compare_runs(tmp_path, run_entry("new"), run_entry("old"))["verdict"] == "worse"

    def test_the_same_model_twice_is_within_the_noise(self, tmp_path, rows):
        truth, noise = rows
        scores = noise + (truth == "yes") * 0.8
        called = np.where(scores > 0.5, "yes", "no")
        keep(tmp_path, "old", truth, called, scores)
        keep(tmp_path, "new", truth, called, scores)
        result = model_store.compare_runs(tmp_path, run_entry("new"), run_entry("old"))
        assert result["verdict"] == "within_the_noise" and result["difference"] == 0.0

    def test_the_rows_may_arrive_in_a_different_order(self, tmp_path, rows):
        truth, noise = rows
        weak, strong = noise + (truth == "yes") * 0.3, noise + (truth == "yes") * 1.2
        order = np.random.default_rng(6).permutation(400)
        keep(tmp_path, "old", truth[order], np.where(weak > 0.5, "yes", "no")[order], weak[order], row_ids=order)
        keep(tmp_path, "new", truth, np.where(strong > 0.5, "yes", "no"), strong)
        result = model_store.compare_runs(tmp_path, run_entry("new"), run_entry("old"))
        assert result["method"] == "paired" and result["verdict"] == "better"

    def test_a_different_split_rule_cannot_be_paired_so_the_intervals_are_compared(self, tmp_path, rows):
        truth, noise = rows
        scores = noise + (truth == "yes") * 0.8
        keep(tmp_path, "old", truth, np.where(scores > 0.5, "yes", "no"), scores)
        keep(tmp_path, "new", truth, np.where(scores > 0.5, "yes", "no"), scores)
        result = model_store.compare_runs(
            tmp_path,
            run_entry("new", interval={"low": 0.80, "high": 0.90}, split_key="a"),
            run_entry("old", interval={"low": 0.60, "high": 0.70}, split_key="b"),
        )
        assert result == {"method": "intervals", "verdict": "better"}

    def test_overlapping_intervals_are_within_the_noise(self, tmp_path):
        result = model_store.compare_runs(
            tmp_path,
            run_entry("new", interval={"low": 0.70, "high": 0.84}, split_key="a"),
            run_entry("old", interval={"low": 0.66, "high": 0.80}, split_key="b"),
        )
        assert result == {"method": "intervals", "verdict": "within_the_noise"}

    def test_a_clear_drop_between_intervals_is_worse(self, tmp_path):
        result = model_store.compare_runs(
            tmp_path,
            run_entry("new", interval={"low": 0.55, "high": 0.65}, split_key="a"),
            run_entry("old", interval={"low": 0.75, "high": 0.85}, split_key="b"),
        )
        assert result["verdict"] == "worse"

    def test_runs_with_nothing_to_judge_by_get_no_verdict_at_all(self, tmp_path):
        """Not the old fixed threshold: no evidence means no colour."""
        result = model_store.compare_runs(tmp_path, run_entry("new"), run_entry("old"))
        assert result == {"method": None, "verdict": None}

    def test_runs_trained_on_different_data_are_never_paired(self, tmp_path, rows):
        truth, noise = rows
        scores = noise + (truth == "yes") * 0.8
        keep(tmp_path, "old", truth, np.where(scores > 0.5, "yes", "no"), scores)
        keep(tmp_path, "new", truth, np.where(scores > 0.5, "yes", "no"), scores)
        result = model_store.compare_runs(
            tmp_path, run_entry("new", fingerprint="after"), run_entry("old", fingerprint="before")
        )
        assert result["method"] is None

    def test_missing_prediction_files_fall_back_rather_than_fail(self, tmp_path):
        result = model_store.compare_runs(
            tmp_path,
            run_entry("new", interval={"low": 0.8, "high": 0.9}),
            run_entry("old", interval={"low": 0.5, "high": 0.6}),
        )
        assert result == {"method": "intervals", "verdict": "better"}


class TestThePredictionsAreKept:
    def test_a_run_leaves_its_predictions_beside_the_history(self, tmp_path):
        keep(tmp_path, "abc", np.array(["yes", "no"]), np.array(["yes", "yes"]), np.array([0.9, 0.6]))
        loaded = model_store.load_run_predictions(tmp_path, "abc")
        assert loaded["truth"].tolist() == ["yes", "no"] and loaded["scores"].tolist() == [0.9, 0.6]

    def test_clearing_the_model_keeps_them_because_the_history_is_kept(self, tmp_path):
        keep(tmp_path, "abc", np.array(["yes", "no"]), np.array(["yes", "no"]), None)
        model_store.clear(tmp_path)
        assert model_store.load_run_predictions(tmp_path, "abc") is not None

    def test_a_run_that_falls_off_the_history_takes_its_predictions_with_it(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_store, "MAX_HISTORY_ENTRIES", 2)
        for run_id in ("a", "b", "c"):
            keep(tmp_path, run_id, np.array(["yes"]), np.array(["yes"]), None)
            model_store.append_history(tmp_path, {"run_id": run_id})
        assert model_store.load_run_predictions(tmp_path, "a") is None
        assert model_store.load_run_predictions(tmp_path, "c") is not None

    def test_an_unreadable_file_is_none_not_an_error(self, tmp_path):
        (tmp_path / model_store.PREDICTIONS_DIR).mkdir()
        (tmp_path / model_store.PREDICTIONS_DIR / "bad.npz").write_bytes(b"not a zip")
        assert model_store.load_run_predictions(tmp_path, "bad") is None


# ------------------------------------------------------------------ over the API


def _session(n: int = 500) -> str:
    rng = np.random.default_rng(17)
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "signal": signal.round(4),
            "noise": rng.normal(0, 1, n).round(4),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


def _train(session_id, **extra):
    body = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False, **extra}
    response = client.post(f"/api/model/{session_id}/train", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def _history(session_id):
    return client.get(f"/api/model/{session_id}/history").json()


class TestOverTheApi:
    def test_a_run_keeps_its_predictions_and_its_split_rule(self):
        session_id = _session()
        run = _train(session_id)
        directory = session_dir(get_settings().upload_dir, session_id)
        assert (directory / model_store.PREDICTIONS_DIR / f"{run['run_id']}.npz").exists()
        assert _history(session_id)["runs"][0]["split_key"]

    def test_the_same_configuration_twice_is_within_the_noise(self):
        session_id = _session()
        _train(session_id)
        second = _train(session_id)
        history = _history(session_id)
        comparison = history["comparisons"][second["run_id"]]
        assert comparison["method"] == "paired"
        assert comparison["verdict"] == "within_the_noise" and comparison["difference"] == 0.0

    def test_dropping_the_one_column_that_matters_is_called_worse(self):
        session_id = _session()
        _train(session_id)
        worse = _train(session_id, excluded_columns=["signal"])
        comparison = _history(session_id)["comparisons"][worse["run_id"]]
        assert comparison["verdict"] == "worse" and comparison["high"] < 0

    def test_putting_it_back_is_called_better(self):
        session_id = _session()
        _train(session_id, excluded_columns=["signal"])
        better = _train(session_id)
        assert _history(session_id)["comparisons"][better["run_id"]]["verdict"] == "better"

    def test_the_first_run_has_nothing_to_compare_with(self):
        session_id = _session()
        first = _train(session_id)
        assert first["run_id"] not in _history(session_id)["comparisons"]

    def test_a_run_from_before_a_cleaning_change_is_not_compared(self):
        from app.services.session_store import record_dataset_version_in

        session_id = _session()
        directory = session_dir(get_settings().upload_dir, session_id)
        _train(session_id)
        pd.read_csv(directory / "original.csv").iloc[:-30].to_csv(directory / "cleaned.csv", index=False)
        record_dataset_version_in(directory, "cleaning_run")
        model_store.clear(directory)
        after = _train(session_id)
        assert after["run_id"] not in _history(session_id)["comparisons"]

    def test_an_old_history_entry_without_a_split_rule_falls_back_to_intervals_or_nothing(self):
        session_id = _session()
        _train(session_id)
        second = _train(session_id)
        directory = session_dir(get_settings().upload_dir, session_id)
        import json

        runs = model_store.load_history(directory)
        for run in runs:
            run.pop("split_key", None)
        (directory / model_store.HISTORY_FILENAME).write_text(json.dumps(runs))
        comparison = _history(session_id)["comparisons"][second["run_id"]]
        assert comparison["method"] == "intervals"  # the stored intervals overlap
        assert comparison["verdict"] == "within_the_noise"
