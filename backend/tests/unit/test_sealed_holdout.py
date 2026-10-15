"""A sealed final holdout: a third partition nothing but an explicit `finalise` ever opens.

Counting looks at the held-out rows (decision 0001) warns when a person has used them up. A
sealed holdout is what to do about it, if it is taken before the first look: rows set aside before
anything is decided, never scored while iterating, opened once. These pin that it is genuinely
sealed -- nothing but `finalise` reads it -- that it keeps every guarantee the ordinary split gives
(whole entities, forward in time, copies together), that it stays fixed however the work around it
changes, and that a file too small to support three parts is told so rather than shortchanged.
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

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import improvement, model_export, model_store, modeling
from app.services.session_store import session_dir

client = TestClient(app)
FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def learnable(n: int = 800, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "x": x.round(4),
            "z": rng.normal(0, 1, n).round(4),
            "when": pd.date_range("2022-01-01", periods=n, freq="D").astype(str),
        }
    )
    df["y"] = np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")
    return df


def daily(rows_per_day=(8, 15), days: int = 80, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.integers(rows_per_day[0], rows_per_day[1], days)
    when = np.repeat(pd.date_range("2022-01-01", periods=days, freq="D"), sizes)
    n = len(when)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "when": when.astype(str),
            "x": x.round(4),
            "y": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no"),
        }
    )


def entities(customers: int = 120, per: int = 8, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, customers)
    return pd.DataFrame(
        {
            "customer": np.repeat([f"c{i:03d}" for i in range(customers)], per),
            "x": (np.repeat(base, per) + rng.normal(0, 0.4, customers * per)).round(4),
            "y": np.repeat(np.where(base + rng.normal(0, 0.8, customers) > 0, "yes", "no"), per),
        }
    )


def three_way(df, **kwargs):
    return modeling.split_with_sealed_holdout(df, "y", "classification", **kwargs)


def indices(parts):
    return [set(parts[0].index), set(parts[1].index), set(parts[2].index)]


class TestThePartitions:
    def test_three_disjoint_parts_in_roughly_seventy_fifteen_fifteen(self):
        df = learnable()
        parts = three_way(df)
        train, dev, final = indices(parts)
        assert not (train & dev) and not (train & final) and not (dev & final)
        assert len(train) + len(dev) + len(final) == len(df)
        assert len(final) / len(df) == pytest.approx(0.15, abs=0.02)
        assert len(dev) / len(df) == pytest.approx(0.15, abs=0.02)

    def test_a_classification_target_is_stratified_in_every_part(self):
        parts = three_way(learnable())
        for y in (parts[3], parts[4], parts[5]):
            assert set(y) == {"yes", "no"}
            share = (y == "yes").mean()
            assert share == pytest.approx((learnable()["y"] == "yes").mean(), abs=0.06)

    def test_no_entity_is_in_more_than_one_part(self):
        df = entities()
        parts = three_way(df, strategy="group", group_column="customer")
        owners = [set(df.loc[list(idx), "customer"]) for idx in indices(parts)]
        assert not (owners[0] & owners[1]) and not (owners[0] & owners[2]) and not (owners[1] & owners[2])

    def test_a_date_split_is_forward_in_time_across_all_three(self):
        df = daily()
        parts = three_way(df, strategy="time", time_column="when")
        newest = lambda X: pd.to_datetime(X["when"]).max()  # noqa: E731
        oldest = lambda X: pd.to_datetime(X["when"]).min()  # noqa: E731
        assert newest(parts[0]) < oldest(parts[1])
        assert newest(parts[1]) < oldest(parts[2])

    def test_a_day_is_never_split_across_parts(self):
        df = daily()
        parts = three_way(df, strategy="time", time_column="when")
        days = [set(X["when"]) for X in parts[:3]]
        assert not (days[0] & days[1]) and not (days[0] & days[2]) and not (days[1] & days[2])

    def test_exact_copies_stay_in_one_part(self):
        rng = np.random.default_rng(6)
        base = pd.DataFrame({"x": rng.normal(0, 1, 300).round(4), "z": rng.normal(0, 1, 300).round(4)})
        base["y"] = np.where(base["x"] + rng.normal(0, 0.7, 300) > 0, "yes", "no")
        df = pd.concat([base, base], ignore_index=True)
        parts = three_way(df, duplicate_columns=["x", "z"], copy_columns=["x", "z"])
        for i in range(300):
            holders = {p for p, idx in enumerate(indices(parts)) if i in idx or i + 300 in idx}
            assert len(holders) == 1, f"row {i} and its copy are in different parts"

    def test_a_regression_target_is_split_too(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({"x": rng.normal(0, 1, 600).round(4)})
        df["price"] = (df["x"] * 3).round(3)
        parts = modeling.split_with_sealed_holdout(df, "price", "regression")
        assert [len(p) for p in parts[:3]] == [len(parts[3]), len(parts[4]), len(parts[5])]


class TestItStaysFixed:
    def test_the_same_data_always_seals_the_same_rows(self):
        a = three_way(learnable())
        b = three_way(learnable())
        assert list(a[2].index) == list(b[2].index)

    def test_which_inputs_are_used_cannot_move_it(self):
        """The sealed rows depend on the data, not on any choice made during development."""
        df = learnable()
        a = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, sealed_holdout=True,
        )
        b = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, sealed_holdout=True, excluded_columns=["z"],
        )
        assert a["split"]["sealed_rows_hash"] == b["split"]["sealed_rows_hash"]

    def test_the_model_tried_cannot_move_it(self):
        df = learnable()
        hashes = {
            modeling.train_candidates(
                df, target_column="y", task="classification", model_names=[name],
                include_deep_diagnostics=False, sealed_holdout=True,
            )["split"]["sealed_rows_hash"]
            for name in ("logistic_regression", "random_forest_classifier")
        }
        assert len(hashes) == 1

    def test_the_improvement_loop_seals_the_same_rows(self, monkeypatch):
        df = learnable()
        seen = {}
        real = modeling.split_with_sealed_holdout

        def spy(*args, **kwargs):
            parts = real(*args, **kwargs)
            seen["final"] = list(parts[2].index)
            return parts

        monkeypatch.setattr(modeling, "split_with_sealed_holdout", spy)
        improvement.improve(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            budget=4, sealed_holdout=True,
        )
        assert seen["final"] == list(three_way(df)[2].index)


class TestNothingButFinaliseReadsIt:
    """Flip every answer in the sealed partition and nothing else may change."""

    def _pinned(self, monkeypatch, df):
        final = three_way(df)[2].index
        monkeypatch.setattr(modeling, "sealed_holdout_rows", lambda labelled, *a, **k: final)
        flipped = df.copy()
        flipped.loc[final, "y"] = flipped.loc[final, "y"].map({"yes": "no", "no": "yes"})
        return flipped

    def test_training_is_unchanged(self, monkeypatch):
        df = learnable()
        flipped = self._pinned(monkeypatch, df)

        def run(frame):
            return modeling.train_candidates(
                frame, target_column="y", task="classification",
                model_names=["logistic_regression", "random_forest_classifier"],
                include_deep_diagnostics=True, sealed_holdout=True,
            )

        honest, rigged = run(df), run(flipped)
        assert rigged["best_model"] == honest["best_model"]
        assert [(r["model"], r["cv_score"], r["metrics"]["roc_auc"]) for r in rigged["results"]] == [
            (r["model"], r["cv_score"], r["metrics"]["roc_auc"]) for r in honest["results"]
        ]
        assert rigged["deep_diagnostics"]["cross_validation"] == honest["deep_diagnostics"]["cross_validation"]
        assert rigged["assessment"] == honest["assessment"]
        assert rigged["split"]["sealed_rows_hash"] == honest["split"]["sealed_rows_hash"]

    def test_the_improvement_loop_is_unchanged(self, monkeypatch):
        df = learnable()
        flipped = self._pinned(monkeypatch, df)

        def run(frame):
            return improvement.improve(
                frame, target_column="y", task="classification",
                model_names=["logistic_regression", "random_forest_classifier"], budget=4,
                sealed_holdout=True,
            )

        honest, rigged = run(df), run(flipped)
        assert rigged["rounds"] == honest["rounds"]
        assert rigged["best"] == honest["best"]
        assert rigged["test_metrics"] == honest["test_metrics"]

    def test_the_flip_is_not_a_no_op(self, monkeypatch):
        """The fixture must be able to fail: the sealed rows do hold different answers."""
        df = learnable()
        flipped = self._pinned(monkeypatch, df)
        final = three_way(df)[2].index
        assert (df.loc[final, "y"] != flipped.loc[final, "y"]).all()


class TestWhenTheFileIsTooSmall:
    def _train(self, df):
        return modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, sealed_holdout=True,
        )

    def test_the_ordinary_split_is_used_and_the_reason_is_given(self):
        small = learnable(120)
        result = self._train(small)
        assert result["split"]["sealed"] is False
        assert "too small to judge" in result["split"]["sealed_unavailable_reason"]
        assert result["split"]["train_rows"] + result["split"]["test_rows"] == len(small)

    def test_a_rare_class_that_cannot_fill_both_parts_is_the_reason(self):
        rng = np.random.default_rng(9)
        df = pd.DataFrame({"x": rng.normal(0, 1, 300).round(4)})
        df["y"] = ["no"] * 270 + ["yes"] * 30  # 10% rare: 30 rows, ~4 per held-out part
        result = self._train(df)
        assert result["split"]["sealed"] is False
        assert "rarer" in result["split"]["sealed_unavailable_reason"].lower() or "10" in result["split"]["sealed_unavailable_reason"]

    def test_one_timestamp_holding_over_half_the_rows_cannot_be_sealed(self):
        """There is no honest way to seal 'the most recent rows' when most rows share a moment."""
        rows = 400
        df = pd.DataFrame(
            {
                "when": ["2022-01-01"] * 300 + [f"2022-02-{d:02d}" for d in range(1, 29)] * 3 + ["2022-03-01"] * 16,
                "x": np.random.default_rng(1).normal(0, 1, rows).round(4),
            }
        )
        df["y"] = np.random.default_rng(2).choice(["yes", "no"], rows)
        with pytest.raises(modeling.ModelingError, match="share one timestamp"):
            modeling.split_with_sealed_holdout(
                df, "y", "classification", strategy="time", time_column="when"
            )

    def test_a_date_split_that_cannot_be_sealed_falls_back_with_the_reason(self):
        """Two dates of 200 rows each: the ordinary split copes (train on the first, test on the
        second) but there is no third group of dates to seal, so the seal is withheld and the run
        is not."""
        rng = np.random.default_rng(3)
        days = np.repeat(pd.date_range("2022-01-01", periods=2, freq="D"), 200)
        df = pd.DataFrame({"when": days.astype(str), "x": rng.normal(0, 1, 400).round(4)})
        df["y"] = rng.choice(["yes", "no"], 400)
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, sealed_holdout=True, split_strategy="time",
            time_column="when",
        )
        assert result["split"]["sealed"] is False
        assert result["split"]["sealed_unavailable_reason"]

    def test_not_asking_is_the_ordinary_two_way_split(self):
        df = learnable()
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
        )
        assert result["split"]["sealed"] is False
        assert "sealed_unavailable_reason" not in result["split"]


class TestTheExport:
    def _result(self, df, **kwargs):
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False, sealed_holdout=True, **kwargs,
        )
        assert result["split"]["sealed"] is True
        result["features"]["excluded"] = []
        return result

    def _exec(self, df, result):
        from sklearn.model_selection import GroupShuffleSplit, train_test_split

        labelled = df[df["y"].notna()]
        features = result["features"]
        namespace = {
            "pd": pd, "np": np, "labelled": labelled, "TARGET": "y",
            "NUMERIC_FEATURES": features["numeric"], "CATEGORICAL_FEATURES": features["categorical"],
            "FREQUENT_FEATURES": features["frequent"],
            "train_test_split": train_test_split, "GroupShuffleSplit": GroupShuffleSplit,
        }
        namespace["features"] = labelled[
            features["numeric"] + features["categorical"] + features["frequent"]
        ]
        namespace["labels"] = labelled["y"]
        exec(model_export._split_block(result["split"], "classification"), namespace)  # noqa: S102
        return namespace

    def test_the_script_reproduces_all_three_partitions_at_random(self):
        df = learnable()
        result = self._result(df)
        ns = self._exec(df, result)
        _, dev, final, *_ = three_way(df)
        assert list(ns["X_test"].index) == list(dev.index)
        assert sorted(ns["final_index"]) == sorted(final.index)

    def test_the_script_reproduces_a_date_split(self):
        df = daily()
        result = self._result(df, split_strategy="time", time_column="when")
        ns = self._exec(df, result)
        _, dev, final, *_ = three_way(df, strategy="time", time_column="when")
        assert list(ns["X_test"].index) == list(dev.index)
        assert sorted(ns["final_index"]) == sorted(final.index)

    def test_the_script_reproduces_a_grouped_split(self):
        df = entities()
        result = self._result(df, split_strategy="group", group_column="customer")
        ns = self._exec(df, result)
        _, dev, final, *_ = three_way(df, strategy="group", group_column="customer")
        assert list(ns["X_test"].index) == list(dev.index)
        assert sorted(ns["final_index"]) == sorted(final.index)

    def test_the_script_reproduces_copies_kept_together(self):
        rng = np.random.default_rng(6)
        base = pd.DataFrame({"x": rng.normal(0, 1, 300).round(4), "z": rng.normal(0, 1, 300).round(4)})
        base["y"] = np.where(base["x"] + rng.normal(0, 0.7, 300) > 0, "yes", "no")
        df = pd.concat([base, base], ignore_index=True)
        result = self._result(df, keep_identical_rows_together=True)
        ns = self._exec(df, result)
        _, dev, final, *_ = three_way(df, duplicate_columns=["x", "z"], copy_columns=["x", "z"])
        assert list(ns["X_test"].index) == list(dev.index)
        assert sorted(ns["final_index"]) == sorted(final.index)

    def test_the_script_runs_reports_the_development_metrics_and_never_opens_the_seal(self, tmp_path):
        df = learnable()
        result = self._result(df)
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), "utf-8")
        done = subprocess.run(
            [sys.executable, str(script), str(csv)], capture_output=True, text=True, cwd=tmp_path, timeout=600
        )
        assert done.returncode == 0, f"{done.stdout}\n{done.stderr}"
        assert "Sealed 120 rows as a final holdout" in done.stdout
        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match and float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["accuracy"], abs=1e-4
        )

    def test_the_script_carries_the_apps_functions_not_copies(self):
        script = model_export.build_training_script(self._result(learnable()))
        assert model_export._function_source(modeling.sealed_holdout_rows) in script

    def test_a_script_without_the_seal_is_untouched(self):
        result = modeling.train_candidates(
            learnable(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        result["features"]["excluded"] = []
        assert "sealed_holdout_rows" not in model_export.build_training_script(result)


# ------------------------------------------------------------------ over the API


def _session(df: pd.DataFrame) -> tuple[str, object]:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id, session_dir(get_settings().upload_dir, session_id)


def train(session_id, **extra):
    response = client.post(f"/api/model/{session_id}/train", json={**FAST, **extra})
    assert response.status_code == 200, response.text
    return response.json()


class TestThePlanOffersIt:
    def test_it_is_offered_with_the_part_sizes_when_the_file_can_afford_it(self):
        plan = client.get(f"/api/model/{_session(learnable())[0]}/plan").json()
        offer = plan["sealed_holdout_offer"]
        assert offer["available"] is True
        assert offer["train_rows"] + offer["development_rows"] + offer["final_rows"] == 800
        assert offer["final_rows"] == 120
        assert offer["development_rows"] == pytest.approx(120, abs=2)
        assert offer["train_rows"] == pytest.approx(560, abs=2)
        assert plan["sealed_holdout"] is False  # opt-in

    def test_it_is_withheld_with_the_reason_when_it_cannot(self):
        offer = client.get(f"/api/model/{_session(learnable(120))[0]}/plan").json()["sealed_holdout_offer"]
        assert offer["available"] is False
        assert "too small to judge" in offer["reason"]

    def test_the_plan_comes_back_showing_what_the_last_run_used(self):
        session_id, _ = _session(learnable())
        train(session_id, sealed_holdout=True)
        assert client.get(f"/api/model/{session_id}/plan").json()["sealed_holdout"] is True


class TestTrainingWithIt:
    def test_the_result_says_what_was_sealed(self):
        session_id, directory = _session(learnable())
        body = train(session_id, sealed_holdout=True)
        split = body["split"]
        assert split["sealed"] is True and split["final_rows"] == 120
        assert split["train_rows"] + split["test_rows"] + split["final_rows"] == 800
        assert split["sealed_after_looks"] == 0
        assert (directory / model_store.SEALED_FILENAME).exists()

    def test_it_is_counted_as_its_own_split_rule(self):
        session_id, directory = _session(learnable())
        train(session_id)
        train(session_id, sealed_holdout=True)
        assert len({e["split_key"] for e in ledger.looks(directory)}) == 2

    def test_a_seal_taken_after_looks_says_so(self):
        """Still unopened, but the rows informed earlier decisions as training or development
        rows -- so it is not presented as pristine."""
        session_id, _ = _session(learnable())
        train(session_id)
        train(session_id)
        body = train(session_id, sealed_holdout=True)
        assert body["split"]["sealed_after_looks"] == 2
        assert train(session_id, sealed_holdout=True)["split"]["sealed_after_looks"] == 2  # first record wins

    def test_the_improvement_route_seals_too(self):
        session_id, _ = _session(learnable())
        body = client.post(
            f"/api/model/{session_id}/improve", json={**FAST, "budget": 4, "sealed_holdout": True}
        ).json()
        assert body["split"]["sealed"] is True


class TestOpeningIt:
    def test_finalise_scores_the_model_on_the_sealed_rows(self):
        session_id, directory = _session(learnable())
        trained = train(session_id, sealed_holdout=True)

        opened = client.post(f"/api/model/{session_id}/finalise")

        assert opened.status_code == 200, opened.text
        body = opened.json()
        assert body["already_opened"] is False and body["for_this_model"] is True
        assert body["rows"] == 120
        assert 0 < body["metrics"]["roc_auc"] <= 1
        assert body["assessment"]["verdict"] in {"useful", "marginal", "no_better_than_guessing"}
        assert body["run_id"] == trained["run_id"]
        # A different set of rows than the development score was measured on.
        assert body["metrics"]["roc_auc"] != trained["results"][0]["metrics"]["roc_auc"]

    def test_opening_it_marks_a_written_audit_report_stale(self):
        from app.graph.nodes.audit_agent import STALE_MARKER, is_report_stale

        session_id, directory = _session(learnable())
        train(session_id, sealed_holdout=True)
        (directory / "audit_report.md").write_text("# report", encoding="utf-8")
        (directory / STALE_MARKER).unlink(missing_ok=True)
        assert client.post(f"/api/model/{session_id}/finalise").status_code == 200
        assert is_report_stale(directory)

    def test_it_is_recorded_as_its_own_event_not_a_look(self):
        session_id, directory = _session(learnable())
        body = train(session_id, sealed_holdout=True)
        looks_before = ledger.look_count(directory, None, body["split"])
        client.post(f"/api/model/{session_id}/finalise")
        assert [e["event"] for e in ledger.looks(directory)].count(ledger.FINAL) == 1
        assert ledger.look_count(directory, None, body["split"]) == looks_before

    def test_a_second_call_returns_the_stored_result_with_a_note(self):
        session_id, directory = _session(learnable())
        train(session_id, sealed_holdout=True)
        first = client.post(f"/api/model/{session_id}/finalise").json()
        second = client.post(f"/api/model/{session_id}/finalise").json()

        assert second["already_opened"] is True and second["for_this_model"] is True
        assert second["metrics"] == first["metrics"]
        assert "stored result, not a new evaluation" in second["note"]
        assert [e["event"] for e in ledger.looks(directory)].count(ledger.FINAL) == 1

    def test_a_different_model_cannot_open_the_same_rows_again(self):
        """Scoring a second model on rows the first result may already have informed is not an
        independent evaluation."""
        session_id, directory = _session(learnable())
        train(session_id, sealed_holdout=True)
        first = client.post(f"/api/model/{session_id}/finalise").json()
        retrained = train(session_id, sealed_holdout=True)
        assert retrained["run_id"] != first["run_id"]

        again = client.post(f"/api/model/{session_id}/finalise").json()

        assert again["already_opened"] is True and again["for_this_model"] is False
        assert again["run_id"] == first["run_id"]
        assert "would not be independent" in again["note"]
        assert [e["event"] for e in ledger.looks(directory)].count(ledger.FINAL) == 1

    def test_the_result_shows_the_final_evaluation_once_opened(self):
        session_id, _ = _session(learnable())
        train(session_id, sealed_holdout=True)
        assert client.get(f"/api/model/{session_id}/result").json()["evaluation"]["final"] is None
        opened = client.post(f"/api/model/{session_id}/finalise").json()
        final = client.get(f"/api/model/{session_id}/result").json()["evaluation"]["final"]
        assert final["rows"] == 120 and final["metrics"] == opened["metrics"]

    def test_opening_it_changes_nothing_else(self):
        session_id, _ = _session(learnable())
        before = train(session_id, sealed_holdout=True)
        client.post(f"/api/model/{session_id}/finalise")
        after = client.get(f"/api/model/{session_id}/result").json()
        assert after["results"] == before["results"]
        assert after["evaluation"]["look_count"] == before["evaluation"]["look_count"]
        assert after["evaluation"]["status"] == before["evaluation"]["status"]

    def test_there_is_nothing_to_open_without_a_sealed_holdout(self):
        session_id, _ = _session(learnable())
        train(session_id)
        response = client.post(f"/api/model/{session_id}/finalise")
        assert response.status_code == 409
        assert "sealed final holdout" in response.json()["detail"]

    def test_the_reason_a_seal_was_not_possible_is_repeated(self):
        session_id, _ = _session(learnable(120))
        train(session_id, sealed_holdout=True)
        response = client.post(f"/api/model/{session_id}/finalise")
        assert response.status_code == 409 and "too small to judge" in response.json()["detail"]

    def test_there_is_nothing_to_open_before_training(self):
        session_id, _ = _session(learnable())
        assert client.post(f"/api/model/{session_id}/finalise").status_code == 404

    def test_it_is_refused_if_the_data_no_longer_produces_the_sealed_rows(self):
        session_id, directory = _session(learnable())
        train(session_id, sealed_holdout=True)
        stored = model_store.load_result(directory)
        stored["split"]["sealed_rows_hash"] = "0" * 16
        (directory / model_store.RESULT_FILENAME).write_text(__import__("json").dumps(stored))

        response = client.post(f"/api/model/{session_id}/finalise")

        assert response.status_code == 409
        assert "not what was sealed" in response.json()["error"].lower()
        assert [e["event"] for e in ledger.looks(directory)].count(ledger.FINAL) == 0
