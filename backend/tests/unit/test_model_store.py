"""Run history: whether the thing you changed helped.

`save_result` used to write one file and overwrite it, so retraining silently discarded the
previous run. The verdict on a single run answers "is this model worth anything"; only the
sequence answers "was that decision a good one", and that is the question people actually have
while they work.

The awkward case these tests pin down is a run that predates a cleaning change. Deleting it
would throw away the most interesting comparison there is; comparing it as though nothing had
happened would be a lie. It is kept, and labelled.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.services import model_store


def result(
    *,
    model: str = "logistic_regression",
    auc: float = 0.8,
    excluded: list[str] | None = None,
    improvement: dict | None = None,
) -> dict:
    payload = {
        "task": "classification",
        "target_column": "outcome",
        "primary_metric": "roc_auc",
        "best_model": model,
        "baseline": {"accuracy": 0.6},
        "features": {"numeric": ["a", "b"], "categorical": [], "excluded": excluded or []},
        "split": {"strategy": "random", "train_rows": 240, "test_rows": 60},
        "assessment": {"verdict": "useful", "headline": "It works."},
        "results": [{"model": model, "label": "Logistic regression", "metrics": {"roc_auc": auc}}],
    }
    if improvement is not None:
        payload["improvement"] = improvement
    return payload


class TestEveryRunIsKept:
    def test_a_second_run_does_not_replace_the_first(self, tmp_path):
        model_store.save_result(tmp_path, result(auc=0.80), fingerprint="aaa")
        model_store.save_result(tmp_path, result(auc=0.86), fingerprint="aaa")

        runs = model_store.load_history(tmp_path)
        assert len(runs) == 2
        assert [r["metrics"]["roc_auc"] for r in runs] == [0.86, 0.80]  # newest first

    def test_the_stored_result_is_still_the_latest(self, tmp_path):
        """History is an addition, not a replacement -- the result endpoint still serves the
        most recent run in full."""
        model_store.save_result(tmp_path, result(auc=0.80), fingerprint="aaa")
        model_store.save_result(tmp_path, result(auc=0.86), fingerprint="aaa")
        assert model_store.load_result(tmp_path)["results"][0]["metrics"]["roc_auc"] == 0.86

    def test_history_is_capped(self, tmp_path):
        for i in range(model_store.MAX_HISTORY_ENTRIES + 5):
            model_store.save_result(tmp_path, result(auc=0.5 + i / 100), fingerprint="aaa")
        assert len(model_store.load_history(tmp_path)) == model_store.MAX_HISTORY_ENTRIES

    def test_each_run_records_what_made_it_different(self, tmp_path):
        model_store.save_result(
            tmp_path, result(excluded=["employee_id", "noise"]), fingerprint="aaa"
        )
        run = model_store.load_history(tmp_path)[0]
        assert run["excluded_columns"] == ["employee_id", "noise"]
        assert run["feature_count"] == 2
        assert run["split"]["strategy"] == "random"

    def test_a_tuned_run_says_so_and_what_it_settled_on(self):
        """"Tuning did not help" is a finding, and it is only visible by comparing runs."""
        summary = model_store.summarise(
            result(improvement={"settings": {"model__C": 0.1}, "stopped_because": "no_further_gains"})
        )
        assert summary["tuned"] is True
        assert summary["tuned_settings"] == {"model__C": 0.1}
        assert summary["stopped_because"] == "no_further_gains"

    def test_an_ordinary_run_is_not_marked_as_tuned(self):
        assert model_store.summarise(result())["tuned"] is False


class TestRunsThatPredateACleaningChange:
    def test_clearing_the_model_keeps_the_history(self, tmp_path):
        """`clear` runs when the data changes underneath the model. That is the exact moment
        the history is most useful -- the user has just changed something and wants to know
        whether it helped."""
        model_store.save_result(tmp_path, result(), fingerprint="aaa")
        model_store.clear(tmp_path)

        assert model_store.load_result(tmp_path) is None
        assert len(model_store.load_history(tmp_path)) == 1

    def test_the_run_carries_the_data_it_was_trained_on(self, tmp_path):
        model_store.save_result(tmp_path, result(auc=0.80), fingerprint="before")
        model_store.save_result(tmp_path, result(auc=0.86), fingerprint="after")
        assert [r["data_fingerprint"] for r in model_store.load_history(tmp_path)] == [
            "after",
            "before",
        ]


class TestFingerprinting:
    def test_the_same_frame_fingerprints_the_same(self):
        df = pd.DataFrame({"a": range(50), "b": list("xy") * 25})
        assert model_store.fingerprint_frame(df) == model_store.fingerprint_frame(df.copy())

    def test_dropping_rows_changes_it(self):
        df = pd.DataFrame({"a": range(50), "b": list("xy") * 25})
        assert model_store.fingerprint_frame(df) != model_store.fingerprint_frame(df.head(40))

    def test_dropping_a_column_changes_it(self):
        df = pd.DataFrame({"a": range(50), "b": list("xy") * 25})
        assert model_store.fingerprint_frame(df) != model_store.fingerprint_frame(df[["a"]])

    def test_rewriting_values_changes_it(self):
        """A fix that fills missing values keeps the shape and the columns, and is exactly the
        kind of change that makes an earlier run non-comparable."""
        df = pd.DataFrame({"a": [1.0, None, 3.0] * 20})
        filled = df.fillna(2.0)
        assert model_store.fingerprint_frame(df) != model_store.fingerprint_frame(filled)

    def test_it_never_raises(self):
        """A fingerprint is a nicety. Failing a completed training run over one would be
        losing a result that exists."""
        assert model_store.fingerprint_frame(object()) == ""


class TestBookkeepingNeverLosesAResult:
    def test_an_unreadable_history_file_does_not_break_saving(self, tmp_path):
        (tmp_path / model_store.HISTORY_FILENAME).write_text("{not json", encoding="utf-8")
        payload = model_store.save_result(tmp_path, result(), fingerprint="aaa")

        assert payload["best_model"] == "logistic_regression"
        assert len(model_store.load_history(tmp_path)) == 1  # started over rather than failing

    def test_history_of_the_wrong_shape_is_ignored(self, tmp_path):
        (tmp_path / model_store.HISTORY_FILENAME).write_text('{"runs": []}', encoding="utf-8")
        assert model_store.load_history(tmp_path) == []

    def test_numpy_values_survive_the_round_trip(self, tmp_path):
        """sklearn's output is full of numpy scalars, and json.dumps refuses them."""
        payload = result()
        payload["results"][0]["metrics"]["roc_auc"] = np.float64(0.8123)
        model_store.save_result(tmp_path, payload, fingerprint="aaa")

        stored = json.loads((tmp_path / model_store.HISTORY_FILENAME).read_text(encoding="utf-8"))
        assert stored[0]["metrics"]["roc_auc"] == pytest.approx(0.8123)
