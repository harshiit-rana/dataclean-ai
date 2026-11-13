"""The robustness benchmark's corruption and summary (#128). Offline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


robust = _load("robustness_benchmark")
openml_tasks = _load("openml_tasks")


class TestRobustness:
    @staticmethod
    def frame(rows: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return openml_tasks.as_uploaded(pd.DataFrame({
            "a": rng.normal(10, 2, rows).round(3), "b": rng.normal(5, 1, rows).round(3),
            "plan": rng.choice(["basic", "plus", "premium"], rows), "y": rng.choice(["yes", "no"], rows),
        }))

    @pytest.mark.parametrize("kind", robust.KINDS)
    def test_each_kind_of_mess_changes_only_the_columns_it_names(self, kind):
        frame = self.frame()
        messy, columns = robust.corrupt(frame, ["a", "b", "plan"], kind, seed=0)
        assert columns and "y" not in columns
        untouched = [c for c in frame.columns if c not in columns]
        assert messy[untouched].equals(frame[untouched])
        changed = (messy[columns].astype(str) != frame[columns].astype(str)).mean()
        assert ((changed > 0.15) & (changed < 0.25)).all()  # about a fifth of the rows

    def test_a_kind_with_no_column_to_go_in_is_skipped(self):
        frame = self.frame()[["a", "b", "y"]]
        assert robust.corrupt(frame, ["a", "b"], "stray_whitespace", seed=0) == (frame, [])

    def test_the_summary_averages_falls_relative_to_the_clean_score_and_counts_warnings(self):
        results = {"seeds": 1, "share": 0.2, "datasets": {"d": {"kind": "binary", "runs": [{
            "metric": "auc", "clean": 0.8,
            "kinds": {"outliers": {"columns": ["a"], "warned": True, "score": 0.72, "fall": 0.08},
                      "type_mismatch": {"columns": ["b"], "warned": False, "error": "ValueError: boom"}},
        }]}}}
        summary = robust.summarise(results)
        assert summary["outliers"] == {"runs": 1, "warned": 1, "failed": 0, "mean_relative_fall": 0.1,
                                       "median_relative_fall": 0.1}
        assert summary["type_mismatch"]["failed"] == 1 and summary["type_mismatch"]["warned"] == 0
        report = robust.markdown({**results, "summary": summary})
        assert "| outliers | 1 | +10.0% | +10.0% | 1 of 1 | 0 |" in report and "failed" in report

    def test_the_median_fall_is_not_carried_by_one_file_with_a_tiny_clean_score(self):
        def run(clean, fall):
            return {"metric": "log_loss", "clean": clean,
                    "kinds": {"outliers": {"columns": ["a"], "warned": True, "score": clean + fall, "fall": fall}}}
        results = {"seeds": 3, "share": 0.2, "datasets": {"d": {"kind": "binary", "runs": [
            run(0.5, 0.05), run(0.5, 0.05), run(0.002, 1.0)]}}}
        summary = robust.summarise(results)["outliers"]
        assert summary["median_relative_fall"] == 0.1 and summary["mean_relative_fall"] > 100


def test_an_interrupted_run_carries_on_from_its_last_finished_seed(tmp_path, monkeypatch):
    task = openml_tasks.Task(1, "one", "binary")
    seeds_run = []

    def counting_run_one(loaded, seed):
        seeds_run.append(seed)
        return {"metric": "auc", "clean": 0.8, "kinds": {}}

    monkeypatch.setattr(robust.openml_tasks, "select", lambda names: [task])
    monkeypatch.setattr(robust.openml_tasks, "load", lambda t, cache: openml_tasks.LoadedTask(
        task=task, frame=pd.DataFrame({"y": [0, 1]}), target="y", folds=[]))
    monkeypatch.setattr(robust, "run_one", counting_run_one)
    progress = tmp_path / "run.json"
    robust.accuracy.save_progress(progress, {"seeds": 3, "share": robust.SHARE, "datasets": {
        "one": {"kind": "binary", "runs": [{"metric": "auc", "clean": 0.8, "kinds": {}}]}}})
    results = robust.run(["one"], 3, tmp_path, progress=progress)
    assert seeds_run == [1, 2] and len(results["datasets"]["one"]["runs"]) == 3
