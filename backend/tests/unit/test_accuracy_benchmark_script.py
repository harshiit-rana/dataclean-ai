"""The accuracy benchmark's task loading, scoring and summary (#124). Offline: nothing here reaches openml.org."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


openml_tasks = _load("openml_tasks")
bench = _load("accuracy_benchmark")

SPLITS = """@relation x_splits

@attribute type {TRAIN,TEST}
@attribute rowid numeric
@attribute repeat numeric
@attribute fold numeric

@data
TRAIN,2,0,0
TRAIN,3,0,0
TEST,0,0,0
TEST,1,0,0
TRAIN,0,0,1
TRAIN,1,0,1
TEST,2,0,1
TEST,3,0,1
TRAIN,9,1,0
"""


class TestTasks:
    def test_the_split_file_becomes_train_and_test_rows_per_fold_of_the_first_repeat(self):
        assert openml_tasks.parse_splits(SPLITS) == [([2, 3], [0, 1]), ([0, 1], [2, 3])]

    def test_tasks_are_picked_by_name_or_id_and_by_set(self):
        assert [t.name for t in openml_tasks.select(["credit-g", "359950"])] == ["credit-g", "boston"]
        held = openml_tasks.select(held_out=True)
        assert held and all(t.held_out for t in held)
        assert len(held) + len(openml_tasks.select(held_out=False)) == len(openml_tasks.TASKS)
        with pytest.raises(KeyError, match="nonsense"):
            openml_tasks.select(["nonsense"])

    def test_every_kind_of_target_is_in_both_sets(self):
        for held_out in (False, True):
            kinds = {t.kind for t in openml_tasks.select(held_out=held_out)}
            assert kinds == {"binary", "multiclass", "regression"}

    def test_the_tabarena_suite_is_all_held_out_and_repeats_no_amlb_dataset(self):
        tabarena = openml_tasks.select(suite="tabarena")
        assert len(tabarena) == 28 and all(t.held_out for t in tabarena)
        assert {t.kind for t in tabarena} == {"binary", "multiclass", "regression"}
        assert not {t.name.lower() for t in tabarena} & {t.name.lower() for t in openml_tasks.TASKS}
        assert openml_tasks.by_id(363629).name == "diabetes"
        with pytest.raises(KeyError):
            openml_tasks.select(["credit-g"], suite="tabarena")

    def test_a_busy_server_is_tried_again_and_its_error_page_never_cached(self, monkeypatch):
        import io as _io
        import urllib.error

        replies = [urllib.error.URLError("503"), b"<div><h4>A PHP Error was encountered</h4>", b'{"ok": 1}']

        def fake_urlopen(request, timeout):
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return _io.BytesIO(reply)

        monkeypatch.setattr(openml_tasks.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(openml_tasks.time, "sleep", lambda seconds: None)
        assert openml_tasks._fetch("https://www.openml.org/x") == b'{"ok": 1}' and not replies

    def test_a_cached_task_loads_without_the_network(self, tmp_path):
        meta = {"task": {"input": [
            {"name": "source_data", "data_set": {"data_set_id": "7", "target_feature": "y"}},
            {"name": "estimation_procedure", "estimation_procedure": {"data_splits_url": "https://unused.invalid"}},
        ]}}
        (tmp_path / "task_1.json").write_text(json.dumps(meta))
        (tmp_path / "task_1_splits.arff").write_text(SPLITS)
        pd.DataFrame({"a": [1, 2, 3, 4], "y": ["p", "q", "p", "q"]}).to_csv(tmp_path / "data_7.csv", index=False)
        loaded = openml_tasks.load(openml_tasks.Task(1, "tiny", "binary"), tmp_path)
        assert loaded.target == "y" and loaded.frame.shape == (4, 2) and len(loaded.folds) == 2


class TestScoring:
    def test_auc_reads_the_column_of_the_later_label_whatever_order_the_system_gives(self):
        y = pd.Series(["no", "yes", "yes", "no"])
        proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])  # columns: yes, no
        got = bench.score("binary", y, bench.Prediction(proba=proba, classes=["yes", "no"]))
        assert got["auc"] == 1.0 and got["accuracy"] == 1.0

    def test_number_labels_match_a_system_that_turned_them_into_text(self):
        y = pd.Series([1, 2, 3, 1])
        proba = np.eye(3)[[0, 1, 2, 0]]
        got = bench.score("multiclass", y, bench.Prediction(proba=proba, classes=["1", "2", "3"]), labels=[1, 2, 3])
        assert got["accuracy"] == 1.0 and got["log_loss"] < 1e-6

    def test_a_label_the_system_never_saw_costs_a_large_finite_log_loss(self):
        y = pd.Series(["a", "b", "c"])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        got = bench.score("multiclass", y, bench.Prediction(proba=proba, classes=["a", "b"]), labels=["a", "b", "c"])
        assert np.isfinite(got["log_loss"]) and got["log_loss"] > 5

    def test_regression_is_rmse_and_r2(self):
        got = bench.score("regression", pd.Series([1.0, 2.0, 3.0]), bench.Prediction(values=np.array([1.0, 2.0, 4.0])))
        assert got["rmse"] == pytest.approx((1 / 3) ** 0.5) and got["r2"] == pytest.approx(0.5)


class TestSummary:
    def test_normalised_is_zero_at_the_constant_and_one_at_the_forest_both_ways_round(self):
        assert bench.normalised(0.5, 0.5, 0.9, True) == 0
        assert bench.normalised(0.9, 0.5, 0.9, True) == 1
        assert bench.normalised(1.0, 2.0, 1.5, False) == pytest.approx(2.0)
        assert bench.normalised(0.7, 0.7, 0.7, True) is None

    def test_a_difference_inside_the_fold_spread_is_a_tie(self):
        assert bench.compare([0.80, 0.90, 0.70], [0.79, 0.88, 0.74], True) == "tie"
        assert bench.compare([0.90, 0.91, 0.92], [0.80, 0.81, 0.82], True) == "win"
        assert bench.compare([1.0, 1.1, 1.2], [0.5, 0.6, 0.7], False) == "loss"

    @staticmethod
    def results(improve_auc: list[float], held_out: bool = False, task_id: str = "1") -> dict:
        folds = [
            {"constant": {"auc": 0.5}, "random_forest": {"auc": 0.8}, "dataclean_improve": {"auc": a}}
            for a in improve_auc
        ]
        return {"folds": 2, "budget_seconds": 10, "systems": ["constant", "random_forest", "dataclean_improve"],
                "cpus": 4, "python": "3.12", "scikit_learn": "1.9",
                "tasks": {task_id: {"name": f"t{task_id}", "kind": "binary", "held_out": held_out, "metric": "auc",
                                    "folds": folds}}}

    def test_a_failed_fold_counts_as_the_constant_predictor(self):
        results = self.results([0.9, 0.9])
        results["tasks"]["1"]["folds"][1]["dataclean_improve"] = {"error": "ValueError: boom"}
        assert bench.fold_values(results["tasks"]["1"], "dataclean_improve") == [0.9, 0.5]
        summary = bench.summarise(results)["all"]
        assert summary["failed_folds"]["dataclean_improve"] == 1
        assert summary["normalised_score"]["dataclean_improve"] == pytest.approx((0.7 - 0.5) / 0.3, abs=1e-4)

    def test_the_median_normalised_score_is_not_carried_by_one_task(self):
        # Three ordinary tasks a little under the forest, and one where the forest barely beats the constant
        # predictor, so a small lead there scales to a huge score: the mean follows that task, the median does not.
        results = self.results([0.78, 0.78])
        results["tasks"].update(self.results([0.78, 0.78], task_id="2")["tasks"])
        results["tasks"].update(self.results([0.78, 0.78], task_id="3")["tasks"])
        near_floor = self.results([0.56, 0.56], task_id="4")
        for fold in near_floor["tasks"]["4"]["folds"]:
            fold["random_forest"]["auc"] = 0.51
        results["tasks"].update(near_floor["tasks"])
        summary = bench.summarise(results)["all"]
        assert summary["normalised_score"]["dataclean_improve"] > 2
        assert summary["median_normalised_score"]["dataclean_improve"] == pytest.approx(28 / 30, abs=1e-4)
        assert "| dataclean_improve | 2." in bench.markdown({**results, "summary": bench.summarise(results)})

    def test_systems_that_differ_only_by_rounding_share_a_rank_in_the_significance_tests(self):
        results = self.results([0.9, 0.92])
        results["tasks"].update(self.results([0.7, 0.72], task_id="2")["tasks"])
        results["systems"].append("dataclean_train")
        for entry in results["tasks"].values():
            for fold in entry["folds"]:
                fold["dataclean_train"] = {"auc": fold["dataclean_improve"]["auc"] + 1e-15}
        ranks = bench.significance(results, ["1", "2"])["average_rank"]
        assert ranks["dataclean_train"] == ranks["dataclean_improve"]

    def test_the_summary_ranks_and_counts_wins_per_set(self):
        results = self.results([0.9, 0.92])
        results["tasks"].update(self.results([0.6, 0.62], held_out=True, task_id="2")["tasks"])
        summary = bench.summarise(results)
        assert summary["development"]["tasks"] == 1 and summary["held_out"]["tasks"] == 1
        assert summary["development"]["average_rank"]["dataclean_improve"] == 1
        assert summary["development"]["dataclean_improve_against"]["random_forest"] == {"win": 1, "tie": 0, "loss": 0}
        assert summary["held_out"]["dataclean_improve_against"]["random_forest"]["loss"] == 1
        report = bench.markdown({**results, "summary": summary})
        assert "### Held-out tasks (1)" in report and "| t2 (held out) | auc |" in report

    def test_a_fall_in_normalised_score_beyond_the_tolerance_is_flagged(self):
        before, after = self.results([0.8, 0.8]), self.results([0.7, 0.7])
        assert bench.regressions(after, before, 0.1) == [
            "dataclean_improve on t1: normalised score 0.667 against 1.000"
        ]
        assert bench.regressions(after, before, 0.5) == []

    def test_runs_over_different_tasks_merge_and_must_agree_on_the_settings(self):
        merged = bench.merge([self.results([0.9, 0.9]), self.results([0.7, 0.7], held_out=True, task_id="2")])
        assert set(merged["tasks"]) == {"1", "2"} and merged["summary"]["all"]["tasks"] == 2
        other = self.results([0.9, 0.9], task_id="3")
        other["budget_seconds"] = 99
        with pytest.raises(ValueError, match="budget_seconds"):
            bench.merge([self.results([0.9, 0.9]), other])


def test_every_system_runs_on_a_small_task_and_beats_the_constant_predictor():
    rng = np.random.default_rng(0)
    rows = 240
    frame = pd.DataFrame({
        "x1": rng.normal(size=rows), "x2": rng.normal(size=rows), "colour": rng.choice(["red", "blue", "green"], rows),
    })
    frame.loc[rng.random(rows) < 0.05, "x2"] = np.nan
    frame["y"] = np.where(frame["x1"] + (frame["colour"] == "red") + rng.normal(0, 0.5, rows) > 0.5, 1, 0)
    order = rng.permutation(rows)
    loaded = openml_tasks.LoadedTask(
        task=openml_tasks.Task(0, "tiny", "binary"), frame=frame, target="y",
        folds=[(sorted(order[:180].tolist()), sorted(order[180:].tolist()))],
    )
    systems = ["constant", "random_forest", "hist_gradient_boosting", "dataclean_train", "dataclean_improve"]
    scores = bench.run_fold(loaded, 0, systems, budget=3)
    assert all("error" not in scores[s] for s in systems), scores
    assert scores["constant"]["auc"] == 0.5
    for s in systems[1:]:
        assert scores[s]["auc"] > 0.75, (s, scores[s])


class TestSignificance:
    """The Friedman test, the Nemenyi critical difference and the Holm-corrected Wilcoxon tests (Demsar, 2006)."""

    @staticmethod
    def results(skill: dict, tasks: int = 12, noise: float = 0.02) -> dict:
        rng = np.random.default_rng(1)
        entries = {}
        for t in range(tasks):
            folds = [{s: {"auc": 0.5 if s == "constant" else skill[s] + rng.normal(0, noise), "brier": 0.2}
                      for s in skill} for _ in range(2)]
            entries[str(t)] = {"name": f"t{t}", "kind": "binary", "held_out": False, "metric": "auc", "folds": folds}
        return {"systems": list(skill), "tasks": entries}

    def test_the_friedman_statistic_is_scipys(self):
        from scipy import stats

        skill = {"constant": 0, "random_forest": 0.80, "dataclean_improve": 0.82, "flaml": 0.83}
        results = self.results(skill)
        sig = bench.significance(results, list(results["tasks"]))
        means = np.array([[np.mean(bench.fold_values(e, s)) for s in skill] for e in results["tasks"].values()])
        chi2 = stats.friedmanchisquare(*means.T).statistic
        n, k = means.shape
        assert sig["friedman_f"] == pytest.approx((n - 1) * chi2 / (n * (k - 1) - chi2), abs=1e-3)
        assert sig["critical_difference"] == pytest.approx(2.569 * (k * (k + 1) / (6 * n)) ** 0.5, abs=1e-3)

    def test_systems_of_equal_skill_are_not_told_apart(self):
        skill = {"constant": 0, "random_forest": 0.8, "hist_gradient_boosting": 0.8, "dataclean_improve": 0.8}
        results = self.results(skill, noise=0.03)
        sig = bench.significance(results, list(results["tasks"]))
        wilcoxon = sig["dataclean_improve_wilcoxon"]
        assert wilcoxon["random_forest"]["p_holm"] > 0.05 and wilcoxon["hist_gradient_boosting"]["p_holm"] > 0.05
        assert wilcoxon["constant"]["p_holm"] < 0.05
        assert ["random_forest", "hist_gradient_boosting", "dataclean_improve"] == sorted(
            next(g for g in bench.cd_runs(sig["average_rank"], sig["critical_difference"]) if "constant" not in g),
            key=["random_forest", "hist_gradient_boosting", "dataclean_improve"].index)

    def test_holm_never_lowers_a_later_p_value(self):
        skill = {"constant": 0, "random_forest": 0.75, "hist_gradient_boosting": 0.79, "dataclean_improve": 0.8}
        results = self.results(skill)
        ps = sorted(w["p_holm"] for w in bench.significance(results, list(results["tasks"]))
                    ["dataclean_improve_wilcoxon"].values())
        assert ps == sorted(ps) and all(0 <= p <= 1 for p in ps)

    def test_too_few_tasks_give_no_test(self):
        results = self.results({"constant": 0, "random_forest": 0.8, "dataclean_improve": 0.8}, tasks=1)
        assert bench.significance(results, list(results["tasks"])) is None

    def test_the_diagram_names_every_system_and_draws_a_bar_per_group(self):
        ranks = {"a": 1.2, "b": 1.9, "c": 3.5, "d": 3.9}
        svg = bench.cd_diagram(ranks, 1.0, "ranks")
        assert svg.startswith("<svg") and all(f">{s} (" in svg for s in ranks)
        assert bench.cd_runs(ranks, 1.0) == [["a", "b"], ["c", "d"]]
        assert svg.count('stroke-width="3"') == 2

    def test_the_summary_carries_the_brier_score_and_the_tests(self):
        skill = {"constant": 0, "random_forest": 0.8, "dataclean_improve": 0.82}
        results = {**self.results(skill), "folds": 2, "budget_seconds": 1, "cpus": 1, "python": "3", "scikit_learn": "1"}
        results["summary"] = bench.summarise(results)
        assert results["summary"]["all"]["brier"]["dataclean_improve"] == 0.2
        assert "Friedman test over 12 tasks" in bench.markdown(results)


def test_an_interrupted_run_carries_on_from_its_last_finished_task(tmp_path, monkeypatch):
    frame = pd.DataFrame({"x": range(40), "y": ["a", "b"] * 20})
    tasks = [openml_tasks.Task(1, "one", "binary"), openml_tasks.Task(2, "two", "binary")]
    loaded_ids = []

    def fake_load(task, cache):
        loaded_ids.append(task.id)
        return openml_tasks.LoadedTask(task=task, frame=frame, target="y", folds=[(list(range(30)), list(range(30, 40)))])

    monkeypatch.setattr(bench.openml_tasks, "load", fake_load)
    progress = tmp_path / "run.json"
    bench.run(tasks[:1], 1, ["constant"], 1, tmp_path, progress=progress)  # stopped after the first task
    assert set(json.loads(progress.read_text())["tasks"]) == {"1"}
    loaded_ids.clear()
    results = bench.run(tasks, 1, ["constant"], 1, tmp_path, progress=progress)
    assert loaded_ids == [2] and set(results["tasks"]) == {"1", "2"}
    loaded_ids.clear()
    bench.run(tasks, 1, ["constant"], 99, tmp_path, progress=progress)  # other settings: nothing is reused
    assert loaded_ids == [1, 2]


def test_a_finished_run_is_not_picked_up_so_the_same_command_measures_a_change(tmp_path):
    settings = {"folds": 1, "tasks": {"1": {}}}
    progress = tmp_path / "run.json"
    bench.save_progress(progress, settings)
    assert bench.interrupted_run(progress, settings, ("folds",)) == settings
    bench.save_progress(progress, {**settings, "summary": {}})
    assert bench.interrupted_run(progress, settings, ("folds",)) is None
    assert bench.interrupted_run(tmp_path / "none.json", settings, ("folds",)) is None
    assert [f.name for f in tmp_path.iterdir()] == ["run.json"]
