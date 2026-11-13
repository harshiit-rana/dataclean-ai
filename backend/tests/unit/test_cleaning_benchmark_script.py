"""The cleaning benchmark's injectors, detection scoring and fixes (#125). Offline: built on a made-up file."""

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


openml_tasks = _load("openml_tasks")
bench = _load("cleaning_benchmark")

ROWS = 400


def clean_file(seed: int = 0) -> pd.DataFrame:
    """Four complete number columns with a spread, three tidy text columns, a two-class target."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "income": rng.normal(50_000, 8_000, ROWS).round(0),
        "age": rng.integers(18, 80, ROWS),
        "tenure": rng.normal(10, 3, ROWS).round(2),
        "visits": rng.normal(30, 6, ROWS).round(1),
        "plan": rng.choice(["basic", "plus", "premium"], ROWS),
        "region": rng.choice(["north", "south", "east", "west"], ROWS),
        "channel": rng.choice(["web", "store", "phone"], ROWS),
    })
    frame["churned"] = np.where(frame["tenure"] + rng.normal(0, 2, ROWS) < 9, "yes", "no")
    return openml_tasks.as_uploaded(frame)


def corrupted(seed: int = 0):
    clean = bench.with_dates(clean_file(), seed)
    dirty, injections = bench.corrupt(clean, "churned", seed)
    return clean, dirty, injections


class TestInjecting:
    def test_every_kind_of_problem_goes_into_its_own_column(self):
        _, _, injections = corrupted()
        kinds = [i.kind for i in injections]
        assert sorted(set(kinds)) == sorted(set(bench.KINDS) - {"duplicates"})
        columns = [i.column for i in injections]
        assert len(columns) == len(set(columns)) and "churned" not in columns

    def test_the_cells_said_to_be_changed_are_the_ones_changed(self):
        clean, dirty, injections = corrupted()
        for injection in injections:
            before, after = clean.loc[injection.rows, injection.column], dirty.loc[injection.rows, injection.column]
            if injection.kind == "missing_values":
                assert after.isna().all()
            else:
                assert (before.astype(str) != after.astype(str)).all(), injection.kind
        untouched = [c for c in clean.columns if c not in {i.column for i in injections}]
        assert dirty[untouched].equals(clean[untouched])

    def test_the_same_seed_gives_the_same_problems(self):
        _, first, _ = corrupted(3)
        _, again, _ = corrupted(3)
        _, other, _ = corrupted(4)
        assert first.equals(again) and not first.equals(other)

    def test_text_in_a_number_column_survives_the_csv_as_text(self):
        _, dirty, injections = corrupted()
        column = next(i.column for i in injections if i.kind == "type_mismatch")
        assert openml_tasks.as_uploaded(dirty)[column].dtype == object

    def test_copies_are_added_at_the_end(self):
        frame = clean_file()
        doubled, injection = bench.add_duplicates(frame, 0)
        assert len(doubled) == ROWS + len(injection.rows) and doubled.duplicated().tail(len(injection.rows)).all()


class TestDetection:
    def test_every_injected_problem_is_found_on_a_file_built_for_it(self):
        clean, dirty, injections = corrupted()
        doubled, copies = bench.add_duplicates(dirty, 0)
        result = bench.score_detection(openml_tasks.as_uploaded(clean), openml_tasks.as_uploaded(doubled),
                                       [*injections, copies])
        missed = [row for row in result["injected"] if not row["found"]]
        assert not missed, missed

    def test_a_finding_the_clean_file_already_had_counts_only_when_it_grew(self):
        class Issue:
            def __init__(self, **stats):
                self.stats = stats

        assert bench.grew(Issue(null_rate=0.2), None)
        assert not bench.grew(None, Issue(null_rate=0.2))
        assert not bench.grew(Issue(null_rate=0.205), Issue(null_rate=0.2))  # copies nudge a share; not a finding
        assert bench.grew(Issue(null_rate=0.25), Issue(null_rate=0.2))
        assert bench.grew(Issue(padded_values=12), Issue(padded_values=3))

    def test_nothing_injected_means_no_false_alarms(self):
        clean = openml_tasks.as_uploaded(bench.with_dates(clean_file(), 0))
        assert bench.score_detection(clean, clean, [])["false_alarms"] == []


class TestRepair:
    def test_fixes_are_learned_on_the_training_rows_and_replayed_on_the_test_rows(self):
        train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 100.0] * 20, "c": ["a", " a", "b", "B", "b"] * 20,
                              "y": ["p", "q"] * 50})
        test = pd.DataFrame({"x": [np.nan, 3.0], "c": [" b", "B"], "y": ["p", "q"]})
        fixed_train, fixed_test, applied = bench.fix(train, test, "y")
        assert "fill_median(x)" in applied and "trim_whitespace(c)" in applied and "drop_duplicate_rows()" in applied
        assert fixed_test.loc[0, "x"] == train["x"].median()  # the training rows' median, not the test row's
        assert set(fixed_test["c"]) == {"b"}  # trimmed, and "B" merged into the training rows' commoner "b"
        assert len(fixed_train) < len(train) and len(fixed_test) == len(test)  # copies dropped from training only

    def test_won_back_is_the_share_of_the_loss_and_none_when_nothing_was_lost(self):
        assert bench.recovered(0.90, 0.80, 0.85, True) == pytest.approx(0.5)
        assert bench.recovered(1.0, 2.0, 1.0, False) == pytest.approx(1.0)
        assert bench.recovered(0.90, 0.895, 0.95, True) is None  # under a point of AUC: noise
        assert bench.recovered(0.80, 0.85, 0.82, True) is None  # the dirty file scored better


def test_the_summary_counts_recall_and_precision_per_kind():
    results = {"seeds": 1, "repair": False, "datasets": {"d": {"runs": [{"detection": {
        "injected": [{"kind": "outliers", "column": "a", "found": True, "cells_outside_range": 0.8},
                     {"kind": "missing_values", "column": "b", "found": False}],
        "false_alarms": [{"kind": "outliers", "column": "c"}],
        "on_clean_file": {"outliers": 2},
    }}]}}}
    summary = bench.summarise(results)
    assert summary["per_kind"]["outliers"] == {"injected": 1, "found": 1, "false_alarms": 1, "recall": 1.0,
                                               "precision": 0.5}
    assert summary["per_kind"]["missing_values"]["recall"] == 0.0
    assert summary["recall"] == 0.5 and summary["precision"] == 0.5
    assert summary["findings_on_clean_files"] == {"outliers": 2}
    report = bench.markdown({**results, "summary": summary})
    assert "| outliers | 1 | 1 | 1.0 | 1 | 0.5 |" in report and "outliers on d.c" in report


def test_an_interrupted_run_carries_on_from_its_last_finished_seed(tmp_path, monkeypatch):
    task = openml_tasks.Task(1, "one", "binary")
    loaded = openml_tasks.LoadedTask(task=task, frame=clean_file(), target="churned", folds=[])
    seeds_run = []

    def counting_corrupt(frame, target, seed):
        seeds_run.append(seed)
        return frame, []

    monkeypatch.setattr(bench.openml_tasks, "select", lambda names: [task])
    monkeypatch.setattr(bench.openml_tasks, "load", lambda t, cache: loaded)
    monkeypatch.setattr(bench, "corrupt", counting_corrupt)
    monkeypatch.setattr(bench, "score_detection", lambda clean, dirty, injected: {"injected": [], "false_alarms": []})
    finished = {"seed": 0, "detection": {"injected": [], "false_alarms": []}, "seconds": 1}
    progress = tmp_path / "run.json"
    bench.accuracy.save_progress(progress, {"seeds": 3, "repair": False, "datasets": {
        "one": {"kind": "binary", "rows": ROWS, "runs": [finished]}}})
    results = bench.run(["one"], 3, False, tmp_path, progress=progress)
    assert seeds_run == [1, 2] and len(results["datasets"]["one"]["runs"]) == 3
