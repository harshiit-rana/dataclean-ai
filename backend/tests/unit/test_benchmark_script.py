"""The benchmark script's data and its regression check (the timings themselves run in their own CI job)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_the_synthetic_file_is_twenty_columns_with_blanks_and_a_two_class_target():
    frame = benchmark.synthetic(500)
    assert frame.shape == (500, 21) and frame.isna().any().any()
    assert set(frame["outcome"]) == {"yes", "no"}
    assert benchmark.synthetic(500).equals(frame)  # the same file every run


def test_a_timing_over_the_tolerance_is_flagged_and_one_within_it_is_not():
    baseline = {"timings": {"1000": {"train": 10.0, "improve": 50.0}}}
    results = {"timings": {"1000": {"train": 12.0, "improve": 70.0}}}
    slower = benchmark.regressions(results, baseline, tolerance=0.3)
    assert slower == ["improve on 1,000 rows: 70.0s against 50.0s (+40%)"]


def test_a_machine_that_is_slower_at_everything_is_not_a_regression():
    # 9 November 2026: the same commit as the week before, every timing 45-104% slower, on a slower runner.
    baseline = {"reference_seconds": 1.0, "timings": {"1000": {"train": 10.0, "improve": 50.0}}}
    results = {"reference_seconds": 1.7, "timings": {"1000": {"train": 17.5, "improve": 84.0}}}
    assert benchmark.regressions(results, baseline, tolerance=0.3) == []


def test_a_step_that_got_slower_on_its_own_is_still_flagged_on_a_slower_machine():
    baseline = {"reference_seconds": 1.0, "timings": {"1000": {"train": 10.0, "improve": 50.0}}}
    results = {"reference_seconds": 1.5, "timings": {"1000": {"train": 15.0, "improve": 120.0}}}
    assert benchmark.regressions(results, baseline, tolerance=0.3) == [
        "improve on 1,000 rows: 120.0s against 50.0s, 75.0s expected on this machine (+60%)"]


def test_slower_timings_are_reported_without_failing_unless_strict(tmp_path, monkeypatch):
    baseline = tmp_path / "old.json"
    baseline.write_text('{"reference_seconds": 1.0, "timings": {"1000": {"train": 1.0}}}', encoding="utf-8")
    monkeypatch.setattr(benchmark, "run", lambda sizes: {"reference_seconds": 1.0, "timings": {"1000": {"train": 9.0}}})
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    args = ["benchmark.py", "--sizes", "1000", "--out", str(tmp_path / "new.json"), "--baseline", str(baseline)]
    monkeypatch.setattr(benchmark.sys, "argv", args)
    benchmark.main()  # reported, not an exit
    assert "SLOWER: train on 1,000 rows" in summary.read_text(encoding="utf-8")
    monkeypatch.setattr(benchmark.sys, "argv", [*args, "--strict"])
    with pytest.raises(SystemExit) as stop:
        benchmark.main()
    assert stop.value.code == 1


def test_a_size_missing_from_the_baseline_is_not_a_regression():
    assert benchmark.regressions({"timings": {"100000": {"train": 99.0}}}, {"timings": {}}, 0.3) == []
