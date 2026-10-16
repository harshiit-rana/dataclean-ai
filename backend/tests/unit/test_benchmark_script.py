"""The benchmark script's data and its regression check (the timings themselves run in their own CI job)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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


def test_a_size_missing_from_the_baseline_is_not_a_regression():
    assert benchmark.regressions({"timings": {"100000": {"train": 99.0}}}, {"timings": {}}, 0.3) == []
