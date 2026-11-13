"""The real-error benchmark's per-cell scoring (#127). Offline: built on small made-up pairs of files."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


real = _load("real_errors_benchmark")


def write_pair(folder: Path, name: str, dirty: pd.DataFrame, clean: pd.DataFrame) -> None:
    (folder / "raha").mkdir(parents=True, exist_ok=True)
    dirty.to_csv(folder / "raha" / f"{name}_dirty.csv", index=False)
    clean.to_csv(folder / "raha" / f"{name}_clean.csv", index=False)


NUMBERS = ["1", "2", "x", "4", "5", "6", "7", "8", "9", "10"] * 2  # nine in ten numbers: "numbers stored as text"


class TestRealErrors:
    def test_comparison_is_raha_s_entities_decoded_and_whitespace_collapsed(self):
        assert real._normalised("  a &amp;\t b \n") == "a & b"

    def test_a_cached_pair_loads_without_the_network_with_columns_by_position(self, tmp_path):
        write_pair(tmp_path, "toy", pd.DataFrame({"a": ["1"], "b": ["x"]}), pd.DataFrame({"A": ["1"], "B": ["y"]}))
        as_read, dirty, clean = real.load("toy", tmp_path)
        assert list(clean.columns) == ["a", "b"] and (dirty != clean).to_numpy().tolist() == [[False, True]]

    def test_findings_point_at_the_cells_they_are_about(self):
        frame = pd.DataFrame({"n": NUMBERS, "c": ["Paris", "paris", "Paris", "Rome", "Rome"] * 4})
        issues = {(i.column_name, i.issue_type): i for i in real.profiling.detect_issues(frame)}
        numbers = real.flagged_cells(frame, issues[("n", "type_mismatch")])
        assert numbers[numbers].index.tolist() == [2, 12]
        casing = real.flagged_cells(frame, issues[("c", "inconsistent_format")])
        assert casing.tolist() == [False, True, False, False, False] * 4  # the rarer spelling only

    def test_detection_is_scored_per_cell(self):
        as_read = pd.DataFrame({"n": NUMBERS, "ok": list("abcde") * 4})
        errors = pd.DataFrame(False, index=as_read.index, columns=as_read.columns)
        errors.loc[[2, 5], "n"] = True  # one "x" is a real error, the other is not; a well-formed wrong number is unseen
        errors.loc[0, "ok"] = True
        got = real.score_detection(as_read, errors)
        assert got["flagged_cells"] == 2 and got["right"] == 1 and got["erroneous_cells"] == 3
        assert got["precision"] == 0.5 and got["recall"] == pytest.approx(1 / 3, abs=1e-4)
        assert got["columns_with_errors"] == 2 and got["columns_with_errors_flagged"] == 1

    def test_repair_counts_right_overwritten_and_invented_apart(self):
        as_read = pd.DataFrame({"city": ["Paris", "paris", "Paris", "PARIS"], "n": [1.0, None, 3.0, None]})
        clean = pd.DataFrame({"city": ["Paris", "Paris", "Paris", "PARIS"], "n": ["1", "2", "3", ""]})
        errors = pd.DataFrame({"city": [False, True, False, False], "n": [False, True, False, False]})
        got = real.score_repair(as_read, clean, errors)
        # "paris" -> "Paris" is right. "PARIS" is a third spelling in its own right here but differs only in
        # capitals, so it is merged and a correct value is overwritten. The blank that should be 2 is filled with the
        # median, 2.0: right. The blank that was really blank is filled: invented, not overwritten.
        assert got["right"] == 2 and got["damaged"] == 1 and got["invented"] == 1

    def test_dates_and_numbers_compare_by_meaning(self):
        assert real._same("2015-01-01", "1/1/2015") and real._same(12.0, "12") and not real._same("7:16", "")
