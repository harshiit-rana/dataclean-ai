"""The exported cleaning script.

The load-bearing test is `test_the_script_reproduces_the_apps_cleaning_byte_for_byte`: the
script is written from hand-kept templates beside the operations they mirror, and templates
drift. So it is run in a real subprocess, over a log that uses every one of the eighteen
operations, and its output file is compared with the app's own replay -- byte for byte, the
way someone diffing the two files would.
"""

from __future__ import annotations

import ast
import subprocess
import sys

import pandas as pd
import pytest

from app.services import operations, replay
from app.services.cleaning_export import build_cleaning_script


def _entry(i: int, op: str, column: str, **params) -> dict:
    return {"action_id": f"a{i}", "operation": op, "column_name": column,
            "operation_params": params, "status": "applied"}


ORIGINAL = pd.DataFrame(
    {
        "spend": [10.0, None, 30.0, 25.0, 1000.0, 12.0, 14.0, 13.0, 30.0, 11.0],
        "visits": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 4.0, 900.0],
        "score": ["1.5", "2.25", "x", "4", "5", "6", "7", "8", "4", "9"],
        "plan": [" Basic", "basic ", None, "PRO", "pro", "Pro", "basic", "pro", "PRO", "basic"],
        "region": ["north", None, "south", "north", "east", "north", "south", "north", "north", "east"],
        "joined": ["2021-01-05", "05/02/2021", "2021-03-01", "2021-04-01", "2021-05-01",
                   "2021-06-01", "2021-07-01", "2021-08-01", "2021-04-01", "2021-09-01"],
        "code": ["A", "B", "C", "D", "E", "F", "G", "H", "D", "J"],
        "note": ["keep", None, "keep", "keep", "keep", "keep", "keep", "keep", "keep", "keep"],
        "unused": list(range(10)),
    }
)

# Every operation in the registry, in an order where each one has something to do.
EVERY_OPERATION = [
    _entry(0, "trim_whitespace", "plan"),
    _entry(1, "standardize_case", "plan", case="lower"),
    _entry(2, "fill_mode", "region"),
    _entry(3, "fill_median", "spend"),
    _entry(4, "fill_mean", "visits"),
    _entry(5, "fill_constant", "plan", value="unknown"),
    _entry(6, "cast_to_numeric", "score"),
    _entry(7, "round_numeric", "score", decimals="1"),
    _entry(8, "cap_outliers", "visits"),
    _entry(9, "standardize_dates", "joined", format="%Y-%m-%d"),
    _entry(10, "replace_values", "region", find="east", replace="eastern"),
    _entry(11, "drop_duplicate_rows", "DATASET"),
    _entry(12, "drop_outlier_rows", "spend"),
    _entry(13, "filter_rows", "code", comparison="ne", value="C", mode="keep"),
    _entry(14, "drop_rows_with_nulls", "note"),
    _entry(15, "cast_to_datetime", "joined"),
    _entry(16, "rename_column", "code", new_name="customer_code"),
    _entry(17, "drop_column", "unused"),
]


def test_the_fixture_really_covers_every_operation():
    assert {e["operation"] for e in EVERY_OPERATION} == set(operations.OPERATIONS)


def _run(tmp_path, script: str, source) -> str:
    source_path = tmp_path / "input.csv"
    source.to_csv(source_path, index=False)
    script_path = tmp_path / "clean.py"
    script_path.write_text(script, encoding="utf-8")
    out = tmp_path / "out.csv"
    completed = subprocess.run(
        [sys.executable, str(script_path), str(source_path), str(out)],
        capture_output=True, text=True, cwd=tmp_path, timeout=300,
    )
    assert completed.returncode == 0, f"the script failed:\n{completed.stdout}\n{completed.stderr}"
    return out.read_text(encoding="utf-8")


def _app_output(tmp_path, frame: pd.DataFrame) -> str:
    path = tmp_path / "app.csv"
    frame.to_csv(path, index=False)
    return path.read_text(encoding="utf-8")


def test_the_script_is_valid_python():
    ast.parse(build_cleaning_script(ORIGINAL, EVERY_OPERATION))


def test_the_script_reproduces_the_apps_cleaning_byte_for_byte(tmp_path):
    script = build_cleaning_script(ORIGINAL, EVERY_OPERATION)
    _steps, app_frame = replay.freeze(ORIGINAL, EVERY_OPERATION)
    assert "NOT REPRODUCED" not in script
    assert _run(tmp_path, script, ORIGINAL) == _app_output(tmp_path, app_frame)


def test_on_a_new_file_it_cleans_the_way_the_app_does(tmp_path):
    """The script's second job: run on a later export, it must match what the app's own
    "use this on a new file" produces -- learned values and all."""
    later = ORIGINAL.copy()
    later["spend"] = [500.0, None, 600.0, 700.0, 800.0, 900.0, 950.0, 990.0, 600.0, 999.0]
    script = build_cleaning_script(ORIGINAL, EVERY_OPERATION)
    steps, _ = replay.freeze(ORIGINAL, EVERY_OPERATION)
    app_frame = replay.apply(later, steps).frame
    assert _run(tmp_path, script, later) == _app_output(tmp_path, app_frame)


def test_learned_values_are_written_in_as_numbers():
    script = build_cleaning_script(ORIGINAL, EVERY_OPERATION)
    median = float(ORIGINAL["spend"].median())
    assert f".fillna({median!r})" in script
    assert "as learned from the original file" in script
    assert ".median()" not in script  # never recomputed on whatever file it is given


def test_undone_and_rejected_fixes_are_not_in_the_script():
    log = [_entry(0, "drop_column", "unused"), _entry(1, "drop_column", "note")]
    log[1]["status"] = "undone"
    script = build_cleaning_script(ORIGINAL, log)
    assert "'unused'" in script
    assert "'note'" not in script
    assert "1 fix, in the order" in script


def test_a_fix_that_cannot_be_reproduced_is_said_not_dropped():
    log = [_entry(0, "fill_median", "no_such_column")]
    script = build_cleaning_script(ORIGINAL, log)
    ast.parse(script)
    assert "NOT REPRODUCED" in script


def test_a_session_with_no_fixes_still_produces_a_working_script(tmp_path):
    script = build_cleaning_script(ORIGINAL, [])
    assert "No fixes were applied" in script
    assert _run(tmp_path, script, ORIGINAL) == _app_output(tmp_path, ORIGINAL)


@pytest.mark.parametrize("comparison,value", [("gt", "12"), ("contains", "b"), ("is_null", None)])
def test_filters_compare_the_way_the_operation_does(tmp_path, comparison, value):
    log = [_entry(0, "filter_rows", "spend" if comparison == "gt" else "plan",
                  comparison=comparison, value=value, mode="drop")]
    script = build_cleaning_script(ORIGINAL, log)
    _steps, app_frame = replay.freeze(ORIGINAL, log)
    assert _run(tmp_path, script, ORIGINAL) == _app_output(tmp_path, app_frame)
