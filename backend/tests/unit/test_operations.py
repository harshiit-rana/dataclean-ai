"""Whitelisted cleaning operations. These are what actually transform user data, so the
failure modes matter as much as the happy paths."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services import operations as ops


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "salary": [100.0, 200.0, None, 300.0, 1_000_000.0],
            "name": ["  Alice ", "BOB", "bob", "Dana", "Eve"],
            "joined": ["2024-01-15", "02/20/2024", "2024-03-05", "04/10/2024", "2024-05-01"],
            "qty": ["1", "2", "3", "4", "not a number"],
        }
    )


# --- parsing ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ('fill_median(column="salary")', ("fill_median", {"column": "salary"})),
        ("drop_duplicate_rows()", ("drop_duplicate_rows", {})),
        ("  fill_mode(column='name')  ", ("fill_mode", {"column": "name"})),
        (
            'standardize_case(column="name", case="upper")',
            ("standardize_case", {"column": "name", "case": "upper"}),
        ),
    ],
)
def test_parse_operation_valid(text, expected):
    assert ops.parse_operation(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "not an operation at all!", 'rm_rf(path="/")'])
def test_parse_operation_rejects_anything_not_whitelisted(text):
    """Anything the model invents that isn't in the registry must be dropped, not executed."""
    assert ops.parse_operation(text) is None


# --- execution ----------------------------------------------------------------------------


def test_fill_median_removes_nulls_without_changing_row_count(df):
    out, before, after = ops.apply_operation(df, "fill_median", {"column": "salary"})
    assert before["null_count"] == 1
    assert after["null_count"] == 0
    assert after["row_count"] == before["row_count"]


def test_drop_rows_with_nulls_shrinks_the_frame(df):
    out, before, after = ops.apply_operation(df, "drop_rows_with_nulls", {"column": "salary"})
    assert after["row_count"] == before["row_count"] - 1
    assert out["salary"].isna().sum() == 0


def test_cap_outliers_pulls_in_the_extreme_value(df):
    out, before, after = ops.apply_operation(df, "cap_outliers", {"column": "salary"})
    assert before["max"] == 1_000_000.0
    assert after["max"] < before["max"]
    assert after["row_count"] == before["row_count"]  # capping keeps every row


def test_standardize_case_collapses_case_variants(df):
    out, _, after = ops.apply_operation(df, "standardize_case", {"column": "name", "case": "lower"})
    assert "bob" in out["name"].tolist()
    assert "BOB" not in out["name"].tolist()
    assert after["unique_value_count"] == 4  # BOB and bob merge


def test_trim_whitespace(df):
    out, _, _ = ops.apply_operation(df, "trim_whitespace", {"column": "name"})
    assert out["name"].iloc[0] == "Alice"


def test_standardize_dates_normalizes_mixed_formats(df):
    out, _, _ = ops.apply_operation(df, "standardize_dates", {"column": "joined"})
    assert out["joined"].tolist() == [
        "2024-01-15",
        "2024-02-20",
        "2024-03-05",
        "2024-04-10",
        "2024-05-01",
    ]


def test_cast_to_numeric_coerces_unparseable_to_null(df):
    out, _, after = ops.apply_operation(df, "cast_to_numeric", {"column": "qty"})
    assert pd.api.types.is_numeric_dtype(out["qty"])
    assert after["null_count"] == 1


def test_drop_duplicate_rows():
    frame = pd.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    out, before, after = ops.apply_operation(frame, "drop_duplicate_rows", {})
    assert before["row_count"] == 3
    assert after["row_count"] == 2


def test_operations_never_mutate_the_input(df):
    """Every operation returns a new frame -- the caller chains them, so in-place mutation
    would corrupt the before/after stats."""
    snapshot = df.copy()
    for name, params in [
        ("fill_median", {"column": "salary"}),
        ("drop_rows_with_nulls", {"column": "salary"}),
        ("trim_whitespace", {"column": "name"}),
        ("drop_column", {"column": "qty"}),
    ]:
        ops.apply_operation(df, name, params)
    pd.testing.assert_frame_equal(df, snapshot)


# --- failure modes ------------------------------------------------------------------------


def test_unknown_operation_raises(df):
    with pytest.raises(ops.UnknownOperationError):
        ops.apply_operation(df, "definitely_not_real", {"column": "salary"})


def test_missing_column_raises_operation_failed(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(df, "fill_median", {"column": "no_such_column"})


def test_median_of_non_numeric_column_raises_operation_failed(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(df, "fill_median", {"column": "name"})


def test_operation_requiring_column_without_one_raises(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(df, "fill_median", {})


def test_catalog_lists_every_registered_operation():
    catalog = ops.operation_catalog()
    for name in ops.OPERATIONS:
        assert name in catalog
