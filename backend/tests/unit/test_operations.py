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


def test_full_catalog_lists_every_registered_operation():
    catalog = ops.operation_catalog(include_conversational=True)
    for name in ops.OPERATIONS:
        assert name in catalog


# --- conversational-only operations (FR-37..40) -------------------------------------------


def test_filter_rows_drop_by_numeric_comparison(df):
    """The canonical free-form request: "drop rows where age is negative"."""
    frame = pd.DataFrame({"age": [30, -5, 42, -1]})
    out, before, after = ops.apply_operation(
        frame, "filter_rows", {"column": "age", "comparison": "lt", "value": "0", "mode": "drop"}
    )
    assert before["row_count"] == 4
    assert after["row_count"] == 2
    assert (out["age"] >= 0).all()


def test_filter_rows_keeps_matching_rows(df):
    frame = pd.DataFrame({"status": ["active", "closed", "active"]})
    out, _, after = ops.apply_operation(
        frame, "filter_rows", {"column": "status", "comparison": "eq", "value": "active"}
    )
    assert after["row_count"] == 2


def test_filter_rows_compares_numerically_not_as_text(df):
    """String comparison would put "100" < "9"; the column is numeric so it must not."""
    frame = pd.DataFrame({"n": [9, 100]})
    out, _, _ = ops.apply_operation(
        frame, "filter_rows", {"column": "n", "comparison": "gt", "value": "50"}
    )
    assert out["n"].tolist() == [100]


def test_filter_rows_rejects_a_non_numeric_value_for_a_numeric_column(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(
            df, "filter_rows", {"column": "salary", "comparison": "gt", "value": "abc"}
        )


def test_filter_rows_rejects_unknown_comparison(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(
            df, "filter_rows", {"column": "salary", "comparison": "approximately", "value": "1"}
        )


def test_filter_rows_null_checks_need_no_value(df):
    out, _, after = ops.apply_operation(
        df, "filter_rows", {"column": "salary", "comparison": "not_null"}
    )
    assert after["null_count"] == 0


def test_rename_column(df):
    out, _, _ = ops.apply_operation(df, "rename_column", {"column": "qty", "new_name": "quantity"})
    assert "quantity" in out.columns and "qty" not in out.columns


def test_rename_column_refuses_to_clobber_an_existing_column(df):
    with pytest.raises(ops.OperationFailedError):
        ops.apply_operation(df, "rename_column", {"column": "qty", "new_name": "salary"})


def test_round_numeric(df):
    frame = pd.DataFrame({"x": [1.2345, 2.6789]})
    out, _, _ = ops.apply_operation(frame, "round_numeric", {"column": "x", "decimals": "2"})
    assert out["x"].tolist() == [1.23, 2.68]


def test_replace_values(df):
    out, _, _ = ops.apply_operation(
        df, "replace_values", {"column": "name", "find": "BOB", "replace": "Bob"}
    )
    assert "Bob" in out["name"].tolist() and "BOB" not in out["name"].tolist()


def test_conversational_operations_are_hidden_from_the_issue_fixing_catalog():
    """The Explanation Agent fixes detected issues; none of these map to a detectable issue
    type, so it must not be able to propose them."""
    issue_catalog = ops.operation_catalog()
    for name in ops.CONVERSATIONAL_ONLY:
        assert name not in issue_catalog
    full_catalog = ops.operation_catalog(include_conversational=True)
    for name in ops.OPERATIONS:
        assert name in full_catalog
