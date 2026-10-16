"""Generates `clean.py`: the session's cleaning as a script that runs without the app.

`train_model.py` already let someone rebuild the model outside the app; the cleaning that
produced the model's training data had no such thing. A model you can rebuild on data you
cannot rebuild is only half reproducible.

Each applied step becomes the pandas lines that do exactly what its whitelisted operation did,
in order. Statistics a step learned from the data -- a median, a mode, outlier limits -- are
written in as literals, read off the session's data at the point that step ran. That makes
the script do two jobs: run on the original upload, it reproduces the session's cleaned file;
run on next month's export, it cleans that file the way the original was cleaned, which is
the same guarantee the app's own "use this on a new file" makes.

Nothing here is written by the LLM. The templates sit beside the operations they mirror, and a
test runs the generated script in a subprocess and compares its output with the app's.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.services import operations
from app.services.operations import OperationFailedError


def _r(value) -> str:
    return repr(value)


def _col(params: dict) -> str:
    return f"df[{_r(params['column'])}]"


def _comparison_value(frame: pd.DataFrame, params: dict):
    """The value filter_rows compares against, coerced exactly as the operation does it --
    against the column as it stood when the step ran."""
    comparison = str(params.get("comparison", "")).lower()
    value = params.get("value")
    if comparison in {"is_null", "not_null"}:
        return None
    return operations._coerce_for_comparison(frame[params["column"]], value)


_MASKS = {
    "eq": "{c} == {v}",
    "ne": "{c} != {v}",
    "gt": "{c} > {v}",
    "gte": "{c} >= {v}",
    "lt": "{c} < {v}",
    "lte": "{c} <= {v}",
    "contains": '{c}.astype("string").str.contains(str({v}), case=False, na=False)',
    "not_contains": '~{c}.astype("string").str.contains(str({v}), case=False, na=False)',
    "is_null": "{c}.isna()",
    "not_null": "{c}.notna()",
}


def _lines(name: str, params: dict, learned: dict | None, frame: pd.DataFrame) -> list[str]:
    """The pandas that does what `operations.OPERATIONS[name]` did to `frame`."""
    if name in ("fill_median", "fill_mean"):
        return [f"{_col(params)} = pd.to_numeric({_col(params)}, errors=\"coerce\").fillna({_r(learned['fill_value'])})"]
    if name == "fill_mode":
        return [f"{_col(params)} = {_col(params)}.fillna({_r(learned['fill_value'])})"]
    if name == "fill_constant":
        return [f"{_col(params)} = {_col(params)}.fillna({_r(params['value'])})"]
    if name == "drop_rows_with_nulls":
        return [f"df = df[{_col(params)}.notna()].copy()"]
    if name == "drop_duplicate_rows":
        subset = params.get("columns") or None
        return [f"df = df.drop_duplicates(subset={_r(subset)}, keep=\"first\").copy()"]
    if name == "cast_to_numeric":
        return [f"{_col(params)} = pd.to_numeric({_col(params)}, errors=\"coerce\")"]
    if name == "cast_to_datetime":
        return [f"{_col(params)} = pd.to_datetime({_col(params)}, errors=\"coerce\", format=\"mixed\")"]
    if name == "standardize_case":
        case = params.get("case", "lower")
        # Written into code, so checked here as well as when the operation ran (#114): the script must never depend on
        # another module having refused a value first.
        if case not in {"lower", "upper", "title"}:
            raise OperationFailedError(f"Unsupported case {case!r}")
        return [f"{_col(params)} = {_col(params)}.astype(\"string\").str.{case}()"]
    if name == "trim_whitespace":
        return [f"{_col(params)} = {_col(params)}.astype(\"string\").str.strip()"]
    if name == "standardize_dates":
        fmt = params.get("format", "%Y-%m-%d")
        return [
            f"_parsed = pd.to_datetime({_col(params)}, errors=\"coerce\", format=\"mixed\")",
            f"{_col(params)} = _parsed.dt.strftime({_r(fmt)}).where(_parsed.notna(), None)",
        ]
    if name == "cap_outliers":
        return [
            f"{_col(params)} = pd.to_numeric({_col(params)}, errors=\"coerce\").clip("
            f"lower={_r(learned['lower'])}, upper={_r(learned['upper'])})"
        ]
    if name == "drop_outlier_rows":
        return [
            f"_values = pd.to_numeric({_col(params)}, errors=\"coerce\")",
            f"df = df[_values.isna() | ((_values >= {_r(learned['lower'])}) & "
            f"(_values <= {_r(learned['upper'])}))].copy()",
        ]
    if name == "drop_column":
        return [f"df = df.drop(columns=[{_r(params['column'])}]).copy()"]
    if name == "rename_column":
        return [f"df = df.rename(columns={{{_r(params['column'])}: {_r(str(params['new_name']).strip())}}}).copy()"]
    if name == "round_numeric":
        return [f"{_col(params)} = pd.to_numeric({_col(params)}, errors=\"coerce\").round({int(params.get('decimals', 0))})"]
    if name == "replace_values":
        return [f"{_col(params)} = {_col(params)}.replace({_r(params['find'])}, {_r(params.get('replace'))})"]
    if name == "filter_rows":
        comparison = str(params.get("comparison", "")).lower()
        value = _comparison_value(frame, params)
        if hasattr(value, "item"):
            value = value.item()
        mask = _MASKS[comparison].format(c=_col(params), v=_r(value))
        keep = str(params.get("mode", "keep")).lower() == "keep"
        return [f"df = df[{mask if keep else '~(' + mask + ')'}].copy()"]
    raise OperationFailedError(f"No script template for {name!r}")


_LABELS = {
    "fill_median": "Fill missing {col} with the median",
    "fill_mean": "Fill missing {col} with the mean",
    "fill_mode": "Fill missing {col} with the most common value",
    "fill_constant": "Fill missing {col} with a fixed value",
    "drop_rows_with_nulls": "Drop rows where {col} is empty",
    "drop_duplicate_rows": "Remove duplicate rows, keeping the first",
    "cast_to_numeric": "Convert {col} to numbers",
    "cast_to_datetime": "Convert {col} to dates",
    "standardize_case": "Make capitalisation in {col} consistent",
    "trim_whitespace": "Remove spaces around the text in {col}",
    "standardize_dates": "Rewrite the dates in {col} in one format",
    "cap_outliers": "Clamp extreme values in {col}",
    "drop_outlier_rows": "Drop rows with extreme values in {col}",
    "drop_column": "Remove the column {col}",
    "rename_column": "Rename {col}",
    "round_numeric": "Round {col}",
    "replace_values": "Replace a value in {col}",
    "filter_rows": "Keep or drop rows by {col}",
}


def _learned_note(name: str, learned: dict | None) -> str:
    if not learned:
        return ""
    if "fill_value" in learned:
        return f" -- {learned['fill_value']!r}, as learned from the original file"
    return f" -- limits {learned['lower']!r} to {learned['upper']!r}, as learned from the original file"


def build_cleaning_script(original: pd.DataFrame, action_log: list[dict], *, file_name: str = "original.csv") -> str:
    """Renders the applied steps of `action_log` as a runnable script.

    Replays the log over `original` as it renders, so each step is written against the data as
    it stood at that point -- which is where learned values and filter comparisons come from.
    A step that cannot be reproduced is written as a comment saying so, not left out silently.
    """
    frame = original
    blocks: list[str] = []
    applied = 0
    for entry in action_log:
        if entry.get("status") != "applied":
            continue
        name = entry.get("operation", "")
        params = dict(entry.get("operation_params") or {})
        spec = operations.OPERATIONS.get(name)
        if spec is not None and spec.needs_column:
            params.setdefault("column", entry.get("column_name"))
        label = _LABELS.get(name, name).format(col=repr(params.get("column")))
        try:
            learned = operations.learn(frame, name, params)
            lines = _lines(name, params, learned, frame)
            frame, _before, _after = operations.apply_operation(frame, name, params, learned=learned)
        except (OperationFailedError, operations.UnknownOperationError, KeyError) as exc:
            blocks.append(f"# NOT REPRODUCED: {label} ({exc}). Recorded as applied, but it could not\n"
                          f"# be run again against the original file.")
            continue
        applied += 1
        source = " (from a chat instruction)" if entry.get("source") == "conversational" else ""
        blocks.append(f"# {applied}. {label}{_learned_note(name, learned)}{source}\n" + "\n".join(lines))

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    from app.services import run_metadata

    # The cleaning ran in this process, so its versions are this process's.
    env = run_metadata.environment()
    body = "\n\n".join(blocks) if blocks else "# No fixes were applied in this session."
    return f'''"""Reproduces the cleaning DataClean AI applied to {file_name}.

Generated {generated_at}. {applied} fix{"es" if applied != 1 else ""}, in the order they were applied.

Run it on the original upload and it reproduces the session's cleaned file. Run it on a newer
export with the same columns and it cleans that file the same way: every value a step learned
from the data -- a median, a most-common value, outlier limits -- is written in below as a
fixed number rather than recomputed, so new rows are filled and clipped exactly as the rows a
model was trained on were.

Produced with Python {env.get("python")}, pandas {env.get("pandas")}, numpy {env.get("numpy")}. The
script says so when the installed versions differ; it runs either way.

    {run_metadata.install_line(env, ("pandas", "numpy"))}
    python clean.py {file_name} cleaned.csv

The output is written exactly as the data is, because it is the input a model is trained on. A
cell that begins with =, +, -, @ or a tab is read by a spreadsheet as a formula: before opening
the output in one, prefix such cells with a ' (the app's own CSV downloads already do).
"""

import sys

import pandas as pd

{run_metadata.versions_block(env, ("pandas", "numpy"))}

source = sys.argv[1] if len(sys.argv) > 1 else {file_name!r}
destination = sys.argv[2] if len(sys.argv) > 2 else "cleaned.csv"
df = pd.read_csv(source)

{body}

df.to_csv(destination, index=False)
print(f"Wrote {{len(df)}} rows to {{destination}}")
'''
