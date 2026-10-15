"""What a model expects as input, written down: a schema saved with every run.

What a model needed used to be implicit -- the column lists in its result plus whatever types happened to be in the
training file -- and the apply flow inferred and coerced as it went. There was no contract a user, a script or a
later service could check a new file against. `build` writes one: for every *source* column the model needs (the
file's own columns, before derived inputs are made, including the ones only a derived input reads), its kind, its
type at training, how often it was blank, its range and quantiles or the categories seen with their counts, and
which derived inputs read it; plus the target's name and labels.

Built from the training rows only. A schema is a description of what the model learned from, and the held-out rows
are not that; a value seen only in a test row must not appear in it. The integrity suite changes the test rows and
checks the schema does not move.
"""

from __future__ import annotations

import pandas as pd

SCHEMA_VERSION = 1
# Categories listed per column, most common first. Past this the schema says how many there were.
MAX_CATEGORIES = 50
QUANTILES = (0.01, 0.25, 0.5, 0.75, 0.99)


def recipe_sources(recipe: dict) -> list[str]:
    """The file columns a derived input is made from: its column, and both halves of an interaction."""
    evidence = recipe.get("evidence") or {}
    found = [recipe.get("column"), evidence.get("a"), evidence.get("b")]
    return [c for c in dict.fromkeys(found) if c]


def _number(value) -> float | None:
    return None if pd.isna(value) else round(float(value), 6)


def _describe(series: pd.Series, kind: str) -> dict:
    blank = series.isna()
    entry: dict = {
        "dtype": str(series.dtype),
        "nullable": bool(blank.any()),
        "blank_share": round(float(blank.mean()), 4) if len(series) else 0.0,
    }
    present = series[~blank]
    if kind == "numeric":
        values = pd.to_numeric(present, errors="coerce").dropna()
        entry["range"] = {"min": _number(values.min()), "max": _number(values.max())} if len(values) else None
        entry["quantiles"] = (
            {str(q): _number(values.quantile(q)) for q in QUANTILES} if len(values) else None
        )
    elif kind == "date":
        parsed = pd.to_datetime(present, errors="coerce", format="mixed").dropna()
        entry["range"] = (
            {"min": parsed.min().isoformat(), "max": parsed.max().isoformat()} if len(parsed) else None
        )
    elif kind == "text":
        lengths = present.astype(str).str.len()
        entry["length"] = {"mean": _number(lengths.mean()), "max": _number(lengths.max())} if len(lengths) else None
    else:
        counts = present.astype(str).value_counts()
        entry["distinct"] = int(len(counts))
        entry["categories"] = {str(k): int(v) for k, v in counts.head(MAX_CATEGORIES).items()}
        entry["categories_truncated"] = bool(len(counts) > MAX_CATEGORIES)
    return entry


def build(train: pd.DataFrame, features: dict, target_column: str, task: str, y_train: pd.Series) -> dict:
    """The input schema of a model trained on `train` (the training rows of the file, with its own columns)."""
    recipes = features.get("derived") or []
    derived = {o for r in recipes for o in r.get("outputs") or []}
    used = [*(features.get("numeric") or []), *(features.get("categorical") or []), *(features.get("frequent") or [])]
    read_by: dict[str, list[str]] = {}
    kinds_from_recipes: dict[str, str] = {}
    for recipe in recipes:
        for column in recipe_sources(recipe):
            read_by.setdefault(column, []).append(recipe.get("id"))
            if recipe.get("kind") == "date_parts":
                kinds_from_recipes[column] = "date"
            elif recipe.get("kind") == "text_length":
                kinds_from_recipes[column] = "text"

    sources = [c for c in dict.fromkeys([*[c for c in used if c not in derived], *read_by]) if c in train.columns]
    columns = []
    for name in sources:
        series = train[name]
        kind = kinds_from_recipes.get(name) or ("numeric" if pd.api.types.is_numeric_dtype(series) else "categorical")
        columns.append({
            "name": name,
            "kind": kind,
            "used_directly": name in used,
            "read_by": read_by.get(name, []),
            **_describe(series, kind),
        })

    labels = pd.Series(y_train)
    target: dict = {"name": target_column, "task": task}
    if task == "classification":
        target["labels"] = {str(k): int(v) for k, v in labels.astype(str).value_counts().sort_index().items()}
    else:
        values = pd.to_numeric(labels, errors="coerce").dropna()
        target["range"] = {"min": _number(values.min()), "max": _number(values.max())} if len(values) else None
    return {
        "schema_version": SCHEMA_VERSION,
        "built_from": "training rows",
        "rows": int(len(train)),
        "columns": columns,
        "target": target,
    }
