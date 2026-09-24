"""Derived inputs: turning columns the model cannot use into ones it can.

The improvement loop proved where the ceiling is. On the churn sample the settings search found
nothing worth keeping, which means the ceiling is the columns, not the settings -- and the app
was leaving real signal on the table. `signup_date` was dropped entirely ("too many values")
when tenure derived from it is obviously predictive of churn. A column of free text was dropped
as an "identifier" because nearly every value was distinct, which reaches the right action for
the wrong reason and, in a tool whose premise is that the reason is the product, a wrong reason
is the defect.

Four kinds of recipe, each a declared piece of data rather than code, so the exported script
can reproduce it line for line:

- **date_parts**: year, month, weekday and days-since-epoch from a date column.
- **text_length**: characters and words in a free-text column.
- **missing_flag**: 0/1 for whether a value was blank. We already know blankness can carry
  signal -- the leakage detector looks for exactly this -- so where it is legitimate it is
  worth an input.
- **frequent_categories**: a high-cardinality column kept rather than dropped, with its most
  common values one-hot encoded and everything else pooled. Done inside the pipeline with
  scikit-learn's own encoder, so the fitted model stays loadable anywhere.

The first three are row-wise and stateless: a row's weekday does not depend on any other row,
so deriving it before the split cannot leak. The fourth learns which values are common, so it
is fitted inside the pipeline on the training fold like every other learned statistic.

Nothing here applies itself. `propose` returns candidates with the evidence behind each; the
plan shows them ticked; only the ids the user sends back are applied. Same shape as the
cleaning cards and the business rules, for the same reason.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

KINDS = ("date_parts", "text_length", "missing_flag", "frequent_categories")

# A column is free text when the values are mostly distinct *and* look like prose. Distinct
# alone is what the identifier rule uses, and a paragraph of clinical notes is distinct too.
FREE_TEXT_MIN_DISTINCT_SHARE = 0.5
FREE_TEXT_MIN_MEAN_CHARS = 25
FREE_TEXT_MIN_MEAN_WORDS = 4

DATE_PARSE_SHARE = 0.9  # share of non-null values that must read as dates

# A missing flag on a column that is blank in 1 row of 10,000 is a constant with a footnote;
# on a column that is blank in all but a handful it is the same thing inverted.
MISSING_FLAG_MIN_SHARE = 0.01
MISSING_FLAG_MAX_SHARE = 0.99

# How many of a high-cardinality column's values get their own input inside the pipeline.
# The rest are pooled as "other". Fixed rather than scaled with row count so the exported
# script can state it as a constant.
MAX_FREQUENT_CATEGORIES = 20
# ...and the column has to be worth it: the top values must cover this much of the rows, or
# nearly every row lands in "other" and the input is a constant.
FREQUENT_MIN_COVERAGE = 0.3

_EPOCH = pd.Timestamp("1970-01-01")


@dataclass(frozen=True)
class Recipe:
    """One derived input, described rather than coded."""

    id: str
    kind: str
    column: str
    outputs: list[str]
    reason: str
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# --- shape tests -----------------------------------------------------------------------------


def parse_dates(series: pd.Series) -> pd.Series | None:
    """The column as datetimes if enough of it reads as dates, else None."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return None
    non_null = series.dropna()
    if non_null.empty:
        return None
    try:
        parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return None
    if parsed.notna().mean() < DATE_PARSE_SHARE:
        return None
    return parsed


def looks_like_dates(series: pd.Series) -> bool:
    parsed = parse_dates(series)
    return parsed is not None and parsed.dt.normalize().nunique() > 1


def looks_like_free_text(series: pd.Series) -> bool:
    """Prose, not a category and not a key.

    An order number is distinct per row and short; a city is repeated and short; a
    complaint is distinct per row and long. Only the last is text."""
    if series.dtype != object:
        return False
    non_null = series.dropna().astype(str)
    if len(non_null) < 20:
        return False
    if non_null.nunique() / len(non_null) < FREE_TEXT_MIN_DISTINCT_SHARE:
        return False
    if parse_dates(series) is not None:
        return False
    mean_chars = float(non_null.str.len().mean())
    mean_words = float(non_null.str.split().str.len().mean())
    return mean_chars >= FREE_TEXT_MIN_MEAN_CHARS and mean_words >= FREE_TEXT_MIN_MEAN_WORDS


# --- proposing -------------------------------------------------------------------------------


def _date_parts(column: str, series: pd.Series) -> Recipe | None:
    parsed = parse_dates(series)
    if parsed is None or parsed.dt.normalize().nunique() <= 1:
        return None
    # Only the parts that vary. A file covering one month has no year or month to learn from,
    # and offering them as inputs would be padding the list.
    parts = {
        "year": parsed.dt.year,
        "month": parsed.dt.month,
        "weekday": parsed.dt.weekday,
        "days": (parsed.dt.normalize() - _EPOCH).dt.days,
    }
    outputs = [f"{column}__{name}" for name, values in parts.items() if values.nunique() > 1]
    if not outputs:
        return None
    span = parsed.max() - parsed.min()
    return Recipe(
        id=f"date_parts:{column}",
        kind="date_parts",
        column=column,
        outputs=outputs,
        reason=(
            f"'{column}' holds dates spanning {span.days:,} days, which a model can't use as "
            f"a category -- nearly every row is a different day. Split into "
            f"{', '.join(o.split('__', 1)[1] for o in outputs)}, it becomes "
            f"{len(outputs)} usable input{'s' if len(outputs) != 1 else ''}: how long ago, "
            f"and any seasonal or weekly pattern."
        ),
        evidence={
            "distinct_days": int(parsed.dt.normalize().nunique()),
            "span_days": int(span.days),
            "earliest": str(parsed.min().date()),
            "latest": str(parsed.max().date()),
        },
    )


def _text_length(column: str, series: pd.Series) -> Recipe | None:
    if not looks_like_free_text(series):
        return None
    non_null = series.dropna().astype(str)
    mean_chars = float(non_null.str.len().mean())
    return Recipe(
        id=f"text_length:{column}",
        kind="text_length",
        column=column,
        outputs=[f"{column}__length", f"{column}__words"],
        reason=(
            f"'{column}' is free text averaging {mean_chars:,.0f} characters. A model can't "
            f"read prose as a category, but how much was written is often informative -- a "
            f"long note and a blank one are different events. Its length in characters and "
            f"words become inputs; the text itself is left out."
        ),
        evidence={
            "mean_chars": round(mean_chars, 1),
            "mean_words": round(float(non_null.str.split().str.len().mean()), 1),
            "distinct_share": round(float(non_null.nunique() / len(non_null)), 3),
        },
    )


def _missing_flag(column: str, series: pd.Series) -> Recipe | None:
    share = float(series.isna().mean())
    if not (MISSING_FLAG_MIN_SHARE <= share <= MISSING_FLAG_MAX_SHARE):
        return None
    return Recipe(
        id=f"missing_flag:{column}",
        kind="missing_flag",
        column=column,
        outputs=[f"{column}__missing"],
        reason=(
            f"'{column}' is blank in {share:.0%} of rows. The model fills those in, which "
            f"hides the fact that they were blank -- and that fact is often a signal in its "
            f"own right (a blank spend can mean a different kind of customer). A 0/1 input "
            f"keeps it."
        ),
        evidence={"missing_share": round(share, 4), "missing_rows": int(series.isna().sum())},
    )


def frequent_categories_recipe(column: str, series: pd.Series, limit: int) -> Recipe | None:
    """Public so classify_features can say, when it drops a column for having too many
    values, whether keeping the common ones is on offer."""
    return _frequent_categories(column, series, limit)


def _frequent_categories(column: str, series: pd.Series, limit: int) -> Recipe | None:
    if series.dtype != object or looks_like_free_text(series) or parse_dates(series) is not None:
        return None
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return None
    distinct = int(non_null.nunique())
    rows = len(non_null)
    if distinct <= limit or distinct / rows > 0.95:
        return None  # ordinary categorical, or an identifier -- neither is this recipe's job
    counts = non_null.value_counts()
    top = counts.head(MAX_FREQUENT_CATEGORIES)
    coverage = float(top.sum() / rows)
    if coverage < FREQUENT_MIN_COVERAGE:
        return None
    return Recipe(
        id=f"frequent_categories:{column}",
        kind="frequent_categories",
        column=column,
        outputs=[],  # encoded inside the pipeline; no new column in the frame
        reason=(
            f"'{column}' has {distinct:,} different values -- too many to give each its own "
            f"input without the model memorising rows. But its {len(top)} most common values "
            f"cover {coverage:.0%} of the rows. Those get an input each and the rest are "
            f"pooled as 'other', which keeps what is learnable and drops what is not."
        ),
        evidence={
            "distinct": distinct,
            "top_values": int(len(top)),
            "coverage": round(coverage, 3),
            "most_common": [str(v) for v in top.index[:5]],
        },
    )


def propose(
    df: pd.DataFrame, target_column: str, excluded: list[str] | None = None
) -> list[Recipe]:
    """Every derived input this file could support, with the evidence for each.

    Skips excluded columns: a missing flag on a column the user withheld for leaking the
    answer would leak it straight back in.
    """
    from app.services.modeling import category_limit

    skip = {target_column, *(excluded or [])}
    limit = category_limit(len(df))
    recipes: list[Recipe] = []
    for column in df.columns:
        if column in skip:
            continue
        series = df[column]
        if series.notna().sum() == 0:
            continue
        for candidate in (
            _date_parts(str(column), series),
            _text_length(str(column), series),
            _missing_flag(str(column), series),
            _frequent_categories(str(column), series, limit),
        ):
            if candidate is not None:
                recipes.append(candidate)
    return recipes


def select(
    df: pd.DataFrame,
    target_column: str,
    excluded: list[str] | None,
    wanted: list[str] | None,
) -> list[Recipe]:
    """The proposals whose ids the user sent back. Re-proposed rather than trusted from the
    request, so an id for a column that no longer supports the recipe is dropped quietly and
    a recipe on an excluded column cannot be smuggled in."""
    if not wanted:
        return []
    wanted_ids = set(wanted)
    return [r for r in propose(df, target_column, excluded) if r.id in wanted_ids]


# --- applying --------------------------------------------------------------------------------


def apply(df: pd.DataFrame, recipes: list[Recipe]) -> pd.DataFrame:
    """Adds the stateless derived columns. Returns a new frame; the input is untouched.

    `frequent_categories` adds nothing here -- it is a pipeline step, not a frame step.
    """
    if not recipes:
        return df
    out = df.copy()
    for recipe in recipes:
        if recipe.column not in out.columns:
            continue
        series = out[recipe.column]
        if recipe.kind == "date_parts":
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            parts = {
                "year": parsed.dt.year,
                "month": parsed.dt.month,
                "weekday": parsed.dt.weekday,
                "days": (parsed.dt.normalize() - _EPOCH).dt.days,
            }
            for output in recipe.outputs:
                part = output.rsplit("__", 1)[1]
                out[output] = parts[part].astype("float64")
        elif recipe.kind == "text_length":
            text = series.fillna("").astype(str)
            out[f"{recipe.column}__length"] = text.str.len().astype("float64")
            out[f"{recipe.column}__words"] = text.str.split().str.len().astype("float64")
        elif recipe.kind == "missing_flag":
            out[f"{recipe.column}__missing"] = series.isna().astype("float64")
    return out


def frequent_columns(recipes: list[Recipe]) -> list[str]:
    return [r.column for r in recipes if r.kind == "frequent_categories"]


def derived_from(recipes: list[Recipe]) -> dict[str, Recipe]:
    """Source column -> the recipe that replaces it in the inputs, for the ones that do."""
    return {r.column: r for r in recipes if r.kind in ("date_parts", "text_length")}


# --- reproducing -----------------------------------------------------------------------------


def as_pandas(recipes: list[Recipe], frame: str = "df") -> str:
    """The recipes as the pandas lines the exported script runs. Kept here beside `apply` so
    the two cannot drift: a test applies both to the same frame and compares."""
    lines: list[str] = []
    for recipe in recipes:
        col = repr(recipe.column)
        if recipe.kind == "date_parts":
            lines.append(f"_dates = pd.to_datetime({frame}[{col}], errors=\"coerce\", format=\"mixed\")")
            for output in recipe.outputs:
                part = output.rsplit("__", 1)[1]
                expr = {
                    "year": "_dates.dt.year",
                    "month": "_dates.dt.month",
                    "weekday": "_dates.dt.weekday",
                    "days": '(_dates.dt.normalize() - pd.Timestamp("1970-01-01")).dt.days',
                }[part]
                lines.append(f"{frame}[{output!r}] = {expr}.astype(\"float64\")")
        elif recipe.kind == "text_length":
            lines.append(f"_text = {frame}[{col}].fillna(\"\").astype(str)")
            lines.append(f"{frame}[{recipe.column + '__length'!r}] = _text.str.len().astype(\"float64\")")
            lines.append(f"{frame}[{recipe.column + '__words'!r}] = _text.str.split().str.len().astype(\"float64\")")
        elif recipe.kind == "missing_flag":
            lines.append(f"{frame}[{recipe.column + '__missing'!r}] = {frame}[{col}].isna().astype(\"float64\")")
    return "\n".join(lines)
