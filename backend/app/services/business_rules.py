"""Facts about the data that no statistic can discover.

Everything else in this app reasons from the numbers in the file. That finds a great deal --
missing values, duplicates, outliers, leakage -- and it is structurally incapable of finding
the largest category of real-world data error, because those errors are not statistically
unusual. `age = -5` is not an outlier; it is impossible. A 150% discount sits comfortably
inside the interquartile range of a column that runs 0 to 200 because half the rows are wrong.
A delivery date before its order date is two perfectly ordinary dates. The IQR check will
happily flag a legitimate premium salary and walk straight past all three.

The missing ingredient is not a better statistic. It is knowledge of what the columns *mean*,
and that knowledge lives with the user. So this module gives them somewhere to put it.

Three deliberate constraints:

**A rule is data, never code.** No `eval`, no expression strings, no `df.query`. A rule is a
frozen dataclass with a named kind, and evaluating one is a match statement over four cases.
This is the same reason the cleaning path selects from a whitelist of operations rather than
generating pandas: a rule arriving from an LLM, a saved session or an API request is untrusted
input, and untrusted input that can express arbitrary computation is a hole no amount of
validation closes reliably.

**A proposed rule is a suggestion, not a rule.** `propose` reads column names and profile
statistics and guesses. Guesses are wrong often enough that acting on one unapproved would be
worse than having no rules at all -- a rule the user never saw, quietly marking a fifth of
their rows as violations, is exactly the failure this app exists to avoid. Proposals carry
`confidence` and the evidence behind them, and nothing evaluates until the user says so.

**A violation is a finding, not a fix.** This module never drops or edits a row. It reports
what breaks the rule and how much of the file that is; what to do about it is a decision made
the same way every other one is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from app.services.readiness_checks import Check

# How close two floats have to be for `price * quantity == total` to count as holding. Money
# is the overwhelmingly common case for arithmetic rules and money is rounded, so an exact
# comparison fires on almost every row of almost every real file -- which trains people to
# ignore the rule. Relative, because a penny of drift means something different on a £3 order
# than on a £30,000 one.
ARITHMETIC_RELATIVE_TOLERANCE = 0.01
ARITHMETIC_ABSOLUTE_TOLERANCE = 0.01

# Below this share of rows a violation is a data-entry mistake worth showing; above it, the
# rule itself is the more likely error and saying so is more useful than reporting 60% of the
# file as broken.
RULE_LIKELY_WRONG_ABOVE = 0.30

RuleKind = Literal["range", "comparison", "arithmetic"]
Comparison = Literal["lte", "lt", "gte", "gt"]

# Ordering words rather than time words. Dates are the motivating case, but a comparison rule
# is equally valid between two numbers -- `min_price <= max_price` -- and "must not be later
# than 'max_price'" would be nonsense. "come after" reads naturally for dates and acceptably
# for anything else that has an order.
_COMPARISON_WORDS: dict[str, str] = {
    "lte": "must not come after",
    "lt": "must come before",
    "gte": "must not come before",
    "gt": "must come after",
}

_COMPARISON_OPERATORS: dict[str, str] = {"lte": "<=", "lt": "<", "gte": ">=", "gt": ">"}


class InvalidRule(ValueError):
    """A rule that cannot be evaluated as written.

    Raised at construction rather than at evaluation time, so a malformed rule is rejected
    when it is submitted and not halfway through a readiness check.
    """


@dataclass(frozen=True)
class Rule:
    """One statement about what the data is allowed to contain.

    Deliberately a closed set of kinds rather than an expression. See the module docstring.

    - `range`: one column against a floor and/or a ceiling. `age >= 0`, `discount <= 100`.
    - `comparison`: one column against another. `order_date <= delivery_date`.
    - `arithmetic`: two columns multiplied against a third. `price * quantity == total`.
    """

    kind: RuleKind
    column: str
    label: str = ""
    minimum: float | None = None
    maximum: float | None = None
    other_column: str | None = None
    comparison: Comparison | None = None
    multiplier_column: str | None = None
    result_column: str | None = None
    # Set on proposals only. A rule the user has approved has no confidence attached to it,
    # because it is no longer a guess -- it is something they told us.
    confidence: float | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.column:
            raise InvalidRule("A rule has to name a column.")
        if self.kind == "range":
            if self.minimum is None and self.maximum is None:
                raise InvalidRule(f"'{self.column}' needs a minimum, a maximum, or both.")
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise InvalidRule(
                    f"'{self.column}' cannot be both at least {self.minimum} and at most "
                    f"{self.maximum} -- no value satisfies that."
                )
        elif self.kind == "comparison":
            if not self.other_column or not self.comparison:
                raise InvalidRule(f"'{self.column}' needs another column to compare against.")
            if self.other_column == self.column:
                raise InvalidRule(f"'{self.column}' cannot be compared against itself.")
        elif self.kind == "arithmetic":
            if not self.multiplier_column or not self.result_column:
                raise InvalidRule(
                    f"'{self.column}' needs a column to multiply by and a column holding "
                    f"the result."
                )
        else:  # pragma: no cover -- guarded by the Literal, kept for hand-built dicts
            raise InvalidRule(f"Unknown rule kind {self.kind!r}.")

    @property
    def name(self) -> str:
        """Stable identifier, used as the check name so a rule keeps its identity."""
        if self.kind == "range":
            return f"rule_range_{self.column}"
        if self.kind == "comparison":
            return f"rule_compare_{self.column}_{self.other_column}"
        return f"rule_arithmetic_{self.result_column}"

    @property
    def sentence(self) -> str:
        """The rule as the user would say it out loud."""
        if self.label:
            return self.label
        if self.kind == "range":
            if self.minimum is not None and self.maximum is not None:
                return f"'{self.column}' is between {_number(self.minimum)} and {_number(self.maximum)}"
            if self.minimum is not None:
                return f"'{self.column}' is never below {_number(self.minimum)}"
            return f"'{self.column}' is never above {_number(self.maximum)}"
        if self.kind == "comparison":
            word = _COMPARISON_WORDS.get(self.comparison or "lte", "compares to")
            return f"'{self.column}' {word} '{self.other_column}'"
        return (
            f"'{self.column}' x '{self.multiplier_column}' equals '{self.result_column}'"
        )

    @property
    def expression(self) -> str:
        """The rule in the compact form, for the audit report's table."""
        if self.kind == "range":
            parts = []
            if self.minimum is not None:
                parts.append(f"{self.column} >= {_number(self.minimum)}")
            if self.maximum is not None:
                parts.append(f"{self.column} <= {_number(self.maximum)}")
            return " and ".join(parts)
        if self.kind == "comparison":
            operator = _COMPARISON_OPERATORS.get(self.comparison or "lte", "<=")
            return f"{self.column} {operator} {self.other_column}"
        return f"{self.column} * {self.multiplier_column} == {self.result_column}"

    def columns_used(self) -> list[str]:
        named = [self.column, self.other_column, self.multiplier_column, self.result_column]
        return [c for c in named if c]

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "column": self.column,
            "label": self.label,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "other_column": self.other_column,
            "comparison": self.comparison,
            "multiplier_column": self.multiplier_column,
            "result_column": self.result_column,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "name": self.name,
            "sentence": self.sentence,
            "expression": self.expression,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Rule":
        """Rebuild a rule from stored or submitted JSON.

        Unknown keys are dropped rather than raising: rules are persisted with derived fields
        (`name`, `sentence`, `expression`) so the frontend can render them without
        recomputing, and those come back on a round trip.
        """
        allowed = {
            "kind", "column", "label", "minimum", "maximum", "other_column",
            "comparison", "multiplier_column", "result_column", "confidence", "rationale",
        }
        return cls(**{k: v for k, v in payload.items() if k in allowed})


def _number(value: float | None) -> str:
    if value is None:
        return ""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.4g}"


@dataclass
class Violation:
    """What a rule found, with enough detail to act on it."""

    rule: Rule
    rows: int
    checked: int
    examples: list[str] = field(default_factory=list)

    @property
    def share(self) -> float:
        return self.rows / self.checked if self.checked else 0.0


def evaluate(df: pd.DataFrame, rule: Rule) -> Violation | None:
    """Count the rows that break a rule.

    Returns None when the rule cannot be applied at all -- a column it names is gone, or the
    columns are the wrong type. That is different from "no violations", and the caller renders
    the two differently: a rule that could not run is reported as not run, on the same
    principle as every other check in this app.
    """
    missing = [c for c in rule.columns_used() if c not in df.columns]
    if missing:
        return None

    if rule.kind == "range":
        series = pd.to_numeric(df[rule.column], errors="coerce")
        usable = series.notna()
        if not usable.any():
            return None
        broken = pd.Series(False, index=df.index)
        if rule.minimum is not None:
            broken |= series < rule.minimum
        if rule.maximum is not None:
            broken |= series > rule.maximum
        broken &= usable
        examples = [_number(v) for v in series[broken].head(3).tolist()]
        return Violation(rule, int(broken.sum()), int(usable.sum()), examples)

    if rule.kind == "comparison":
        left, right = _comparable_pair(df[rule.column], df[rule.other_column or ""])
        if left is None or right is None:
            return None
        usable = left.notna() & right.notna()
        if not usable.any():
            return None
        held = {
            "lte": left <= right,
            "lt": left < right,
            "gte": left >= right,
            "gt": left > right,
        }[rule.comparison or "lte"]
        broken = usable & ~held
        examples = [
            f"{_render(a)} vs {_render(b)}"
            for a, b in zip(
                df.loc[broken, rule.column].head(3),
                df.loc[broken, rule.other_column or ""].head(3),
            )
        ]
        return Violation(rule, int(broken.sum()), int(usable.sum()), examples)

    left = pd.to_numeric(df[rule.column], errors="coerce")
    right = pd.to_numeric(df[rule.multiplier_column or ""], errors="coerce")
    result = pd.to_numeric(df[rule.result_column or ""], errors="coerce")
    usable = left.notna() & right.notna() & result.notna()
    if not usable.any():
        return None
    expected = left * right
    # Relative *and* absolute, because a purely relative tolerance is infinitely strict at
    # zero and a purely absolute one is meaninglessly loose on large numbers.
    slack = (expected.abs() * ARITHMETIC_RELATIVE_TOLERANCE).clip(
        lower=ARITHMETIC_ABSOLUTE_TOLERANCE
    )
    broken = usable & ((result - expected).abs() > slack)
    examples = [
        f"{_number(l)} x {_number(r)} = {_number(l * r)}, recorded as {_number(a)}"
        for l, r, a in zip(
            left[broken].head(3), right[broken].head(3), result[broken].head(3)
        )
    ]
    return Violation(rule, int(broken.sum()), int(usable.sum()), examples)


def _comparable_pair(left: pd.Series, right: pd.Series):
    """Coerce two columns to a type they can be ordered in, or give up.

    Dates first: `order_date <= delivery_date` is the motivating case, and dates arriving as
    strings is the normal state of a CSV. Numbers second. Anything else -- two text columns --
    has no meaningful ordering and the rule is reported as not run rather than compared
    alphabetically, which would produce confident nonsense.
    """
    as_dates = (
        pd.to_datetime(left, errors="coerce", format="mixed"),
        pd.to_datetime(right, errors="coerce", format="mixed"),
    )
    if as_dates[0].notna().any() and as_dates[1].notna().any():
        return as_dates

    as_numbers = (
        pd.to_numeric(left, errors="coerce"),
        pd.to_numeric(right, errors="coerce"),
    )
    if as_numbers[0].notna().any() and as_numbers[1].notna().any():
        return as_numbers

    return None, None


def _render(value) -> str:
    if pd.isna(value):
        return "blank"
    return str(value)[:24]


def check_rule(df: pd.DataFrame, rule: Rule) -> Check:
    """One rule, as a Check -- so it reaches the panel, the summary and the audit report.

    That is the whole reason the provenance refactor came first. A rule does not need its own
    display, its own summary format or its own section in the report; it needs to be a Check,
    and everything downstream already knows what to do with one.
    """
    label = f"Rule: {rule.sentence}"
    violation = evaluate(df, rule)

    if violation is None:
        missing = [c for c in rule.columns_used() if c not in df.columns]
        reason = (
            f"'{missing[0]}' is not in the data."
            if missing
            else f"The columns this rule compares hold no values it can order."
        )
        return Check(
            name=rule.name,
            label=label,
            status="not_run",
            finding=f"This rule could not be checked. {reason}",
        )

    if violation.rows == 0:
        return Check(
            name=rule.name,
            label=label,
            status="passed",
            finding=f"Every row satisfies {rule.expression}.",
            evidence=f"Checked {violation.checked:,} rows.",
        )

    examples = f" For example: {'; '.join(violation.examples)}." if violation.examples else ""

    # Past a certain share the rule is the more likely thing to be wrong. Saying so is more
    # useful than reporting most of the file as broken, and it is the honest reading: a
    # constraint two thirds of the data disagrees with was probably mis-stated.
    if violation.share > RULE_LIKELY_WRONG_ABOVE:
        return Check(
            name=rule.name,
            label=label,
            status="warning",
            finding=f"{violation.rows:,} rows ({violation.share:.0%}) break {rule.expression}.",
            evidence=f"That is more than {RULE_LIKELY_WRONG_ABOVE:.0%} of the "
            f"{violation.checked:,} rows checked, which more often means the rule is wrong "
            f"than that the data is.{examples}",
            remedy="Check the rule states what you meant before treating these as errors.",
            severity="warning",
        )

    return Check(
        name=rule.name,
        label=label,
        status="failed",
        finding=f"{violation.rows:,} of {violation.checked:,} rows "
        f"({violation.share:.1%}) break {rule.expression}.",
        evidence=f"These are not unusual values -- they are impossible ones, and no "
        f"statistical check finds them.{examples}",
        remedy="Decide whether these rows are wrong or the rule is, then drop or correct "
        "them from the cleaning box.",
        severity="warning",
    )


def check_all(df: pd.DataFrame, rules: list[Rule]) -> list[Check]:
    return [check_rule(df, rule) for rule in rules]


# --- proposing rules -------------------------------------------------------------------
#
# Everything below guesses, and is written to guess narrowly. A proposal costs the user
# attention whether it is right or wrong, and a list padded with plausible-looking rules that
# find nothing is how people learn to approve without reading -- which would make the approval
# step, the thing this whole app is built around, worthless.
#
# So the bar is: propose a rule only when it would find something *in this file*. A rule that
# holds on every row may still be worth recording, but it is not worth interrupting someone
# to ask about, and it is not what they came here for.

# Columns whose names leave no room for a negative value. Deliberately short. `balance`,
# `profit`, `margin`, `change`, `delta`, `adjustment`, `score` and `temperature` are all
# legitimately negative and are the reason this is a list of known-safe words rather than a
# guess at what sounds quantity-like.
_NEVER_NEGATIVE = (
    "age", "quantity", "qty", "count", "price", "cost", "salary", "income", "tenure",
    "duration", "distance", "weight", "height", "units", "stock", "views", "clicks",
    "visits", "attempts", "mileage", "volume",
)

# Words that make a 0-100 bound near-certain. `rate` and `ratio` are excluded: `hourly_rate`
# and `debt_to_income_ratio` are both common and neither is a percentage.
_PERCENTAGE = ("percent", "pct", "percentage", "discount")

# A pair of dates has to disagree somewhere to be worth proposing, but it also has to mostly
# agree -- two unrelated date columns cross each other about half the time, and proposing an
# ordering between them would be noise.
MIN_SHARE_FOR_IMPLIED_ORDER = 0.90

_EARLIER_WORDS = ("start", "begin", "open", "order", "created", "birth", "signup",
                  "registration", "admission", "hire", "issue", "first", "from")
_LATER_WORDS = ("end", "close", "delivery", "shipped", "dispatch", "updated", "death",
                "churn", "cancel", "discharge", "termination", "exit", "expiry",
                "expiration", "due", "last", "to")

_UNIT_WORDS = ("price", "unit_price", "unit_cost", "rate", "cost_per")
_COUNT_WORDS = ("quantity", "qty", "units", "count", "hours", "days", "items")
_TOTAL_WORDS = ("total", "amount", "revenue", "subtotal", "line_total", "gross", "net")


def _has_word(column: str, words: tuple[str, ...]) -> bool:
    lowered = column.lower()
    return any(word in lowered for word in words)


def _as_dates(series: pd.Series) -> pd.Series | None:
    """Parse a column as dates, or decide it isn't one.

    Requires most non-null values to parse. A column of free text where three entries happen
    to look like years is not a date column, and treating it as one produces date-ordering
    proposals about nothing.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None

    # Numeric columns are excluded outright. `pd.to_datetime` reads a float as nanoseconds
    # since the epoch and succeeds on every row, so without this every numeric column is a
    # date column -- which produced the proposal "'unit_price' must not be later than 'age'".
    # A date stored as a bare integer (20240115) exists but is rare and ambiguous; guessing
    # at it is not worth generating confident nonsense about every price and count in the
    # file. Same reasoning as excluding numerics from the repeated-entity check.
    if pd.api.types.is_numeric_dtype(series):
        return None
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except Exception:
        try:
            parsed = pd.to_datetime(series, errors="coerce")
        except Exception:
            return None
    if parsed.notna().sum() < len(non_null) * 0.8:
        return None
    return parsed


def propose(df: pd.DataFrame, limit: int = 8) -> list[Rule]:
    """Rules worth asking the user about, most confident first.

    Nothing here is applied. These are questions, and the answer is the user's.
    """
    proposals: list[Rule] = []
    numeric = {
        c: pd.to_numeric(df[c], errors="coerce")
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) or _has_word(c, _NEVER_NEGATIVE + _PERCENTAGE)
    }
    numeric = {c: s for c, s in numeric.items() if s.notna().any()}

    # --- percentages outside 0-100 -------------------------------------------------------
    #
    # Run before the non-negative pass so a column that gets a 0-100 bound does not also get a
    # bare "never below 0". The wider rule already says everything the narrower one would, and
    # two overlapping rules on one column is two approvals for one decision.
    bounded: set[str] = set()
    for column, series in numeric.items():
        if not _has_word(column, _PERCENTAGE):
            continue
        # A column recorded as a fraction (0.0-1.0) is not broken, just on a different scale,
        # and bounding it at 100 would be true but useless. It still gets the non-negative
        # rule below, which is the part that means something at that scale.
        if series.max() <= 1.0:
            continue
        outside = int(((series < 0) | (series > 100)).sum())
        if outside == 0:
            continue
        bounded.add(column)
        proposals.append(
            Rule(
                kind="range",
                column=column,
                minimum=0,
                maximum=100,
                confidence=0.85,
                rationale=f"'{column}' reads as a percentage and {outside:,} "
                f"{'row falls' if outside == 1 else 'rows fall'} outside 0-100.",
            )
        )

    # --- values that cannot be negative --------------------------------------------------
    for column, series in numeric.items():
        if column in bounded or not _has_word(column, _NEVER_NEGATIVE + _PERCENTAGE):
            continue
        negatives = int((series < 0).sum())
        if negatives == 0:
            continue
        proposals.append(
            Rule(
                kind="range",
                column=column,
                minimum=0,
                confidence=0.9,
                rationale=f"'{column}' holds {negatives:,} negative "
                f"{'value' if negatives == 1 else 'values'}, and nothing this column could "
                f"reasonably measure goes below zero.",
            )
        )

    # --- dates that should not cross -----------------------------------------------------
    dates = {c: parsed for c in df.columns if (parsed := _as_dates(df[c])) is not None}
    for first in dates:
        for second in dates:
            if first >= second:
                continue
            for earlier, later in ((first, second), (second, first)):
                both = dates[earlier].notna() & dates[later].notna()
                if both.sum() < 10:
                    continue
                holds = (dates[earlier] <= dates[later]) & both
                share = holds.sum() / both.sum()
                broken = int(both.sum() - holds.sum())
                if share < MIN_SHARE_FOR_IMPLIED_ORDER or broken == 0:
                    continue
                named = _has_word(earlier, _EARLIER_WORDS) and _has_word(later, _LATER_WORDS)
                proposals.append(
                    Rule(
                        kind="comparison",
                        column=earlier,
                        other_column=later,
                        comparison="lte",
                        confidence=min(0.95, share) if named else min(0.7, share),
                        rationale=f"'{earlier}' comes before '{later}' in {share:.0%} of rows "
                        f"and after it in {broken:,}"
                        + (
                            ", and the names say which should come first."
                            if named
                            else " -- which looks like an ordering the data mostly keeps."
                        ),
                    )
                )

    # --- arithmetic that does not add up -------------------------------------------------
    units = [c for c in numeric if _has_word(c, _UNIT_WORDS)]
    counts = [c for c in numeric if _has_word(c, _COUNT_WORDS)]
    totals = [c for c in numeric if _has_word(c, _TOTAL_WORDS)]
    for unit in units:
        for count in counts:
            for total in totals:
                if len({unit, count, total}) != 3:
                    continue
                rule = Rule(
                    kind="arithmetic",
                    column=unit,
                    multiplier_column=count,
                    result_column=total,
                )
                violation = evaluate(df, rule)
                if violation is None or violation.checked < 10:
                    continue
                agreement = 1 - violation.share
                if agreement < MIN_SHARE_FOR_IMPLIED_ORDER or violation.rows == 0:
                    continue
                proposals.append(
                    Rule(
                        kind="arithmetic",
                        column=unit,
                        multiplier_column=count,
                        result_column=total,
                        confidence=min(0.95, agreement),
                        rationale=f"'{unit}' x '{count}' equals '{total}' in {agreement:.0%} "
                        f"of rows and misses in {violation.rows:,}.",
                    )
                )

    proposals.sort(key=lambda r: r.confidence or 0, reverse=True)
    return proposals[:limit]
