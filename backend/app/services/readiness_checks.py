"""Why the app believes what it believes about your data.

The rest of the tool spends four fields justifying a suggestion to fill one column's median,
and then asserts "Nothing found that would stop you training a model on this data" with no
working shown. That is backwards. The individual fix is the low-stakes claim; the verdict is
the one someone actually relies on -- and it is a claim of *absence*, which is precisely the
kind a reader cannot verify by looking at the data. Either they trust it or they don't.

So every check records what it concluded, the numbers behind it, and what would change it.
Three statuses matter, not two:

- **passed** -- it ran and found nothing. Worth saying out loud: "we looked for duplicate
  rows with contradictory answers and there are none" is information, and silence isn't.
- **failed** -- it ran and found something, with the evidence attached.
- **not_run** -- and *why*. This is the one that earns the module its place. Leakage
  detection needs a named target; class balance needs an inferred task. The old code returned
  early when the target was missing, so six checks silently never happened and the report was
  indistinguishable from one where they had all passed.

Deliberately a flat list with a few "blocked by" edges rather than a graph. The checks are
not causally chained -- they are independent conditions that all have to hold -- and drawing
arrows between them would encode a dependency that mostly isn't there.

Nothing here is stored. It is derived on every read, the same as the verdict it explains,
because the moment evidence is persisted an undo turns it into a lie.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Status = Literal["passed", "failed", "warning", "not_run"]
Severity = Literal["blocker", "warning"]

# Comparing every row against every other is O(rows) with hashing, but the frame copy and
# grouping still cost real time on a large file. Above this the check reports itself as not
# run rather than quietly making the readiness call slow.
MAX_ROWS_FOR_CONTRADICTION_CHECK = 200_000

# What "an entity repeating across rows" looks like: many distinct values, each appearing a
# handful of times. A two-value flag repeats far more often and means nothing here; a
# row-unique column repeats not at all. Named because the model plan offers exactly these
# columns as candidates for a grouped split -- proposing a different set there than the panel
# warns about would be two answers to one question.
MIN_DISTINCT_FOR_ENTITY = 20
MIN_REPETITION_FOR_ENTITY = 2.0


@dataclass(frozen=True)
class Check:
    """One question asked of the data, and the answer with its working.

    `remedy` is empty on a pass. On anything else it says what would change the answer --
    a finding the user cannot act on is only half-reported.
    """

    name: str
    label: str
    status: Status
    finding: str
    evidence: str = ""
    remedy: str = ""
    # Set when status is not_run: the name of the check whose failure made this one
    # impossible. The only kind of edge these checks genuinely have.
    blocked_by: str | None = None
    severity: Severity | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "finding": self.finding,
            "evidence": self.evidence,
            "remedy": self.remedy,
            "blocked_by": self.blocked_by,
            "severity": self.severity,
        }


@dataclass
class CheckLog:
    """Collects results and answers "did the thing this depends on hold up?".

    Checks append in the order they run, so the list reads as a transcript.
    """

    checks: list[Check] = field(default_factory=list)

    def add(self, check: Check) -> Check:
        self.checks.append(check)
        return check

    def passed(self, name: str) -> bool:
        return any(c.name == name and c.status in {"passed", "warning"} for c in self.checks)

    def skip(self, name: str, label: str, because: str, reason: str) -> Check:
        """Records a check that could not run, and what stopped it."""
        return self.add(
            Check(
                name=name,
                label=label,
                status="not_run",
                finding=reason,
                blocked_by=because,
            )
        )

    def as_dicts(self) -> list[dict]:
        return [c.as_dict() for c in self.checks]

    # --- deriving the older, flatter shape -------------------------------------------------
    #
    # The panel, the audit report and the model card all consume blockers/warnings/notes.
    # Those are now views over the checks rather than the primary record, so there is one
    # source of truth and the two can't drift apart.

    @property
    def blockers(self) -> list[str]:
        return [
            self._sentence(c)
            for c in self.checks
            if c.status == "failed" and c.severity == "blocker"
        ]

    @property
    def warnings(self) -> list[str]:
        return [
            self._sentence(c)
            for c in self.checks
            if c.status in {"failed", "warning"} and c.severity == "warning"
        ]

    @staticmethod
    def _sentence(check: Check) -> str:
        """Finding, evidence and remedy as one paragraph.

        The structured form is for the provenance panel, which can lay the three out
        separately. Anything consuming `blockers`/`warnings` wants the whole thought in one
        string -- dropping the evidence there would make the summary vaguer than the version
        it replaced."""
        return " ".join(part for part in (check.finding, check.evidence, check.remedy) if part)

    @property
    def counts(self) -> dict:
        return {
            "passed": sum(1 for c in self.checks if c.status == "passed"),
            "warning": sum(1 for c in self.checks if c.status == "warning"),
            "failed": sum(1 for c in self.checks if c.status == "failed"),
            "not_run": sum(1 for c in self.checks if c.status == "not_run"),
            "total": len(self.checks),
        }


# --- individual checks -----------------------------------------------------------------
#
# Each takes what it needs and returns a Check. Kept as free functions rather than methods so
# a new one is a new function and nothing else, which is the point of the refactor: the
# validation stage adds checks here and gets provenance for free.


def check_contradictory_labels(
    df: pd.DataFrame, target: str, feature_columns: list[str]
) -> Check:
    """Identical rows with different answers.

    A dataset-killer that no statistical check catches. If two rows agree on every input and
    disagree on the outcome, no model can separate them -- they are a hard ceiling on
    accuracy, and usually a sign of a labelling process that was inconsistent, a missing
    column that would have distinguished them, or rows merged from sources that disagreed.

    Only exact matches count. Near-duplicates are a different problem (entity resolution) and
    treating them as contradictions here would fire constantly on continuous columns.
    """
    label = "Rows that contradict each other"

    if not feature_columns:
        return Check(
            name="contradictory_labels",
            label=label,
            status="not_run",
            finding="There are no input columns to compare rows on.",
            blocked_by="has_features",
        )

    if len(df) > MAX_ROWS_FOR_CONTRADICTION_CHECK:
        return Check(
            name="contradictory_labels",
            label=label,
            status="not_run",
            finding=f"Skipped on a file this size ({len(df):,} rows) to keep this page fast.",
        )

    subset = df[feature_columns + [target]].dropna(subset=[target])
    duplicated = subset.duplicated(subset=feature_columns, keep=False)
    if not duplicated.any():
        return Check(
            name="contradictory_labels",
            label=label,
            status="passed",
            finding="No two rows have identical inputs, so nothing contradicts anything else.",
            evidence=f"Compared {len(subset):,} rows across {len(feature_columns)} input columns.",
        )

    groups = subset[duplicated].groupby(feature_columns, dropna=False, observed=True)[target]
    disagreeing = groups.nunique()
    conflicted = disagreeing[disagreeing > 1]

    if conflicted.empty:
        repeats = int(duplicated.sum())
        return Check(
            name="contradictory_labels",
            label=label,
            status="passed",
            finding="Rows with identical inputs all carry the same answer.",
            evidence=f"{repeats:,} rows share their inputs with another row, and every one "
            f"of those groups agrees on '{target}'.",
        )

    affected = int(sum(int(g) for g in groups.size()[conflicted.index]))
    share = affected / max(len(subset), 1)
    return Check(
        name="contradictory_labels",
        label=label,
        status="failed",
        finding=f"{len(conflicted):,} groups of rows have identical inputs but different "
        f"answers for '{target}'.",
        evidence=f"{affected:,} rows ({share:.1%}) are involved. No model can tell these "
        f"apart -- they set a hard ceiling on how accurate any model can be.",
        remedy="Check whether a column that would distinguish them was dropped, or whether "
        "these rows came from sources that disagreed. Deduplicating without deciding which "
        "answer is right just picks one at random.",
        # A warning rather than a blocker below 5%: a handful of contradictions is normal in
        # real data and blocking on them would be the kind of false alarm that teaches people
        # to ignore the panel.
        severity="blocker" if share >= 0.05 else "warning",
    )


def check_validation_budget(df: pd.DataFrame, target: str, task_type: str | None) -> Check | None:
    """Whether the file can be split into enough to train on and enough to judge on.

    Said only when it cannot. Every fraction the app will hold back leaves either too few test rows
    for a score to mean anything or too few training rows to learn from, so the verdict the model
    page gives will be "not enough test data" however well it does -- better to say so before the
    model is trained than after.
    """
    from app.services import validation_advice

    advice = validation_advice.recommend_for(df[target], task_type) if target in df.columns else None
    if advice is None or advice["feasible"]:
        return None
    return Check(
        name="validation_budget",
        label="Enough to hold back a test set",
        status="warning",
        finding=advice["reason"],
        evidence="A test set is only worth having if it is big enough to judge on (at least "
        "30 rows, and 10 of the rarer answer for two answers) and the model still has enough left "
        "to learn from. This file cannot give both.",
        remedy="Collect more labelled rows, especially of the rarer answer.",
        severity="warning",
    )


def check_repeated_entities(df: pd.DataFrame, target: str) -> Check:
    """Columns whose values repeat across many rows, which a random split would scatter.

    If the same customer, patient or machine appears in both the training and test sets, the
    model can memorise that entity rather than learn the pattern, and every score is
    inflated. The fix is to split by group, which the model plan now offers on exactly the
    columns named here -- this check used to be able to describe the problem and not solve it.
    """
    label = "Rows that belong together"
    rows = len(df)
    candidates: list[tuple[str, int]] = []

    for column in df.columns:
        if column == target:
            continue
        series = df[column].dropna()
        if series.empty:
            continue

        # Numeric columns are excluded outright. An entity identifier is a label -- a
        # customer number, an email, a machine name -- and its values carry no order. A
        # numeric column with moderate cardinality is almost always a measurement:
        # `last_login_days_ago` has a few hundred distinct integers each repeating a dozen
        # times, which is the exact shape this check looks for and nothing to do with
        # entities. Telling someone their scores are inflated when they aren't is the kind of
        # false alarm that teaches people to ignore the panel, so the rule stays narrow. A
        # numeric id that is near-unique is caught by the identifier check instead.
        if pd.api.types.is_numeric_dtype(series):
            continue

        distinct = int(series.nunique())
        if distinct < MIN_DISTINCT_FOR_ENTITY or distinct > rows / 2:
            continue
        if rows / distinct < MIN_REPETITION_FOR_ENTITY:
            continue
        candidates.append((column, distinct))

    if not candidates:
        return Check(
            name="repeated_entities",
            label=label,
            status="passed",
            finding="No column looks like a repeated entity, so rows can be treated as "
            "independent.",
            evidence=f"Checked {max(len(df.columns) - 1, 0)} columns for identifier-shaped "
            f"repetition.",
        )

    named = ", ".join(f"'{c}' ({n:,} distinct)" for c, n in candidates[:3])
    return Check(
        name="repeated_entities",
        label=label,
        status="warning",
        finding=f"{named} repeats across rows, so the same entity may appear more than once.",
        evidence="A random split can put the same entity in both training and test data, "
        "which lets the model recognise it rather than learn the pattern -- and every score "
        "comes out better than it should.",
        remedy=f"If those rows really are repeated measurements of the same thing, choose "
        f"\"Keep related rows together\" on the model plan and group on '{candidates[0][0]}'. "
        f"A random split will otherwise report a score the model cannot repeat on an entity "
        f"it has not seen.",
        severity="warning",
    )
