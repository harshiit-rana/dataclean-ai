"""Does the new file look like the data the model learned from?

A model scores whatever it is given. If February's `monthly_spend` runs twice as high as
January's, or a region appears that never existed in training, the predictions still come back
looking exactly as confident as before -- nothing inside the model notices. This is the same
principle as the readiness verdict and the honest model verdict, applied at the point of use:
the app is allowed to say "these rows are not what I was trained on".

Per column:

- **Numbers**: the population stability index over ten bins cut at the training data's
  deciles, plus the share of new values outside the training range entirely.
- **Categories**: PSI over the training categories (anything else pooled as "unseen"), plus
  the share of new rows carrying a value the model has never seen -- the encoder ignores those,
  so for that row the column contributes nothing at all.

PSI thresholds are the long-standing credit-scoring convention: under 0.1 stable, 0.1 to 0.25
a shift worth knowing about, over 0.25 a different population. They are conventions, not statistics:
on a small new file PSI crosses 0.1 by chance alone, and with thirty inputs a few will. So each column
is also given the PSI that chance reaches for samples of its sizes -- simulated, by drawing a training-
sized and a new-sized sample from the training distribution many times -- at a 5% error rate spread
across every column compared. A column is called shifted only when its PSI is beyond that *and* past
the convention, so the convention is a floor on how big a shift has to be to matter, and chance is a
floor on how sure the call has to be.

Like the model verdict, it declines to judge a sample too small to mean anything -- the file as a
whole, and each column on its own filled-in values: a column that is 95% blank in a 500-row file
has 25 values to judge, not 500. How often a column is blank is compared too, because a column that
stops being filled in has drifted as surely as one whose values moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_MODERATE = 0.1
PSI_MAJOR = 0.25
UNSEEN_MAJOR = 0.2  # share of rows with a category the model never saw
OUTSIDE_RANGE_MAJOR = 0.2  # share of rows outside the training range
# Below this, a few unseen categories or values past the training extremes are ordinary: a
# training maximum is an extreme by definition, so almost any new file has some values beyond
# it. A warning that fires on every column of every file teaches people to ignore it.
SHIFT_SHARE = 0.05
MIN_ROWS_TO_JUDGE = 30
# The same bar for each column's own filled-in values, in either file.
MIN_VALUES_TO_JUDGE = 30
# A change in how often a column is blank, in percentage points of the rows, and only when it is
# more than chance on these row counts (a two-proportion z of at least 3).
BLANK_SHIFT_MODERATE = 0.10
BLANK_SHIFT_MAJOR = 0.25
_BLANK_Z = 3.0
_ORDER = {"not_judged": -1, "stable": 0, "shifted": 1, "different": 2}
_BINS = 10
_EPS = 1e-4  # floor for an empty bin, so the log term stays finite
# Simulated sample pairs per column for the chance threshold. Multinomial draws over the bins, so
# this costs milliseconds whatever the file size.
NULL_DRAWS = 4000
# The overall error rate the chance thresholds are set for, spread across the columns compared.
FAMILY_ERROR = 0.05


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def chance_threshold(expected: np.ndarray, reference_rows: int, new_rows: int, error: float, seed: int = 0) -> float:
    """The PSI that two samples from the *same* population reach by chance, at this error rate.

    The population is the training distribution over the bins. Each draw takes a training-sized and a
    new-sized sample from it -- both, because the training proportions are themselves a sample -- and
    measures PSI between them; the threshold is the (1 - error) quantile of those draws.
    """
    p = np.asarray(expected, dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    ref = rng.multinomial(max(int(reference_rows), 1), p, size=NULL_DRAWS) / max(int(reference_rows), 1)
    new = rng.multinomial(max(int(new_rows), 1), p, size=NULL_DRAWS) / max(int(new_rows), 1)
    ref, new = np.clip(ref, _EPS, None), np.clip(new, _EPS, None)
    null = np.sum((new - ref) * np.log(new / ref), axis=1)
    return float(np.quantile(null, 1 - error))


def _blanks(ref_all: pd.Series, new_all: pd.Series) -> dict:
    """How often the column is blank in each file, and whether that moved by more than chance."""
    ref_share = float(ref_all.isna().mean()) if len(ref_all) else 0.0
    new_share = float(new_all.isna().mean()) if len(new_all) else 0.0
    change = new_share - ref_share
    pooled = (ref_all.isna().sum() + new_all.isna().sum()) / max(len(ref_all) + len(new_all), 1)
    spread = float(np.sqrt(pooled * (1 - pooled) * (1 / max(len(ref_all), 1) + 1 / max(len(new_all), 1))))
    beyond_chance = spread > 0 and abs(change) / spread >= _BLANK_Z
    status = "stable"
    if beyond_chance and abs(change) >= BLANK_SHIFT_MAJOR:
        status = "different"
    elif beyond_chance and abs(change) >= BLANK_SHIFT_MODERATE:
        status = "shifted"
    return {
        "training_blank_share": round(ref_share, 4),
        "new_blank_share": round(new_share, 4),
        "blank_status": status,
    }


def _too_few(column: str, kind: str, ref_values: int, new_values: int, blanks: dict) -> dict | None:
    """The column's own verdict when it has too few filled-in values to compare, or None.

    The values cannot be compared, but the blank share still can: a column that stopped being
    filled in is a finding, not a gap in the evidence."""
    if ref_values >= MIN_VALUES_TO_JUDGE and new_values >= MIN_VALUES_TO_JUDGE:
        return None
    where = "the new file" if new_values < MIN_VALUES_TO_JUDGE else "the training data"
    note = (
        f"Only {min(ref_values, new_values):,} filled-in values in {where} "
        f"({ref_values:,} in training, {new_values:,} here); at least {MIN_VALUES_TO_JUDGE} are "
        f"needed to compare them."
    )
    status = blanks["blank_status"] if blanks["blank_status"] != "stable" else "not_judged"
    return {
        "column": column, "kind": kind, "status": status, "note": note,
        "training_values": ref_values, "new_values": new_values, **blanks,
    }


def _numeric(column: str, ref_all: pd.Series, new_all: pd.Series, blanks: dict) -> dict:
    ref = pd.to_numeric(ref_all, errors="coerce").dropna()
    new = pd.to_numeric(new_all, errors="coerce").dropna()
    sparse = _too_few(column, "numeric", len(ref), len(new), blanks)
    if sparse is not None:
        return sparse

    edges = np.unique(np.quantile(ref, np.linspace(0, 1, _BINS + 1)))
    if len(edges) < 3:
        # A near-constant column: compare the share equal to the usual value instead.
        common = ref.mode().iloc[0]
        expected = np.array([(ref == common).mean(), (ref != common).mean()])
        actual = np.array([(new == common).mean(), (new != common).mean()])
    else:
        inner = edges[1:-1]
        expected = np.bincount(np.searchsorted(inner, ref, side="right"), minlength=len(inner) + 1) / len(ref)
        actual = np.bincount(np.searchsorted(inner, new, side="right"), minlength=len(inner) + 1) / len(new)
    psi = _psi(expected, actual)
    outside = float(((new < ref.min()) | (new > ref.max())).mean())
    return {
        "column": column,
        "kind": "numeric",
        "psi": round(psi, 4),
        "outside_training_range": round(outside, 4),
        "training_range": [float(ref.min()), float(ref.max())],
        "training_median": float(ref.median()),
        "new_median": float(new.median()),
        "training_values": int(len(ref)),
        "new_values": int(len(new)),
        **blanks,
        "_expected": expected,
        "_share": outside,
        "_share_major": OUTSIDE_RANGE_MAJOR,
    }


def _categorical(column: str, ref_all: pd.Series, new_all: pd.Series, blanks: dict) -> dict:
    ref = ref_all.dropna().astype(str)
    new = new_all.dropna().astype(str)
    sparse = _too_few(column, "categorical", len(ref), len(new), blanks)
    if sparse is not None:
        return sparse
    categories = ref.value_counts().index
    expected = ref.value_counts(normalize=True).reindex(categories).to_numpy()
    seen = new.isin(set(categories))
    actual_seen = new[seen].value_counts().reindex(categories, fill_value=0).to_numpy() / len(new)
    unseen_share = float((~seen).mean())
    psi = _psi(np.append(expected, 0.0), np.append(actual_seen, unseen_share))
    unseen_values = new[~seen].value_counts().index[:5].tolist()
    return {
        "column": column,
        "kind": "categorical",
        "psi": round(psi, 4),
        "unseen_share": round(unseen_share, 4),
        "unseen_examples": [str(v) for v in unseen_values],
        "training_values": int(len(ref)),
        "new_values": int(len(new)),
        **blanks,
        # The unseen bin has no weight in the training distribution, so chance never fills it.
        "_expected": np.append(expected, 0.0),
        "_share": unseen_share,
        "_share_major": UNSEEN_MAJOR,
    }


def _worse(a: str, b: str) -> str:
    return a if _ORDER[a] >= _ORDER[b] else b


def _status(psi: float, share: float, share_major: float, threshold: float = 0.0) -> str:
    """Past the convention *and* past chance for the PSI; the shares of unseen or out-of-range
    values keep their own bars."""
    beyond = psi > threshold
    if (beyond and psi > PSI_MAJOR) or share > share_major:
        return "different"
    if (beyond and psi > PSI_MODERATE) or share > SHIFT_SHARE:
        return "shifted"
    return "stable"


def _judge(columns: list[dict]) -> None:
    """Set each compared column's chance threshold and status, now that the number of columns --
    which the error rate is spread across -- is known."""
    compared = [c for c in columns if "_expected" in c]
    error = FAMILY_ERROR / max(len(compared), 1)
    for index, c in enumerate(compared):
        threshold = chance_threshold(c["_expected"], c["training_values"], c["new_values"], error, seed=index)
        c["chance_threshold"] = round(threshold, 4)
        c["beyond_chance"] = bool(c["psi"] > threshold)
        value_status = _status(c["psi"], c["_share"], c["_share_major"], threshold)
        c["status"] = _worse(value_status, c["blank_status"])
    for c in compared:
        for key in ("_expected", "_share", "_share_major"):
            c.pop(key, None)


def compare(
    reference: pd.DataFrame,
    new: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    blanks_from: dict[str, tuple[pd.Series, pd.Series]] | None = None,
) -> dict:
    """Column-by-column comparison and an overall verdict.

    `blanks_from` gives, for a derived input, the source column whose blankness it inherits: a
    length or a date part is never blank itself, so a source that stops being filled in would
    otherwise pass unnoticed.

    Verdicts: `not_enough_rows`, `similar`, `shifted` (worth knowing, predictions still
    usable), `different` (at least one input no longer looks like what the model learned
    from -- treat its predictions as unreliable until it is retrained on data like this).
    """
    if len(new) < MIN_ROWS_TO_JUDGE:
        return {
            "verdict": "not_enough_rows",
            "headline": f"{len(new)} rows is too few to tell whether this data has shifted -- "
            f"at least {MIN_ROWS_TO_JUDGE} are needed. The predictions are still made; there "
            f"is just no evidence either way about whether they can be trusted.",
            "columns": [],
        }

    sources = blanks_from or {}

    def blanks(c: str) -> dict:
        ref_all, new_all = sources.get(c, (reference[c], new[c]))
        return _blanks(ref_all, new_all)

    columns = [
        _numeric(c, reference[c], new[c], blanks(c)) for c in numeric if c in reference and c in new
    ] + [
        _categorical(c, reference[c], new[c], blanks(c)) for c in categorical if c in reference and c in new
    ]
    _judge(columns)
    different = [c["column"] for c in columns if c["status"] == "different"]
    shifted = [c["column"] for c in columns if c["status"] == "shifted"]
    unjudged = [c["column"] for c in columns if c["status"] == "not_judged"]
    compared = [c for c in columns if "chance_threshold" in c]
    beyond = [c for c in compared if c["beyond_chance"]]

    if different:
        verdict = "different"
        headline = (
            f"{_list(different)} no longer look{'s' if len(different) == 1 else ''} like the "
            f"data the model learned from. Its predictions on these rows are extrapolation, "
            f"and nothing inside the model will warn you -- treat them as unreliable until it "
            f"is retrained on data like this."
        )
    elif shifted:
        verdict = "shifted"
        headline = (
            f"{_list(shifted)} {'has' if len(shifted) == 1 else 'have'} moved a little from the "
            f"training data. The predictions are usable, but worth checking against real "
            f"outcomes once they arrive."
        )
    else:
        verdict = "similar"
        headline = (
            "Every input that could be compared looks like the data the model learned from."
            if unjudged
            else "Every input looks like the data the model learned from."
        )
    if compared and verdict != "similar":
        headline += (
            f" {len(beyond)} of the {len(compared)} inputs compared moved more than chance explains on "
            f"files of these sizes."
        )
    if unjudged:
        headline += (
            f" {_list(unjudged)} had too few filled-in values to compare, so "
            f"{'it was' if len(unjudged) == 1 else 'they were'} left out of this verdict."
        )
    # The most changed first: that is the one to look at.
    columns.sort(key=lambda c: c.get("psi", -1), reverse=True)
    return {"verdict": verdict, "headline": headline, "columns": columns}


def _list(names: list[str]) -> str:
    quoted = [f"'{n}'" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
