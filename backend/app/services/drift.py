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
a shift worth knowing about, over 0.25 a different population. They are conventions, not
statistics, and the report says so by showing the numbers behind every call.

Like the model verdict, it declines to judge a sample too small to mean anything.
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
_BINS = 10
_EPS = 1e-4  # floor for an empty bin, so the log term stays finite


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric(column: str, ref: pd.Series, new: pd.Series) -> dict:
    ref = pd.to_numeric(ref, errors="coerce").dropna()
    new = pd.to_numeric(new, errors="coerce").dropna()
    if ref.empty or new.empty:
        return {"column": column, "kind": "numeric", "status": "not_judged",
                "note": "No numeric values to compare."}

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
        "status": _status(psi, outside, OUTSIDE_RANGE_MAJOR),
    }


def _categorical(column: str, ref: pd.Series, new: pd.Series) -> dict:
    ref = ref.dropna().astype(str)
    new = new.dropna().astype(str)
    if ref.empty or new.empty:
        return {"column": column, "kind": "categorical", "status": "not_judged",
                "note": "No values to compare."}
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
        "status": _status(psi, unseen_share, UNSEEN_MAJOR),
    }


def _status(psi: float, share: float, share_major: float) -> str:
    if psi > PSI_MAJOR or share > share_major:
        return "different"
    if psi > PSI_MODERATE or share > SHIFT_SHARE:
        return "shifted"
    return "stable"


def compare(reference: pd.DataFrame, new: pd.DataFrame, numeric: list[str], categorical: list[str]) -> dict:
    """Column-by-column comparison and an overall verdict.

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

    columns = [
        _numeric(c, reference[c], new[c]) for c in numeric if c in reference and c in new
    ] + [
        _categorical(c, reference[c], new[c]) for c in categorical if c in reference and c in new
    ]
    different = [c["column"] for c in columns if c["status"] == "different"]
    shifted = [c["column"] for c in columns if c["status"] == "shifted"]

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
        headline = "Every input looks like the data the model learned from."
    # The most changed first: that is the one to look at.
    columns.sort(key=lambda c: c.get("psi", -1), reverse=True)
    return {"verdict": verdict, "headline": headline, "columns": columns}


def _list(names: list[str]) -> str:
    quoted = [f"'{n}'" for n in names]
    return quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " and " + quoted[-1]
