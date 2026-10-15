"""Any two runs side by side: what differed between them, and whether the score moved beyond the noise.

The history compares each run with the one before it. The question people actually ask is often wider -- "did
dropping that column help, compared with three runs ago?" -- and answering it by eye means diffing settings across
rows and treating a bare difference as a result. `compare` lists every setting that differs, says whether the two
can be compared at all (the same data, the same metric) and on the same held-out rows (the same split too), and gives
the primary metric's difference with its uncertainty: paired on the shared rows when it can be, the two stored
intervals otherwise, and no verdict at all for runs that cannot be compared.
"""

from __future__ import annotations

from typing import Any, Callable

from app.services import model_store

# (field, label, how to read it off a history entry). Every field that decides what a run was.
SETTINGS: list[tuple[str, str, Callable[[dict], Any]]] = [
    ("target_column", "Column predicted", lambda r: r.get("target_column")),
    ("excluded_columns", "Columns left out", lambda r: sorted(r.get("excluded_columns") or [])),
    ("feature_count", "Inputs used", lambda r: r.get("feature_count")),
    ("feature_recipes", "Derived inputs", lambda r: sorted(r.get("feature_recipes") or [])),
    ("clipped_columns", "Clipped columns", lambda r: sorted(r.get("clipped_columns") or [])),
    ("split.strategy", "How the rows were split", lambda r: (r.get("split") or {}).get("strategy")),
    ("split.time_column", "Date column", lambda r: (r.get("split") or {}).get("time_column")),
    ("split.group_column", "Grouping column", lambda r: (r.get("split") or {}).get("group_column")),
    ("split.requested_test_size", "Share held back", lambda r: (r.get("split") or {}).get("requested_test_size")),
    ("split.gap_days", "Gap before the test rows (days)", lambda r: (r.get("split") or {}).get("gap_days")),
    ("models_tried", "Models tried", lambda r: sorted(r.get("models_tried") or [])),
    ("best_model", "Model chosen", lambda r: r.get("best_model")),
    ("tuned", "Improvement loop", lambda r: bool(r.get("tuned"))),
    ("tuned_settings", "Settings found", lambda r: r.get("tuned_settings") or {}),
    ("search_strategy", "Search strategy", lambda r: r.get("search_strategy")),
    ("calibration", "Calibration", lambda r: r.get("calibration")),
    ("operating_point.objective", "Cut-off chosen by",
     lambda r: (r.get("operating_point") or {}).get("objective") or "default"),
    ("operating_point.threshold", "Cut-off", lambda r: (r.get("operating_point") or {}).get("threshold")),
]


def settings_diff(a: dict, b: dict) -> list[dict]:
    """Every setting that differs between two history entries, with both values."""
    found = []
    for field, label, read in SETTINGS:
        left, right = read(a), read(b)
        if left != right:
            found.append({"field": field, "label": label, "a": left, "b": right})
    return found


def comparability(a: dict, b: dict) -> dict:
    """Whether the two scores can be compared at all, and whether on the same held-out rows."""
    reasons = []
    if a.get("data_fingerprint") != b.get("data_fingerprint"):
        reasons.append("they were trained on different data")
    if a.get("primary_metric") != b.get("primary_metric"):
        reasons.append(f"they are scored on different metrics ({b.get('primary_metric')} and {a.get('primary_metric')})")
    if a.get("target_column") != b.get("target_column"):
        reasons.append("they predict different columns")
    same_rows = not reasons and bool(a.get("split_key")) and a.get("split_key") == b.get("split_key")
    return {"comparable": not reasons, "reasons": reasons, "same_rows": same_rows}


def _metrics(a: dict, b: dict) -> list[dict]:
    left, right = a.get("metrics") or {}, b.get("metrics") or {}
    rows = []
    for key in [k for k in left if k in right]:
        x, y = left[key], right[key]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            rows.append({"metric": key, "a": x, "b": y, "difference": round(float(x) - float(y), 6)})
    return rows


def _describe_change(diff: list[dict]) -> str:
    # The number of inputs follows from the columns left out and the derived inputs; it is not a change of its own.
    fields = {d["field"] for d in diff}
    if fields & {"excluded_columns", "feature_recipes"}:
        diff = [d for d in diff if d["field"] != "feature_count"]
    if len(diff) == 1 and diff[0]["field"] == "excluded_columns":
        added = [c for c in diff[0]["a"] if c not in diff[0]["b"]]
        returned = [c for c in diff[0]["b"] if c not in diff[0]["a"]]
        if added and not returned:
            return f"Leaving out {', '.join(repr(c) for c in added)}"
        if returned and not added:
            return f"Bringing back {', '.join(repr(c) for c in returned)}"
    if len(diff) == 1:
        return f"Changing {diff[0]['label'].lower()}"
    return f"These {len(diff)} changes together"


def summary(diff: list[dict], able: dict, primary: dict | None, metric: str | None) -> str:
    """One sentence a reader can act on."""
    if not able["comparable"]:
        return "These two runs cannot be compared: " + "; ".join(able["reasons"]) + "."
    verdict = (primary or {}).get("verdict")
    if not diff:
        if verdict == "within_the_noise":
            return "Nothing in the settings differs, and the two scores agree within the noise."
        if verdict in ("better", "worse"):
            return (f"Nothing in the settings differs, yet {metric} moved beyond the noise: something outside "
                    f"these settings changed.")
        return "Nothing in the settings differs; there is no range to judge the difference in score by."
    what = _describe_change(diff)
    if verdict == "within_the_noise":
        return f"{what} did not change {metric} beyond the noise."
    if verdict in ("better", "worse"):
        direction = "raised" if verdict == "better" else "lowered"
        detail = ""
        if primary.get("method") == "paired" and primary.get("low") is not None:
            detail = f" (by {primary['difference']:+.3f}; 95% range {primary['low']:+.3f} to {primary['high']:+.3f})"
        return f"{what} {direction} {metric} beyond the noise{detail}."
    return f"{what} moved {metric}, but there is no range to judge the difference by."


def compare(directory, a: dict, b: dict) -> dict:
    """Run `a` against run `b` (differences are a minus b)."""
    able = comparability(a, b)
    diff = settings_diff(a, b)
    metric = a.get("primary_metric")
    primary = model_store.compare_runs(directory, a, b) if able["comparable"] else None
    return {
        "a": {k: a.get(k) for k in ("run_id", "at", "model_label", "best_model", "primary_metric")},
        "b": {k: b.get(k) for k in ("run_id", "at", "model_label", "best_model", "primary_metric")},
        **able,
        "settings": diff,
        "metrics": _metrics(a, b),
        # Only for runs that can be compared: a difference between scores on different data or metrics is not one.
        "primary": None if primary is None else {"metric": metric, **primary},
        "summary": summary(diff, able, primary, metric),
    }
