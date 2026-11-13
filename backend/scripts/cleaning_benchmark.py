"""How many known problems the profiler finds, how many it invents, and how much fixing them wins back.  (issue #125)

The method is REIN's (Abdelaal et al., 2023) and CleanML's (Li et al., 2019): start from clean public data, inject
problems whose every cell is known, and score the tool against that truth. The data are OpenML AutoML Benchmark tasks
(`openml_tasks.py`), read back through CSV as an upload would be.

Injected problems (seeded; each column gets at most one, so every finding has one cause):
- `missing_values`: 10% of a complete numeric column, and of a complete text column, blanked.
- `outliers`: 2% of a numeric column multiplied by 10, 100 or 1000, the shape of a unit or decimal-point mistake.
- `type_mismatch`: 3% of a numeric column replaced by "?", "unknown" or "--", so the column arrives as text.
- `inconsistent_casing`: 10% of a text column's values in capitals or title case.
- `stray_whitespace`: 10% of a text column's values with spaces before or after.
- `date_formats`: a date column (added, the same in the clean file) with 20% of its dates written day-first.
- `duplicates`: 5% of the rows repeated.

**Detection** compares `profiling.detect_issues` on the dirty file with its findings on the clean one. A finding on an
injected column counts as found when the clean file had no such finding there, or when the dirty file's count for it
(the share of blanks or copies, by more than a point; padded values, spellings or formats) grew; injected outliers
count as found when the column is flagged and at least half of them sit outside the profiler's range. Every other
new finding is a false alarm. What the profiler reports on the clean file is listed separately, not counted. Recall
is found / injected, precision is found / (found + false alarms), per kind of problem. For outliers, the share of the
injected cells outside the profiler's own range is reported too: a column can be flagged while most of its bad
values sit inside the range.

**Repair** trains the app's plain train (`modeling.train_candidates`, as in the accuracy benchmark) three times on one
of the task's official folds (a different one for each seed): on the clean file, on the dirty file, and on the dirty file after the fix each finding
implies. Each is scored on the test rows in the same state (clean, dirty, dirty with the training rows' fixes
replayed on them, as the app replays a recipe). Copies are added to the training rows only; the added date column is
left out of training. The share of the lost score the fixes win back is reported when there was a loss to win back.
Blanks are information destroyed, not hidden: filling them with a median or the commonest value cannot bring it back,
so on a file whose answer depends on every column, most of the loss is beyond any fix.

The fixes are a fixed mapping from finding to operation (`REFERENCE_FIX`), not the explanation agent's choice: this
measures the detectors and the operations, not the LLM. Fixes are applied in one pass, casts first.

Usage (from `backend/`):
    python scripts/cleaning_benchmark.py                          # every dataset, 3 seeds
    python scripts/cleaning_benchmark.py --datasets credit-g --seeds 1 --no-repair
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "benchmark-key-not-real")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import accuracy_benchmark as accuracy  # noqa: E402
import openml_tasks  # noqa: E402
from app.services import operations, profiling  # noqa: E402

DATASETS = ["credit-g", "churn", "vehicle", "abalone", "car", "house_prices_nominal"]
KINDS = ["missing_values", "outliers", "type_mismatch", "inconsistent_casing", "stray_whitespace", "date_formats",
         "duplicates"]
DATE_COLUMN = "benchmark_date"
# Not "n/a", "NA" or "null": read_csv, like the app, reads those as blanks, not as text.
TEXT_FOR_NUMBER = ["?", "unknown", "--"]


@dataclass
class Injection:
    kind: str
    column: str  # "DATASET" for duplicates
    rows: list[int] = field(default_factory=list)


# --- injecting ----------------------------------------------------------------------------------------------------


def _numeric_candidates(frame: pd.DataFrame, target: str) -> list[str]:
    return [c for c in frame.columns if c != target and pd.api.types.is_numeric_dtype(frame[c])
            and frame[c].notna().all() and frame[c].nunique() > 10]


def _text_candidates(frame: pd.DataFrame, target: str, *, with_letters: bool = False) -> list[str]:
    """Complete text columns with a handful of values and no casing or spacing variants already. `with_letters`
    keeps only columns where most values change in capitals, so a casing problem can be injected at all."""
    out = []
    for c in frame.columns:
        if c == target or frame[c].dtype != object or not frame[c].notna().all():
            continue
        values = frame[c].astype(str)
        if 2 <= values.nunique() <= 30 and values.nunique() == values.str.strip().str.lower().nunique():
            if not with_letters or (values.str.upper() != values).mean() >= 0.5:
                out.append(c)
    return out


def _some(rng: np.random.Generator, rows: int, share: float, among=None) -> np.ndarray:
    pool = np.arange(rows) if among is None else np.asarray(among)
    count = max(1, int(round(len(pool) * share)))
    return np.sort(rng.choice(pool, size=min(count, len(pool)), replace=False))


def with_dates(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The clean file with a tidy ISO date column, so a date-format problem has somewhere to go."""
    rng = np.random.default_rng(seed + 1000)
    days = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1500, len(frame)), unit="D")
    out = frame.copy()
    out[DATE_COLUMN] = days.strftime("%Y-%m-%d")
    return out


def corrupt(frame: pd.DataFrame, target: str, seed: int) -> tuple[pd.DataFrame, list[Injection]]:
    """The file with every applicable problem injected, and where each went. `frame` already has the date column.
    Duplicates are not added here (see `add_duplicates`), so rows keep their positions."""
    rng = np.random.default_rng(seed)
    dirty = frame.copy()
    injections: list[Injection] = []
    numeric = list(rng.permutation(_numeric_candidates(frame, target)))
    text = list(rng.permutation([c for c in _text_candidates(frame, target) if c != DATE_COLUMN]))
    n = len(frame)

    if numeric:
        column = numeric.pop()
        rows = _some(rng, n, 0.10)
        dirty.loc[rows, column] = np.nan
        injections.append(Injection("missing_values", column, rows.tolist()))
    if text:
        column = text.pop()
        rows = _some(rng, n, 0.10)
        dirty[column] = dirty[column].astype(object)
        dirty.loc[rows, column] = np.nan
        injections.append(Injection("missing_values", column, rows.tolist()))
    if numeric:
        column = numeric.pop()
        rows = _some(rng, n, 0.02, among=np.flatnonzero(frame[column].to_numpy() != 0))
        dirty[column] = dirty[column].astype(float)
        dirty.loc[rows, column] = frame.loc[rows, column].astype(float) * rng.choice([10, 100, 1000], len(rows))
        injections.append(Injection("outliers", column, rows.tolist()))
    if numeric:
        column = numeric.pop()
        rows = _some(rng, n, 0.03)
        dirty[column] = dirty[column].astype(object)
        dirty.loc[rows, column] = rng.choice(TEXT_FOR_NUMBER, len(rows))
        injections.append(Injection("type_mismatch", column, rows.tolist()))
    lettered = [c for c in text if c in _text_candidates(frame, target, with_letters=True)]
    if lettered:
        column = lettered[-1]
        text.remove(column)
        rows = _some(rng, n, 0.10)
        values = dirty[column].astype(str)
        # Alternately in capitals and in title case (capitals where title case changes nothing).
        dirty.loc[rows, column] = [v.upper() if i % 2 or v.title() == v else v.title()
                                   for i, v in enumerate(values.iloc[rows])]
        injections.append(Injection("inconsistent_casing", column, rows.tolist()))
    if text:
        column = text.pop()
        rows = _some(rng, n, 0.10)
        dirty.loc[rows, column] = [f" {v} " if i % 2 else f"{v}  " for i, v in enumerate(dirty[column].iloc[rows])]
        injections.append(Injection("stray_whitespace", column, rows.tolist()))
    rows = _some(rng, n, 0.20)
    dirty.loc[rows, DATE_COLUMN] = pd.to_datetime(dirty.loc[rows, DATE_COLUMN]).dt.strftime("%d/%m/%Y")
    injections.append(Injection("date_formats", DATE_COLUMN, rows.tolist()))
    return dirty, injections


def add_duplicates(frame: pd.DataFrame, seed: int, share: float = 0.05) -> tuple[pd.DataFrame, Injection]:
    rng = np.random.default_rng(seed + 2000)
    rows = _some(rng, len(frame), share)
    out = pd.concat([frame, frame.iloc[rows]], ignore_index=True)
    return out, Injection("duplicates", "DATASET", rows.tolist())


# --- detection ----------------------------------------------------------------------------------------------------


def finding_kind(issue) -> str:
    """The profiler's finding in the benchmark's terms: inconsistent_format is split by which detector raised it."""
    if issue.issue_type != "inconsistent_format":
        return issue.issue_type
    if "padded_values" in issue.stats:
        return "stray_whitespace"
    if "formats_found" in issue.stats:
        return "date_formats"
    return "inconsistent_casing"


# How much of a finding there is. Shares, where the profiler gives one, so the copies added for the duplicates problem
# (which repeat existing blanks and outliers) do not make an old finding look new; counts otherwise.
RATE_KEYS = ["null_rate", "duplicate_rate", "outlier_rate"]
COUNT_KEYS = ["padded_values", "formats_found", "distinct_values"]
RATE_MARGIN = 0.01  # a share must rise by more than this to count as grown


def grew(now, before) -> bool:
    """Whether a finding on the dirty file is more than the same finding on the clean one."""
    if before is None:
        return now is not None
    if now is None:
        return False
    for key in RATE_KEYS:
        if key in now.stats and key in before.stats:
            return float(now.stats[key]) > float(before.stats[key]) + RATE_MARGIN
    for key in COUNT_KEYS:
        if key in now.stats and key in before.stats:
            return float(now.stats[key]) > float(before.stats[key])
    return False


def findings(frame: pd.DataFrame) -> dict[tuple[str, str], object]:
    return {(str(i.column_name), finding_kind(i)): i for i in profiling.detect_issues(frame)}


def outlier_cells_caught(issue, dirty: pd.DataFrame, injection: Injection) -> float | None:
    """The share of injected outlier cells outside the profiler's own range (its quartiles, 1.5 IQR out)."""
    if issue is None:
        return 0.0
    q1, q3 = issue.stats.get("q1"), issue.stats.get("q3")
    if q1 is None or q3 is None:
        return None
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    values = pd.to_numeric(dirty.loc[injection.rows, injection.column], errors="coerce")
    return float(((values < low) | (values > high)).mean())


def score_detection(clean: pd.DataFrame, dirty: pd.DataFrame, injections: list[Injection]) -> dict:
    before, after = findings(clean), findings(dirty)
    injected = {(i.column, i.kind): i for i in injections}
    rows = []
    for (column, kind), injection in injected.items():
        found_now = after.get((column, kind))
        row = {"kind": kind, "column": column, "found": grew(found_now, before.get((column, kind)))}
        if kind == "outliers":
            # Judged by the injected cells: on a column the clean file already had outliers in, most of the bad
            # values can fall in the flagged range without the share of outliers rising by a point.
            caught = outlier_cells_caught(found_now, dirty, injection)
            row["cells_outside_range"] = caught
            row["found"] = found_now is not None and caught is not None and caught >= 0.5
        rows.append(row)
    false_alarms = [
        {"kind": kind, "column": column}
        for (column, kind), issue in after.items()
        if (column, kind) not in injected and grew(issue, before.get((column, kind)))
    ]
    # What the profiler already says about the clean file. Not counted against it -- blanks in a public dataset are
    # real blanks, and a value far out in the tail is an outlier whether or not it is a mistake -- but worth seeing.
    on_clean: dict[str, int] = {}
    for _, kind in before:
        on_clean[kind] = on_clean.get(kind, 0) + 1
    return {"injected": rows, "false_alarms": false_alarms, "on_clean_file": on_clean}


# --- repair -------------------------------------------------------------------------------------------------------


def REFERENCE_FIX(issue, frame: pd.DataFrame) -> tuple[str, dict] | None:  # noqa: N802 -- a table, in effect
    """The operation each finding implies, the one a careful person would approve."""
    kind, column = finding_kind(issue), str(issue.column_name)
    if kind == "duplicates":
        return "drop_duplicate_rows", {}
    if kind == "type_mismatch":
        return ("cast_to_numeric", {"column": column}) if issue.stats.get("detected_as") == "numeric" else None
    if kind == "inconsistent_casing":
        return "unify_case_variants", {"column": column}
    if kind == "stray_whitespace":
        return "trim_whitespace", {"column": column}
    if kind == "date_formats":
        return "standardize_dates", {"column": column}
    if kind == "missing_values":
        numeric = pd.api.types.is_numeric_dtype(frame[column])
        return ("fill_median" if numeric else "fill_mode"), {"column": column}
    if kind == "outliers":
        return "cap_outliers", {"column": column}
    return None


# Casts first, so the later steps see numbers; outliers before blanks, so the range is learned from the values
# actually written -- filling blanks with the median first squeezes the range and caps ordinary values.
ORDER = ["type_mismatch", "stray_whitespace", "inconsistent_casing", "date_formats", "outliers", "missing_values",
         "duplicates"]


def fix(train: pd.DataFrame, test: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The reference fixes for what the profiler finds on the training rows, learned there and replayed on the test
    rows. Copies are dropped from the training rows only."""
    issues = [i for i in profiling.detect_issues(train) if str(i.column_name) != target]
    issues.sort(key=lambda i: ORDER.index(finding_kind(i)) if finding_kind(i) in ORDER else len(ORDER))
    applied = []
    for issue in issues:
        chosen = REFERENCE_FIX(issue, train)
        if chosen is None:
            continue
        name, params = chosen
        try:
            learned = operations.learn(train, name, params)
            train = operations.apply_operation(train, name, params, learned=learned)[0]
            if name != "drop_duplicate_rows":
                test = operations.apply_operation(test, name, params, learned=learned)[0]
            applied.append(f"{name}({params.get('column', '')})")
        except operations.OperationFailedError:
            continue
    return train.reset_index(drop=True), test.reset_index(drop=True), applied


def trained_score(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str) -> float:
    train, test = train.drop(columns=[DATE_COLUMN], errors="ignore"), test.drop(columns=[DATE_COLUMN], errors="ignore")
    labels = None if kind == "regression" else sorted(pd.concat([train[target], test[target]]).dropna().unique().tolist(),
                                                      key=str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = accuracy.run_dataclean_train(train, test, target, kind, budget=0)
    return accuracy.score(kind, test[target], prediction, labels)[accuracy.METRICS[kind]]


def recovered(clean: float, dirty: float, fixed: float, higher_better: bool) -> float | None:
    """The share of the score lost to the problems that the fixes won back; None when the loss is too small to tell
    from the noise of one test fold (under 0.01 of AUC, or 1% of a log loss or RMSE above 1)."""
    lost = (clean - dirty) if higher_better else (dirty - clean)
    if lost < 0.01 * max(abs(clean), 1.0):
        return None
    won = (fixed - dirty) if higher_better else (dirty - fixed)
    return won / lost


def score_repair(loaded: openml_tasks.LoadedTask, clean: pd.DataFrame, dirty: pd.DataFrame, seed: int) -> dict:
    target, kind = loaded.target, loaded.task.kind
    # Each seed on a different official fold, so the repair numbers are not all about one small set of test rows.
    train_rows, test_rows = loaded.folds[seed % len(loaded.folds)]
    clean_train, clean_test = clean.iloc[train_rows].reset_index(drop=True), clean.iloc[test_rows].reset_index(drop=True)
    dirty_train, dirty_test = dirty.iloc[train_rows].reset_index(drop=True), dirty.iloc[test_rows].reset_index(drop=True)
    dirty_train, _ = add_duplicates(dirty_train, seed)
    # Through CSV, as the app reads each file: a number column holding "n/a" arrives as text, and so on.
    dirty_train, dirty_test = openml_tasks.as_uploaded(dirty_train), openml_tasks.as_uploaded(dirty_test)
    fixed_train, fixed_test, applied = fix(dirty_train, dirty_test, target)
    metric = accuracy.METRICS[kind]
    scores = {
        "clean": trained_score(clean_train, clean_test, target, kind),
        "dirty": trained_score(dirty_train, dirty_test, target, kind),
        "fixed": trained_score(fixed_train, fixed_test, target, kind),
    }
    return {"metric": metric, **scores, "fixes": applied,
            "recovered": recovered(scores["clean"], scores["dirty"], scores["fixed"], accuracy.HIGHER_IS_BETTER[metric])}


# --- running and summarising -------------------------------------------------------------------------------------


def run(datasets: list[str], seeds: int, repair: bool, cache: Path, progress: Path | None = None) -> dict:
    """Every dataset at every seed. With `progress`, the results so far are written there after each seed, and an
    interrupted run with the same settings carries on from them (see `accuracy_benchmark.interrupted_run`)."""
    results: dict = {"python": platform.python_version(), "cpus": os.cpu_count(), "seeds": seeds, "repair": repair,
                     "datasets": {}}
    earlier = accuracy.interrupted_run(progress, results, ("seeds", "repair"))
    if earlier:
        results["datasets"] = {k: v for k, v in earlier.get("datasets", {}).items() if k in datasets}
        done = sum(len(e["runs"]) for e in results["datasets"].values())
        print(f"Picking up {done} finished run(s) from {progress}.", flush=True)
    for name in datasets:
        task = openml_tasks.select([name])[0]
        loaded = openml_tasks.load(task, cache)
        entry: dict = results["datasets"].setdefault(name, {"kind": task.kind, "rows": len(loaded.frame), "runs": []})
        for seed in range(len(entry["runs"]), seeds):
            started = time.perf_counter()
            clean = with_dates(loaded.frame, seed)
            dirty, injections = corrupt(clean, loaded.target, seed)
            dirty_full, duplicates = add_duplicates(dirty, seed)
            # Both files through CSV, as the app reads an upload.
            clean_csv, dirty_csv = openml_tasks.as_uploaded(clean), openml_tasks.as_uploaded(dirty_full)
            run_entry = {"seed": seed, "detection": score_detection(clean_csv, dirty_csv, [*injections, duplicates])}
            if repair:
                run_entry["repair"] = score_repair(loaded, clean, dirty, seed)
            run_entry["seconds"] = round(time.perf_counter() - started, 1)
            entry["runs"].append(run_entry)
            found = sum(r["found"] for r in run_entry["detection"]["injected"])
            line = f"{name} seed {seed}: found {found}/{len(run_entry['detection']['injected'])}, " \
                   f"{len(run_entry['detection']['false_alarms'])} false alarm(s)"
            if repair:
                r = run_entry["repair"]
                line += f"; {r['metric']} clean {r['clean']:.4f} dirty {r['dirty']:.4f} fixed {r['fixed']:.4f}"
            print(line, flush=True)
            accuracy.save_progress(progress, results)
    results["summary"] = summarise(results)
    return results


def summarise(results: dict) -> dict:
    per_kind = {k: {"injected": 0, "found": 0, "false_alarms": 0} for k in KINDS}
    outlier_cells: list[float] = []
    on_clean: dict[str, int] = {}
    recoveries: dict[str, list[float]] = {}
    for name, entry in results["datasets"].items():
        for run_entry in entry["runs"]:
            for row in run_entry["detection"]["injected"]:
                per_kind[row["kind"]]["injected"] += 1
                per_kind[row["kind"]]["found"] += int(row["found"])
                if row.get("cells_outside_range") is not None:
                    outlier_cells.append(row["cells_outside_range"])
            for kind, count in run_entry["detection"].get("on_clean_file", {}).items():
                on_clean[kind] = on_clean.get(kind, 0) + count
            for alarm in run_entry["detection"]["false_alarms"]:
                per_kind.setdefault(alarm["kind"], {"injected": 0, "found": 0, "false_alarms": 0})
                per_kind[alarm["kind"]]["false_alarms"] += 1
            if run_entry.get("repair") and run_entry["repair"]["recovered"] is not None:
                recoveries.setdefault(name, []).append(run_entry["repair"]["recovered"])
    for counts in per_kind.values():
        counts["recall"] = round(counts["found"] / counts["injected"], 3) if counts["injected"] else None
        flagged = counts["found"] + counts["false_alarms"]
        counts["precision"] = round(counts["found"] / flagged, 3) if flagged else None
    injected = sum(c["injected"] for c in per_kind.values())
    found = sum(c["found"] for c in per_kind.values())
    alarms = sum(c["false_alarms"] for c in per_kind.values())
    return {
        "per_kind": per_kind,
        "recall": round(found / injected, 3) if injected else None,
        "precision": round(found / (found + alarms), 3) if found + alarms else None,
        "outlier_cells_outside_range": round(float(np.mean(outlier_cells)), 3) if outlier_cells else None,
        "findings_on_clean_files": on_clean,
        "recovered": {name: round(float(np.mean(v)), 3) for name, v in recoveries.items()},
    }


def markdown(results: dict) -> str:
    s = results["summary"]
    lines = [f"{len(results['datasets'])} datasets, {results['seeds']} seed(s) each.", "",
             f"Overall: recall {s['recall']}, precision {s['precision']}.", "",
             "| Problem | Injected | Found | Recall | False alarms | Precision |", "|---|---:|---:|---:|---:|---:|"]
    for kind, c in s["per_kind"].items():
        if c["injected"] or c["false_alarms"]:
            lines.append(f"| {kind} | {c['injected']} | {c['found']} | {c['recall'] if c['recall'] is not None else ''}"
                         f" | {c['false_alarms']} | {c['precision'] if c['precision'] is not None else ''} |")
    if s["outlier_cells_outside_range"] is not None:
        lines += ["", f"Injected outlier cells outside the profiler's range: {s['outlier_cells_outside_range']:.0%}."]
    if s["findings_on_clean_files"]:
        lines += ["", "Findings on the clean files, not counted above: " + ", ".join(
            f"{kind} {count}" for kind, count in sorted(s["findings_on_clean_files"].items())) + "."]
    alarms = [(name, a) for name, e in results["datasets"].items() for r in e["runs"] for a in r["detection"]["false_alarms"]]
    if alarms:
        lines += ["", "False alarms: " + "; ".join(sorted({f"{a['kind']} on {name}.{a['column']}" for name, a in alarms}))]
    if results["repair"]:
        lines += ["", "| Dataset | Metric | Clean | Dirty | Fixed | Won back |", "|---|---|---:|---:|---:|---:|"]
        for name, entry in results["datasets"].items():
            reps = [r["repair"] for r in entry["runs"] if r.get("repair")]
            if not reps:
                continue
            mean = {k: float(np.mean([r[k] for r in reps])) for k in ("clean", "dirty", "fixed")}
            won = s["recovered"].get(name)
            lines.append(f"| {name} | {reps[0]['metric']} | {mean['clean']:.4f} | {mean['dirty']:.4f} | "
                         f"{mean['fixed']:.4f} | {'no measurable loss' if won is None else f'{won:.0%}'} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--no-repair", action="store_true", help="detection only (seconds, not minutes)")
    parser.add_argument("--cache", type=Path, default=openml_tasks.DEFAULT_CACHE)
    parser.add_argument("--out", default="cleaning_benchmark.json")
    parser.add_argument("--markdown", help="also write the summary tables to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("app").setLevel(logging.ERROR)
    # The output file doubles as the progress file, so running the same command again carries on.
    results = run(args.datasets, args.seeds, not args.no_repair, args.cache, progress=Path(args.out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = markdown(results)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
