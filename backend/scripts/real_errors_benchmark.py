"""How the profiler and the cleaning operations do on errors people actually made, cell by cell.  (issue #127)

The cleaning benchmark injects problems the profiler was built to find. These datasets carry real ones: each comes as
a dirty file and a clean version made by hand, from the error-detection research of Mahdavi et al. (Raha, SIGMOD
2019; Baran, VLDB 2020), published at github.com/BigDaMa/raha (Apache-2.0).

| Dataset | Rows | Kinds of error |
|---|---|---|
| hospital | 1,000 | typos (letters swapped for "x") in almost every column |
| flights | 2,376 | missing and conflicting departure and arrival times from different sources |
| beers | 2,410 | units glued to numbers ("12.0 oz"), blanks, city and state in one field |
| rayyan | 1,000 | blanks, wrong dates and volumes in bibliographic records |
| movies_1 | 7,390 | thousands separators, inconsistent formats, swapped values |

An erroneous cell is one whose dirty value differs from its clean one, compared the way Raha compares them (as text,
with HTML entities decoded and runs of whitespace collapsed).

**Detection**, per cell. The profiler reports problems per column; each finding is turned into the cells it points at:
the blanks for missing values, the values that do not parse for numbers or dates stored as text, the values outside
its range for outliers, the less common spelling for inconsistent capitals, the padded values for stray spaces, the
less common shape for mixed date formats. Duplicates and high cardinality point at no cell. Precision, recall and F1
against the erroneous cells, as in the papers, and per column: of the columns with errors, how many get a finding.

**Repair**, per cell. The reference fix for each finding (`cleaning_benchmark.REFERENCE_FIX`, without dropping
duplicate rows, which would misalign the files) is applied to the whole dirty file, and each cell compared with the
clean one (dates by the moment they name, numbers by value). A repair is right when an erroneous cell now matches.
A correct written value that no longer matches is damaged; a blank that was really blank and is now filled is
invented, which is what filling does and is counted apart. Precision is right repairs over cells changed, recall
right repairs over erroneous cells.

The profiler was never meant to catch typos or values that are wrong but well formed, and most of these errors are of
that kind, so low recall here is expected. What the benchmark shows is how much of a real file's damage the app can
see, and whether what it fixes it fixes without breaking the rest.

Usage (from `backend/`):
    python scripts/real_errors_benchmark.py
    python scripts/real_errors_benchmark.py --datasets hospital beers
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import platform
import re
import sys
import urllib.request
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "benchmark-key-not-real")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cleaning_benchmark as cleaning  # noqa: E402
import openml_tasks  # noqa: E402
from app.services import operations, profiling  # noqa: E402

DATASETS = ["hospital", "flights", "beers", "rayyan", "movies_1"]
SOURCE = "https://raw.githubusercontent.com/BigDaMa/raha/master/datasets/{name}/{kind}.csv"


# --- the files ----------------------------------------------------------------------------------------------------


def _normalised(value: str) -> str:
    """Raha's comparison: HTML entities decoded, whitespace runs collapsed, ends stripped."""
    return re.sub("[\t\n ]+", " ", html.unescape(value)).strip("\t\n ")


def load(name: str, cache: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(the dirty file as the app reads it, the dirty file as text, the clean file as text), columns by position
    under the dirty file's names (some clean files name them differently)."""
    paths = {}
    for kind in ("dirty", "clean"):
        path = Path(cache) / "raha" / f"{name}_{kind}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            request = urllib.request.Request(SOURCE.format(name=name, kind=kind),
                                             headers={"User-Agent": "dataclean-ai-benchmark"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 -- fixed GitHub URL
                path.write_bytes(response.read())
        paths[kind] = path
    as_read = pd.read_csv(paths["dirty"], low_memory=False)
    dirty_text = pd.read_csv(paths["dirty"], dtype=str, keep_default_na=False).map(_normalised)
    clean_text = pd.read_csv(paths["clean"], dtype=str, keep_default_na=False).map(_normalised)
    clean_text.columns = dirty_text.columns
    return as_read, dirty_text, clean_text


# --- detection ----------------------------------------------------------------------------------------------------


def flagged_cells(frame: pd.DataFrame, issue) -> pd.Series:
    """The rows of `issue.column_name` the finding points at, as a boolean Series (all False when it points at no
    cell, or its column is not in the frame)."""
    column = str(issue.column_name)
    none = pd.Series(False, index=frame.index)
    if column not in frame.columns:
        return none
    values = frame[column]
    kind = cleaning.finding_kind(issue)
    present = values.notna()
    if kind == "missing_values":
        return values.isna()
    if kind == "type_mismatch":
        if issue.stats.get("detected_as") == "numeric":
            return present & pd.to_numeric(values, errors="coerce").isna()
        return present & pd.to_datetime(values, errors="coerce", format="mixed").isna()
    if kind == "outliers":
        q1, q3 = issue.stats.get("q1"), issue.stats.get("q3")
        numbers = pd.to_numeric(values, errors="coerce")
        if q1 is None or q3 is None:
            return none
        spread = q3 - q1
        return (numbers < q1 - 1.5 * spread) | (numbers > q3 + 1.5 * spread)
    text = values.astype(str).where(present)
    if kind == "stray_whitespace":
        return present & (text != text.str.strip())
    if kind == "inconsistent_casing":
        folded = text.str.lower()
        commonest = text.groupby(folded).agg(lambda s: s.value_counts().index[0])
        return present & (text != folded.map(commonest))
    if kind == "date_formats":
        parsed = pd.to_datetime(values, errors="coerce", format="mixed")
        shapes = text.where(parsed.notna()).dropna().map(profiling._date_shape)
        if shapes.empty:
            return none
        return (text.where(parsed.notna()).map(lambda v: profiling._date_shape(v) if isinstance(v, str) else None)
                .ne(shapes.value_counts().index[0]) & parsed.notna())
    return none


def score_detection(as_read: pd.DataFrame, errors: pd.DataFrame) -> dict:
    flags = pd.DataFrame(False, index=as_read.index, columns=as_read.columns)
    by_kind: dict[str, dict[str, int]] = {}
    for issue in profiling.detect_issues(as_read):
        cells = flagged_cells(as_read, issue)
        if not cells.any():
            continue
        column = str(issue.column_name)
        flags[column] |= cells.to_numpy()
        kind = cleaning.finding_kind(issue)
        counts = by_kind.setdefault(kind, {"flagged": 0, "right": 0})
        counts["flagged"] += int(cells.sum())
        counts["right"] += int((cells.to_numpy() & errors[column].to_numpy()).sum())
    truth = errors.to_numpy()
    flagged = flags.to_numpy()
    right = int((truth & flagged).sum())
    columns_with_errors = [c for c in errors.columns if errors[c].any()]
    return {
        "erroneous_cells": int(truth.sum()),
        "flagged_cells": int(flagged.sum()),
        "right": right,
        **prf(right, int(flagged.sum()), int(truth.sum())),
        "columns_with_errors": len(columns_with_errors),
        "columns_with_errors_flagged": sum(bool(flags[c].any()) for c in columns_with_errors),
        "by_kind": by_kind,
    }


def prf(right: int, claimed: int, actual: int) -> dict:
    precision = right / claimed if claimed else None
    recall = right / actual if actual else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else 0.0
    return {"precision": None if precision is None else round(precision, 4),
            "recall": None if recall is None else round(recall, 4), "f1": round(f1, 4)}


# --- repair -------------------------------------------------------------------------------------------------------


def _same(a, b: str) -> bool:
    """A repaired value against the clean text: equal as text, as numbers when both read as numbers, or as the same
    moment when both read as dates -- a date rewritten in another format is not damaged."""
    text = "" if a is None or (isinstance(a, float) and np.isnan(a)) or a is pd.NA else _normalised(str(a))
    if text == b:
        return True
    if not text or not b:
        return False
    try:
        return bool(np.isclose(float(text), float(b)))
    except ValueError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first, second = pd.to_datetime([text, b], errors="coerce", format="mixed")
    return bool(pd.notna(first) and pd.notna(second) and first == second)


def repaired(as_read: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The dirty file with the reference fix for every finding applied, in the cleaning benchmark's order."""
    issues = profiling.detect_issues(as_read)
    issues.sort(key=lambda i: cleaning.ORDER.index(cleaning.finding_kind(i))
                if cleaning.finding_kind(i) in cleaning.ORDER else len(cleaning.ORDER))
    frame, applied = as_read, []
    for issue in issues:
        chosen = cleaning.REFERENCE_FIX(issue, frame)
        if chosen is None or chosen[0] == "drop_duplicate_rows":
            continue
        name, params = chosen
        try:
            frame = operations.apply_operation(frame, name, params)[0]
            applied.append(f"{name}({params.get('column', '')})")
        except operations.OperationFailedError:
            continue
    return frame, applied


def _unchanged(a, b) -> bool:
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return str(a) == str(b)


def score_repair(as_read: pd.DataFrame, clean_text: pd.DataFrame, errors: pd.DataFrame) -> dict:
    """Only cells the fixes changed are judged, against the file as the app read it, so how pandas reads a value
    ("TRUE" as True) is not mistaken for a repair."""
    fixed, applied = repaired(as_read)
    right = changed = damaged = invented = 0
    damage_by_column: dict[str, int] = {}
    for column in as_read.columns:
        if column not in fixed.columns:  # a fix dropped it: every cell changed
            continue
        for value, was, truth, is_error in zip(fixed[column].tolist(), as_read[column].tolist(),
                                               clean_text[column].tolist(), errors[column].tolist()):
            if _unchanged(value, was):
                continue
            changed += 1
            if _same(value, truth):
                right += int(is_error)
            elif not is_error and truth == "":
                invented += 1  # a blank that was really blank, filled with a made-up value
            elif not is_error:
                damaged += 1  # a correct written value, overwritten
                damage_by_column[column] = damage_by_column.get(column, 0) + 1
    return {"fixes": applied, "changed_cells": changed, "right": right, "damaged": damaged, "invented": invented,
            "damaged_by_column": damage_by_column, **prf(right, changed, int(errors.to_numpy().sum()))}


# --- running ------------------------------------------------------------------------------------------------------


def run(datasets: list[str], cache: Path) -> dict:
    results: dict = {"python": platform.python_version(), "datasets": {}}
    for name in datasets:
        as_read, dirty_text, clean_text = load(name, cache)
        errors = dirty_text != clean_text
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            entry = {"rows": len(as_read), "columns": as_read.shape[1],
                     "detection": score_detection(as_read, errors),
                     "repair": score_repair(as_read, clean_text, errors)}
        results["datasets"][name] = entry
        d, r = entry["detection"], entry["repair"]
        print(f"{name}: {d['erroneous_cells']} erroneous cells; detection P {d['precision']} R {d['recall']} "
              f"F1 {d['f1']}; repair P {r['precision']} R {r['recall']}, {r['damaged']} damaged", flush=True)
    return results


def markdown(results: dict) -> str:
    def cell(value) -> str:
        return "" if value is None else f"{value:.3f}"

    lines = ["| Dataset | Erroneous cells | Detection P | R | F1 | Columns with errors flagged | Repair P | R | "
             "Correct values overwritten | Real blanks filled |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, e in results["datasets"].items():
        d, r = e["detection"], e["repair"]
        lines.append(f"| {name} | {d['erroneous_cells']:,} | {cell(d['precision'])} | {cell(d['recall'])} | "
                     f"{cell(d['f1'])} | {d['columns_with_errors_flagged']} of {d['columns_with_errors']} | "
                     f"{cell(r['precision'])} | {cell(r['recall'])} | {r['damaged']:,} | {r['invented']:,} |")
    lines += ["", "Cells flagged and right, by kind of finding (all datasets):", ""]
    totals: dict[str, list[int]] = {}
    for e in results["datasets"].values():
        for kind, c in e["detection"]["by_kind"].items():
            t = totals.setdefault(kind, [0, 0])
            t[0] += c["flagged"]
            t[1] += c["right"]
    for kind, (flagged, right) in sorted(totals.items()):
        lines.append(f"- {kind}: {flagged:,} flagged, {right:,} erroneous ({right / flagged:.0%})")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS + ["tax"])
    parser.add_argument("--cache", type=Path, default=openml_tasks.DEFAULT_CACHE)
    parser.add_argument("--out", default="real_errors_benchmark.json")
    parser.add_argument("--markdown", help="also write the table to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    results = run(args.datasets, args.cache)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = markdown(results)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
