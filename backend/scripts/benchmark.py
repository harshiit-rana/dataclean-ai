"""How long profiling, a plain train and the improvement loop take, at three sizes of file.  (issue #107)

Synthetic files of 1,000, 10,000 and 100,000 rows by 20 columns (numeric, categorical, some blanks, a two-class
target), each timed through the same service functions the routes call: `profiling.detect_issues`,
`modeling.train_candidates` and `improvement.improve` with a fixed budget. The results go to a JSON file. Given a
baseline file from an earlier run, every timing more than `--tolerance` slower than its baseline is flagged and the
script exits non-zero, so a CI job can mark the run without the job blocking anything.

Timings depend on the machine, so compare runs from the same kind of machine: the CI job keeps its results as an
artifact, which is the baseline for the next.

Usage (from `backend/`):
    python scripts/benchmark.py                                   # all three sizes; the 100k improve takes minutes
    python scripts/benchmark.py --sizes 1000 10000 --out bench.json
    python scripts/benchmark.py --baseline old.json --tolerance 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANTHROPIC_API_KEY", "benchmark-key-not-real")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services import improvement, modeling, profiling  # noqa: E402

COLUMNS = 20
BUDGET = 4
MODELS = ["logistic_regression", "random_forest_classifier"]


def synthetic(rows: int, seed: int = 0) -> pd.DataFrame:
    """Twenty columns that look like a real export: mostly numeric, a few categories, some blanks, a target that
    depends on three of the inputs."""
    rng = np.random.default_rng(seed)
    data: dict = {}
    for i in range(15):
        values = rng.normal(size=rows)
        values[rng.random(rows) < 0.03] = np.nan
        data[f"num_{i}"] = values.round(4)
    for i in range(5):
        data[f"cat_{i}"] = rng.choice([f"v{j}" for j in range(4 + i * 3)], rows)
    signal = np.nan_to_num(data["num_0"]) + 0.7 * np.nan_to_num(data["num_1"]) - 0.5 * (data["cat_0"] == "v1")
    data["outcome"] = np.where(signal + rng.normal(0, 1, rows) > 0, "yes", "no")
    return pd.DataFrame(data)


def timed(work) -> float:
    started = time.perf_counter()
    work()
    return round(time.perf_counter() - started, 3)


def run(sizes: list[int]) -> dict:
    results: dict = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "columns": COLUMNS,
        "budget": BUDGET,
        "timings": {},
    }
    for rows in sizes:
        frame = synthetic(rows)
        timings = {
            "profiling": timed(lambda: profiling.detect_issues(frame)),
            "train": timed(lambda: modeling.train_candidates(
                frame, target_column="outcome", task="classification", model_names=MODELS,
                include_deep_diagnostics=False,
            )),
            "improve": timed(lambda: improvement.improve(
                frame, target_column="outcome", task="classification", model_names=MODELS, budget=BUDGET,
            )),
        }
        results["timings"][str(rows)] = timings
        print(f"{rows:>7,} rows: " + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items()), flush=True)
    return results


def regressions(results: dict, baseline: dict, tolerance: float) -> list[str]:
    """Every timing slower than its baseline by more than `tolerance` (a share: 0.3 is 30%)."""
    found = []
    for rows, timings in results["timings"].items():
        for step, seconds in timings.items():
            before = (baseline.get("timings") or {}).get(rows, {}).get(step)
            if before and seconds > before * (1 + tolerance):
                found.append(f"{step} on {int(rows):,} rows: {seconds:.1f}s against {before:.1f}s (+{seconds / before - 1:.0%})")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--out", default="benchmark.json")
    parser.add_argument("--baseline", help="an earlier run's JSON to compare with")
    parser.add_argument("--tolerance", type=float, default=0.3)
    args = parser.parse_args()

    results = run(args.sizes)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.baseline and Path(args.baseline).exists():
        slower = regressions(results, json.loads(Path(args.baseline).read_text(encoding="utf-8")), args.tolerance)
        for line in slower:
            print(f"SLOWER: {line}")
        if slower:
            sys.exit(1)
        print(f"No timing more than {args.tolerance:.0%} slower than the baseline.")


if __name__ == "__main__":
    main()
