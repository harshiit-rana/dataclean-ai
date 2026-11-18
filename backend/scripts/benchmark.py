"""How long profiling, a plain train and the improvement loop take, at three sizes of file.  (issue #107)

Synthetic files of 1,000, 10,000 and 100,000 rows by 20 columns (numeric, categorical, some blanks, a two-class
target), each timed through the same service functions the routes call: `profiling.detect_issues`,
`modeling.train_candidates` and `improvement.improve` with a fixed budget. The results go to a JSON file. Given a
baseline file from an earlier run, every timing more than `--tolerance` slower than its baseline is reported.

Timings depend on the machine, and two CI runners of the same image are not equally fast: on 9 November 2026 every
timing came out 45-104% slower than the week before, on the same commit. So each run also times a fixed reference
workload that has nothing to do with the app, and the baseline is scaled by how much slower or faster that ran before
comparing. A uniformly slower machine is then not a regression; a step that got slower on its own still is. Slower
timings are reported, not failed (`--strict` makes them exit non-zero): a timing is evidence about a change, not a
pass or a fail for it.

Usage (from `backend/`):
    python scripts/benchmark.py                                   # all three sizes; the 100k improve takes minutes
    python scripts/benchmark.py --sizes 1000 10000 --out bench.json
    python scripts/benchmark.py --baseline old.json --tolerance 0.3   # add --strict to exit non-zero when slower
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


def reference_seconds(repeats: int = 5) -> float:
    """How fast this machine is, apart from the app: the median of a few runs of a fixed mix of interpreter work and
    numpy work, the two things the app's timings are made of. The fastest of a few runs, so an interruption does not count."""
    def once() -> None:
        total = 0
        for i in range(6_000_000):
            total += i % 7
        rng = np.random.default_rng(0)
        np.sort(rng.normal(size=1_000_000))
        a = rng.normal(size=(300, 300))
        for _ in range(20):
            a = a @ a / np.linalg.norm(a)
    return round(min(timed(once) for _ in range(repeats)), 4)


def run(sizes: list[int]) -> dict:
    results: dict = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "columns": COLUMNS,
        "budget": BUDGET,
        "reference_seconds": reference_seconds(),
        "timings": {},
    }
    print(f"Reference workload: {results['reference_seconds']:.2f}s", flush=True)
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


def machine_speed(results: dict, baseline: dict) -> float:
    """How much longer the same work takes on this machine than on the baseline's: 1.0 when either run has no
    reference timing, as runs from before it was added do not."""
    now, before = results.get("reference_seconds"), baseline.get("reference_seconds")
    return now / before if now and before else 1.0


def regressions(results: dict, baseline: dict, tolerance: float) -> list[str]:
    """Every timing slower than its baseline by more than `tolerance` (a share: 0.3 is 30%), after scaling the
    baseline by how much slower or faster this machine ran the reference workload."""
    speed = machine_speed(results, baseline)
    found = []
    for rows, timings in results["timings"].items():
        for step, seconds in timings.items():
            before = (baseline.get("timings") or {}).get(rows, {}).get(step)
            if not before:
                continue
            expected = before * speed
            if seconds > expected * (1 + tolerance):
                note = "" if speed == 1.0 else f", {expected:.1f}s expected on this machine"
                found.append(f"{step} on {int(rows):,} rows: {seconds:.1f}s against {before:.1f}s{note} "
                             f"(+{seconds / expected - 1:.0%})")
    return found


def report(results: dict, baseline: dict, slower: list[str], tolerance: float) -> str:
    """The comparison as Markdown, for the CI run's page."""
    speed = machine_speed(results, baseline)
    lines = ["## Timings", "", "| Rows | Step | Seconds | Baseline |", "|---:|---|---:|---:|"]
    for rows, timings in results["timings"].items():
        for step, seconds in timings.items():
            before = (baseline.get("timings") or {}).get(rows, {}).get(step)
            lines.append(f"| {int(rows):,} | {step} | {seconds:.1f} | {'' if before is None else f'{before:.1f}'} |")
    lines.append("")
    lines.append(f"This machine took {speed:.2f}x the baseline's time on the reference workload; the baseline is "
                 "scaled by that." if speed != 1.0 else "No reference timing to scale by: timings compared as they are.")
    lines.append("")
    lines += [f"- SLOWER: {line}" for line in slower] or [f"No timing more than {tolerance:.0%} slower than expected."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--out", default="benchmark.json")
    parser.add_argument("--baseline", help="an earlier run's JSON to compare with")
    parser.add_argument("--tolerance", type=float, default=0.3)
    parser.add_argument("--strict", action="store_true", help="exit non-zero when a timing is slower than expected")
    args = parser.parse_args()

    results = run(args.sizes)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        slower = regressions(results, baseline, args.tolerance)
        for line in slower:
            print(f"SLOWER: {line}")
            if os.environ.get("GITHUB_ACTIONS"):
                print(f"::warning title=Slower timing::{line}")
        if not slower:
            print(f"No timing more than {args.tolerance:.0%} slower than expected.")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(report(results, baseline, slower, args.tolerance))
        if slower and args.strict:
            sys.exit(1)


if __name__ == "__main__":
    main()
