"""Does the verdict's multiple-candidate correction still earn its place?  (issue #35)

`modeling._auc_noise_threshold` raises the AUC bar a model must clear to be called better than
guessing with the number of candidates trained (z = 1.645 for one, 2.13 for three). That was
compensating for reporting the *best of several draws on the test set*. Since the winner is chosen
on training folds, the reported test AUC is arguably one draw, and the correction arguably too strict.
This script measures it instead of arguing it.

For each test-set size and each scenario, it generates data, runs the real `train_candidates` (all
three classifiers, so the production candidate count of three applies), and then re-scores the very
same run under each rule by calling `assess_result` again with a different `model_count`:

    current   the rule as shipped: z by the number of candidates trained (three -> z = 2.13)
    relaxed   one draw: z = 1.645 whatever the number of candidates

Scenarios:
    noise   the target is a coin flip unrelated to every column. Any "marginal" or "useful" verdict
            is a false positive.
    weak    the target depends weakly on one column (population AUC printed below). A "marginal" or
            "useful" verdict is a correct detection; this is the price of the stricter rule.

It also prints, for the noise scenario, the chosen model's test AUC against the two thresholds, and
how it compares with the average over all candidates: if choosing on folds really leaves the test
score a single draw, the chosen model's AUC is no higher than a candidate picked at random.

Usage (from `backend/`):
    python scripts/simulate_verdict_correction.py                 # about 8 minutes on a laptop
    python scripts/simulate_verdict_correction.py --runs 8 --weak-runs 4        # a quick look
    python scripts/simulate_verdict_correction.py --runs 30 --weak-runs 12      # the run in the decision note, ~14 minutes

The defaults are small enough to finish in under ten minutes, which is too few datasets to tell a 3% rate
from a 7% one; the intervals in the table say so. Raise --runs when the question is a close one.

Nothing here changes the app. The reporting extras that do not affect the verdict (the seed check and
the best-single-column baseline) are switched off for speed.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "not-needed-for-this-script")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services import modeling  # noqa: E402

TEST_FRACTION = 0.2  # train_candidates' default hold-out, so total rows = test rows / 0.2
FEATURES = 5
MODELS = ["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"]
RULES = {"current": None, "relaxed": 1}  # None = as shipped (the number of candidates trained)
VERDICTS = ("no_better_than_guessing", "marginal", "useful", "not_enough_test_data")


def make_data(rows: int, scenario: str, signal: float, positive_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    frame = pd.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(FEATURES)})
    frame["group"] = rng.choice(["a", "b", "c"], rows)
    if scenario == "noise":
        probability = np.full(rows, positive_rate)
    else:
        intercept = math.log(positive_rate / (1 - positive_rate))
        probability = 1 / (1 + np.exp(-(intercept + signal * frame["x0"].to_numpy())))
    frame["outcome"] = np.where(rng.random(rows) < probability, "yes", "no")
    return frame


def population_auc(signal: float, positive_rate: float, seed: int = 0) -> float:
    """The AUC of the weak scenario's best possible score, from a large sample, for the report."""
    rng = np.random.default_rng(seed)
    frame = make_data(60_000, "weak", signal, positive_rate, rng)
    positive = frame["outcome"].to_numpy() == "yes"
    ranks = pd.Series(frame["x0"].to_numpy()).rank().to_numpy()
    n1, n0 = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    centre = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return (max(0.0, centre - half), min(1.0, centre + half))


def one_run(df: pd.DataFrame) -> dict:
    """Trains once and returns the verdict under each rule, plus the test AUCs for the selection check."""
    captured: dict = {}
    real_assess = modeling.assess_result

    def spy(metrics, baseline, task, **kwargs):
        captured["call"] = (metrics, baseline, task, kwargs)
        return real_assess(metrics, baseline, task, **kwargs)

    modeling.assess_result = spy
    try:
        result = modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=MODELS,
            include_deep_diagnostics=False,
        )
    finally:
        modeling.assess_result = real_assess

    metrics, baseline, task, kwargs = captured["call"]
    verdicts = {}
    for name, count in RULES.items():
        override = dict(kwargs)
        if count is not None:
            override["model_count"] = count
        verdicts[name] = real_assess(metrics, baseline, task, **override)["verdict"]

    aucs = [r["metrics"].get("roc_auc") for r in result["results"] if r["metrics"].get("roc_auc") is not None]
    class_counts = kwargs.get("class_counts") or []
    return {
        "verdicts": verdicts,
        "chosen_auc": metrics.get("roc_auc"),
        "mean_candidate_auc": float(np.mean(aucs)) if aucs else None,
        "class_counts": class_counts,
        "test_rows": kwargs.get("test_rows"),
        "candidates": kwargs.get("model_count"),
    }


def thresholds(class_counts: list[int]) -> dict[str, float]:
    return {
        name: modeling._auc_noise_threshold(class_counts, count if count is not None else len(MODELS))
        for name, count in RULES.items()
    }


def table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=18, help="noise datasets per size (default 18)")
    parser.add_argument("--weak-runs", type=int, default=6, help="weak-signal datasets per size (default 6)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[30, 60, 120, 500], help="test-set sizes")
    parser.add_argument("--signal", type=float, default=0.5, help="weak scenario: logit slope on x0 (default 0.5)")
    parser.add_argument("--positive-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, help="also write the report here")
    args = parser.parse_args()

    # Reporting extras that cannot change the verdict.
    modeling.SEED_CHECK_MAX_ROWS = 0
    modeling.BASELINE_MAX_COLUMNS = 0

    started = time.time()
    rng = np.random.default_rng(args.seed)
    auc_pop = population_auc(args.signal, args.positive_rate)
    sections: list[str] = []
    selection_rows: list[list[str]] = []

    for scenario in ("noise", "weak"):
        rows: list[list[str]] = []
        runs = args.runs if scenario == "noise" else args.weak_runs
        for size in args.sizes:
            total = int(round(size / TEST_FRACTION))
            outcomes = {name: dict.fromkeys(VERDICTS, 0) for name in RULES}
            chosen, average, counts = [], [], []
            for _ in range(runs):
                done = one_run(make_data(total, scenario, args.signal, args.positive_rate, rng))
                for name in RULES:
                    outcomes[name][done["verdicts"][name]] += 1
                if done["chosen_auc"] is not None:
                    chosen.append(done["chosen_auc"])
                    average.append(done["mean_candidate_auc"])
                    counts.append(done["class_counts"])
            for name in RULES:
                tally = outcomes[name]
                judged = runs - tally["not_enough_test_data"]
                flagged = tally["marginal"] + tally["useful"]
                low, high = wilson(flagged, judged)
                rows.append([
                    str(size), name, str(runs), str(tally["no_better_than_guessing"]), str(tally["marginal"]),
                    str(tally["useful"]), str(tally["not_enough_test_data"]),
                    f"{flagged / judged:.0%} ({low:.0%}-{high:.0%})" if judged else "n/a",
                ])
            if scenario == "noise" and chosen:
                typical = counts[len(counts) // 2]
                limits = thresholds(typical)
                selection_rows.append([
                    str(size), f"{np.mean(chosen):.3f}", f"{np.mean(average):.3f}",
                    f"{np.percentile(chosen, 95):.3f}", f"{limits['relaxed']:.3f}", f"{limits['current']:.3f}",
                ])
        heading = (
            "### Pure noise: the target is a coin flip, so every marginal or useful call is a false positive"
            if scenario == "noise"
            else f"### Weak real signal (population AUC about {auc_pop:.2f}): marginal or useful is a correct detection"
        )
        rate = "False-positive rate" if scenario == "noise" else "Detection rate"
        sections += [
            heading, "",
            table(rows, ["Test rows", "Rule", "Runs", "Guessing", "Marginal", "Useful", "Unjudgeable", f"{rate} (95% range)"]),
            "",
        ]

    sections += [
        "### Is the chosen model's test AUC still the best of several draws? (pure noise)",
        "",
        table(
            selection_rows,
            ["Test rows", "Mean AUC, chosen", "Mean AUC, all candidates", "95th percentile, chosen",
             "Bar, relaxed", "Bar, current"],
        ),
        "",
        "If choosing on training folds leaves the test score a single draw, the chosen model's mean AUC "
        "matches the all-candidates mean, and its 95th percentile sits near the relaxed bar. If it were the "
        "best of three test draws, the chosen mean would sit clearly above the all-candidates mean.",
    ]

    header = (
        f"## Verdict-correction simulation\n\n{args.runs} noise and {args.weak_runs} weak-signal datasets per size, "
        f"{len(MODELS)} candidates, "
        f"test fraction {TEST_FRACTION:.0%}, positive rate {args.positive_rate:.0%}, seed {args.seed}; "
        f"took {time.time() - started:.0f}s.\n"
    )
    report = header + "\n" + "\n".join(sections) + "\n"
    print(report)
    if args.out:
        args.out.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
