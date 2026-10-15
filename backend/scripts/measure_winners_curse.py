"""How often does the settings round accept a change on data with no signal?  (issue #51)

The settings search returns the best of up to a dozen noisy fold scores, so on data where nothing predicts the
target, the luckiest setting can still clear the gain rule against the incumbent. The winner is now re-scored on
folds the search did not select on, with the incumbent, and has to clear the rule there too.

This runs the improvement loop on pure-noise data for 25 seeds, once with the confirmation off (how it was) and once
with it on (how it is), and counts how many seeds accepted a setting in the settings round. The same seed gives the
same data, the same search and the same first decision, so a seed accepted with the confirmation is always one the
old rule accepted too.

Usage (from `backend/`):
    python scripts/measure_winners_curse.py                  # about 3 minutes
    python scripts/measure_winners_curse.py --seeds 100 --model random_forest_classifier
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ANTHROPIC_API_KEY", "not-needed-for-this-script")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.services import improvement  # noqa: E402


def noise(rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"x{i}": rng.normal(size=rows) for i in range(4)})
    frame["outcome"] = rng.choice(["yes", "no"], rows)
    return frame


def accepted_in_settings_round(rows: int, seed: int, model: str, budget: int) -> bool:
    found = improvement.improve(
        noise(rows, seed), target_column="outcome", task="classification", model_names=[model], budget=budget,
    )
    return next(r for r in found["rounds"] if r["name"] == "hyperparameters")["accepted"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=25)
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--model", default="logistic_regression")
    args = parser.parse_args()

    started = time.time()
    counts = {}
    for label, confirm in (("before (no confirmation)", False), ("after (confirmed on fresh folds)", True)):
        improvement.CONFIRM_SEARCH_WINNERS = confirm
        accepted = [accepted_in_settings_round(args.rows, seed, args.model, args.budget) for seed in range(args.seeds)]
        counts[label] = accepted
        print(f"{label}: accepted a setting in {sum(accepted)} of {args.seeds} seeds")
    old, new = counts.values()
    print(f"seeds the confirmation refused: {sum(o and not n for o, n in zip(old, new))}; "
          f"seeds it accepted that the old rule refused: {sum(n and not o for o, n in zip(old, new))}")
    print(f"({args.model}, {args.rows} rows, budget {args.budget}; took {time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
