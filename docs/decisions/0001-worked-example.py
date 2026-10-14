"""Reproduces the row counts in 0001-evaluation-protocol.md.

Run from the repository root:

    backend/.venv/Scripts/python.exe docs/decisions/0001-worked-example.py

It reads samples/subscription_churn.csv and prints, for each design the ADR weighed, how many
rows land in training, development evaluation and final evaluation -- and whether each held-out
partition clears the app's own bar for a score to mean anything (`MIN_TEST_ROWS_TO_JUDGE` and
`MIN_TEST_ROWS_PER_CLASS_TO_JUDGE` in `modeling.py`). It also shows the same arithmetic for a
300-row file with a 20% rarer class, the case the ADR must not make unusable.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from app.services import modeling  # noqa: E402

TARGET = "churned"
FINAL_FRACTION = 0.15  # the sealed final holdout in option (b)
DEV_FRACTION = 0.15  # the development evaluation set in option (b)


def verdict(rows: int, rarer: int) -> str:
    ok = rows >= modeling.MIN_TEST_ROWS_TO_JUDGE and rarer >= modeling.MIN_TEST_ROWS_PER_CLASS_TO_JUDGE
    return "can be judged" if ok else "TOO SMALL TO JUDGE"


def counts(labels: pd.Series) -> tuple[int, int]:
    return len(labels), int(labels.value_counts().min())


def two_way(y: pd.Series) -> dict:
    train, test = train_test_split(
        y, test_size=modeling.DEFAULT_TEST_SIZE, random_state=modeling.RANDOM_STATE, stratify=y
    )
    return {"train": counts(train), "evaluation": counts(test)}


def three_way(y: pd.Series) -> dict:
    rest, final = train_test_split(
        y, test_size=FINAL_FRACTION, random_state=modeling.RANDOM_STATE, stratify=y
    )
    train, dev = train_test_split(
        rest,
        test_size=DEV_FRACTION / (1 - FINAL_FRACTION),
        random_state=modeling.RANDOM_STATE,
        stratify=rest,
    )
    return {"train": counts(train), "development": counts(dev), "final": counts(final)}


def show(title: str, parts: dict) -> None:
    print(f"\n{title}")
    for name, (rows, rarer) in parts.items():
        held_out = name != "train"
        note = f"  -> {verdict(rows, rarer)}" if held_out else ""
        print(f"  {name:<12} {rows:>5} rows   rarer class {rarer:>4}{note}")


def main() -> None:
    frame = pd.read_csv(ROOT / "samples" / "subscription_churn.csv")
    y = frame[TARGET].dropna()
    print(f"samples/subscription_churn.csv: {len(frame):,} rows, {len(y):,} labelled")
    print(f"  {TARGET}: {y.value_counts().to_dict()}")

    show("(a) two-way split, what the app does today (count and warn)", two_way(y))
    show(
        f"(b) three-way split, {1 - FINAL_FRACTION - DEV_FRACTION:.0%} / {DEV_FRACTION:.0%} / "
        f"{FINAL_FRACTION:.0%}, sealed final holdout",
        three_way(y),
    )

    folds = 5
    outer_test = len(y) // folds
    print("\n(c) nested cross-validation, 5 outer x 5 inner folds")
    print(f"  each outer test fold   {outer_test:>5} rows   (every row is tested once)")
    print(f"  each inner training    {len(y) - outer_test - outer_test // 4:>5} rows (approximately)")
    print(f"  fits per model         {folds * folds + folds:>5}   (vs 6 for the two-way split)")
    print("  no sealed set exists afterwards: every row has informed some decision")

    small = pd.Series(["no"] * 240 + ["yes"] * 60)  # 300 rows, 20% rarer class
    print("\n--- the same arithmetic for a 300-row file with a 20% rarer class ---")
    show("(a) two-way split", two_way(small))
    show("(b) three-way split", three_way(small))


if __name__ == "__main__":
    main()
