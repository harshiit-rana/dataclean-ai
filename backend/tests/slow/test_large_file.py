"""A real run on a very big file stays within the processing limit and says what it reduced.

Takes minutes, so it is marked `slow` and left out of the default run: `pytest -m slow`.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from app.config import Settings
from app.services import modeling

pytestmark = pytest.mark.slow

ROWS = 200_000


def big_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=ROWS)
    return pd.DataFrame({
        "x": x, "a": rng.normal(size=ROWS), "b": rng.normal(size=ROWS), "group": rng.choice(["p", "q", "r"], ROWS),
        "outcome": np.where(x + rng.normal(0, 1, ROWS) > 0, "yes", "no"),
    })


def test_a_200000_row_train_is_reduced_to_the_limit_and_says_so():
    limit = Settings().training_cost_limit
    started = time.time()
    result = modeling.train_candidates(
        big_frame(), target_column="outcome", task="classification",
        model_names=["logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"],
        include_deep_diagnostics=False, cost_limit=limit,
    )
    elapsed = time.time() - started

    report = result["resource_limits"]
    assert report is not None and report["estimated_before"] > limit >= report["estimated"]
    assert report["reductions"][0]["kind"] == "selection_rows"
    # The reduction touched only the comparison: every candidate was fitted on every training row.
    assert result["selection"]["rows_used"] < result["selection"]["rows_available"] == result["split"]["train_rows"]
    assert all(r["metrics"] for r in result["results"])
    print(f"\n200,000 rows: {elapsed:.0f}s; compared on {result['selection']['rows_used']:,} of "
          f"{result['selection']['rows_available']:,} rows; estimate {report['estimated_before']:,} -> {report['estimated']:,}")
