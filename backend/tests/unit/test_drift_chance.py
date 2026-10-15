"""The drift verdict accounts for testing many columns at once.

Every input whose PSI crossed 0.1 was called shifted. With thirty inputs and a moderately sized new
file, several cross it by chance, so a fresh sample from the very same population came back "shifted"
most of the time. Each column now also has to clear the PSI chance reaches for samples of its sizes,
at a 5% error rate spread across all the columns compared.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import drift

NUMERIC = [f"n{i}" for i in range(25)]
CATEGORICAL = [f"c{i}" for i in range(5)]


def table(n: int, seed: int, shift: str | None = None, by: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {name: rng.normal(0, 1, n) for name in NUMERIC}
    data.update({name: rng.choice(["a", "b", "c", "d"], n, p=[0.4, 0.3, 0.2, 0.1]) for name in CATEGORICAL})
    frame = pd.DataFrame(data)
    if shift:
        frame[shift] = frame[shift] + by
    return frame


REFERENCE = table(1000, 0)


class TestSameDataIsCalledSimilar:
    def test_a_fresh_sample_from_the_same_population_is_similar_in_at_least_19_of_20_seeds(self):
        verdicts = [drift.compare(REFERENCE, table(200, seed), NUMERIC, CATEGORICAL)["verdict"] for seed in range(1, 21)]
        assert verdicts.count("similar") >= 19, verdicts

    def test_the_fixed_convention_alone_would_have_cried_wolf(self):
        """What the change is for: on the same twenty files, some column crosses 0.1 in most of them."""
        crossings = sum(
            any(c["psi"] > drift.PSI_MODERATE for c in drift.compare(REFERENCE, table(200, seed), NUMERIC, CATEGORICAL)["columns"])
            for seed in range(1, 21)
        )
        assert crossings >= 10


class TestARealShiftIsStillCaught:
    def test_a_genuine_shift_in_one_column_is_detected_and_named(self):
        report = drift.compare(REFERENCE, table(200, 99, shift="n3"), NUMERIC, CATEGORICAL)
        n3 = next(c for c in report["columns"] if c["column"] == "n3")
        assert n3["beyond_chance"] is True and n3["status"] == "different"
        assert report["verdict"] == "different" and "'n3'" in report["headline"]

    def test_a_moderate_shift_is_a_shift(self):
        report = drift.compare(REFERENCE, table(400, 98, shift="n7", by=0.45), NUMERIC, CATEGORICAL)
        n7 = next(c for c in report["columns"] if c["column"] == "n7")
        assert n7["status"] in {"shifted", "different"} and n7["beyond_chance"]

    def test_a_category_that_changes_mix_is_caught(self):
        new = table(300, 97)
        new["c2"] = np.random.default_rng(5).choice(["a", "b", "c", "d"], 300, p=[0.1, 0.2, 0.3, 0.4])
        report = drift.compare(REFERENCE, new, NUMERIC, CATEGORICAL)
        assert next(c for c in report["columns"] if c["column"] == "c2")["status"] == "different"

    def test_the_headline_says_how_many_inputs_moved_more_than_chance(self):
        report = drift.compare(REFERENCE, table(200, 99, shift="n3"), NUMERIC, CATEGORICAL)
        assert "1 of the 30 inputs compared moved more than chance explains" in report["headline"]


class TestTheThresholds:
    def test_every_compared_column_reports_its_psi_and_chance_threshold(self):
        report = drift.compare(REFERENCE, table(200, 3), NUMERIC, CATEGORICAL)
        assert all("psi" in c and "chance_threshold" in c and "beyond_chance" in c for c in report["columns"])

    def test_a_smaller_new_file_has_a_higher_threshold(self):
        expected = np.full(10, 0.1)
        assert drift.chance_threshold(expected, 1000, 50, 0.01) > drift.chance_threshold(expected, 1000, 2000, 0.01)

    def test_more_columns_raise_each_columns_threshold(self):
        expected = np.full(10, 0.1)
        assert drift.chance_threshold(expected, 1000, 200, 0.05 / 30) > drift.chance_threshold(expected, 1000, 200, 0.05)

    def test_it_is_deterministic(self):
        expected = np.full(10, 0.1)
        assert drift.chance_threshold(expected, 1000, 200, 0.01) == drift.chance_threshold(expected, 1000, 200, 0.01)

    def test_a_large_but_trivial_shift_on_a_huge_file_is_still_not_called_a_shift(self):
        """Beyond chance is not enough: the convention stays as a floor on how much it moved."""
        reference = table(20000, 1)
        new = table(20000, 2, shift="n1", by=0.05)
        n1 = next(c for c in drift.compare(reference, new, NUMERIC, CATEGORICAL)["columns"] if c["column"] == "n1")
        assert n1["psi"] < drift.PSI_MODERATE and n1["status"] == "stable"

    def test_not_judged_columns_do_not_count_towards_the_number_of_tests(self):
        sparse = REFERENCE.assign(n0=np.where(np.arange(1000) < 990, np.nan, 1.0))
        report = drift.compare(sparse, table(200, 4).assign(n0=np.nan), NUMERIC, CATEGORICAL)
        compared = [c for c in report["columns"] if "chance_threshold" in c]
        assert len(compared) == 29
