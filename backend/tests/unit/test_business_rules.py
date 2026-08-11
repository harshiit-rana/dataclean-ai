"""Business rules: the errors that are wrong rather than unusual.

Every other detector in this app reasons from the distribution of the data, which makes it
structurally blind to the largest category of real-world error. `age = -5` is not an outlier.
A 150% discount sits inside the interquartile range of a column that runs 0 to 200 because
half the rows are wrong. A delivery date before its order date is two ordinary dates.

These tests pin down three things: that a rule is data rather than code, that a proposal is
inert until someone approves it, and that the engine declines to answer when it cannot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import business_rules as br


@pytest.fixture
def orders() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 300
    order = pd.Series(
        pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    )
    delivery = pd.Series(order + pd.to_timedelta(rng.integers(1, 9, n), unit="D"))
    delivery.iloc[:6] = order.iloc[:6] - pd.Timedelta(days=3)  # 6 impossible deliveries
    price = rng.choice([2.5, 10.0, 7.25, 4.0], n)
    quantity = rng.integers(1, 9, n)
    total = pd.Series((price * quantity).round(2))
    total.iloc[:5] = 999.0  # 5 totals that don't add up
    age = pd.Series(rng.integers(18, 80, n).astype(float))
    age.iloc[:3] = [-5.0, -1.0, -40.0]
    return pd.DataFrame(
        {
            "age": age,
            "order_date": order,
            "delivery_date": delivery,
            "unit_price": price,
            "quantity": quantity,
            "total_amount": total,
            "balance": rng.normal(0, 500, n).round(2),  # legitimately negative
            "notes": [f"note {i}" for i in range(n)],
        }
    )


class TestTheEngineFindsWhatStatisticsCannot:
    def test_an_impossible_value_is_caught(self, orders):
        check = br.check_rule(orders, br.Rule(kind="range", column="age", minimum=0))
        assert check.status == "failed"
        assert "3 of 300" in check.finding

    def test_a_bound_on_both_sides_counts_breaches_of_either(self):
        values = [10.0, 50.0, 80.0, 25.0] * 25  # 100 rows, all legitimate
        values[0] = 150.0  # over the ceiling
        values[1] = -20.0  # under the floor
        check = br.check_rule(
            pd.DataFrame({"discount": values}),
            br.Rule(kind="range", column="discount", minimum=0, maximum=100),
        )
        assert check.status == "failed"
        assert "2 of 100" in check.finding

    def test_dates_that_cross(self, orders):
        check = br.check_rule(
            orders,
            br.Rule(
                kind="comparison",
                column="order_date",
                other_column="delivery_date",
                comparison="lte",
            ),
        )
        assert check.status == "failed"
        assert "6 of 300" in check.finding

    def test_arithmetic_that_does_not_add_up(self, orders):
        check = br.check_rule(
            orders,
            br.Rule(
                kind="arithmetic",
                column="unit_price",
                multiplier_column="quantity",
                result_column="total_amount",
            ),
        )
        assert check.status == "failed"
        assert "5 of 300" in check.finding

    def test_rounding_does_not_count_as_a_violation(self):
        """Money is rounded. An exact comparison fires on nearly every row of nearly every
        real file, which is how people learn to ignore a rule."""
        df = pd.DataFrame({"price": [3.333] * 50, "qty": [3] * 50, "total": [10.0] * 50})
        check = br.check_rule(
            df,
            br.Rule(
                kind="arithmetic", column="price", multiplier_column="qty", result_column="total"
            ),
        )
        assert check.status == "passed"

    def test_a_rule_that_holds_says_so(self, orders):
        check = br.check_rule(orders, br.Rule(kind="range", column="quantity", minimum=0))
        assert check.status == "passed"
        assert "Checked 300 rows" in check.evidence


class TestTheEngineDeclinesRatherThanGuessing:
    def test_a_missing_column_is_not_run(self, orders):
        check = br.check_rule(orders, br.Rule(kind="range", column="gone", minimum=0))
        assert check.status == "not_run"
        assert "'gone' is not in the data" in check.finding

    def test_two_text_columns_have_no_ordering(self, orders):
        """Comparing them alphabetically would produce confident nonsense."""
        check = br.check_rule(
            orders,
            br.Rule(
                kind="comparison", column="notes", other_column="age", comparison="lte"
            ),
        )
        assert check.status == "not_run"

    def test_a_rule_most_of_the_data_breaks_blames_the_rule(self):
        """Past a certain share the rule is the likelier error, and saying so is more useful
        than reporting two thirds of the file as broken."""
        df = pd.DataFrame({"score": list(range(100))})
        check = br.check_rule(df, br.Rule(kind="range", column="score", maximum=10))
        assert check.status == "warning"
        assert "the rule is wrong" in check.evidence


class TestARuleIsDataNotCode:
    def test_a_rule_with_no_bound_is_rejected_at_construction(self):
        with pytest.raises(br.InvalidRule):
            br.Rule(kind="range", column="age")

    def test_an_impossible_bound_is_rejected(self):
        with pytest.raises(br.InvalidRule):
            br.Rule(kind="range", column="age", minimum=100, maximum=0)

    def test_a_column_cannot_be_compared_against_itself(self):
        with pytest.raises(br.InvalidRule):
            br.Rule(kind="comparison", column="a", other_column="a", comparison="lte")

    def test_a_rule_survives_a_round_trip_through_json(self):
        rule = br.Rule(kind="range", column="age", minimum=0, maximum=120)
        assert br.Rule.from_dict(rule.as_dict()) == rule

    def test_derived_fields_do_not_break_the_round_trip(self):
        """as_dict emits name/sentence/expression for the client to render. Those come back on
        a save and must not be mistaken for constructor arguments."""
        payload = br.Rule(kind="range", column="age", minimum=0).as_dict()
        assert {"name", "sentence", "expression"} <= set(payload)
        assert br.Rule.from_dict(payload).column == "age"


class TestProposalsAreNarrow:
    def test_it_proposes_what_it_can_evidence(self, orders):
        sentences = [r.sentence for r in br.propose(orders)]
        assert any("order_date" in s and "delivery_date" in s for s in sentences)
        assert any("unit_price" in s and "total_amount" in s for s in sentences)
        assert any("age" in s for s in sentences)

    def test_it_leaves_legitimately_negative_columns_alone(self, orders):
        """`balance`, `profit` and `change` are the reason the non-negative list is a list of
        known-safe words rather than a guess at what sounds quantity-like."""
        assert all(r.column != "balance" for r in br.propose(orders))

    def test_clean_data_produces_no_proposals(self):
        """A list padded with rules that find nothing is how people learn to approve without
        reading, which would make the approval step worthless."""
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {"age": rng.integers(18, 80, 200), "quantity": rng.integers(1, 9, 200)}
        )
        assert br.propose(df) == []

    def test_numeric_columns_are_never_read_as_dates(self):
        """`pd.to_datetime` reads a float as nanoseconds since the epoch and succeeds on every
        row, which produced the proposal "'unit_price' must not be later than 'age'"."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {"unit_price": rng.random(200) * 50, "age": rng.integers(18, 80, 200)}
        )
        assert all(r.kind != "comparison" for r in br.propose(df))

    def test_one_column_does_not_get_two_overlapping_rules(self):
        """A 0-100 bound already says everything a bare "never below 0" would. Two rules on
        one column is two approvals for one decision."""
        values = pd.Series([10.0, 150.0, -20.0] * 40)
        df = pd.DataFrame({"discount_percent": values})
        for_column = [r for r in br.propose(df) if r.column == "discount_percent"]
        assert len(for_column) == 1
        assert for_column[0].maximum == 100

    def test_a_fraction_scale_percentage_still_gets_its_floor(self):
        """Bounding a 0-1 column at 100 would be true and useless -- but the negatives in it
        are still impossible."""
        rng = np.random.default_rng(3)
        values = pd.Series(rng.random(200).round(3))
        values.iloc[:2] = [-0.2, -0.05]
        proposals = [r for r in br.propose(pd.DataFrame({"discount_percent": values}))]
        assert len(proposals) == 1
        assert proposals[0].minimum == 0 and proposals[0].maximum is None

    def test_every_proposal_carries_its_evidence(self, orders):
        """"We think age can't be negative" is an opinion. "age holds 3 negative values" is a
        fact the user can check in a second."""
        for rule in br.propose(orders):
            assert rule.rationale
            assert rule.confidence is not None

    def test_unrelated_date_columns_are_not_given_an_ordering(self):
        """Two unrelated dates cross each other about half the time, and proposing an order
        between them would be noise."""
        rng = np.random.default_rng(5)
        base = pd.to_datetime("2024-01-01")
        df = pd.DataFrame(
            {
                "seen_at": base + pd.to_timedelta(rng.integers(0, 300, 200), unit="D"),
                "surveyed_at": base + pd.to_timedelta(rng.integers(0, 300, 200), unit="D"),
            }
        )
        assert all(r.kind != "comparison" for r in br.propose(df))


class TestTheShippedSample:
    """`retail_orders.csv` exists so the feature can be seen at all.

    Every other sample in the repo is clean of semantic errors, so the proposal engine
    correctly offers nothing on them -- which meant a first-time user could run the whole app
    and never learn this existed. A feature nobody can reach is not delivered, and a demo
    dataset that quietly stops demonstrating is worse than none.
    """

    @pytest.fixture
    def orders(self) -> pd.DataFrame:
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "samples" / "retail_orders.csv"
        assert path.exists(), f"the shipped sample is missing: {path}"
        return pd.read_csv(path)

    def test_it_demonstrates_every_kind_of_rule(self, orders):
        kinds = {rule.kind for rule in br.propose(orders)}
        assert kinds == {"range", "comparison", "arithmetic"}

    def test_each_proposed_rule_actually_finds_something(self, orders):
        for check in br.check_all(orders, br.propose(orders)):
            assert check.status == "failed", check.finding

    def test_the_violations_stay_a_believable_minority(self, orders):
        """Every one of these is under 10% of the file. A demo where a third of the rows break
        a rule teaches the wrong lesson -- that rules are noisy."""
        for rule in br.propose(orders):
            violation = br.evaluate(orders, rule)
            assert violation is not None
            assert 0 < violation.share < 0.10
