"""Derived inputs.

Two properties carry this file. `test_the_script_lines_produce_the_same_columns` pins
`apply` and `as_pandas` to each other -- they are two renderings of one recipe, and a model
the export cannot rebuild is a model trapped in the app. `TestNothingIsAppliedByItself`
pins the approval shape the whole project is built on: a proposal does nothing until its id
comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import features, modeling


@pytest.fixture
def messy() -> pd.DataFrame:
    """The shapes each recipe exists for, in one frame."""
    rng = np.random.default_rng(11)
    n = 300
    words = ["the", "customer", "called", "about", "a", "billing", "issue", "again", "today"]
    return pd.DataFrame(
        {
            "signal": rng.normal(0, 1, n),
            # 20% blank, which is worth a flag.
            "spend": np.where(rng.random(n) < 0.2, np.nan, rng.normal(50, 10, n)),
            "signup_date": pd.date_range("2019-01-01", periods=n, freq="3D").astype(str),
            "notes": [
                " ".join(rng.choice(words, size=int(rng.integers(6, 20)))) for _ in range(n)
            ],
            # 60 values, but the top ones cover most rows.
            "city": rng.choice(
                [f"city{i}" for i in range(60)],
                n,
                p=np.r_[np.full(10, 0.06), np.full(50, 0.4 / 50)],
            ),
            "order_id": [f"O{i:05d}" for i in range(n)],
            "region": rng.choice(["north", "south"], n),
            "outcome": rng.choice(["yes", "no"], n),
        }
    )


class TestWhatItProposes:
    def test_a_date_column_becomes_its_parts(self, messy):
        recipe = next(r for r in features.propose(messy, "outcome") if r.column == "signup_date")
        assert recipe.kind == "date_parts"
        assert recipe.outputs == [
            "signup_date__year",
            "signup_date__month",
            "signup_date__weekday",
            "signup_date__days",
        ]
        assert "897 days" in recipe.reason
        assert recipe.evidence["span_days"] == 897

    def test_only_the_parts_that_vary_are_offered(self):
        """A file covering three weeks has no year to learn from, and offering one would be
        padding the list -- the same reason business rules are filtered to rules that would
        find something."""
        df = pd.DataFrame(
            {"when": pd.date_range("2024-03-01", periods=21, freq="D").astype(str), "y": range(21)}
        )
        recipe = next(r for r in features.propose(df, "y") if r.kind == "date_parts")
        assert "when__year" not in recipe.outputs
        assert "when__month" not in recipe.outputs
        assert "when__weekday" in recipe.outputs and "when__days" in recipe.outputs

    def test_free_text_becomes_its_length(self, messy):
        recipe = next(r for r in features.propose(messy, "outcome") if r.column == "notes")
        assert recipe.kind == "text_length"
        assert recipe.outputs == ["notes__length", "notes__words"]

    def test_an_identifier_is_not_free_text(self, messy):
        """Distinct-per-row is what the identifier rule tests. A paragraph of prose is
        distinct too -- the difference is that it is long, and the RSNA run dropped a column
        of radiology reports on exactly this confusion."""
        assert not any(r.column == "order_id" for r in features.propose(messy, "outcome"))
        assert features.looks_like_free_text(messy["notes"])
        assert not features.looks_like_free_text(messy["order_id"])

    def test_a_short_repeated_category_is_not_free_text(self, messy):
        assert not features.looks_like_free_text(messy["region"])

    def test_a_blank_heavy_column_gets_a_flag(self, messy):
        recipe = next(r for r in features.propose(messy, "outcome") if r.kind == "missing_flag")
        assert recipe.column == "spend"
        assert recipe.outputs == ["spend__missing"]
        assert 0.15 < recipe.evidence["missing_share"] < 0.25

    def test_a_column_with_no_gaps_gets_no_flag(self, messy):
        flagged = {r.column for r in features.propose(messy, "outcome") if r.kind == "missing_flag"}
        assert "signal" not in flagged
        assert "region" not in flagged

    def test_a_high_cardinality_column_is_kept_rather_than_dropped(self, messy):
        recipe = next(
            r for r in features.propose(messy, "outcome") if r.kind == "frequent_categories"
        )
        assert recipe.column == "city"
        assert recipe.outputs == []  # encoded in the pipeline; no new column in the frame
        assert recipe.evidence["coverage"] > features.FREQUENT_MIN_COVERAGE

    def test_a_column_whose_values_never_repeat_is_not_worth_pooling(self, messy):
        """Pooling an order number leaves every row in 'other', which is a constant."""
        kinds = {
            r.column for r in features.propose(messy, "outcome") if r.kind == "frequent_categories"
        }
        assert "order_id" not in kinds

    def test_the_target_is_never_derived_from(self, messy):
        assert not any(r.column == "outcome" for r in features.propose(messy, "outcome"))

    def test_an_excluded_column_is_never_derived_from(self, messy):
        """A missing flag on a column withheld for giving away the answer would leak it
        straight back in through the back door."""
        proposed = features.propose(messy, "outcome", excluded=["spend", "signup_date"])
        assert not any(r.column in {"spend", "signup_date"} for r in proposed)

    def test_clean_ordinary_data_proposes_nothing(self):
        df = pd.DataFrame(
            {"a": range(100), "b": [f"k{i % 4}" for i in range(100)], "y": [i % 2 for i in range(100)]}
        )
        assert features.propose(df, "y") == []

    def test_every_proposal_carries_its_evidence(self, messy):
        for recipe in features.propose(messy, "outcome"):
            assert recipe.reason and recipe.column in recipe.reason
            assert recipe.kind in features.KINDS
            assert recipe.id.endswith(recipe.column)


def _proposals(df: pd.DataFrame, target: str = "outcome", excluded=None) -> list[dict]:
    return [r.as_dict() for r in features.propose(df, target, excluded)]


def _target_variants(df: pd.DataFrame, target: str):
    """The same frame with the target's *values* replaced by things that share nothing with
    the real answers: shuffles, a constant, the reverse, labels from another vocabulary, a
    numeric regression-style target, and a column of blanks."""
    n = len(df)
    for seed in (0, 1, 2, 3, 4):
        yield f"permutation {seed}", df.assign(
            **{target: np.random.default_rng(seed).permutation(df[target].to_numpy())}
        )
    yield "constant", df.assign(**{target: "yes"})
    yield "reversed", df.assign(**{target: df[target].to_numpy()[::-1]})
    yield "other labels", df.assign(**{target: np.random.default_rng(9).choice(list("abc"), n)})
    yield "numeric", df.assign(**{target: np.random.default_rng(10).normal(0, 1, n)})
    yield "all blank", df.assign(**{target: np.nan})


def assert_independent_of_the_target(propose, df: pd.DataFrame, target: str = "outcome") -> None:
    """Fails if `propose` returns anything different when only the target's values change."""
    reference = [r.as_dict() for r in propose(df, target)]
    for label, variant in _target_variants(df, target):
        assert [r.as_dict() for r in propose(variant, target)] == reference, (
            f"the proposals changed when the target was replaced by: {label}"
        )


class TestProposalsNeverReadTheTarget:
    """Derived inputs are offered from the whole file, test rows included. That is only
    harmless while no proposal depends on the answers. It is true today -- offering is decided
    by dates, text length, how much is blank and how often categories repeat -- and nothing
    enforced it, so a future recipe that looked at the target (a target-correlated binning,
    say) would leak into the proposals with every test still green."""

    def test_replacing_the_targets_values_changes_no_proposal(self, messy):
        assert_independent_of_the_target(features.propose, messy)

    def test_the_proposals_exist_to_be_compared(self, messy):
        """A property over an empty list holds for any implementation."""
        assert {r["kind"] for r in _proposals(messy)} == set(features.KINDS)

    def test_the_selection_the_server_makes_is_independent_of_it_too(self, messy):
        ids = [r["id"] for r in _proposals(messy)]
        reference = [r.as_dict() for r in features.select(messy, "outcome", [], ids)]
        for label, variant in _target_variants(messy, "outcome"):
            assert [r.as_dict() for r in features.select(variant, "outcome", [], ids)] == reference, label

    def test_the_check_would_notice_a_proposal_that_read_the_target(self, messy):
        """The test above is only worth having if it can fail. This stands in for the future
        change it guards against: a proposer that offers an extra input when the answers lean
        one way."""

        def peeking_propose(df, target, excluded=None):
            proposals = features.propose(df, target, excluded)
            if (df[target].astype(str) == "yes").mean() > 0.5:
                proposals.append(
                    features.Recipe(
                        id="peek:signal", kind="missing_flag", column="signal",
                        outputs=["signal_peek"], reason="signal, because the answers lean yes",
                    )
                )
            return proposals

        # A frame whose answers lean one way, so that the canary actually fires.
        leaning = messy.assign(outcome=np.where(np.arange(len(messy)) % 5 == 0, "no", "yes"))
        with pytest.raises(AssertionError, match="the proposals changed when the target"):
            assert_independent_of_the_target(peeking_propose, leaning)


class TestNothingIsAppliedByItself:
    def test_a_proposal_alone_changes_nothing(self, messy):
        before = messy.copy()
        features.propose(messy, "outcome")
        pd.testing.assert_frame_equal(messy, before)

    def test_only_the_ids_sent_back_are_applied(self, messy):
        chosen = features.select(messy, "outcome", None, ["missing_flag:spend"])
        assert [r.id for r in chosen] == ["missing_flag:spend"]
        out = features.apply(messy, chosen)
        assert "spend__missing" in out.columns
        assert "signup_date__year" not in out.columns

    def test_an_id_for_a_recipe_the_data_no_longer_supports_is_dropped_quietly(self, messy):
        """The data can change under a saved plan -- a cleaning fix can fill every blank.
        That is a reason to drop the recipe, not to fail the run."""
        messy["spend"] = messy["spend"].fillna(0)
        assert features.select(messy, "outcome", None, ["missing_flag:spend"]) == []

    def test_a_recipe_on_an_excluded_column_cannot_be_smuggled_in(self, messy):
        assert features.select(messy, "outcome", ["spend"], ["missing_flag:spend"]) == []

    def test_applying_nothing_returns_the_frame_untouched(self, messy):
        assert features.apply(messy, []) is messy

    def test_the_source_frame_is_never_modified(self, messy):
        before = messy.copy()
        features.apply(messy, features.propose(messy, "outcome"))
        pd.testing.assert_frame_equal(messy, before)


class TestApplying:
    def test_date_parts_are_the_real_parts(self):
        df = pd.DataFrame({"when": ["2021-03-04", "2020-12-25"], "y": [0, 1]})
        recipe = features.Recipe(
            id="date_parts:when",
            kind="date_parts",
            column="when",
            outputs=["when__year", "when__month", "when__weekday", "when__days"],
            reason="",
        )
        out = features.apply(df, [recipe])
        assert list(out["when__year"]) == [2021.0, 2020.0]
        assert list(out["when__month"]) == [3.0, 12.0]
        assert list(out["when__weekday"]) == [3.0, 4.0]  # a Thursday and a Friday
        assert out["when__days"].iloc[0] == (pd.Timestamp("2021-03-04") - pd.Timestamp("1970-01-01")).days

    def test_text_length_counts_characters_and_words(self):
        df = pd.DataFrame({"note": ["two words", None], "y": [0, 1]})
        recipe = features.Recipe(
            id="text_length:note", kind="text_length", column="note",
            outputs=["note__length", "note__words"], reason="",
        )
        out = features.apply(df, [recipe])
        assert list(out["note__length"]) == [9.0, 0.0]
        assert list(out["note__words"]) == [2.0, 0.0]

    def test_a_missing_flag_is_one_where_the_value_was_blank(self):
        df = pd.DataFrame({"x": [1.0, None, 3.0], "y": [0, 1, 0]})
        recipe = features.Recipe(
            id="missing_flag:x", kind="missing_flag", column="x",
            outputs=["x__missing"], reason="",
        )
        assert list(features.apply(df, [recipe])["x__missing"]) == [0.0, 1.0, 0.0]

    def test_the_script_lines_produce_the_same_columns(self, messy):
        """`apply` and `as_pandas` are two renderings of the same recipe. If they drift, the
        exported script silently builds different inputs from the ones that were scored, and
        every number in the app becomes unreproducible."""
        recipes = [r for r in features.propose(messy, "outcome") if r.outputs]
        applied = features.apply(messy, recipes)

        scripted = messy.copy()
        exec(features.as_pandas(recipes, frame="scripted"), {"pd": pd}, {"scripted": scripted})

        derived = [o for r in recipes for o in r.outputs]
        assert derived  # or the test proves nothing
        pd.testing.assert_frame_equal(applied[derived], scripted[derived])


class TestTheReasonIsTheProduct:
    def test_a_date_column_is_not_called_an_identifier(self, messy):
        """It used to be dropped for having "too many values", which is true of every date
        column ever written and tells the reader nothing about what to do."""
        dropped = {d["column"]: d["reason"] for d in modeling.classify_features(messy, "outcome")["dropped"]}
        assert "Dates" in dropped["signup_date"]
        assert "derived inputs" in dropped["signup_date"]

    def test_free_text_is_not_called_an_identifier(self, messy):
        """The RSNA run dropped a column of radiology reports because "nearly every row has
        a different value, so this identifies rows rather than describing them". For a
        paragraph of clinical prose that is precisely backwards."""
        dropped = {d["column"]: d["reason"] for d in modeling.classify_features(messy, "outcome")["dropped"]}
        assert "Free text" in dropped["notes"]
        assert "identif" not in dropped["notes"].lower()

    def test_a_pooled_column_says_that_keeping_it_is_on_offer(self, messy):
        dropped = {d["column"]: d["reason"] for d in modeling.classify_features(messy, "outcome")["dropped"]}
        assert "most common values" in dropped["city"]

    def test_an_actual_identifier_still_reads_like_one(self, messy):
        dropped = {d["column"]: d["reason"] for d in modeling.classify_features(messy, "outcome")["dropped"]}
        assert "an ID or a name" in dropped["order_id"]

    def test_a_replaced_column_says_what_replaced_it(self, messy):
        recipes = features.select(messy, "outcome", None, ["date_parts:signup_date"])
        classified = modeling.classify_features(
            features.apply(messy, recipes), "outcome", None, recipes
        )
        dropped = {d["column"]: d["reason"] for d in classified["dropped"]}
        assert "Used through its derived inputs" in dropped["signup_date"]
        assert "signup_date__weekday" in dropped["signup_date"]


class TestTrainingWithThem:
    def test_derived_columns_become_model_inputs(self, messy):
        ids = [r.id for r in features.propose(messy, "outcome")]
        result = modeling.train_candidates(
            messy, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            feature_recipes_wanted=ids,
        )
        numeric = result["features"]["numeric"]
        assert "signup_date__days" in numeric
        assert "notes__length" in numeric
        assert "spend__missing" in numeric
        assert result["features"]["frequent"] == ["city"]
        assert {r["id"] for r in result["features"]["derived"]} == set(ids)

    def test_without_them_nothing_changes(self, messy):
        result = modeling.train_candidates(
            messy, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        assert result["features"]["derived"] == []
        assert result["features"]["frequent"] == []
        assert not any("__" in c for c in result["features"]["numeric"])

    def test_a_derived_input_can_carry_the_signal(self):
        """The point of the whole exercise: a target that depends only on the weekday of a
        date column is unlearnable until the date is split into parts."""
        n = 600
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {
                "noise": np.random.default_rng(2).normal(0, 1, n),
                "when": dates.astype(str),
                "outcome": np.where(dates.weekday >= 5, "yes", "no"),
            }
        )
        without = modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
        )
        with_parts = modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
            feature_recipes_wanted=["date_parts:when"],
        )
        assert without["assessment"]["verdict"] == "no_better_than_guessing"
        assert with_parts["results"][0]["metrics"]["roc_auc"] > 0.95
        assert with_parts["assessment"]["verdict"] == "useful"
