"""Saying when the folds cannot tell the chosen model from its closest rival.

Even with the one-standard-error rule the page showed one winner, and a ranking reads as meaningful
whether or not the evidence separates the models. The folds are the same for every candidate, so
their scores pair up: how much the best beat the other on each fold, and whether the range of that
difference includes zero, is the question. The sentence appears exactly when it does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as selection

from app.services import modeling, tuning, uncertainty
from app.services.model_card import build_model_card

LOGISTIC, FOREST = "logistic_regression", "random_forest_classifier"


class TestTheFoldByFoldDifference:
    def test_a_lead_on_every_fold_by_a_steady_amount_excludes_zero(self):
        found = uncertainty.fold_difference([0.82, 0.83, 0.81, 0.84, 0.82], [0.80, 0.81, 0.79, 0.82, 0.80])
        assert found["difference"] == pytest.approx(0.02)
        assert found["low"] > 0

    def test_a_lead_that_flips_from_fold_to_fold_includes_zero(self):
        found = uncertainty.fold_difference([0.80, 0.70, 0.90, 0.60, 0.85], [0.81, 0.69, 0.88, 0.62, 0.84])
        assert found["low"] < 0 < found["high"]

    def test_identical_scores_differ_by_nothing_and_include_zero(self):
        found = uncertainty.fold_difference([0.7, 0.8, 0.75], [0.7, 0.8, 0.75])
        assert found == {"difference": 0.0, "low": 0.0, "high": 0.0}

    def test_swapping_the_models_negates_the_interval(self):
        a, b = [0.82, 0.83, 0.81, 0.84, 0.82], [0.80, 0.82, 0.79, 0.83, 0.80]
        forward, backward = uncertainty.fold_difference(a, b), uncertainty.fold_difference(b, a)
        assert backward["difference"] == pytest.approx(-forward["difference"])
        assert backward["low"] == pytest.approx(-forward["high"])

    def test_it_is_wider_than_the_plain_paired_t_interval(self):
        """The folds share training rows, so the plain interval would call differences real
        far too often (Nadeau and Bengio, 2003)."""
        from scipy import stats

        a, b = np.array([0.82, 0.80, 0.83, 0.79, 0.84]), np.array([0.80, 0.81, 0.80, 0.78, 0.80])
        d = a - b
        plain_half = stats.t.ppf(0.975, 4) * d.std(ddof=1) / np.sqrt(5)
        found = uncertainty.fold_difference(a, b)
        assert (found["high"] - found["low"]) / 2 > plain_half

    def test_three_folds_give_a_very_wide_interval_which_is_the_honest_answer(self):
        two_five = uncertainty.fold_difference([0.82, 0.83, 0.81, 0.84, 0.82], [0.80, 0.81, 0.78, 0.82, 0.80])
        three = uncertainty.fold_difference([0.82, 0.83, 0.81], [0.80, 0.81, 0.78])
        assert (three["high"] - three["low"]) > (two_five["high"] - two_five["low"])

    @pytest.mark.parametrize(
        "first,second",
        [([0.8, 0.9], [0.8]), ([0.8], [0.7]), ([], []), ([0.8, float("nan"), 0.7], [0.7, 0.7, 0.7]), (None, [0.8, 0.7])],
        ids=["different lengths", "one fold", "none", "a nan fold", "missing"],
    )
    def test_scores_that_do_not_line_up_give_no_answer(self, first, second):
        assert uncertainty.fold_difference(first, second) is None


class TestEveryCandidateIsScoredOnTheSameFolds:
    """The pairing is only valid if fold i means the same rows for every model."""

    @staticmethod
    def _splits(splitter, X, y, groups=None):
        return [(train.tolist(), test.tolist()) for train, test in splitter.split(X, y, groups)]

    def test_two_calls_make_identical_splits_for_each_kind_of_split(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.normal(size=120)})
        classes = pd.Series(np.where(rng.random(120) < 0.4, "yes", "no"))
        numbers = pd.Series(rng.normal(size=120))
        groups = pd.Series(np.repeat(np.arange(30), 4))
        cases = {
            "stratified": (("classification", classes, 5, None, False), classes, None),
            "plain": (("regression", numbers, 5, None, False), numbers, None),
            "grouped": (("classification", classes, 5, groups, False), classes, groups),
            "time": (("regression", numbers, 5, None, True), numbers, None),
        }
        for label, (args, y, g) in cases.items():
            first, second = tuning.cv_splitter(*args), tuning.cv_splitter(*args)
            assert self._splits(first, X, y, g) == self._splits(second, X, y, g), label

    def test_selection_hands_every_candidate_the_same_folds(self, monkeypatch):
        seen: list[list] = []
        real = selection.cross_val_score

        def spy(estimator, X, y=None, *, cv=None, **kwargs):
            seen.append(self._splits(cv, X, y, kwargs.get("groups")))
            return real(estimator, X, y, cv=cv, **kwargs)

        monkeypatch.setattr(selection, "cross_val_score", spy)
        modeling.train_candidates(
            two_class(), target_column="y", task="classification",
            model_names=[LOGISTIC, FOREST], include_deep_diagnostics=False,
        )
        assert len(seen) == 2 and seen[0] == seen[1]

    def test_score_config_returns_the_score_of_each_fold(self):
        df = two_class()
        found = tuning.score_config(
            lambda: modeling.build_pipeline(LOGISTIC, ["x", "z"], []),
            df[["x", "z"]], df["y"], "classification",
        )
        # 400 rows is a small file, so the five folds are repeated three times.
        assert len(found.fold_scores) == found.fits == 5 * found.repeats and found.repeats == 3
        assert np.mean(found.fold_scores) == pytest.approx(found.score, abs=1e-4)
        assert np.std(found.fold_scores) == pytest.approx(found.std, abs=1e-4)


def two_class(n: int = 400, signal: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.where(signal * x + rng.normal(0, 1, n) > 0, "yes", "no")}
    )


def folds(*values):
    return {"score": float(np.mean(values)), "std": float(np.std(values)), "folds": len(values), "fold_scores": list(values)}


class TestWhetherTheRivalCanBeToldApart:
    def test_a_rival_the_folds_cannot_separate_is_within_the_noise(self):
        scores = {
            LOGISTIC: folds(0.80, 0.70, 0.90, 0.60, 0.85),
            FOREST: folds(0.81, 0.69, 0.88, 0.62, 0.84),
        }
        found = modeling.runner_up_comparison(scores, chosen=FOREST, highest_scorer=FOREST)
        assert found["runner_up"] == LOGISTIC and found["distinguishable"] is False

    def test_a_rival_beaten_on_every_fold_is_distinguishable(self):
        scores = {
            LOGISTIC: folds(0.60, 0.61, 0.59, 0.60, 0.62),
            FOREST: folds(0.82, 0.83, 0.81, 0.84, 0.82),
        }
        found = modeling.runner_up_comparison(scores, chosen=FOREST, highest_scorer=FOREST)
        assert found["distinguishable"] is True
        assert found["paired_difference"]["low"] > 0

    def test_the_rival_is_the_second_highest_scorer_when_the_best_was_kept(self):
        scores = {
            LOGISTIC: folds(0.70, 0.71, 0.69, 0.70, 0.72),
            FOREST: folds(0.82, 0.83, 0.81, 0.84, 0.82),
            "gradient_boosting_classifier": folds(0.80, 0.81, 0.79, 0.82, 0.80),
        }
        found = modeling.runner_up_comparison(scores, chosen=FOREST, highest_scorer=FOREST)
        assert found["runner_up"] == "gradient_boosting_classifier"

    def test_the_rival_is_the_highest_scorer_when_the_simpler_model_was_kept(self):
        scores = {
            LOGISTIC: folds(0.80, 0.79, 0.81, 0.80, 0.80),
            FOREST: folds(0.81, 0.80, 0.82, 0.81, 0.81),
        }
        found = modeling.runner_up_comparison(scores, chosen=LOGISTIC, highest_scorer=FOREST)
        assert found["runner_up"] == FOREST
        assert (found["leader"], found["other"]) == (FOREST, LOGISTIC)

    def test_nothing_to_compare_with_is_none(self):
        assert modeling.runner_up_comparison({LOGISTIC: folds(0.8, 0.7, 0.9)}, LOGISTIC, LOGISTIC) is None

    def test_scores_from_a_different_number_of_folds_are_not_compared(self):
        scores = {LOGISTIC: folds(0.8, 0.7, 0.9), FOREST: folds(0.8, 0.7, 0.9, 0.85, 0.8)}
        assert modeling.runner_up_comparison(scores, FOREST, FOREST) is None


class TestTheSentence:
    def test_it_appears_exactly_when_the_interval_includes_zero(self):
        """A noise target: neither model has anything to find, so the folds cannot tell them apart."""
        rng = np.random.default_rng(11)
        df = pd.DataFrame({"x": rng.normal(size=300), "z": rng.normal(size=300), "y": rng.choice(["a", "b"], 300)})
        chosen = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=[LOGISTIC, FOREST]
        )["selection"]
        gap = chosen["paired_difference"]
        assert chosen["distinguishable"] is (not (gap["low"] <= 0 <= gap["high"]))
        assert (chosen["distinguishable_note"] is not None) is (not chosen["distinguishable"])
        assert chosen["distinguishable"] is False
        assert "within the noise of each other" in chosen["distinguishable_note"]

    def test_it_is_absent_when_one_model_clearly_wins(self):
        """An interaction a linear model cannot represent but a forest can: not a close call."""
        rng = np.random.default_rng(12)
        a, b = rng.normal(size=600), rng.normal(size=600)
        df = pd.DataFrame({"a": a, "b": b, "y": np.where(a * b > 0, "yes", "no")})
        found = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=[LOGISTIC, FOREST]
        )["selection"]
        assert found["chosen"] == FOREST
        assert found["distinguishable"] is True
        assert found["distinguishable_note"] is None and found["paired_difference"]["low"] > 0

    def test_a_kept_simpler_model_is_said_to_be_kept_for_being_simpler(self, monkeypatch):
        """Build the selection with controlled fold scores: the forest leads a little, the folds
        disagree, so the simpler model is kept and the sentence says that is why."""
        monkeypatch.setattr(tuning, "score_config", _controlled({
            LOGISTIC: (0.80, 0.70, 0.90, 0.60, 0.85),
            FOREST: (0.81, 0.69, 0.92, 0.62, 0.86),
        }))
        found = _selection()
        assert found["rule"] == "one_standard_error" and found["chosen"] == LOGISTIC
        assert found["distinguishable"] is False
        note = found["distinguishable_note"]
        assert note.startswith("Random forest and logistic regression are within the noise of each other")
        assert "Logistic regression was kept because it is simpler" in note
        assert "95% range" in note

    def test_a_best_model_kept_only_for_scoring_higher_does_not_claim_more(self, monkeypatch):
        monkeypatch.setattr(tuning, "score_config", _controlled({
            LOGISTIC: (0.70, 0.75, 0.72, 0.80, 0.68),
            FOREST: (0.80, 0.74, 0.80, 0.79, 0.82),
        }))
        found = _selection()
        assert found["chosen"] == FOREST and found["rule"] == "highest_score"
        assert found["distinguishable"] is False
        assert "cannot show is more than chance" in found["distinguishable_note"]

    def test_a_simpler_model_kept_against_a_real_lead_says_the_lead_is_not_noise(self, monkeypatch):
        """The forest leads by 0.01 on every fold, yet the folds themselves swing by 0.15, so
        the one-standard-error rule keeps the simpler model. The paired comparison sees the
        steady lead, and the note must not claim it is indistinguishable."""
        monkeypatch.setattr(tuning, "score_config", _controlled({
            LOGISTIC: (0.80, 0.70, 0.90, 0.60, 0.85),
            FOREST: (0.81, 0.71, 0.91, 0.61, 0.86),
        }))
        found = _selection()
        assert found["chosen"] == LOGISTIC and found["rule"] == "one_standard_error"
        assert found["distinguishable"] is True and found["distinguishable_note"] is None
        assert "not noise" in found["rule_note"] and "ahead by +0.010" in found["rule_note"]
        assert "not distinguishable" not in found["rule_note"]

    def test_a_single_model_has_nothing_to_say(self):
        found = modeling.train_candidates(
            two_class(), target_column="y", task="classification", model_names=[LOGISTIC]
        )["selection"]
        assert found.get("distinguishable") is None and found.get("distinguishable_note") is None

    def test_the_fold_scores_are_not_stored_in_the_result(self):
        found = modeling.train_candidates(
            two_class(), target_column="y", task="classification", model_names=[LOGISTIC, FOREST]
        )["selection"]
        assert "scores" not in found


def _controlled(by_model: dict[str, tuple]):
    """A `score_config` that returns fixed fold scores for each model, so the decision is
    tested on numbers chosen for the case rather than on whatever a random forest does."""
    def fake(build, X, y, task, **kwargs):
        name = _model_of(build)
        values = by_model[name]
        return tuning.SearchResult(
            params={}, score=round(float(np.mean(values)), 4), std=round(float(np.std(values)), 4),
            metric="roc_auc", fits=len(values), fold_scores=tuple(values),
        )

    return fake


def _model_of(build) -> str:
    return LOGISTIC if "Logistic" in type(build().named_steps["model"]).__name__ else FOREST


def _selection() -> dict:
    return modeling.train_candidates(
        two_class(), target_column="y", task="classification", model_names=[LOGISTIC, FOREST]
    )["selection"]


class TestTheCardCarriesIt:
    BASE = dict(
        session_id="s", file_name="f.csv", summary="A model.", interpretation="It works.",
    )

    def _card(self, **selection):
        result = modeling.train_candidates(
            two_class(), target_column="y", task="classification", model_names=[LOGISTIC, FOREST]
        )
        result["selection"] = {**result["selection"], **selection}
        return build_model_card(result=result, **self.BASE)

    def test_the_sentence_is_in_the_card_under_the_comparison_table(self):
        card = self._card(distinguishable_note="Random forest and logistic regression are within the noise of each other.")
        assert "are within the noise of each other." in card
        table_end = card.index("| Random forest")
        assert card.index("within the noise of each other") > table_end
        assert card.index("within the noise of each other") < card.index("### Held-out evaluation")

    def test_the_card_says_nothing_when_the_models_can_be_told_apart(self):
        assert "within the noise of each other" not in self._card(distinguishable_note=None)
