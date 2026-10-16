"""No "useful" verdict on fewer than 60 test rows (#119).

On 30 test rows the two-answer noise bar is about 0.73 with three candidates, above the 0.70 cut-off
for "useful", so a lucky run that cleared the bar got the strongest verdict the app gives; the #35
simulation saw that happen on pure noise. Between 30 and 59 test rows a result that would have been
useful now says the evidence is insufficient. Sixty rows is a safeguard, not a proof, and the
headline must not suggest otherwise.
"""

from __future__ import annotations

import pytest

from app.services import model_card, modeling

SMALL, ENOUGH = 40, modeling.MIN_TEST_ROWS_FOR_USEFUL


def binary(rows: int, auc: float = 0.85) -> dict:
    half = rows // 2
    return modeling.assess_result(
        {"accuracy": 0.8, "roc_auc": auc}, {"accuracy": 0.5}, "classification",
        test_rows=rows, class_counts=[half, rows - half], model_count=3,
    )


def regression(rows: int) -> dict:
    return modeling.assess_result(
        {"r2": 0.7, "mae": 1.0, "rmse": 1.2}, {"r2": 0.0}, "regression", test_rows=rows,
        intervals={"metrics": {"r2": {"low": 0.55, "high": 0.8}}},
    )


def several_answers(rows: int) -> dict:
    return modeling.assess_result(
        {"accuracy": 0.8, "roc_auc_ovr_macro": 0.86, "labels": ["a", "b", "c"]},
        {"accuracy": 0.34, "strategy": "always answer a"}, "classification",
        test_rows=rows, ovr_auc_floor=0.6,
    )


def accuracy_only(rows: int) -> dict:
    return modeling.assess_result(
        {"accuracy": 0.95}, {"accuracy": 0.5, "strategy": "always answer yes"}, "classification", test_rows=rows,
    )


KINDS = [binary, regression, several_answers, accuracy_only]


@pytest.mark.parametrize("assess", KINDS, ids=lambda f: f.__name__)
class TestEveryKindOfVerdict:
    def test_a_useful_result_on_a_small_test_set_is_held_back(self, assess):
        found = assess(SMALL)
        assert found["verdict"] == "insufficient_evidence"
        assert found["held_back_from"] == "useful"
        assert found["min_test_rows_for_useful"] == 60

    def test_sixty_test_rows_is_unchanged(self, assess):
        assert assess(ENOUGH)["verdict"] == "useful"

    def test_below_thirty_is_still_not_enough_test_data(self, assess):
        assert assess(20)["verdict"] == "not_enough_test_data"


class TestTheOtherVerdictsStand:
    def test_a_weak_result_on_a_small_test_set_stays_marginal(self):
        # 56 balanced rows and one candidate put the noise bar near 0.63, so 0.695 lands in the marginal band.
        found = modeling.assess_result(
            {"accuracy": 0.6, "roc_auc": 0.695}, {"accuracy": 0.5}, "classification",
            test_rows=56, class_counts=[28, 28], model_count=1,
        )
        assert found["verdict"] == "marginal"

    def test_noise_on_a_small_test_set_is_still_no_better_than_guessing(self):
        assert binary(SMALL, auc=0.55)["verdict"] == "no_better_than_guessing"


class TestTheWording:
    def test_it_names_the_score_the_rows_and_the_remedy(self):
        headline = binary(SMALL, auc=0.83)["headline"]
        assert "AUC 0.83" in headline
        assert "40 test rows" in headline
        assert "more labelled rows" in headline

    def test_it_does_not_present_sixty_rows_as_a_guarantee(self):
        headline = binary(SMALL)["headline"]
        assert "would not make it certain" in headline
        for overclaim in ("guarantee", "enough to be reliable", "statistically significant"):
            assert overclaim not in headline

    def test_each_kind_names_its_own_score(self):
        assert "R² 0.70" in regression(SMALL)["headline"]
        assert "macro AUC 0.86" in several_answers(SMALL)["headline"]
        assert "accuracy 95%" in accuracy_only(SMALL)["headline"]


def test_the_model_card_and_audit_report_have_a_heading_for_it():
    assert "insufficient_evidence" in model_card._VERDICT_HEADINGS
