"""The card's "How these numbers were obtained" section.

Assembled in code from the result and its evaluation record. It is in every card, its optional
parts appear only when they apply, and no figure in it comes from a language model.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from app.services import modeling
from app.services.model_card import build_model_card

HEADING = "## How these numbers were obtained"


def result(**overrides) -> dict:
    base = {
        "task": "classification",
        "best_model": "logistic_regression",
        "primary_metric": "roc_auc",
        "features": {"numeric": ["a"], "categorical": [], "frequent": [], "excluded": [], "derived": []},
        "split": {"strategy": "random", "train_rows": 800, "test_rows": 200, "test_fraction": 0.2},
        "selection": {
            "chosen": "logistic_regression", "method": "cross_validation", "metric": "roc_auc",
            "folds": 5, "repeats": 1, "distinguishable": True, "chosen_on": "cross-validated roc_auc",
        },
        "results": [
            {"model": "logistic_regression", "label": "Logistic regression", "metrics": {"roc_auc": 0.8},
             "cv_score": 0.79, "cv_std": 0.02, "failed": None, "diagnostics": {}},
        ],
        "evaluation": {"look_count": 1, "status": "independent", "message": "", "limit": 3},
        "calibration": {"method": "none"},
    }
    base.update(overrides)
    return copy.deepcopy(base)


def section(res: dict) -> str:
    card = build_model_card(session_id="s", file_name="f.csv", result=res, summary="A.", interpretation="B.")
    assert HEADING in card
    body = card.split(HEADING, 1)[1]
    return body.split("\n## ", 1)[0]


class TestAlwaysThere:
    def test_the_section_is_in_every_card_with_the_split_and_the_choice(self):
        text = section(result())
        assert "800 rows to train on and 200 held back for testing (20%), at random" in text
        assert "By cross-validated ROC AUC over 5 folds of the training rows; the held-out rows took no part." in text
        assert "What the numbers support." in text and "What they do not support." in text

    def test_it_comes_before_the_results(self):
        card = build_model_card(session_id="s", file_name="f.csv", result=result(), summary="A.", interpretation="B.")
        assert card.index(HEADING) < card.index("## Results")

    def test_a_bare_result_from_long_ago_still_gets_the_section(self):
        old = {"task": "regression", "best_model": "m", "results": [{"model": "m", "label": "M", "metrics": {}}]}
        text = section(old)
        assert "Not recorded: this result predates the record" in text
        assert "picked by its held-out score" in text


class TestTheSplit:
    def test_a_date_split_names_its_column_and_gap(self):
        res = result()
        res["split"].update(strategy="time", time_column="when", gap_days=7, gap_rows_dropped=3)
        text = section(res)
        assert "by date, on when, leaving a gap of 7 days" in text
        assert "in a later period" not in text  # a date split does test that

    def test_a_recommendation_that_was_followed_is_said(self):
        res = result()
        res["split"].update(recommended={"strategy": "random", "column": None, "confidence": "high"}, overridden=False)
        assert "This is the split the app recommended." in section(res)

    def test_a_recommendation_that_was_overridden_is_said(self):
        res = result()
        res["split"].update(recommended={"strategy": "time", "column": "when", "confidence": "high"}, overridden=True)
        assert "recommended splitting by date on when; this run split differently" in section(res)

    def test_a_random_split_cannot_speak_for_a_later_period(self):
        assert "how the model will do in a later period" in section(result()).lower()


class TestHowTheModelWasChosen:
    def test_one_model_means_no_choice(self):
        res = result()
        res["selection"] = {"chosen": "logistic_regression", "method": "only_candidate", "metric": None, "folds": 0}
        assert "Only one model was trained, so none was chosen between." in section(res)

    def test_default_order_quotes_the_reason(self):
        res = result()
        res["selection"] = {"method": "default_order", "chosen_on": "no candidate could be cross-validated"}
        assert "No candidate could be cross-validated." in section(res)

    def test_repeated_folds_are_said(self):
        res = result()
        res["selection"]["repeats"] = 3
        assert "repeated 3 times" in section(res)

    def test_a_rival_the_folds_could_not_separate_is_said(self):
        res = result()
        res["selection"]["distinguishable"] = False
        assert "could not separate the chosen model from its closest rival" in section(res)

    def test_seed_dependence_is_said(self):
        res = result()
        res["selection"]["seed_stability"] = {"applicable": True, "seed_dependent": True, "note": "Depends on the seed."}
        assert "depends on the random seed" in section(res)

    def test_an_improved_run_says_how_many_rounds(self):
        res = result(improvement={"rounds": [{}, {}, {}]})
        assert "improved by a search of 3 rounds, each judged on the training folds" in section(res)


class TestUseOfTheHeldOutRows:
    def test_the_first_look_is_independent(self):
        assert "first time these rows were used to evaluate a model, so the score is an independent estimate" in section(result())

    def test_a_reused_holdout_quotes_the_ledgers_message_and_limits_the_claim(self):
        res = result()
        res["evaluation"] = {"look_count": 4, "status": "reused", "limit": 6,
                             "message": "These held-out rows have now been used to evaluate a model 4 times in this session."}
        text = section(res)
        assert "used to evaluate a model 4 times in this session" in text
        assert "an independent estimate: the held-out rows had been used 4 times" in text.lower()

    def test_a_sealed_holdout_not_yet_opened(self):
        res = result()
        res["split"].update(sealed=True, final_rows=150, final_fraction=0.15, sealed_after_looks=0)
        assert "A further 150 rows (15%) were sealed away; they have not been opened" in section(res)

    def test_a_sealed_holdout_that_was_opened_gives_its_score(self):
        res = result()
        res["split"].update(sealed=True, final_rows=150, final_fraction=0.15, sealed_after_looks=2)
        res["evaluation"]["final"] = {"metrics": {"roc_auc": 0.77}, "primary_metric": "roc_auc"}
        text = section(res)
        assert "after 2 earlier look(s) at the data" in text
        assert "opened once, scoring 0.77 on ROC AUC" in text

    def test_a_sealed_holdout_that_could_not_be_made(self):
        res = result()
        res["split"]["sealed_unavailable_reason"] = "too few rows of the rarer answer"
        assert "Asked for and not made: too few rows of the rarer answer" in section(res)

    def test_new_data_checks_are_counted(self):
        res = result()
        res["evaluation"]["new_data"] = {
            "count": 2,
            "latest": {"rows": 120, "metric": "roc_auc", "value": 0.74, "reported": 0.8},
        }
        assert "Checked on new data.** 2 time(s)" in section(res)
        assert "120 rows that did not exist when the model was built, ROC AUC 0.74 (0.8 when it was built)" in section(res)

    def test_nothing_is_said_about_new_data_when_there_is_none(self):
        assert "Checked on new data" not in section(result())


class TestProbabilitiesAndCutOff:
    def test_calibration_that_was_applied(self):
        res = result(calibration={"method": "isotonic", "fitted_on": "cross-validation over 5 folds of the training rows"})
        assert "Calibrated with isotonic, cross-validation over 5 folds" in section(res)

    def test_calibration_that_was_refused(self):
        res = result(calibration={"method": "none", "refused_because": "too few rows"})
        assert "Calibration was asked for and not applied: too few rows" in section(res)

    def test_no_calibration_is_said_for_a_classifier(self):
        assert "As the model produced them, not calibrated." in section(result())

    def test_a_regression_says_nothing_about_probabilities(self):
        assert "Probabilities" not in section(result(task="regression"))

    def test_a_chosen_cut_off_was_not_chosen_on_the_test_set(self):
        res = result(operating_point={"objective": "recall", "threshold": 0.31, "chosen_on": "out-of-fold scores"})
        assert "Cut-off.** 0.31, chosen on out-of-fold scores; the held-out rows had no say in it." in section(res)

    def test_the_default_cut_off_is_not_mentioned(self):
        assert "Cut-off" not in section(result(operating_point={"objective": "default", "threshold": 0.5}))


class TestOverlaps:
    def test_each_overlap_is_listed(self):
        res = result()
        res["split"].update(keep_duplicates_together=True, test_rows_seen_in_training=7)
        res["evaluation"]["whole_file_statistics"] = [{"operation": "fill", "column": "a"}]
        text = section(res)
        assert "Exact copies of a row were kept on one side of the split" in text
        assert "7 test rows are exact copies of training rows, answer included" in text
        assert "1 approved fix(es) worked a value out over every row" in text

    def test_no_overlaps_no_line(self):
        assert "Overlaps" not in section(result())


class TestWhatTheNumbersSupport:
    def test_the_plausible_range_is_quoted(self):
        res = result()
        res["results"][0]["metrics_intervals"] = {"level": 0.95, "resamples": 1000, "metrics": {"roc_auc": {"low": 0.71, "high": 0.86}}}
        assert "could plausibly land between 0.71 and 0.86" in section(res)


class TestNoLanguageModelIsInvolved:
    def test_the_section_is_the_same_whatever_prose_is_slotted_in(self):
        res = result()
        a = build_model_card(session_id="s", file_name="f.csv", result=res, summary="One.", interpretation="Two.")
        b = build_model_card(session_id="s", file_name="f.csv", result=res, summary="Totally different.",
                             interpretation="Other words.", cautions=["Something else."])
        assert a.split(HEADING)[1].split("\n## ")[0] == b.split(HEADING)[1].split("\n## ")[0]


class TestAgainstARealRun:
    def test_a_trained_result_gets_a_complete_section(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=300)
        df = pd.DataFrame({"x": x, "z": rng.normal(size=300), "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")})
        res = modeling.train_candidates(
            df, target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        text = section(res)
        assert "rows to train on" in text and "By cross-validated" in text and "not calibrated" in text
