"""The model card.

Structure and every number are assembled in code; only two paragraphs of interpretation come
from the LLM. These tests cover the code-assembled part, which is the part that has to be
right even when the LLM call fails.
"""

from __future__ import annotations

import json

import pytest

from app.services.model_card import build_model_card

BASE_RESULT = {
    "task": "classification",
    "target_column": "attrition",
    "best_model": "logistic_regression",
    "primary_metric": "roc_auc",
    "baseline": {"strategy": "always answer 'No'", "accuracy": 0.68},
    "assessment": {
        "verdict": "useful",
        "headline": "This model separates the two outcomes meaningfully (AUC 0.82).",
    },
    "split": {"strategy": "random", "time_column": None, "train_rows": 640, "test_rows": 160},
    "features": {
        "numeric": ["tenure", "salary"],
        "categorical": ["department"],
        "dropped": [{"column": "employee_id", "reason": "Nearly every row is different."}],
        "excluded": [],
    },
    "results": [
        {
            "model": "logistic_regression",
            "label": "Logistic regression",
            "metrics": {
                "accuracy": 0.79,
                "precision": 0.66,
                "recall": 0.71,
                "f1": 0.68,
                "roc_auc": 0.82,
            },
            "column_importance": [
                {"column": "tenure", "importance": 0.52},
                {"column": "salary", "importance": 0.31},
            ],
            "diagnostics": {
                "per_class": [
                    {"label": "No", "precision": 0.85, "recall": 0.83, "f1": 0.84, "support": 109},
                    {"label": "Yes", "precision": 0.66, "recall": 0.71, "f1": 0.68, "support": 51},
                ],
                "confusion_cells": [
                    {"actual": "No", "predicted": "No", "count": 90, "share_of_actual": 0.83, "correct": True},
                    {"actual": "No", "predicted": "Yes", "count": 19, "share_of_actual": 0.17, "correct": False},
                    {"actual": "Yes", "predicted": "No", "count": 15, "share_of_actual": 0.29, "correct": False},
                    {"actual": "Yes", "predicted": "Yes", "count": 36, "share_of_actual": 0.71, "correct": True},
                ],
            },
            "failed": None,
        },
        {
            "model": "random_forest_classifier",
            "label": "Random forest",
            "metrics": {"accuracy": 0.75, "precision": 0.60, "recall": 0.63, "f1": 0.61, "roc_auc": 0.77},
            "column_importance": [],
            "diagnostics": {},
            "failed": None,
        },
    ],
    "deep_diagnostics": {
        "cross_validation": {"metric": "roc_auc", "folds": [0.81, 0.82, 0.80, 0.83, 0.82],
                             "mean": 0.816, "std": 0.01, "spread": 0.03},
        "learning_curve": [
            {"training_rows": 128, "training_score": 0.81, "validation_score": 0.78},
            {"training_rows": 640, "training_score": 0.80, "validation_score": 0.79},
        ],
    },
}


def _card(**overrides) -> str:
    result = {**BASE_RESULT, **overrides}
    return build_model_card(
        session_id="abc-123", file_name="employees.csv", result=result,
        summary="A model was built.", interpretation="It works reasonably.",
    )


class TestStructure:
    def test_carries_identifying_metadata(self):
        card = _card()
        assert "abc-123" in card and "employees.csv" in card and "attrition" in card

    def test_lists_every_candidate_not_only_the_winner(self):
        """Which models were tried and rejected is part of the record -- otherwise a reader
        can't tell whether the chosen one won or was the only one attempted."""
        card = _card()
        assert "Logistic regression" in card and "Random forest" in card
        assert "(chosen)" in card

    def test_includes_the_do_nothing_baseline_in_the_comparison(self):
        card = _card()
        assert "No model at all" in card
        assert "always answer 'No'" in card

    def test_a_failed_model_is_shown_as_failed(self):
        result = {**BASE_RESULT}
        result["results"] = BASE_RESULT["results"] + [
            {"model": "gradient_boosting_classifier", "label": "Gradient boosting",
             "metrics": {}, "column_importance": [], "diagnostics": {},
             "failed": "ran out of memory"}
        ]
        assert "did not train" in _card(**result)

    def test_explains_what_the_metrics_mean(self):
        card = _card()
        assert "0.50 is a coin flip" in card  # the ROC AUC gloss

    def test_confusion_matrix_is_rendered_as_a_table(self):
        card = _card()
        assert "Rows are what actually happened" in card
        assert "83%" in card  # share_of_actual, not just the raw count

    def test_columns_left_out_carry_their_reason(self):
        card = _card()
        assert "employee_id" in card
        assert "Nearly every row is different." in card


class TestVerdictIsNeverSoftened:
    def test_a_working_model_says_so(self):
        assert "This model works" in _card()

    def test_a_useless_model_says_so_in_the_heading(self):
        """The verdict is settled before the card is written. If the model learned nothing,
        the document has to say that where a reader will see it."""
        card = _card(
            assessment={
                "verdict": "no_better_than_guessing",
                "headline": "This model is within the range of random guessing (AUC 0.54).",
            }
        )
        assert "This model does not work" in card
        assert "random guessing" in card

    def test_a_marginal_model_is_not_dressed_up(self):
        card = _card(assessment={"verdict": "marginal", "headline": "Weak signal."})
        assert "This model is weak" in card


class TestCautions:
    def test_a_small_test_set_is_flagged_without_the_llm(self):
        """Deterministic cautions appear whether or not the prose call succeeded."""
        card = build_model_card(
            session_id="s", file_name="f.csv",
            result={**BASE_RESULT, "split": {**BASE_RESULT["split"], "test_rows": 40}},
        )
        assert "40 held-out rows" in card
        assert "small sample" in card

    def test_a_thinly_supported_class_is_flagged(self):
        result = {**BASE_RESULT}
        result["results"] = [
            {**BASE_RESULT["results"][0],
             "diagnostics": {"per_class": [
                 {"label": "No", "precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 155},
                 {"label": "Yes", "precision": 0.4, "recall": 0.4, "f1": 0.4, "support": 5},
             ]}},
            BASE_RESULT["results"][1],
        ]
        card = build_model_card(session_id="s", file_name="f.csv", result=result)
        assert "Only 5 test rows" in card
        assert "anecdote" in card

    def test_a_random_split_prompts_the_time_question(self):
        assert "date-based split" in _card()

    def test_withheld_columns_are_named(self):
        result = {**BASE_RESULT}
        result["features"] = {**BASE_RESULT["features"], "excluded": ["exit_date"]}
        assert "exit_date" in build_model_card(session_id="s", file_name="f.csv", result=result)

    def test_llm_cautions_are_appended_to_the_derived_ones(self):
        card = build_model_card(
            session_id="s", file_name="f.csv", result=BASE_RESULT,
            cautions=["Salary was recorded inconsistently before 2021."],
        )
        assert "Salary was recorded inconsistently before 2021." in card
        assert "date-based split" in card  # the derived one is still there


class TestTrustSection:
    def test_a_narrow_cross_validation_spread_is_called_stable(self):
        card = _card()
        assert "narrow range" in card

    def test_a_wide_spread_warns_the_headline_is_approximate(self):
        card = _card(
            deep_diagnostics={
                **BASE_RESULT["deep_diagnostics"],
                "cross_validation": {"metric": "roc_auc", "folds": [0.5, 0.75, 0.6, 0.55, 0.7],
                                     "mean": 0.62, "std": 0.09, "spread": 0.25},
            }
        )
        assert "wide range" in card
        assert "approximate" in card

    def test_a_large_train_test_gap_is_read_as_memorizing(self):
        card = _card(
            deep_diagnostics={
                "cross_validation": None,
                "learning_curve": [
                    {"training_rows": 128, "training_score": 0.99, "validation_score": 0.70},
                    {"training_rows": 640, "training_score": 0.99, "validation_score": 0.72},
                ],
            }
        )
        assert "memorizing" in card

    def test_a_flat_curve_says_more_rows_will_not_help(self):
        card = _card()
        assert "flattened out" in card

    def test_a_still_climbing_curve_says_more_rows_would_help(self):
        card = _card(
            deep_diagnostics={
                "cross_validation": None,
                "learning_curve": [
                    {"training_rows": 128, "training_score": 0.72, "validation_score": 0.61},
                    {"training_rows": 640, "training_score": 0.80, "validation_score": 0.78},
                ],
            }
        )
        assert "still improving" in card

    def test_skipped_deep_diagnostics_are_explained_not_omitted(self):
        card = _card(deep_diagnostics={"skipped": "Skipped: 500,000 rows would take too long."})
        assert "500,000 rows" in card


class TestRegressionCard:
    def test_uses_regression_metrics_and_error_ranges(self):
        result = {
            **BASE_RESULT,
            "task": "regression",
            "target_column": "price",
            "best_model": "linear_regression",
            "baseline": {"strategy": "always predict the average (250,000)", "r2": 0.0},
            "results": [
                {
                    "model": "linear_regression",
                    "label": "Ridge regression",
                    "metrics": {"r2": 0.84, "mae": 18400.0, "rmse": 24100.0},
                    "column_importance": [{"column": "size", "importance": 0.9}],
                    "diagnostics": {
                        "error_by_range": [
                            {"range": "50,000 to 150,000", "mean_absolute_error": 9000.0,
                             "mean_error": -1200.0, "count": 40},
                            {"range": "600,000 to 900,000", "mean_absolute_error": 61000.0,
                             "mean_error": 44000.0, "count": 40},
                        ]
                    },
                    "failed": None,
                }
            ],
        }
        card = build_model_card(session_id="s", file_name="homes.csv", result=result)
        assert "R²" in card
        assert "Where it is most wrong" in card
        assert "600,000 to 900,000" in card
        assert "over" in card  # the direction of the bias, not just its size


def test_a_card_is_still_produced_without_any_llm_prose():
    """Every number and warning is assembled in code, so a failed prose call degrades the
    card rather than losing it."""
    card = build_model_card(session_id="s", file_name="f.csv", result=BASE_RESULT)
    assert "No summary was generated" in card
    assert "0.82" in card  # the metrics survived
    assert "## Before you rely on this" in card


class TestTheCutoffSection:
    POINT = {
        "objective": "cost",
        "description": "a missed case costs 5x a false alarm",
        "positive_label": "yes",
        "threshold": 0.31,
        "chosen_on": "out-of-fold predictions over the 640 training rows (5 folds); the test set was not consulted",
        "folds": 5,
        "reachable": True,
        "note": None,
        "expected": {"precision": 0.55, "recall": 0.91, "flagged": 200, "rows": 640, "missed": 12, "false_alarms": 90},
        "observed": {"precision": 0.52, "recall": 0.88, "flagged": 52, "rows": 160, "missed": 4, "false_alarms": 25},
        "at_default": {"precision": 0.70, "recall": 0.60, "flagged": 30, "rows": 160, "missed": 14, "false_alarms": 9},
    }

    def test_the_default_writes_nothing(self):
        assert "Where the cut-off sits" not in _card(operating_point=None)
        assert "Where the cut-off sits" not in _card(operating_point={**self.POINT, "objective": "default"})

    def test_a_chosen_line_is_explained_with_both_columns(self):
        card = _card(operating_point=self.POINT)
        assert "## Where the cut-off sits" in card
        assert "**0.31**" in card
        assert "a missed case costs 5x a false alarm" in card
        assert "test set was not consulted" in card
        # expected, observed and the default, on one row
        assert "| Share of real cases caught (recall) | 0.91 | 0.88 | 0.6 |" in card

    def test_an_unreachable_floor_is_a_caution(self):
        card = _card(operating_point={**self.POINT, "reachable": False, "note": "No cut-off catches 99%."})
        assert "No cut-off catches 99%." in card
        assert "no cut-off achieves that" in card


class TestTheProseKnowsAboutDerivedInputs:
    """The prose is explicitly asked to comment on which columns were used and which were
    not. Anything the payload omits is a column the writer cannot see."""

    RESULT = {
        "target_column": "churn",
        "task": "classification",
        "best_model": "m",
        "results": [{"model": "m", "label": "Logistic regression", "metrics": {"roc_auc": 0.8}}],
        "features": {
            "numeric": ["signal", "signup_date__days"],
            "categorical": [],
            "frequent": ["city"],
            "dropped": [{"column": "signup_date", "reason": "Used through its derived inputs."}],
            "derived": [
                {
                    "id": "date_parts:signup_date",
                    "kind": "date_parts",
                    "column": "signup_date",
                    "outputs": ["signup_date__days"],
                    "reason": "Dates spanning 897 days.",
                    "evidence": {},
                },
                {
                    "id": "frequent_categories:city",
                    "kind": "frequent_categories",
                    "column": "city",
                    "outputs": [],
                    "reason": "77 values, the top 20 cover 65% of rows.",
                    "evidence": {},
                },
            ],
        },
    }

    def _payload(self) -> dict:
        from app.services.prompts.model_prompts import build_model_card_messages

        human = build_model_card_messages(self.RESULT, "data.csv", None)[1].content
        return json.loads(human[human.index("{") : human.rindex("}") + 1])

    def test_a_pooled_column_is_not_invisible(self):
        """It appears in neither `numeric` nor `categorical`, so before this it was in
        neither features_used nor columns_left_out -- a column the model used that the
        writer was never told about."""
        assert "city" in self._payload()["features_used"]

    def test_the_derived_inputs_say_where_they_came_from(self):
        derived = {d["from_column"]: d for d in self._payload()["derived_inputs"]}
        assert derived["signup_date"]["produced"] == ["signup_date__days"]
        assert "897 days" in derived["signup_date"]["why"]

    def test_a_recipe_that_adds_no_column_still_says_what_it_did(self):
        derived = {d["from_column"]: d for d in self._payload()["derived_inputs"]}
        assert derived["city"]["produced"] == "its most common values, the rest pooled"

    def test_the_writer_is_told_not_to_call_a_derived_input_a_column(self):
        from app.services.prompts.model_prompts import SYSTEM_PROMPT

        assert "derived_inputs" in SYSTEM_PROMPT
        assert "not columns of the file" in SYSTEM_PROMPT

    def test_nothing_derived_leaves_the_payload_empty_rather_than_absent(self):
        from app.services.prompts.model_prompts import build_model_card_messages

        bare = {**self.RESULT, "features": {"numeric": ["signal"], "categorical": [], "dropped": []}}
        human = build_model_card_messages(bare, "data.csv", None)[1].content
        payload = json.loads(human[human.index("{") : human.rindex("}") + 1])
        assert payload["derived_inputs"] == []
        assert payload["features_used"] == ["signal"]


SELECTION = {
    "chosen": "logistic_regression",
    "method": "cross_validation",
    "metric": "roc_auc",
    "folds": 5,
    "scores": {},
    "chosen_on": "cross-validated roc_auc over 5 folds of the training rows; the test set was not consulted",
}


def _with_fold_scores(selection=SELECTION) -> dict:
    """The forest scores higher on the test set but lower on the folds -- the case where the
    old table's 'chosen' marker looked like a mistake."""
    results = [dict(r) for r in BASE_RESULT["results"]]
    results[0].update(cv_score=0.8421, cv_std=0.0123)
    results[1].update(cv_score=0.7954, cv_std=0.0312)
    return {"results": results, "selection": selection}


class TestTheChoiceIsShownOnTheNumbersThatMadeIt:
    """The winner is chosen on training folds, so those lead; the held-out score is for the
    chosen model only. A table of every candidate's test score with one marked chosen invites
    choosing by it -- selecting on the test set by hand."""

    def test_the_fold_scores_and_spreads_are_shown(self):
        card = _card(**_with_fold_scores())
        assert "### How the model was chosen" in card
        assert "| Model | Fold score (ROC AUC) | Fold spread |" in card
        assert "0.8421" in card and "± 0.0123" in card
        assert "0.7954" in card and "± 0.0312" in card

    def test_the_chosen_row_says_how_it_was_chosen(self):
        card = _card(**_with_fold_scores())
        assert "(chosen; chosen on training folds)" in card

    def test_the_reason_is_stated_in_words(self):
        card = _card(**_with_fold_scores())
        assert "never on the held-out rows" in card
        assert "the test set was not consulted" in card

    def test_only_the_chosen_models_held_out_scores_appear(self):
        card = _card(**_with_fold_scores())
        held_out = card.split("### Held-out evaluation", 1)[1].split("\n## ", 1)[0]
        assert "Logistic regression" in held_out
        assert "Random forest" not in held_out
        assert "0.77" not in held_out  # the forest's held-out AUC is nowhere on the card's table

    def test_the_do_nothing_baseline_stays_beside_the_held_out_score(self):
        card = _card(**_with_fold_scores())
        held_out = card.split("### Held-out evaluation", 1)[1]
        assert "No model at all" in held_out

    def test_a_single_model_says_nothing_was_chosen(self):
        selection = {**SELECTION, "method": "only_candidate", "metric": None,
                     "chosen_on": "the only model trained"}
        card = _card(**{**_with_fold_scores(selection), "results": _with_fold_scores()["results"][:1]})
        assert "(chosen; the only model trained)" in card

    def test_a_choice_made_by_default_order_gives_its_reason(self):
        selection = {**SELECTION, "method": "default_order", "metric": None,
                     "chosen_on": "no candidate could be cross-validated, so the simplest one was kept"}
        results = _with_fold_scores(selection)
        for r in results["results"]:
            r.update(cv_score=None, cv_std=None)
        card = _card(**results)
        assert "so the simplest one was kept" in card
        assert "(chosen; kept by default order)" in card

    def test_a_failed_candidate_is_still_shown_as_failed(self):
        data = _with_fold_scores()
        data["results"] = data["results"] + [
            {"model": "gradient_boosting_classifier", "label": "Gradient boosting", "metrics": {},
             "column_importance": [], "diagnostics": {}, "failed": "ran out of memory"}
        ]
        assert "| Gradient boosting | did not train | — |" in _card(**data)

    def test_a_result_from_before_fold_selection_keeps_its_old_table(self):
        """Its choice was made on the test set, so its test-score table is the honest account."""
        card = _card()
        assert "### How the model was chosen" not in card
        assert "Random forest" in card and "0.77" in card


class TestTheOneStandardErrorRuleIsExplained:
    NOTE = (
        "Random forest scored highest (0.8421), but Logistic regression (0.8300) is within one "
        "standard error (0.0123) of it, so the simpler model was kept."
    )

    def _kept(self):
        data = _with_fold_scores({**SELECTION, "rule": "one_standard_error", "rule_note": self.NOTE})
        return _card(**data)

    def test_the_card_says_why_the_simpler_model_was_kept(self):
        assert self.NOTE in self._kept()

    def test_the_chosen_row_says_it_is_the_simplest_within_the_noise(self):
        assert "(chosen; simplest within one standard error of the best)" in self._kept()

    def test_a_rule_that_changed_nothing_adds_nothing(self):
        card = _card(**_with_fold_scores({**SELECTION, "rule": "highest_score", "rule_note": None}))
        assert "within one standard error" not in card
        assert "(chosen; chosen on training folds)" in card
