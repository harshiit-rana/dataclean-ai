"""The "Why this model?" explanation, built in code from the result's own fields.

Every statement has to be backed by fields the result holds: the evidence on it names them, and the
tests check that each one resolves to the value shown. No statement may come from a language model.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from app.services import modeling, why_this_model
from app.services.model_card import build_model_card
from app.services.why_this_model import explain, resolve


def result(**overrides) -> dict:
    """A cross-validated run of logistic regression against a forest, the forest ahead clearly."""
    base = {
        "task": "classification",
        "best_model": "random_forest_classifier",
        "primary_metric": "roc_auc",
        "features": {"numeric": ["a", "b"], "categorical": [], "frequent": [], "excluded": [], "derived": []},
        "split": {"strategy": "random", "train_rows": 800, "test_rows": 200},
        "selection": {
            "chosen": "random_forest_classifier",
            "method": "cross_validation",
            "metric": "roc_auc",
            "folds": 5,
            "repeats": 1,
            "rule": "highest_score",
            "highest_scorer": "random_forest_classifier",
            "standard_error": 0.01,
            "rule_note": None,
            "runner_up": "logistic_regression",
            "distinguishable": True,
            "paired_difference": {"difference": 0.06, "low": 0.03, "high": 0.09},
            "distinguishable_note": None,
            "chosen_on": "cross-validated roc_auc over 5 folds of the training rows; the test set was not consulted",
        },
        "results": [
            {"model": "logistic_regression", "label": "Logistic regression", "cv_score": 0.74, "cv_std": 0.02, "failed": None},
            {"model": "random_forest_classifier", "label": "Random forest", "cv_score": 0.80, "cv_std": 0.02, "failed": None},
        ],
    }
    base.update(overrides)
    return copy.deepcopy(base)


def by_id(explanation: dict, id_: str) -> dict:
    return next(s for s in explanation["statements"] if s["id"] == id_)


def assert_every_statement_is_backed(res: dict) -> dict:
    explanation = explain(res)
    assert explanation["statements"]
    for s in explanation["statements"]:
        assert s["text"].strip() and s["section"] in why_this_model.SECTIONS
        assert s["evidence"], f"{s['id']} has no evidence"
        for e in s["evidence"]:
            assert resolve(res, e["field"]) == e["value"], (s["id"], e)
    return explanation


class TestHowItWasChosen:
    def test_a_cross_validated_choice_names_the_metric_folds_and_that_the_test_set_took_no_part(self):
        s = by_id(assert_every_statement_is_backed(result()), "how_chosen_method")
        assert "5 folds" in s["text"] and "area under the curve" in s["text"]
        assert "held-out rows took no part" in s["text"]

    def test_repeated_folds_are_said(self):
        res = result()
        res["selection"]["repeats"] = 3
        assert "repeated 3 times" in by_id(explain(res), "how_chosen_method")["text"]

    def test_the_highest_scorer_is_said_to_be_the_highest(self):
        s = by_id(explain(result()), "how_chosen_rule")
        assert "highest fold score (0.800)" in s["text"]

    def test_the_one_standard_error_rule_quotes_the_code_written_note(self):
        res = result(best_model="logistic_regression")
        res["selection"].update(
            chosen="logistic_regression", rule="one_standard_error",
            rule_note="Random forest scored highest (0.800), but Logistic regression (0.795) is within one standard error.",
            runner_up="random_forest_classifier", distinguishable=False,
            distinguishable_note="Random forest and logistic regression are within the noise of each other.",
            paired_difference={"difference": 0.005, "low": -0.02, "high": 0.03},
        )
        res["results"][0]["cv_score"] = 0.795
        explanation = assert_every_statement_is_backed(res)
        assert by_id(explanation, "how_chosen_rule")["text"] == res["selection"]["rule_note"]

    def test_a_single_candidate_has_no_choice_to_explain(self):
        res = result()
        res["selection"] = {"chosen": "random_forest_classifier", "method": "only_candidate", "metric": None,
                            "folds": 0, "chosen_on": "the only model trained"}
        res["results"] = res["results"][1:]
        explanation = assert_every_statement_is_backed(res)
        assert "no choice between models" in by_id(explanation, "how_chosen_method")["text"]
        assert "alternatives" not in {s["section"] for s in explanation["statements"]}

    def test_the_default_order_fallback_is_a_warning_that_leads(self):
        res = result(best_model="logistic_regression")
        res["selection"] = {"chosen": "logistic_regression", "method": "default_order", "metric": None, "folds": 0,
                            "chosen_on": "no candidate could be cross-validated on the training rows, so the simplest one was kept"}
        explanation = assert_every_statement_is_backed(res)
        s = by_id(explanation, "how_chosen_method")
        assert s["tone"] == "warning" and explanation["lead_id"] == "how_chosen_method"
        assert s["text"].startswith("No candidate could be cross-validated")

    def test_a_result_from_before_folds_chose_the_model_says_the_test_set_did(self):
        res = result()
        del res["selection"]
        explanation = assert_every_statement_is_backed(res)
        s = by_id(explanation, "how_chosen_legacy")
        assert s["tone"] == "warning" and "held-out score" in s["text"]


class TestWhyTheOthersWereNot:
    def test_a_clear_lead_over_the_runner_up_is_quoted_with_its_range(self):
        s = by_id(assert_every_statement_is_backed(result()), "runner_up")
        assert s["tone"] == "plain"
        assert "+0.060" in s["text"] and "+0.030 to +0.090" in s["text"] and "more than noise" in s["text"]

    def test_a_lead_inside_the_noise_is_a_prominent_warning(self):
        res = result()
        res["selection"].update(
            distinguishable=False,
            paired_difference={"difference": 0.004, "low": -0.02, "high": 0.03},
            distinguishable_note="Random forest and logistic regression are within the noise of each other on these folds.",
        )
        explanation = assert_every_statement_is_backed(res)
        s = by_id(explanation, "runner_up")
        assert s["tone"] == "warning" and s["text"].startswith("Too close to call.")
        assert explanation["lead_id"] == "runner_up"

    def test_a_higher_scorer_kept_out_by_simplicity_says_so_without_overstating(self):
        res = result(best_model="logistic_regression")
        res["selection"].update(
            chosen="logistic_regression", rule="one_standard_error", highest_scorer="random_forest_classifier",
            rule_note="Random forest scored highest.", runner_up="random_forest_classifier",
            distinguishable=True, paired_difference={"difference": 0.03, "low": 0.01, "high": 0.05},
        )
        s = by_id(assert_every_statement_is_backed(res), "runner_up")
        assert "scored higher by +0.030" in s["text"] and "simpler" in s["text"]

    def test_a_third_model_behind_by_more_than_one_standard_error_is_said_to_be_behind(self):
        res = result()
        res["results"].append(
            {"model": "gradient_boosting_classifier", "label": "Gradient boosting", "cv_score": 0.70, "cv_std": 0.03, "failed": None}
        )
        s = by_id(assert_every_statement_is_backed(res), "alt_gradient_boosting_classifier")
        assert "further behind the best than one standard error" in s["text"]

    def test_a_third_model_within_one_standard_error_is_not_ruled_out(self):
        res = result()
        res["results"].append(
            {"model": "gradient_boosting_classifier", "label": "Gradient boosting", "cv_score": 0.795, "cv_std": 0.03, "failed": None}
        )
        s = by_id(assert_every_statement_is_backed(res), "alt_gradient_boosting_classifier")
        assert "cannot rule it out" in s["text"] and "because it is simpler" in s["text"]

    def test_a_model_that_did_not_train_says_why(self):
        res = result()
        res["results"].append(
            {"model": "gradient_boosting_classifier", "label": "Gradient boosting", "cv_score": None, "cv_std": None,
             "failed": "boom"}
        )
        s = by_id(assert_every_statement_is_backed(res), "alt_gradient_boosting_classifier")
        assert s["tone"] == "notable" and "could not be trained: boom" in s["text"]

    def test_nothing_is_said_about_alternatives_when_the_choice_was_not_made_on_folds(self):
        res = result()
        res["selection"] = {"chosen": "random_forest_classifier", "method": "only_candidate", "metric": None, "folds": 0}
        assert not [s for s in explain(res)["statements"] if s["section"] == "alternatives"]


class TestWhatConstrainedIt:
    def test_a_date_split_says_the_folds_were_ordered_too(self):
        res = result()
        res["split"] = {"strategy": "time", "time_column": "when", "gap_days": 7}
        s = by_id(assert_every_statement_is_backed(res), "split")
        assert "on when" in s["text"] and "validated on later" in s["text"]

    def test_a_grouped_split_says_entities_stayed_whole(self):
        res = result()
        res["split"] = {"strategy": "group", "group_column": "customer", "train_groups": 40, "test_groups": 10}
        s = by_id(assert_every_statement_is_backed(res), "split")
        assert "Every customer was kept whole" in s["text"] and "inside every fold" in s["text"]

    def test_a_split_that_differs_from_the_advice_is_flagged(self):
        res = result()
        res["split"] = {"strategy": "random", "recommended": {"strategy": "time", "column": "when", "confidence": "high"},
                        "overridden": True}
        s = by_id(assert_every_statement_is_backed(res), "split_override")
        assert "recommended splitting by date on when" in s["text"] and s["tone"] == "notable"

    def test_withheld_columns_are_named_and_the_models_were_compared_without_them(self):
        res = result()
        res["features"]["excluded"] = ["id", "leaky"]
        s = by_id(assert_every_statement_is_backed(res), "excluded")
        assert "id and leaky" in s["text"] and "without them" in s["text"]

    def test_a_cut_off_is_said_not_to_have_decided_the_model(self):
        res = result()
        res["operating_point"] = {"objective": "recall", "threshold": 0.31, "reachable": True,
                                  "description": "it catches 90% of real cases"}
        s = by_id(assert_every_statement_is_backed(res), "operating_point")
        assert "0.31" in s["text"] and "did not influence which model won" in s["text"]

    def test_the_default_cut_off_is_not_mentioned(self):
        res = result()
        res["operating_point"] = {"objective": "default", "threshold": 0.5}
        assert not [s for s in explain(res)["statements"] if s["id"] == "operating_point"]

    def test_an_unreachable_cut_off_is_flagged(self):
        res = result()
        res["operating_point"] = {"objective": "recall", "threshold": 0.1, "reachable": False, "description": "x"}
        s = by_id(explain(res), "operating_point")
        assert s["tone"] == "notable" and "nearest one was used" in s["text"]

    def test_calibration_is_said_not_to_have_changed_the_winner(self):
        res = result()
        res["calibration"] = {"method": "isotonic", "folds": 5}
        assert "did not change which model won" in by_id(assert_every_statement_is_backed(res), "calibration")["text"]


class TestTheImprovementSearch:
    def trail(self):
        return {
            "rounds": [
                {"name": "tune", "label": "Tuning", "what_it_tried": "searched the settings", "score": 0.81,
                 "gain": 0.01, "accepted": True, "outcome": "Kept: it gained 0.010."},
                {"name": "derived", "label": "Derived inputs", "what_it_tried": "added date parts", "score": 0.80,
                 "gain": -0.01, "accepted": False, "outcome": "Rejected: no real gain."},
            ],
            "goal": 0.95, "stopped_because": "goal_out_of_reach", "cv_score": 0.81, "cv_std": 0.02,
        }

    def test_every_round_is_listed_kept_or_rejected_with_its_numbers(self):
        explanation = assert_every_statement_is_backed(result(improvement=self.trail()))
        assert "2 rounds, 1 kept" in by_id(explanation, "improvement_summary")["text"]
        assert "-- kept" in by_id(explanation, "round_0")["text"]
        rejected = by_id(explanation, "round_1")["text"]
        assert "-- rejected" in rejected and "no real gain" in rejected

    def test_a_goal_that_was_not_reached_is_said(self):
        assert "was asked for and not reached" in by_id(explain(result(improvement=self.trail())), "improvement_goal")["text"]

    def test_a_plain_run_has_no_improvement_section(self):
        assert not [s for s in explain(result())["statements"] if s["section"] == "improvement"]


class TestResolve:
    def test_paths_models_positions_and_lengths(self):
        res = result(improvement={"rounds": [{"score": 0.5}, {"score": 0.7}]})
        assert resolve(res, "results[logistic_regression].cv_score") == 0.74
        assert resolve(res, "improvement.rounds[1].score") == 0.7
        assert resolve(res, "improvement.rounds.length") == 2
        assert resolve(res, "selection.nothing.here") is None
        assert resolve(res, "results[no_such_model].cv_score") is None


class TestAgainstARealRun:
    def frame(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=300)
        return pd.DataFrame({"x": x, "z": rng.normal(size=300), "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")})

    def test_a_trained_result_is_explained_and_every_statement_is_backed(self):
        res = modeling.train_candidates(
            self.frame(), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        explanation = assert_every_statement_is_backed(res)
        assert {"how_chosen", "alternatives", "constraints"} <= {s["section"] for s in explanation["statements"]}

    def test_the_card_carries_the_same_sentences(self):
        res = modeling.train_candidates(
            self.frame(), target_column="y", task="classification",
            model_names=["logistic_regression", "random_forest_classifier"], include_deep_diagnostics=False,
        )
        card = build_model_card(session_id="s", file_name="f.csv", result=res, summary="A.", interpretation="B.")
        assert "## Why this model" in card
        for s in explain(res)["statements"]:
            assert s["text"] in card

    def test_a_result_that_cannot_be_explained_still_builds_a_card(self, monkeypatch):
        def boom(_):
            raise RuntimeError("no")

        monkeypatch.setattr(why_this_model, "explain", boom)
        res = result()
        assert build_model_card(session_id="s", file_name="f.csv", result=res, summary="A.", interpretation="B.")


class TestServed:
    """The explanation is on the result a train returns and on the one read back, and they match."""

    def _session(self):
        import io

        from fastapi.testclient import TestClient

        from app.config import get_settings
        from app.main import app

        client = TestClient(app)
        buffer = io.StringIO()
        TestServed.frame(self).to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def frame(self):
        rng = np.random.default_rng(6)
        x = rng.normal(size=300)
        return pd.DataFrame({"x": x.round(4), "z": rng.normal(size=300).round(4),
                             "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")})

    def test_a_train_and_a_read_back_carry_the_same_explanation(self):
        client, sid = self._session()
        trained = client.post(
            f"/api/model/{sid}/train",
            json={"model_names": ["logistic_regression", "random_forest_classifier"], "include_deep_diagnostics": False},
        )
        assert trained.status_code == 200, trained.text
        read = client.get(f"/api/model/{sid}/result").json()
        assert trained.json()["why_this_model"]["statements"]
        assert read["why_this_model"] == trained.json()["why_this_model"]
        for s in read["why_this_model"]["statements"]:
            assert s["evidence"]
            for e in s["evidence"]:
                assert resolve(read, e["field"]) == e["value"]
