"""Every "cannot judge" state shares one block and one sentence pattern.

The verdict, the drift check, the check against a new file's real answers and the cut-off each decline
to judge when evidence is thin. Each carries `{"status": "not_judged", what, have, need, sentence}`, and
the cut-off, which used to vanish silently, now says why it was not chosen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import drift, modeling, operating_point, scoring
from app.services.not_judged import not_judged


class TestTheBlock:
    def test_a_count_gives_the_pattern_sentence(self):
        block = not_judged("held-out rows", have=12, need=30)
        assert block["status"] == "not_judged"
        assert block["sentence"] == "Not enough held-out rows to judge: 12 rows, at least 30 needed."
        assert (block["have"], block["need"]) == (12, 30)

    def test_thousands_are_separated(self):
        assert "1,200 rows, at least 5,000 needed" in not_judged("x", have=1200, need=5000)["sentence"]

    def test_a_unit_other_than_rows(self):
        assert "3 groups, at least 5 needed" in not_judged("customers", have=3, need=5, unit="groups")["sentence"]

    def test_without_counts_it_gives_the_reason(self):
        block = not_judged(because="this model gives no probabilities.")
        assert block["sentence"] == "Cannot judge: this model gives no probabilities."
        assert block["have"] is None and block["need"] is None


class TestTheVerdict:
    def test_too_few_test_rows(self):
        found = modeling._too_small_to_judge(12, None)
        assert found["verdict"] == "not_enough_test_data"
        assert found["not_judged"]["sentence"] == (
            f"Not enough test rows to judge: 12 rows, at least {modeling.MIN_TEST_ROWS_TO_JUDGE} needed."
        )

    def test_too_few_of_the_rarer_answer(self):
        found = modeling._too_small_to_judge(200, [196, 4])
        assert found["not_judged"]["sentence"] == (
            "Not enough test rows with the rarer answer to judge: 4 rows, "
            f"at least {modeling.MIN_TEST_ROWS_PER_CLASS_TO_JUDGE} needed."
        )

    def test_enough_data_is_not_a_not_judged_state(self):
        assert modeling._too_small_to_judge(500, [250, 250]) is None

    def test_the_block_survives_the_reuse_qualification(self):
        from app.services import evaluation_ledger

        assessment = modeling._too_small_to_judge(12, None)
        qualified = evaluation_ledger.qualify(
            assessment, {"status": "reused", "qualification": "Used 4 times."}
        )
        assert qualified["not_judged"] == assessment["not_judged"]


class TestDrift:
    def test_too_few_new_rows(self):
        reference = pd.DataFrame({"a": np.arange(100.0)})
        report = drift.compare(reference, reference.head(10), ["a"], [])
        assert report["verdict"] == "not_enough_rows"
        assert report["not_judged"]["sentence"] == (
            f"Not enough new rows to judge: 10 rows, at least {drift.MIN_ROWS_TO_JUDGE} needed."
        )

    def test_enough_rows_has_no_such_block(self):
        reference = pd.DataFrame({"a": np.arange(100.0)})
        assert "not_judged" not in drift.compare(reference, reference, ["a"], [])


class TestNewAnswers:
    def test_a_file_with_too_few_labelled_rows(self):
        X = pd.DataFrame({"a": np.arange(25.0)})
        y = pd.Series(["yes"] * 12 + ["no"] * 13)
        found = scoring.check_against_answers(object(), X, y, {"task": "classification", "results": []})
        assert found["now"] is None
        assert found["not_judged"]["sentence"] == (
            f"Not enough labelled rows in this file to judge: 25 rows, at least {modeling.MIN_TEST_ROWS_TO_JUDGE} needed."
        )

    def test_a_file_with_too_few_of_the_rarer_answer(self):
        X = pd.DataFrame({"a": np.arange(200.0)})
        y = pd.Series(["yes"] * 4 + ["no"] * 196)
        found = scoring.check_against_answers(object(), X, y, {"task": "classification", "results": []})
        assert found["not_judged"]["sentence"].startswith("Not enough labelled rows in this file with the rarer answer to judge: 4 rows")


class TestTheCutOff:
    class Probabilistic:
        def predict_proba(self, X):  # pragma: no cover -- only its presence matters
            raise NotImplementedError

    def test_a_model_with_no_probabilities(self):
        block = operating_point.not_judged_reason(pipeline=object(), y_train=pd.Series(["a", "b"] * 20))
        assert block["sentence"] == (
            "Cannot judge: this model gives no probabilities, so there is nothing to put a cut-off on."
        )

    def test_a_rarer_answer_too_thin_to_make_two_folds(self):
        y = pd.Series(["no"] * 40 + ["yes"])
        block = operating_point.not_judged_reason(pipeline=self.Probabilistic(), y_train=y)
        assert block["sentence"] == (
            "Not enough training rows with the rarer answer to judge: 1 rows, at least 2 needed."
        )

    def test_too_few_groups(self):
        y = pd.Series(["no", "yes"] * 20)
        block = operating_point.not_judged_reason(
            pipeline=self.Probabilistic(), y_train=y, groups=pd.Series([1, 1] * 20)
        )
        assert block["sentence"] == "Not enough distinct groups in the training rows to judge: 1 groups, at least 2 needed."

    def test_too_few_rows_to_fold_by_date(self):
        block = operating_point.not_judged_reason(
            pipeline=self.Probabilistic(), y_train=pd.Series(["no", "yes"]), ordered=True
        )
        assert block["have"] == 2 and block["need"] == 3

    def test_folds_that_exist_but_could_not_be_scored(self):
        y = pd.Series(["no", "yes"] * 20)
        block = operating_point.not_judged_reason(pipeline=self.Probabilistic(), y_train=y)
        assert block["sentence"].startswith("Cannot judge: the training folds could not be scored")


class TestTheResult:
    def frame(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=200)
        return pd.DataFrame({"x": x, "z": rng.normal(size=200), "y": np.where(x + rng.normal(size=200) > 0, "yes", "no")})

    def train(self, monkeypatch=None, objective="min_recall", rate=0.9, empty=False):
        if empty:
            monkeypatch.setattr(operating_point, "apply", lambda **kwargs: None)
        return modeling.train_candidates(
            self.frame(), target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
            point=operating_point.OperatingPoint(objective=objective, target_rate=rate),
        )

    def test_a_cut_off_that_could_not_be_chosen_says_why_instead_of_vanishing(self, monkeypatch):
        result = self.train(monkeypatch, empty=True)
        assert result["operating_point"] is None
        found = result["operating_point_not_judged"]
        assert found["status"] == "not_judged" and found["sentence"]
        assert found["objective"] == "min_recall" and "catch at least 90%" in found["description"]

    def test_a_cut_off_that_was_chosen_has_no_such_field_set(self):
        result = self.train()
        assert result["operating_point"]["threshold"] is not None
        assert result["operating_point_not_judged"] is None

    def test_no_cut_off_asked_for_is_nothing_to_explain(self, monkeypatch):
        result = self.train(monkeypatch, objective="default", empty=True)
        assert result["operating_point_not_judged"] is None
