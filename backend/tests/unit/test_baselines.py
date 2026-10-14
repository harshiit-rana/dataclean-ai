"""What you would get with no model at all, on every metric the models report.

The only classification baseline was "always answer the most common answer", scored on accuracy.
Its precision, recall, F1 and AUC cells were blank, and on a lopsided target its F1 is 0 -- which
flatters every model, because anything beats zero. The baseline now has two rows, each complete:
the common-answer guesser (judged exactly as a model is, through scikit-learn's DummyClassifier) and
a guesser that answers at random in proportion to how common each answer is, whose expected scores
are computed rather than sampled so the row is the same every time.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score

from app.main import app
from app.services import modeling
from app.services.model_card import build_model_card

client = TestClient(app)
TWO_CLASS_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc", "brier_score", "expected_calibration_error"}
MULTI_CLASS_KEYS = {"accuracy", "precision", "recall", "f1"}


def labels(rate: float, n: int = 1000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.where(rng.random(n) < rate, "yes", "no"))


class TestAlwaysAnswerTheCommonAnswer:
    def test_when_the_common_answer_is_not_the_one_hunted_it_catches_nothing(self):
        train, test = labels(0.2, seed=1), labels(0.2, seed=2)
        base = modeling.baseline_score(train, test, "classification", "yes")
        assert base["strategy"] == "always answer 'no'"
        assert (base["precision"], base["recall"], base["f1"]) == (0.0, 0.0, 0.0)
        assert base["accuracy"] == pytest.approx(float((test == "no").mean()), abs=1e-4)

    def test_when_it_is_the_one_hunted_it_catches_everything_and_is_right_at_the_base_rate(self):
        train, test = labels(0.8, seed=3), labels(0.8, seed=4)
        base = modeling.baseline_score(train, test, "classification", "yes")
        q = float((test == "yes").mean())
        assert base["recall"] == 1.0
        assert base["precision"] == pytest.approx(q, abs=1e-4)
        assert base["f1"] == pytest.approx(2 * q / (1 + q), abs=1e-3)

    def test_it_is_scored_the_way_a_model_is(self):
        """Through `evaluate`, so a column means the same thing for it as for every model."""
        train, test = labels(0.3, seed=5), labels(0.3, seed=6)
        dummy = DummyClassifier(strategy="prior").fit(np.zeros((len(train), 1)), train)
        measured = modeling.evaluate(dummy, np.zeros((len(test), 1)), test, "classification", "yes")
        base = modeling.baseline_score(train, test, "classification", "yes")
        for key in TWO_CLASS_KEYS:
            assert base[key] == measured[key], key

    def test_its_auc_is_a_coin_flip_because_it_ranks_nothing(self):
        assert modeling.baseline_score(labels(0.3), labels(0.3, seed=1), "classification", "yes")["roc_auc"] == 0.5

    def test_its_brier_score_is_that_of_the_training_rate(self):
        train, test = labels(0.3, seed=7), labels(0.3, seed=8)
        rate = float((train == "yes").mean())
        expected = brier_score_loss((test == "yes").astype(int), np.full(len(test), rate))
        base = modeling.baseline_score(train, test, "classification", "yes")
        assert base["brier_score"] == pytest.approx(expected, abs=1e-4)

    def test_its_calibration_error_is_the_gap_between_the_rate_it_states_and_the_one_it_met(self):
        train, test = labels(0.3, seed=9), labels(0.35, seed=10)
        base = modeling.baseline_score(train, test, "classification", "yes")
        assert base["expected_calibration_error"] == pytest.approx(
            abs(float((train == "yes").mean()) - float((test == "yes").mean())), abs=1e-4
        )

    def test_the_verdicts_comparison_figure_is_unchanged(self):
        """`assess_result` reads this accuracy; it is still the share of the commonest answer."""
        train, test = labels(0.3, seed=11), labels(0.3, seed=12)
        common = train.value_counts().index[0]
        assert modeling.baseline_score(train, test, "classification", "yes")["accuracy"] == pytest.approx(
            float((test == common).mean()), abs=1e-4
        )


class TestGuessingAtRandomInProportion:
    def other(self, base):
        assert len(base["others"]) == 1
        return base["others"][0]

    def test_it_is_a_second_row_with_its_own_strategy(self):
        row = self.other(modeling.baseline_score(labels(0.3), labels(0.3, seed=1), "classification", "yes"))
        assert "in proportion to how common each answer is" in row["strategy"]

    def test_its_expected_scores_match_a_simulation_of_actually_guessing(self):
        """The row is computed, not sampled. A large sample of real stratified guesses lands on it."""
        train, test = labels(0.25, 4000, seed=1), labels(0.3, 4000, seed=2)
        row = self.other(modeling.baseline_score(train, test, "classification", "yes"))
        dummy = DummyClassifier(strategy="stratified", random_state=0).fit(np.zeros((len(train), 1)), train)
        draws = [pd.Series(dummy.set_params(random_state=i).predict(np.zeros((len(test), 1)))) for i in range(60)]
        mean = lambda f: float(np.mean([f(d) for d in draws]))  # noqa: E731
        truth = test
        assert row["accuracy"] == pytest.approx(mean(lambda d: (d == truth).mean()), abs=0.01)
        assert row["precision"] == pytest.approx(mean(lambda d: precision_score(truth, d, pos_label="yes")), abs=0.01)
        assert row["recall"] == pytest.approx(mean(lambda d: recall_score(truth, d, pos_label="yes")), abs=0.01)
        assert row["f1"] == pytest.approx(mean(lambda d: f1_score(truth, d, pos_label="yes")), abs=0.015)

    def test_its_brier_score_and_error_are_the_chance_of_being_wrong(self):
        train, test = labels(0.25, 4000, seed=3), labels(0.3, 4000, seed=4)
        row = self.other(modeling.baseline_score(train, test, "classification", "yes"))
        assert row["brier_score"] == pytest.approx(1 - row["accuracy"], abs=1e-3)
        assert row["expected_calibration_error"] == pytest.approx(row["brier_score"], abs=1e-3)
        assert row["roc_auc"] == 0.5

    def test_it_is_the_same_every_time(self):
        train, test = labels(0.3), labels(0.3, seed=1)
        assert modeling.baseline_score(train, test, "classification", "yes") == modeling.baseline_score(
            train, test, "classification", "yes"
        )

    def test_on_a_lopsided_target_it_is_the_floor_the_common_answer_guesser_is_not(self):
        base = modeling.baseline_score(labels(0.1, seed=5), labels(0.1, seed=6), "classification", "yes")
        assert base["f1"] == 0.0 and self.other(base)["f1"] > 0.05

    def test_more_than_two_answers_get_macro_scores(self):
        rng = np.random.default_rng(7)
        train = pd.Series(rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]))
        test = pd.Series(rng.choice(["a", "b", "c"], 3000, p=[0.5, 0.3, 0.2]))
        base = modeling.baseline_score(train, test, "classification")
        row = self.other(base)
        assert set(k for k in base if k not in {"strategy", "others"}) == MULTI_CLASS_KEYS
        assert set(k for k in row if k != "strategy") == MULTI_CLASS_KEYS
        assert row["accuracy"] == pytest.approx(0.5**2 + 0.3**2 + 0.2**2, abs=0.03)
        assert row["recall"] == pytest.approx(1 / 3, abs=0.02) and row["precision"] == pytest.approx(1 / 3, abs=0.02)


class TestFittedOnTheTrainingRowsOnly:
    def test_the_strategy_and_the_probabilities_it_states_come_from_the_training_answers(self):
        train, test = labels(0.2, seed=1), labels(0.8, seed=2)  # the test rows' common answer is the other one
        base = modeling.baseline_score(train, test, "classification", "yes")
        assert base["strategy"] == "always answer 'no'"
        # The probability it states is the training rate, so its Brier score moves with the training
        # rows and not with anything fitted to the test rows.
        flipped = modeling.baseline_score(labels(0.8, seed=1), test, "classification", "yes")
        assert flipped["strategy"] == "always answer 'yes'" and flipped["brier_score"] != base["brier_score"]

    def test_the_test_rows_are_only_ever_what_it_is_scored_on(self):
        """Scrambling the test answers changes how it scores, never what it guesses."""
        train = labels(0.3, seed=3)
        a, b = labels(0.3, seed=4), labels(0.3, seed=5)
        assert modeling.baseline_score(train, a, "classification", "yes")["strategy"] == modeling.baseline_score(
            train, b, "classification", "yes"
        )["strategy"]


class TestEveryColumnHasAValue:
    def two_class(self, n=600):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, n)
        return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > 0.8, "yes", "no")})

    def test_each_row_has_every_metric_a_model_reports_for_two_answers(self):
        result = modeling.train_candidates(
            self.two_class(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        model = set(result["results"][0]["metrics"]) & TWO_CLASS_KEYS
        assert model == TWO_CLASS_KEYS
        for row in [result["baseline"], *result["baseline"]["others"]]:
            assert TWO_CLASS_KEYS <= set(row), set(row)

    def test_and_for_more_than_two(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, 600)
        df = pd.DataFrame({"x": x, "y": np.digitize(x + rng.normal(0, 1, 600), [-0.5, 0.5]).astype(str)})
        result = modeling.train_candidates(
            df, target_column="y", task="classification", model_names=["logistic_regression"],
            include_deep_diagnostics=False,
        )
        for row in [result["baseline"], *result["baseline"]["others"]]:
            assert MULTI_CLASS_KEYS <= set(row)

    def test_a_test_set_holding_one_answer_does_not_crash_it(self):
        base = modeling.baseline_score(labels(0.3), pd.Series(["no"] * 40), "classification", "yes")
        assert base["accuracy"] == 1.0 and "others" in base

    def test_the_brier_baseline_beside_each_models_score_is_still_the_common_answer_rows(self):
        result = modeling.train_candidates(
            self.two_class(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        assert result["results"][0]["metrics"]["brier_baseline"] == result["baseline"]["brier_score"]

    def test_the_card_shows_both_rows_with_no_blank_in_either(self):
        result = modeling.train_candidates(
            self.two_class(), target_column="y", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
        )
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        rows = [line for line in card.splitlines() if line.startswith("| _No model at all")]
        assert len(rows) == 2
        assert "always answer" in rows[0] and "in proportion to how common" in rows[1]
        for row in rows:
            assert "—" not in row, row

    def test_the_api_returns_both(self):
        buffer = io.StringIO()
        self.two_class().to_csv(buffer, index=False)
        session_id = client.post(
            "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
        ).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
        client.post(f"/api/goal/{session_id}", json=goal)
        body = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        ).json()
        assert TWO_CLASS_KEYS <= set(body["baseline"]) and len(body["baseline"]["others"]) == 1
