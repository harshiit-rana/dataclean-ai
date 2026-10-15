"""A chance-adjusted verdict for a target with more than two answers.

With two answers the verdict compares AUC with what luck could produce on a test set of that size. With
more there is no single AUC, so the verdict fell back to accuracy against the commonest answer plus a
binomial margin, which ignores how well the model ranks every answer and treats all mistakes alike. It
now uses the macro one-vs-rest AUC and a noise floor read off the data by shuffling the test answers,
keeping the accuracy comparison as a second line.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.metrics import roc_auc_score

from app.main import app
from app.services import modeling
from app.services.model_card import build_model_card

client = TestClient(app)


def four_classes(seed: int = 0, n: int = 300, signal: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    if signal:
        y = np.array(list("abcd"))[np.digitize(signal * x + rng.normal(size=n), [-1, 0, 1])]
    else:
        y = rng.choice(list("abcd"), n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": y})


def weak_signal(seed: int, n: int = 500) -> pd.DataFrame:
    """The column does inform the answer, but through a lot of noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.array(list("abcd"))[np.digitize(0.6 * x + rng.normal(size=n), [-1.2, 0, 1.2])]
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": y})


def train(df, models=("logistic_regression",), **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=list(models),
        include_deep_diagnostics=False, **kwargs,
    )


def chosen(result):
    return next(r for r in result["results"] if r["model"] == result["best_model"])


class TestTheMetric:
    def test_a_multiclass_result_reports_the_macro_one_vs_rest_auc(self):
        result = train(four_classes(signal=1.5))
        value = chosen(result)["metrics"]["roc_auc_ovr_macro"]
        assert 0.5 < value <= 1

    def test_it_is_scikit_learns_number(self):
        df = four_classes(signal=1.5)
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        result = train(df)
        model = chosen(result)["pipeline"]
        expected = roc_auc_score(y_test.astype(str), model.predict_proba(X_test), multi_class="ovr",
                                 average="macro", labels=[str(c) for c in model.classes_])
        assert chosen(result)["metrics"]["roc_auc_ovr_macro"] == pytest.approx(expected, abs=1e-4)

    def test_two_answers_are_unchanged(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=300)
        result = train(pd.DataFrame({"x": x, "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")}))
        assert "roc_auc_ovr_macro" not in chosen(result)["metrics"] and "roc_auc" in chosen(result)["metrics"]

    def test_a_regression_has_none(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=200)
        result = modeling.train_candidates(
            pd.DataFrame({"x": x, "y": 2 * x}), target_column="y", task="regression",
            model_names=["linear_regression"], include_deep_diagnostics=False,
        )
        assert "roc_auc_ovr_macro" not in chosen(result)["metrics"]

    def test_an_answer_missing_from_the_test_rows_leaves_it_out_rather_than_averaging_what_is_left(self):
        class Stub:
            classes_ = np.array(["a", "b", "c"])

            def predict(self, X):
                return np.array(["a"] * len(X))

            def predict_proba(self, X):
                return np.tile([0.5, 0.3, 0.2], (len(X), 1))

        X = pd.DataFrame({"x": range(30)})
        y = pd.Series(["a", "b"] * 15)  # no "c" among the test answers
        assert "roc_auc_ovr_macro" not in modeling.evaluate(Stub(), X, y, "classification")

    def test_a_model_with_no_probabilities_has_none(self):
        class NoProba:
            classes_ = np.array(["a", "b", "c"])

            def predict(self, X):
                return np.array(["a"] * len(X))

        X = pd.DataFrame({"x": range(30)})
        y = pd.Series(["a", "b", "c"] * 10)
        assert "roc_auc_ovr_macro" not in modeling.evaluate(NoProba(), X, y, "classification")


class TestTheNoiseFloor:
    def fitted(self, n=300, seed=0):
        df = four_classes(seed, n)
        result = train(df)
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        return chosen(result)["pipeline"], X_test, y_test

    def test_it_is_a_little_above_a_coin_flip_on_a_small_test_set(self):
        model, X, y = self.fitted()
        assert 0.5 < modeling.ovr_noise_floor(model, X, y) < 0.8

    def test_it_is_the_same_every_time(self):
        model, X, y = self.fitted()
        assert modeling.ovr_noise_floor(model, X, y) == modeling.ovr_noise_floor(model, X, y)

    def test_more_test_rows_bring_it_closer_to_chance(self):
        small = modeling.ovr_noise_floor(*self.fitted(n=160))
        large = modeling.ovr_noise_floor(*self.fitted(n=3000))
        assert large < small and large < 0.56

    def test_more_candidates_raise_it_because_the_best_of_several_runs_high(self):
        model, X, y = self.fitted()
        assert modeling.ovr_noise_floor(model, X, y, model_count=5) > modeling.ovr_noise_floor(model, X, y, model_count=1)

    def test_it_never_reads_the_answers_as_anything_but_a_shuffle(self):
        """Shuffling is all it does to them, so the model's skill cannot raise it: a model that is
        good and one that is useless get the same floor on the same answers."""
        df = four_classes(0, signal=2.0)
        good = train(df)
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        floor_good = modeling.ovr_noise_floor(chosen(good)["pipeline"], X_test, y_test)
        assert floor_good == pytest.approx(modeling.ovr_noise_floor(chosen(train(four_classes(0)))["pipeline"], X_test, y_test), abs=0.08)

    def test_two_answers_have_no_such_floor_because_they_have_the_analytic_one(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=300)
        df = pd.DataFrame({"x": x, "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")})
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        assert modeling.ovr_noise_floor(chosen(train(df))["pipeline"], X_test, y_test) is None


class TestTheVerdict:
    def verdict(self, macro, floor, accuracy=0.4, baseline=0.3):
        return modeling.assess_result(
            {"accuracy": accuracy, "roc_auc_ovr_macro": macro, "labels": list("abcd")},
            {"accuracy": baseline, "strategy": "always answer 'a'"}, "classification",
            test_rows=200, class_counts=[60, 60, 40, 40], ovr_auc_floor=floor,
        )

    def test_below_what_luck_reaches_is_no_better_than_guessing(self):
        found = self.verdict(0.55, 0.60)
        assert found["verdict"] == "no_better_than_guessing" and "0.60" in found["headline"]

    def test_above_luck_but_weak_is_marginal(self):
        assert self.verdict(0.64, 0.58)["verdict"] == "marginal"

    def test_clearly_ranking_the_answers_is_useful(self):
        assert self.verdict(0.82, 0.58)["verdict"] == "useful"

    def test_the_accuracy_against_the_baseline_stays_as_a_second_line(self):
        headline = self.verdict(0.82, 0.58, accuracy=0.55, baseline=0.28)["headline"]
        assert "55% accuracy" in headline and "28%" in headline and "always answer 'a'" in headline

    def test_the_same_macro_auc_means_different_things_on_different_amounts_of_evidence(self):
        assert self.verdict(0.64, 0.70)["verdict"] == "no_better_than_guessing"
        assert self.verdict(0.64, 0.52)["verdict"] == "marginal"

    def test_without_the_auc_it_falls_back_to_accuracy_and_says_so(self):
        found = modeling.assess_result(
            {"accuracy": 0.8, "labels": list("abcd")}, {"accuracy": 0.3}, "classification",
            test_rows=200, class_counts=[60, 60, 40, 40],
        )
        assert found["verdict"] == "useful" and "one-vs-rest AUC was not available" in found["headline"]

    def test_a_two_answer_verdict_is_untouched(self):
        found = modeling.assess_result(
            {"accuracy": 0.7, "roc_auc": 0.65}, {"accuracy": 0.7}, "classification",
            test_rows=2000, class_counts=[1400, 600], ovr_auc_floor=0.9,
        )
        assert found["verdict"] == "marginal" and "one-vs-rest" not in found["headline"]

    def test_an_accuracy_only_result_with_two_labels_gets_no_such_note(self):
        found = modeling.assess_result(
            {"accuracy": 0.9, "labels": ["a", "b"]}, {"accuracy": 0.5}, "classification",
            test_rows=200, class_counts=[100, 100],
        )
        assert "not available" not in found["headline"]


class TestTheBenchmark:
    def test_noise_is_called_guessing_in_nearly_every_seed_and_never_useful(self):
        """Four answers with no relationship to any column, 240 rows so 48 are held back."""
        verdicts = []
        for seed in range(25):
            result = train(four_classes(seed, n=240), models=("logistic_regression", "random_forest_classifier"))
            assert result["split"]["test_rows"] == 48
            verdicts.append(result["assessment"]["verdict"])
        assert verdicts.count("no_better_than_guessing") >= 23, verdicts
        assert "useful" not in verdicts

    def test_a_weak_but_real_signal_is_found_where_the_accuracy_rule_alone_missed_it(self):
        """What the change is for. A model that ranks the answers better than chance while its
        accuracy barely moves was called "no better than guessing" by the accuracy margin; measured
        on twenty such files the old rule missed all twenty, and this finds them."""
        found, missed_by_the_old_rule = [], 0
        for seed in range(8):
            df = weak_signal(seed)
            result = train(df)
            found.append(result["assessment"]["verdict"])
            metrics = dict(chosen(result)["metrics"])
            metrics.pop("roc_auc_ovr_macro")
            old = modeling.assess_result(
                metrics, result["baseline"], "classification", test_rows=result["split"]["test_rows"],
                class_counts=df["y"].value_counts().tolist(),
            )
            missed_by_the_old_rule += old["verdict"] == "no_better_than_guessing"
        assert "no_better_than_guessing" not in found, found
        assert missed_by_the_old_rule == 8

    def test_a_real_signal_is_still_useful(self):
        verdicts = [train(four_classes(seed, n=400, signal=1.5))["assessment"]["verdict"] for seed in range(5)]
        assert verdicts == ["useful"] * 5


class TestEverywhereItShows:
    def test_the_baselines_have_a_value_under_the_new_column(self):
        baseline = train(four_classes(signal=1.5))["baseline"]
        assert baseline["roc_auc_ovr_macro"] == 0.5 and baseline["others"][0]["roc_auc_ovr_macro"] == 0.5

    def test_the_card_has_the_column_and_explains_it(self):
        card = build_model_card(session_id="s", file_name="f.csv", result=train(four_classes(signal=1.5)),
                                summary="A.", interpretation="B.")
        header = next(line for line in card.splitlines() if line.startswith("| Model |") and "macro" in line)
        assert "ROC AUC (macro)" in header
        assert "**ROC AUC (macro)** — How well it ranks each answer against all the others" in card
        rows = [line for line in card.splitlines() if line.startswith("| _No model at all")]
        assert rows and all("—" not in row for row in rows)

    def test_a_two_class_card_does_not_mention_it(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=300)
        card = build_model_card(
            session_id="s", file_name="f.csv", summary="A.", interpretation="B.",
            result=train(pd.DataFrame({"x": x, "y": np.where(x + rng.normal(size=300) > 0, "yes", "no")})),
        )
        assert "macro" not in card

    def test_the_train_response_carries_it_and_a_verdict_that_uses_it(self):
        buffer = io.StringIO()
        four_classes(signal=1.5, n=400).to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        client.post(f"/api/goal/{sid}", json={"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"})
        body = client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False}).json()
        assert "roc_auc_ovr_macro" in body["results"][0]["metrics"]
        assert "macro AUC" in body["assessment"]["headline"]

    def test_the_sealed_final_evaluation_is_judged_the_same_way(self):
        buffer = io.StringIO()
        four_classes(signal=1.5, n=1400).to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        client.post(f"/api/goal/{sid}", json={"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"})
        trained = client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "sealed_holdout": True})
        assert trained.status_code == 200 and trained.json()["split"]["sealed"] is True
        opened = client.post(f"/api/model/{sid}/finalise").json()
        assert "roc_auc_ovr_macro" in opened["metrics"] and "macro AUC" in opened["assessment"]["headline"]
