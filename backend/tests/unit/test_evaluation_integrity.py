"""The held-out test set must never inform a decision.

This is the project's central integrity property, pinned in one place. For each decision the app
makes, the scenario runs twice -- once honestly, once with **every test-set answer flipped** --
and every decision must be identical: the chosen model, the tuned settings, the derived inputs
kept, the cut-off, the columns suggested for exclusion, the cross-validation spread and learning
curve. The only thing allowed to differ is the *report* of how the final model scored on the
held-out rows, and each case asserts that it did differ, so a patch that silently stopped
working cannot let the equalities pass over nothing.

Why a flip rather than a shuffle or a deletion: a decision that read the held-out answers
anywhere -- to rank candidates, to pick a threshold, to decide a column leaks -- moves when the
answers are reversed, because reversing them turns the best model on that data into the worst.

**Every new decision path adds a case here** (see CONTRIBUTING.md). Feature selection, ensembles,
calibration and new search strategies belong in this file before they are merged. A case is
cheap: run the path on `learnable`/`dated`, flip with `flipped_test_answers`, compare decisions,
assert the report moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services import (
    diagnostics,
    features as feature_recipes,
    improvement,
    ml_readiness,
    modeling,
    operating_point,
)
from app.services.operating_point import OperatingPoint

MODELS = ["logistic_regression", "random_forest_classifier"]


# ------------------------------------------------------------------ the two worlds


@pytest.fixture
def learnable() -> pd.DataFrame:
    """Non-linear, so the tree model and the linear one genuinely differ, with both classes
    well represented -- flipping answers must not leave a model nothing to fit."""
    rng = np.random.default_rng(21)
    n = 400
    a, b = rng.normal(0, 1, n), rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(2.0 * a * b + 0.8 * a)))
    return pd.DataFrame(
        {
            "a": a,
            "b": b,
            "noise": rng.normal(0, 1, n),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


@pytest.fixture
def dated() -> pd.DataFrame:
    """The answer depends on the weekday of a date column, so the derived-inputs round of the
    improvement loop has a real decision to make."""
    rng = np.random.default_rng(8)
    n = 420
    when = pd.date_range("2021-01-01", periods=n, freq="D")
    weekend = when.dayofweek >= 5
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-(1.5 * signal + 2.0 * weekend - 1.0)))
    return pd.DataFrame(
        {
            "signal": signal,
            "noise": rng.normal(0, 1, n),
            "when": when.astype(str),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


@pytest.fixture
def flipped_test_answers(monkeypatch):
    """Call it to switch the rest of the test into the world where every held-out answer is
    reversed. Training rows are untouched; the partition is the same rows."""
    real_split = modeling.split_data

    def activate(task: str = "classification"):
        def flipped(*args, **kwargs):
            X_train, X_test, y_train, y_test = real_split(*args, **kwargs)
            if task == "classification":
                swap = {"yes": "no", "no": "yes"}
                y_test = y_test.map(swap)
            else:
                y_test = y_test.max() + y_test.min() - y_test
            return X_train, X_test, y_train, y_test

        monkeypatch.setattr(modeling, "split_data", flipped)

    return activate


def _chosen(result: dict) -> dict:
    return next(r for r in result["results"] if r["model"] == result["best_model"])


# ------------------------------------------------------------------ which model


class TestChoosingTheModel:
    def _train(self, df, **kwargs):
        return modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=MODELS,
            include_deep_diagnostics=False, **kwargs,
        )

    def test_the_chosen_model_and_every_fold_score_are_unmoved(self, learnable, flipped_test_answers):
        honest = self._train(learnable)
        flipped_test_answers()
        rigged = self._train(learnable)

        assert rigged["best_model"] == honest["best_model"]
        assert [(r["model"], r["cv_score"], r["cv_std"]) for r in rigged["results"]] == [
            (r["model"], r["cv_score"], r["cv_std"]) for r in honest["results"]
        ]
        assert rigged["selection"] == honest["selection"]
        # The report is allowed to move -- and has to, or the flip did nothing.
        assert _chosen(rigged)["metrics"]["roc_auc"] != _chosen(honest)["metrics"]["roc_auc"]
        assert _chosen(rigged)["metrics"]["roc_auc"] < 0.5 < _chosen(honest)["metrics"]["roc_auc"]

    def test_a_target_encoded_column_never_sees_the_test_answers(self, learnable, flipped_test_answers):
        """The encoding is the average answer for each value, so it is the first place a test answer could
        leak in. It is fitted inside each training fold, from training rows; flip every held-out answer
        and the choice, every fold score and the encoded column's place in the model must not move."""
        rng = np.random.default_rng(12)
        wide = learnable.copy()
        wide["city"] = rng.choice([f"c{i:02d}" for i in range(130)], len(wide))
        assert any(r.kind == "target_encoding" for r in feature_recipes.propose(wide, "outcome"))

        def run():
            return modeling.train_candidates(
                wide, target_column="outcome", task="classification", model_names=MODELS,
                include_deep_diagnostics=False, feature_recipes_wanted=["target_encoding:city"],
            )

        honest = run()
        flipped_test_answers()
        rigged = run()

        assert honest["features"]["target_encoded"] == ["city"]
        assert rigged["best_model"] == honest["best_model"]
        assert rigged["selection"] == honest["selection"]
        assert [(r["model"], r["cv_score"], r["cv_std"]) for r in rigged["results"]] == [
            (r["model"], r["cv_score"], r["cv_std"]) for r in honest["results"]
        ]
        assert _chosen(rigged)["metrics"]["roc_auc"] != _chosen(honest)["metrics"]["roc_auc"]

    def test_a_comparison_made_on_a_sample_is_unmoved_too(self, learnable, flipped_test_answers, monkeypatch):
        """Over the processing limit the models are compared on a sample of the training rows. The sample is
        drawn from the training rows only, so flipping every held-out answer moves neither it nor the choice."""
        monkeypatch.setattr(modeling, "MIN_SELECTION_ROWS", 100)
        honest = self._train(learnable, cost_limit=80_000)
        flipped_test_answers()
        rigged = self._train(learnable, cost_limit=80_000)

        assert honest["resource_limits"]["reductions"][0]["kind"] == "selection_rows"
        assert honest["selection"]["rows_used"] < honest["selection"]["rows_available"]
        assert rigged["selection"] == honest["selection"]
        assert rigged["best_model"] == honest["best_model"]
        assert rigged["resource_limits"] == honest["resource_limits"]
        assert _chosen(rigged)["metrics"]["roc_auc"] != _chosen(honest)["metrics"]["roc_auc"]

    def test_the_same_for_a_regression_target(self, flipped_test_answers):
        rng = np.random.default_rng(3)
        n = 400
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "price": 3 * x + rng.normal(0, 0.5, n)})
        names = ["linear_regression", "random_forest_regressor"]

        def train():
            return modeling.train_candidates(
                df, target_column="price", task="regression", model_names=names,
                include_deep_diagnostics=False,
            )

        honest = train()
        flipped_test_answers("regression")
        rigged = train()

        assert rigged["best_model"] == honest["best_model"]
        assert [r["cv_score"] for r in rigged["results"]] == [r["cv_score"] for r in honest["results"]]
        assert _chosen(rigged)["metrics"]["r2"] != _chosen(honest)["metrics"]["r2"]

    def test_the_same_under_a_date_split(self, dated, flipped_test_answers):
        kwargs = {"split_strategy": "time", "time_column": "when"}
        honest = self._train(dated, **kwargs)
        flipped_test_answers()
        rigged = self._train(dated, **kwargs)
        assert rigged["best_model"] == honest["best_model"]
        assert rigged["selection"] == honest["selection"]
        assert _chosen(rigged)["metrics"]["roc_auc"] != _chosen(honest)["metrics"]["roc_auc"]


# ------------------------------------------------------------------ the cut-off


class TestChoosingTheCutOff:
    def _train(self, df):
        return modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            point=OperatingPoint("f1"),
        )

    def test_the_line_and_what_the_folds_promised_are_unmoved(self, learnable, flipped_test_answers):
        honest = self._train(learnable)["operating_point"]
        flipped_test_answers()
        rigged = self._train(learnable)["operating_point"]

        assert rigged["threshold"] == honest["threshold"]
        assert rigged["expected"] == honest["expected"]
        assert rigged["observed"] != honest["observed"]  # what the test set delivered moves

    def test_the_direct_call_agrees(self, learnable):
        """`operating_point.apply` itself, without the training wrapper around it."""
        X_train, X_test, y_train, y_test = modeling.split_data(learnable, "outcome", "classification")
        features = modeling.classify_features(learnable, "outcome")
        used = features["numeric"]

        def build():
            return modeling.build_pipeline("logistic_regression", features["numeric"], [])

        def apply(y_for_test):
            pipeline = build().fit(X_train[used], y_train)
            return operating_point.apply(
                build=build, pipeline=pipeline, X_train=X_train[used], y_train=y_train,
                X_test=X_test[used], y_test=y_for_test, point=OperatingPoint("f1"),
                positive_label="yes",
            )

        honest = apply(y_test)
        rigged = apply(y_test.map({"yes": "no", "no": "yes"}))
        assert rigged["threshold"] == honest["threshold"]
        assert rigged["observed"] != honest["observed"]


# ------------------------------------------------------------------ calibration


class TestCalibratingTheProbabilities:
    """The mapping is fitted on folds of the training rows. Flip every held-out answer and the
    fitted calibrator, the folds it used and the cut-off chosen on its probabilities must not
    move; only the before-and-after report of how it did on the held-out rows may."""

    def _train(self, df, **kwargs):
        return modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
            calibration="sigmoid", **kwargs,
        )

    def test_the_fitted_calibrator_is_unmoved(self, learnable, flipped_test_answers):
        probe = learnable.drop(columns="outcome").iloc[:80]
        honest = self._train(learnable)
        flipped_test_answers()
        rigged = self._train(learnable)

        honest_model, rigged_model = _chosen(honest)["pipeline"], _chosen(rigged)["pipeline"]
        assert np.array_equal(honest_model.predict_proba(probe), rigged_model.predict_proba(probe))
        assert rigged["calibration"]["folds"] == honest["calibration"]["folds"]
        assert rigged["calibration"]["method"] == honest["calibration"]["method"]
        # The report is allowed to move -- and has to, or the flip did nothing.
        assert rigged["calibration"]["after"] != honest["calibration"]["after"]

    def test_the_offer_to_calibrate_is_made_without_the_test_answers_deciding_the_methods(
        self, learnable, flipped_test_answers
    ):
        honest = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
        )["calibration"]
        flipped_test_answers()
        rigged = modeling.train_candidates(
            learnable, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
        )["calibration"]
        assert rigged["options"] == honest["options"]  # which methods the data supports

    def test_the_cut_off_chosen_on_calibrated_probabilities_is_unmoved(self, learnable, flipped_test_answers):
        honest = self._train(learnable, point=OperatingPoint("f1"))["operating_point"]
        flipped_test_answers()
        rigged = self._train(learnable, point=OperatingPoint("f1"))["operating_point"]
        assert rigged["threshold"] == honest["threshold"]
        assert rigged["expected"] == honest["expected"]
        assert rigged["observed"] != honest["observed"]


# ------------------------------------------------------------------ recommending the split


class TestRecommendingTheSplit:
    """The advice on how to divide the rows reads the answers once: to see whether they drift over
    time. It must read only the training side of the default split -- the same convention the
    leakage suggestions follow -- or the rows held back would have a say in how they are held back."""

    @pytest.fixture
    def drifting(self) -> pd.DataFrame:
        rng = np.random.default_rng(31)
        n = 600
        return pd.DataFrame(
            {
                "created_on": pd.date_range("2021-01-01", periods=n).astype(str),
                "x": rng.normal(0, 1, n),
                "outcome": np.where(rng.random(n) < np.linspace(0.05, 0.6, n), "yes", "no"),
            }
        )

    def _pin(self, monkeypatch, df):
        """The honest frame, the frame with the held-back answers flipped, and the partition
        pinned so the only difference between them is those answers."""
        _, X_test, _, _ = modeling.split_data(df, "outcome", "classification")
        test_index = X_test.index
        flipped = df.copy()
        flipped.loc[test_index, "outcome"] = flipped.loc[test_index, "outcome"].map({"yes": "no", "no": "yes"})

        def fixed_partition(frame, target, task, **_kwargs):
            labelled = frame[frame[target].notna()]
            train_index = labelled.index.difference(test_index)
            features = labelled.drop(columns=[target])
            return (
                features.loc[train_index], features.loc[test_index],
                labelled.loc[train_index, target], labelled.loc[test_index, target],
            )

        monkeypatch.setattr(modeling, "split_data", fixed_partition)
        return flipped, test_index

    def test_flipping_the_held_back_answers_does_not_move_the_recommendation(self, drifting, monkeypatch):
        from app.api import routes_model

        flipped, _ = self._pin(monkeypatch, drifting)
        honest = routes_model._recommended_split(drifting, "outcome", "classification", "")
        rigged = routes_model._recommended_split(flipped, "outcome", "classification", "")
        assert honest["strategy"] == "time"  # there is a decision to protect
        assert rigged == honest

    def test_a_computation_over_every_row_would_have_moved(self, drifting, monkeypatch):
        """The flip has to bite: read over the whole file the same drift test gives other numbers."""
        from app.services import validation_advice

        flipped, test_index = self._pin(monkeypatch, drifting)
        train_only = drifting.drop(index=test_index)
        everything = validation_advice.drift_over_time(flipped, "created_on", "outcome", "classification")
        training = validation_advice.drift_over_time(train_only, "created_on", "outcome", "classification")
        assert everything["text"] != training["text"] and everything["effect"] != training["effect"]

    def test_the_drift_test_is_given_no_held_back_row(self, drifting, monkeypatch):
        from app.api import routes_model
        from app.services import validation_advice

        _, test_index = self._pin(monkeypatch, drifting)
        seen = {}
        real = validation_advice.drift_over_time

        def spy(frame, *args, **kwargs):
            seen["index"] = set(frame.index)
            return real(frame, *args, **kwargs)

        monkeypatch.setattr(validation_advice, "drift_over_time", spy)
        routes_model._recommended_split(drifting, "outcome", "classification", "")
        assert seen["index"] and not (seen["index"] & set(test_index))


# ------------------------------------------------------------------ the seed check


class TestCheckingTheChoiceAgainstOtherSeeds:
    """Refitting the chosen model with other seeds is done on the training folds. Flip every
    held-out answer and the seeds, the scores, the spread and the verdict on them must not move."""

    def test_the_seed_check_is_unmoved(self, learnable, flipped_test_answers):
        def check():
            return modeling.train_candidates(
                learnable, target_column="outcome", task="classification",
                model_names=["random_forest_classifier", "gradient_boosting_classifier"],
                include_deep_diagnostics=False,
            )

        honest = check()
        flipped_test_answers()
        rigged = check()
        assert honest["selection"]["seed_stability"]["applicable"] is True
        assert rigged["selection"]["seed_stability"] == honest["selection"]["seed_stability"]
        assert _chosen(rigged)["metrics"] != _chosen(honest)["metrics"]  # the report moved


# ------------------------------------------------------------------ the best single column


class TestComparingWithTheBestSingleColumn:
    """Which column stands alone best, and how it scores, are worked out on the training folds.
    Flip every held-out answer and the column, its fold score and the verdict must not move."""

    def test_the_single_column_comparison_is_unmoved(self, learnable, flipped_test_answers):
        def check():
            return modeling.train_candidates(
                learnable, target_column="outcome", task="classification", model_names=MODELS,
                include_deep_diagnostics=False,
            )

        honest = check()
        flipped_test_answers()
        rigged = check()
        single = honest["baseline"]["best_single_column"]
        assert single["applicable"] is True
        assert rigged["baseline"]["best_single_column"] == single
        assert _chosen(rigged)["metrics"] != _chosen(honest)["metrics"]  # the report moved


# ------------------------------------------------------------------ the improvement loop


class TestTheImprovementLoop:
    def _improve(self, df, **kwargs):
        return improvement.improve(
            df, target_column="outcome", task="classification", model_names=MODELS,
            budget=4, **kwargs,
        )

    def _decisions(self, found: dict) -> dict:
        return {
            "model": found["best"]["model"],
            "settings": found["best"]["params"],
            "recipes": found["best"]["feature_recipes"],
            "cv": (found["best"]["cv_score"], found["best"]["cv_std"]),
            "rounds": found["rounds"],
            "stopped": found["stopped_because"],
        }

    def test_the_model_settings_and_every_round_are_unmoved(self, learnable, flipped_test_answers):
        honest = self._improve(learnable)
        flipped_test_answers()
        rigged = self._improve(learnable)

        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["test_metrics"] != honest["test_metrics"]
        assert len(honest["rounds"]) >= 2  # there were decisions to compare

    def test_a_time_limit_does_not_let_the_test_answers_in(self, learnable, flipped_test_answers):
        """The deadline is checked against the clock, never against a score, so a limit that cuts the
        search short cuts it at the same place whatever the held-out answers are."""
        import sklearn.model_selection as selection

        from app.services import tuning

        class Clock:
            now = 1000.0

            def monotonic(self):
                return self.now

        def run():
            clock = Clock()
            real = selection.cross_val_score

            def slow(*args, **kwargs):
                clock.now += 10.0
                return real(*args, **kwargs)

            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(improvement, "time", clock)
                patch.setattr(tuning, "time", clock)
                patch.setattr(selection, "cross_val_score", slow)
                return self._improve(learnable, time_limit_seconds=25)

        honest = run()
        flipped_test_answers()
        rigged = run()

        assert honest["stopped_because"] == "time_limit"
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["skipped_for_time"] == honest["skipped_for_time"] != []
        assert rigged["test_metrics"] != honest["test_metrics"]

    def test_successive_halving_chooses_the_same_settings_whatever_the_test_answers_are(
        self, learnable, flipped_test_answers
    ):
        """Halving ranks settings on the first folds of the training rows and nothing else: flip every
        held-out answer and the rungs, the survivor and every round must be identical."""
        honest = self._improve(learnable, search_strategy="halving")
        flipped_test_answers()
        rigged = self._improve(learnable, search_strategy="halving")

        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["test_metrics"] != honest["test_metrics"]
        searched = [r for r in honest["rounds"] if r["detail"].get("rungs")]
        assert searched  # the strategy ran, so there were rungs to protect

    def test_whether_the_models_are_averaged_never_depends_on_the_test_answers(self, learnable, flipped_test_answers):
        """The seventh round averages the incumbent with the strongest runner-up and checks the average on the
        training folds and on fresh ones. Flip every held-out answer: the members, the scores and whether the
        average was kept must not move."""
        honest = self._improve(learnable)
        flipped_test_answers()
        rigged = self._improve(learnable)

        averaged = next(r for r in honest["rounds"] if r["name"] == "ensemble")
        assert averaged["detail"]["members"]  # the round ran
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["test_metrics"] != honest["test_metrics"]

    def test_which_combinations_are_kept_never_depends_on_the_test_answers(self, flipped_test_answers):
        """The sixth round ranks the numeric inputs on validation folds, tries products and ratios of the top
        few, and confirms the winner on fresh folds. Give it data where a product helps, flip every held-out
        answer, and the pairs tried, the one kept and every round must be identical."""
        rng = np.random.default_rng(41)
        n = 500
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        logit = x1 + x2 + 2.5 * x1 * x2
        df = pd.DataFrame({
            "x1": x1, "x2": x2, "n0": rng.normal(size=n),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no"),
        })

        def linear():  # a forest finds a product by itself; the linear model needs it handed over
            return improvement.improve(
                df, target_column="outcome", task="classification", model_names=["logistic_regression"], budget=4
            )

        honest = linear()
        flipped_test_answers()
        rigged = linear()

        kept = next(r for r in honest["rounds"] if r["name"] == "interactions")["detail"].get("kept")
        assert kept  # the round ran and kept something, so there was a decision to protect
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["best"]["feature_recipes"] == honest["best"]["feature_recipes"]
        assert rigged["test_metrics"] != honest["test_metrics"]

    def test_which_inputs_are_removed_never_depends_on_the_test_answers(self, learnable, flipped_test_answers):
        """The fifth round ranks inputs by shuffling them on validation folds of the training rows. Give the
        data enough noise columns for it to run, flip every held-out answer, and the importances, the
        columns removed and every round must be identical."""
        rng = np.random.default_rng(31)
        wide = learnable.copy()
        for index in range(5):
            wide[f"noise_{index}"] = rng.normal(0, 1, len(wide))

        honest = self._improve(wide)
        flipped_test_answers()
        rigged = self._improve(wide)

        assert any(r["name"] == "feature_selection" for r in honest["rounds"])  # the round ran
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["best"]["removed_columns"] == honest["best"]["removed_columns"]
        assert rigged["test_metrics"] != honest["test_metrics"]

    def test_the_check_on_fresh_folds_never_reads_the_test_set(self, learnable, flipped_test_answers, monkeypatch):
        """A search winner is re-scored with the incumbent on reshuffled training folds. Force a winner that
        clears the first rule so the check always happens, then flip every held-out answer: the fresh-fold
        scores, and so the decision, must be the same."""
        from app.services import tuning
        from app.services.tuning import SearchResult

        def lucky(build, name, X, y, task, **kwargs):
            return SearchResult(params={"model__C": 3.0} if name == "logistic_regression" else {"model__max_depth": 6},
                                score=0.999, std=0.001, metric="roc_auc", fits=1)

        monkeypatch.setattr(tuning, "search", lucky)
        honest = self._improve(learnable)
        flipped_test_answers()
        rigged = self._improve(learnable)

        checks = [r["detail"]["confirmation"] for r in honest["rounds"] if "confirmation" in r["detail"]]
        assert checks and all(c["status"] in {"confirmed", "not_confirmed"} for c in checks)
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["test_metrics"] != honest["test_metrics"]

    def test_the_test_set_is_scored_exactly_once_whether_or_not_the_clock_runs_out(self, learnable, monkeypatch):
        calls = []
        real = modeling.evaluate

        def counting(pipeline, X, y, *args, **kwargs):
            calls.append(len(X))
            return real(pipeline, X, y, *args, **kwargs)

        monkeypatch.setattr(modeling, "evaluate", counting)
        self._improve(learnable)
        self._improve(learnable, time_limit_seconds=1)
        # One evaluation per run, on the held-out rows -- never on a training fold.
        assert len(calls) == 2

    def test_the_derived_inputs_it_keeps_are_unmoved(self, dated, flipped_test_answers):
        honest = self._improve(dated, split_strategy="time", time_column="when")
        flipped_test_answers()
        rigged = self._improve(dated, split_strategy="time", time_column="when")

        assert rigged["best"]["feature_recipes"] == honest["best"]["feature_recipes"]
        assert self._decisions(rigged) == self._decisions(honest)
        assert rigged["test_metrics"] != honest["test_metrics"]
        # The derived-inputs round ran, so there was a recipe decision to protect.
        assert any("derived" in r["name"] or "input" in r["name"] for r in honest["rounds"])

    def test_the_loop_has_a_test_report_to_move(self, learnable):
        """Documents why the equalities above mean something: the loop does score the final
        model on the held-out rows, once, and reports it separately from its decisions."""
        found = self._improve(learnable)
        assert "roc_auc" in found["test_metrics"]


# ------------------------------------------------------------------ the inputs


@pytest.fixture
def leaky_labelled() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    n = 300
    signal = rng.normal(0, 1, n)
    outcome = np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no")
    months = ["january", "february", "march", "april"]
    return pd.DataFrame(
        {
            "signal": signal.round(4),
            "noise": rng.normal(0, 1, n).round(4),
            # Present exactly when the answer is yes: it gives the answer away.
            "exit_note": [f"left in {months[i % 4]}" if o == "yes" else None for i, o in enumerate(outcome)],
            "outcome": outcome,
        }
    )


class TestSuggestingColumnsToLeaveOut:
    def test_the_plan_s_leakage_suggestions_are_unmoved(self, leaky_labelled, monkeypatch):
        """The plan measures leakage on the training rows of the default split. Flip the held-out
        rows' answers *in the frame* (a patch to split_data cannot reach the frame the check
        reads) and pin the partition so the only difference is those answers."""
        from app.api import routes_model

        _, X_test, _, _ = modeling.split_data(leaky_labelled, "outcome", "classification")
        test_index = X_test.index
        flipped = leaky_labelled.copy()
        flipped.loc[test_index, "outcome"] = flipped.loc[test_index, "outcome"].map(
            {"yes": "no", "no": "yes"}
        )

        def fixed_partition(df, target, task, **_kwargs):
            labelled = df[df[target].notna()]
            train_index = labelled.index.difference(test_index)
            features = labelled.drop(columns=[target])
            return (
                features.loc[train_index], features.loc[test_index],
                labelled.loc[train_index, target], labelled.loc[test_index, target],
            )

        monkeypatch.setattr(modeling, "split_data", fixed_partition)

        def suggestions(frame):
            rows, _ = routes_model._leakage_rows(frame, "outcome", "classification")
            goal = type("G", (), {"target_column": "outcome"})()
            return [s["column"] for s in routes_model._suggested_exclusions(frame, goal, "classification", rows)]

        assert suggestions(flipped) == suggestions(leaky_labelled)
        assert "exit_note" in suggestions(flipped)
        # The flip has to bite: a check over every row would have lost the leak.
        assert ml_readiness.detect_leakage(flipped, "outcome", "classification") == []

    def test_derived_input_proposals_do_not_read_the_answers(self, leaky_labelled):
        flipped = leaky_labelled.assign(outcome=leaky_labelled["outcome"].map({"yes": "no", "no": "yes"}))
        honest = [r.as_dict() for r in feature_recipes.propose(leaky_labelled, "outcome")]
        rigged = [r.as_dict() for r in feature_recipes.propose(flipped, "outcome")]
        assert rigged == honest and honest  # and there is something to compare


# ------------------------------------------------------------------ what is shown beside the score


class TestTheChecksShownBesideTheScore:
    def test_the_spread_and_the_learning_curve_are_unmoved(self, learnable, flipped_test_answers):
        def deep():
            return modeling.train_candidates(
                learnable, target_column="outcome", task="classification",
                model_names=["logistic_regression"], include_deep_diagnostics=True,
            )["deep_diagnostics"]

        honest = deep()
        flipped_test_answers()
        rigged = deep()

        assert rigged["cross_validation"] == honest["cross_validation"]
        assert rigged["learning_curve"] == honest["learning_curve"]
        # Permutation importance explains the evaluated model on the test rows, so it moves.
        assert rigged["permutation_importance"] != honest["permutation_importance"]


# ------------------------------------------------------------------ the sealed rows


class TestTheSealedRows:
    """A sealed final holdout is only sealed if nothing but `finalise` ever reads it. Flip every
    answer in it and everything the app decides, and everything it reports on the development
    rows, must stay exactly as it was."""

    def _pin(self, monkeypatch, df):
        final = modeling.split_with_sealed_holdout(df, "outcome", "classification")[2].index
        monkeypatch.setattr(modeling, "sealed_holdout_rows", lambda labelled, *a, **k: final)
        flipped = df.copy()
        flipped.loc[final, "outcome"] = flipped.loc[final, "outcome"].map({"yes": "no", "no": "yes"})
        return flipped, final

    def test_flipping_the_sealed_answers_moves_no_decision_and_no_report(self, learnable, monkeypatch):
        flipped, final = self._pin(monkeypatch, learnable)

        def run(frame):
            return modeling.train_candidates(
                frame, target_column="outcome", task="classification", model_names=MODELS,
                include_deep_diagnostics=False, sealed_holdout=True, point=OperatingPoint("f1"),
            )

        honest, rigged = run(learnable), run(flipped)

        assert rigged["best_model"] == honest["best_model"]
        assert [(r["model"], r["cv_score"]) for r in rigged["results"]] == [
            (r["model"], r["cv_score"]) for r in honest["results"]
        ]
        assert _chosen(rigged)["metrics"] == _chosen(honest)["metrics"]  # even the report
        assert rigged["operating_point"] == honest["operating_point"]
        assert rigged["assessment"] == honest["assessment"]
        # The flip has to bite: the sealed rows really do hold different answers.
        assert (learnable.loc[final, "outcome"] != flipped.loc[final, "outcome"]).all()


# ------------------------------------------------------------------ the guard on the guard


def test_contributing_requires_a_case_here_for_every_new_decision_path():
    """The suite is only a guard while new decision paths keep landing in it. The rule is
    written where contributors read, and this fails if it is ever edited out."""
    from pathlib import Path

    contributing = (Path(__file__).resolve().parents[3] / "CONTRIBUTING.md").read_text(
        encoding="utf-8"
    )
    flat = " ".join(contributing.split())
    assert "test_evaluation_integrity.py" in flat
    assert "decision path" in flat
