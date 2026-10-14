"""Probability calibration as an action the user approves, fitted on training rows only.

A model can rank well and still be wrong about how likely things are; tree ensembles do it
routinely. Calibration corrects that, and the way it goes wrong is subtle: fitted on rows the model
trained on it learns the model's over-confidence and corrects nothing, and fitted with the test
rows it reports a flattering fix. These pin that it uses training rows only, through the same fold
rule as everything else done to them, that it is refused where the data cannot support it, and that
what it changes is shown.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from app.main import app
from app.services import calibration, modeling, model_store, operating_point
from app.services.model_card import build_model_card
from app.services.operating_point import OperatingPoint

client = TestClient(app)
FOREST = "random_forest_classifier"
LOGISTIC = "logistic_regression"


def frame(n: int = 2000, seed: int = 0, shift: float = 0.4) -> pd.DataFrame:
    """A target a random forest ranks well but states over-confidently."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > shift, "yes", "no")}
    )


def train(df, calibration_method="none", model=FOREST, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=[model],
        include_deep_diagnostics=False, calibration=calibration_method, **kwargs,
    )


def chosen(result):
    return next(r for r in result["results"] if r["model"] == result["best_model"])


@pytest.fixture(scope="module")
def forest_runs():
    df = frame()
    return {method: train(df, method) for method in ("none", "sigmoid", "isotonic")}


class TestWhatIsEligible:
    def labels(self, rows, positives):
        return pd.Series(["yes"] * positives + ["no"] * (rows - positives))

    def test_isotonic_needs_a_thousand_rows_and_says_so(self):
        reason = calibration.eligibility("isotonic", "classification", self.labels(640, 200))
        assert "1,000" in reason and "640" in reason

    def test_isotonic_needs_enough_of_the_rarer_answer_not_just_enough_rows(self):
        reason = calibration.eligibility("isotonic", "classification", self.labels(5000, 60))
        assert "100" in reason and "rarer" in reason and "60" in reason

    def test_isotonic_is_allowed_when_both_are_met(self):
        assert calibration.eligibility("isotonic", "classification", self.labels(1000, 100)) is None

    def test_sigmoid_works_on_much_less_data_but_not_none(self):
        assert calibration.eligibility("sigmoid", "classification", self.labels(100, 20)) is None
        assert "20" in calibration.eligibility("sigmoid", "classification", self.labels(500, 12))

    @pytest.mark.parametrize("task,labels", [("regression", pd.Series(range(50))), ("classification", pd.Series(list("abc") * 40))])
    def test_only_a_two_answer_target_can_be_calibrated(self, task, labels):
        assert "exactly two" in calibration.eligibility("sigmoid", task, labels)

    def test_an_unknown_method_is_named(self):
        assert "platt" in calibration.eligibility("platt", "classification", self.labels(500, 200))

    def test_options_say_which_are_available_and_why_not(self):
        found = calibration.options("classification", self.labels(500, 200))
        assert found["sigmoid"] == {"available": True, "reason": None}
        assert found["isotonic"]["available"] is False and "1,000" in found["isotonic"]["reason"]


class TestWhatItDoes:
    def test_sigmoid_makes_the_probabilities_more_honest(self, forest_runs):
        before, after = chosen(forest_runs["none"])["metrics"], chosen(forest_runs["sigmoid"])["metrics"]
        assert after["expected_calibration_error"] < before["expected_calibration_error"]
        assert after["brier_score"] < before["brier_score"]

    def test_isotonic_does_too(self, forest_runs):
        before, after = chosen(forest_runs["none"])["metrics"], chosen(forest_runs["isotonic"])["metrics"]
        assert after["expected_calibration_error"] < before["expected_calibration_error"]

    def test_sigmoid_leaves_the_auc_exactly_where_it_was(self, forest_runs):
        """A sigmoid is a monotone function of the score: it cannot reorder anything."""
        assert chosen(forest_runs["sigmoid"])["metrics"]["roc_auc"] == chosen(forest_runs["none"])["metrics"]["roc_auc"]

    def test_sigmoid_does_not_reorder_a_single_pair_of_rows(self, forest_runs):
        df = frame()
        _, X_test, _, y_test = modeling.split_data(df, "y", "classification")
        raw = chosen(forest_runs["none"])["pipeline"].predict_proba(X_test)[:, 1]
        calibrated = chosen(forest_runs["sigmoid"])["pipeline"].predict_proba(X_test)[:, 1]
        assert np.array_equal(np.argsort(raw, kind="stable"), np.argsort(calibrated, kind="stable"))
        assert roc_auc_score(y_test == "yes", raw) == pytest.approx(roc_auc_score(y_test == "yes", calibrated), abs=1e-12)

    def test_the_delivered_model_is_the_calibrated_one(self, forest_runs):
        result = forest_runs["sigmoid"]
        assert isinstance(chosen(result)["pipeline"], CalibratedClassifierCV)
        assert chosen(result)["metrics"] == result["calibration"]["after"] | {
            k: v for k, v in chosen(result)["metrics"].items() if k not in result["calibration"]["after"]
        }

    def test_the_charts_describe_the_calibrated_probabilities(self, forest_runs):
        points = lambda r: chosen(r)["diagnostics"]["calibration"]  # noqa: E731
        spread = lambda r: np.mean([abs(p["predicted"] - p["actual"]) for p in points(r)])  # noqa: E731
        assert spread(forest_runs["sigmoid"]) < spread(forest_runs["none"])

    def test_the_other_candidates_are_left_alone(self):
        result = modeling.train_candidates(
            frame(800), target_column="y", task="classification",
            model_names=[LOGISTIC, FOREST], include_deep_diagnostics=False, calibration="sigmoid",
        )
        others = [r for r in result["results"] if r["model"] != result["best_model"]]
        assert others and all(not isinstance(r["pipeline"], CalibratedClassifierCV) for r in others)

    def test_the_importance_still_describes_the_underlying_model(self, forest_runs):
        assert chosen(forest_runs["sigmoid"])["feature_importance"] == chosen(forest_runs["none"])["feature_importance"]


class TestTheErrorChanceWouldShow:
    def test_it_matches_a_simulation_of_perfectly_honest_probabilities(self):
        from app.services import diagnostics

        rng = np.random.default_rng(11)
        p = rng.random(400)
        errors = []
        for _ in range(200):
            y = (rng.random(400) < p).astype(int)
            errors.append(diagnostics.expected_calibration_error(y, p))
        points = diagnostics.calibration_points((rng.random(400) < p).astype(int), p)
        assert calibration.chance_error(points) == pytest.approx(np.mean(errors), rel=0.15)

    def test_it_shrinks_with_more_rows(self):
        def floor(n):
            rng = np.random.default_rng(1)
            p = rng.random(n)
            from app.services import diagnostics
            return calibration.chance_error(diagnostics.calibration_points((rng.random(n) < p).astype(int), p))
        assert floor(5000) < floor(500) / 2

    def test_certain_probabilities_have_no_chance_error(self):
        assert calibration.chance_error([{"predicted": 1.0, "actual": 1.0, "count": 40}]) == 0.0

    def test_nothing_to_measure_is_none(self):
        assert calibration.chance_error([]) is None and calibration.chance_error(None) is None


class TestWhatTheResultSays:
    def test_a_calibrated_result_carries_the_method_the_folds_and_before_and_after(self, forest_runs):
        report = forest_runs["sigmoid"]["calibration"]
        assert report["method"] == "sigmoid" and report["requested"] == "sigmoid"
        assert report["folds"] == 5 and report["ensemble"] is False
        assert report["recommended"] is False and report["refused_because"] is None
        assert "test set was not consulted" in report["fitted_on"]
        for side in ("before", "after"):
            assert {"brier_score", "expected_calibration_error", "roc_auc"} <= set(report[side])

    def test_before_is_the_uncalibrated_model_s_held_out_score(self, forest_runs):
        before = forest_runs["sigmoid"]["calibration"]["before"]
        raw = chosen(forest_runs["none"])["metrics"]
        assert before["brier_score"] == raw["brier_score"] and before["expected_calibration_error"] == raw["expected_calibration_error"]

    def test_an_uncalibrated_result_still_carries_the_offer(self, forest_runs):
        report = forest_runs["none"]["calibration"]
        assert report["method"] == "none" and report["before"] is None and report["after"] is None
        error = chosen(forest_runs["none"])["metrics"]["expected_calibration_error"]
        assert report["recommended"] is (
            error > calibration.OFFER_ECE and error > calibration.CHANCE_MARGIN * report["chance_error"]
        )
        assert report["ece_threshold"] == calibration.OFFER_ECE
        assert set(report["options"]) == {"sigmoid", "isotonic"}

    def test_a_well_calibrated_model_is_not_pushed_to_calibrate(self):
        """An even split, where logistic regression's balanced class weights change nothing."""
        report = train(frame(2000, shift=0.0), model=LOGISTIC)["calibration"]
        assert report["recommended"] is False

    def test_balanced_class_weights_on_a_lopsided_target_are_what_the_offer_is_for(self):
        """The default logistic regression reweights a rare answer, which shifts its
        probabilities: calibrated or not, the ranking is fine and the numbers are not."""
        assert train(frame(2000, shift=1.0), model=LOGISTIC)["calibration"]["recommended"] is True

    def test_a_small_test_set_s_noise_is_not_mistaken_for_miscalibration(self):
        """Ten bins of a few rows each: even honest probabilities score several points."""
        report = train(frame(500, shift=0.0), model=LOGISTIC)["calibration"]
        assert report["chance_error"] > 0.05 and report["recommended"] is False

    def test_the_offer_is_not_repeated_once_it_has_been_taken(self, forest_runs):
        assert forest_runs["sigmoid"]["calibration"]["recommended"] is False

    def test_multi_class_and_regression_have_nothing_to_calibrate(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 500)
        three = pd.DataFrame({"x": x, "y": np.digitize(x + rng.normal(0, 1, 500), [-0.5, 0.5]).astype(str)})
        numeric = pd.DataFrame({"x": x, "y": 2 * x + rng.normal(0, 1, 500)})
        a = modeling.train_candidates(three, target_column="y", task="classification", model_names=[LOGISTIC],
                                      include_deep_diagnostics=False, calibration="sigmoid")
        b = modeling.train_candidates(numeric, target_column="y", task="regression", model_names=["linear_regression"],
                                      include_deep_diagnostics=False, calibration="sigmoid")
        assert a["calibration"] is None and b["calibration"] is None
        assert not isinstance(chosen(a)["pipeline"], CalibratedClassifierCV)


class TestARefusalLeavesTheModelAlone:
    def test_isotonic_on_a_small_file_is_declined_with_the_reason(self):
        small = frame(500)
        asked, plain = train(small, "isotonic"), train(small, "none")
        report = asked["calibration"]
        assert report["method"] == "none" and report["requested"] == "isotonic"
        assert "1,000" in report["refused_because"]
        assert chosen(asked)["metrics"] == chosen(plain)["metrics"]
        assert not isinstance(chosen(asked)["pipeline"], CalibratedClassifierCV)

    def test_a_failure_while_fitting_does_not_lose_the_result(self, monkeypatch):
        def broken(*args, **kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(calibration, "fit", broken)
        result = train(frame(800), "sigmoid")
        assert result["calibration"]["method"] == "none"
        assert "out of memory" in result["calibration"]["refused_because"]
        assert chosen(result)["metrics"]["roc_auc"] > 0.5


class TestOnlyTheTrainingRowsAreUsed:
    def test_the_calibrator_is_fitted_on_exactly_the_training_rows(self, monkeypatch):
        seen = {}
        real = CalibratedClassifierCV.fit

        def spy(self, X, y, *args, **kwargs):
            seen.setdefault("indexes", []).append(set(X.index))
            return real(self, X, y, *args, **kwargs)

        monkeypatch.setattr(CalibratedClassifierCV, "fit", spy)
        df = frame(900)
        X_train, X_test, _, _ = modeling.split_data(df, "y", "classification")
        train(df, "sigmoid")
        assert seen["indexes"][0] == set(X_train.index)
        assert not (seen["indexes"][0] & set(X_test.index))

    def test_the_folds_it_is_fitted_on_never_touch_the_test_rows(self):
        df = frame(900)
        X_train, X_test, y_train, _ = modeling.split_data(df, "y", "classification")
        for fit_rows, check_rows in calibration.TrainingFolds().split(X_train, y_train):
            assert max(fit_rows) < len(X_train) and max(check_rows) < len(X_train)

    def test_a_grouped_split_calibrates_on_folds_that_keep_each_entity_whole(self):
        rng = np.random.default_rng(4)
        entities = np.repeat(np.arange(300), 6)
        base = np.repeat(rng.normal(0, 1, 300), 6)
        df = pd.DataFrame(
            {"customer": entities, "x": base + rng.normal(0, 0.5, 1800),
             "y": np.where(base + rng.normal(0, 0.5, 1800) > 0.3, "yes", "no")}
        )
        groups = df["customer"]
        X, y = df[["x"]], df["y"]
        folds = list(calibration.TrainingFolds(groups).split(X, y))
        assert len(folds) == 5
        for fit_rows, check_rows in folds:
            assert not (set(groups.iloc[fit_rows]) & set(groups.iloc[check_rows]))
        # And the same holds when the rows handed over are a subset, as in the cut-off's own folds.
        subset = X.iloc[200:1500]
        for fit_rows, check_rows in calibration.TrainingFolds(groups).split(subset, y.iloc[200:1500]):
            assert not (set(groups.loc[subset.index[fit_rows]]) & set(groups.loc[subset.index[check_rows]]))

    def test_a_grouped_run_passes_its_groups_to_the_calibrator(self, monkeypatch):
        used = []
        real = calibration.TrainingFolds.split

        def spy(self, X, y=None, groups=None):
            used.append(self.groups is not None)
            yield from real(self, X, y, groups)

        monkeypatch.setattr(calibration.TrainingFolds, "split", spy)
        rng = np.random.default_rng(5)
        entities = np.repeat(np.arange(250), 6)
        base = np.repeat(rng.normal(0, 1, 250), 6)
        df = pd.DataFrame({"customer": entities, "x": base + rng.normal(0, 0.5, 1500),
                           "y": np.where(base > 0.2, "yes", "no")})
        result = train(df, "sigmoid", model=LOGISTIC, split_strategy="group", group_column="customer")
        assert result["calibration"]["method"] == "sigmoid"
        assert used and all(used)

    def test_a_date_split_calibrates_forward_in_time_and_says_the_model_is_averaged(self):
        rng = np.random.default_rng(6)
        n = 1600
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str), "x": x,
                           "y": np.where(x + rng.normal(0, 1, n) > 0.4, "yes", "no")})
        folds = list(calibration.TrainingFolds(ordered=True).split(df[["x"]], df["y"]))
        assert all(max(fit_rows) < min(check_rows) for fit_rows, check_rows in folds)
        result = train(df, "sigmoid", split_strategy="time", time_column="when")
        report = result["calibration"]
        assert report["method"] == "sigmoid" and report["ensemble"] is True
        assert "later than the model trained on" in report["fitted_on"]

    def test_copies_kept_together_stay_together_in_calibrations_folds(self, monkeypatch):
        used = []
        real = calibration.TrainingFolds.split

        def spy(self, X, y=None, groups=None):
            used.append(self.groups is not None)
            yield from real(self, X, y, groups)

        monkeypatch.setattr(calibration.TrainingFolds, "split", spy)
        base = frame(700)
        df = pd.concat([base, base.iloc[:200]], ignore_index=True)  # 200 exact copies
        result = train(df, "sigmoid", model=LOGISTIC, keep_identical_rows_together=True)
        assert result["split"]["keep_duplicates_together"] is True
        assert result["calibration"]["method"] == "sigmoid" and used and all(used)


class TestTheModelThatIsSaved:
    def test_it_does_not_depend_on_this_applications_fold_class_to_be_loaded(self, forest_runs):
        model = chosen(forest_runs["sigmoid"])["pipeline"]
        assert isinstance(model.cv, int)
        assert b"TrainingFolds" not in pickle.dumps(model)

    def test_it_is_one_model_and_one_mapping_not_a_copy_per_fold(self, forest_runs):
        model = chosen(forest_runs["sigmoid"])["pipeline"]
        assert len(model.calibrated_classifiers_) == 1 and model.ensemble is False

    def test_a_round_trip_through_a_pickle_predicts_the_same(self, forest_runs):
        model = chosen(forest_runs["sigmoid"])["pipeline"]
        _, X_test, _, _ = modeling.split_data(frame(), "y", "classification")
        again = pickle.loads(pickle.dumps(model))
        assert np.array_equal(model.predict_proba(X_test), again.predict_proba(X_test))


class TestTheCutOffIsChosenOnCalibratedProbabilities:
    def test_the_out_of_fold_scores_come_from_the_calibrated_estimator(self, monkeypatch):
        built = []
        real = operating_point.apply

        def spy(**kwargs):
            built.append(type(kwargs["build"]()).__name__)
            return real(**kwargs)

        monkeypatch.setattr(operating_point, "apply", spy)
        train(frame(900), "sigmoid", model=LOGISTIC, point=OperatingPoint("f1"))
        train(frame(900), "none", model=LOGISTIC, point=OperatingPoint("f1"))
        assert built == ["CalibratedClassifierCV", "Pipeline"]

    def test_a_cut_off_is_still_chosen_and_reported_with_calibration_on(self):
        result = train(frame(900), "sigmoid", model=LOGISTIC, point=OperatingPoint("f1"))
        point = result["operating_point"]
        assert point is not None and 0 < point["threshold"] < 1 and point["folds"] > 0
        assert point["observed"]["rows"] == result["split"]["test_rows"]

    def test_calibrating_a_forest_moves_the_line_the_same_objective_picks(self):
        df = frame(1500)
        raw = train(df, "none", point=OperatingPoint("f1"))["operating_point"]["threshold"]
        fixed = train(df, "sigmoid", point=OperatingPoint("f1"))["operating_point"]["threshold"]
        assert raw != fixed


class TestTheCardAndTheHistory:
    def card(self, result):
        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A model.", interpretation="Fine.")

    def test_the_card_gives_the_method_and_the_before_and_after(self, forest_runs):
        card = self.card(forest_runs["sigmoid"])
        assert "## Probability calibration" in card and "**sigmoid** calibration" in card
        assert "| Brier score (lower is better) |" in card and "| Calibration error (lower is better) |" in card
        assert "the AUC is unchanged" in card

    def test_it_warns_that_the_default_line_now_sits_on_different_numbers(self, forest_runs):
        assert "default 0.50 line" in self.card(forest_runs["sigmoid"])

    def test_isotonic_does_not_claim_the_auc_is_unchanged(self, forest_runs):
        card = self.card(forest_runs["isotonic"])
        assert "the AUC is unchanged" not in card and "the AUC can move slightly" in card

    def test_a_refused_request_is_in_the_card_so_nobody_assumes_it_was_done(self):
        card = self.card(train(frame(500), "isotonic"))
        assert "was asked for and not applied" in card and "1,000" in card

    def test_an_uncalibrated_model_has_no_section(self, forest_runs):
        assert "Probability calibration" not in self.card(forest_runs["none"])

    def test_a_date_split_says_the_model_is_an_average(self):
        rng = np.random.default_rng(7)
        n = 1600
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"when": pd.date_range("2020-01-01", periods=n, freq="D").astype(str), "x": x,
                           "y": np.where(x + rng.normal(0, 1, n) > 0.4, "yes", "no")})
        card = self.card(train(df, "sigmoid", split_strategy="time", time_column="when"))
        assert "averages one copy per fold" in card

    def test_the_run_history_records_the_method(self, forest_runs):
        assert model_store.summarise({**forest_runs["sigmoid"], "run_id": "a"})["calibration"] == "sigmoid"
        assert model_store.summarise({**forest_runs["none"], "run_id": "b"})["calibration"] == "none"


def _session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


class TestOverTheApi:
    BODY = {"model_names": [FOREST], "include_deep_diagnostics": False}

    def test_train_accepts_it_and_returns_the_report(self):
        session_id = _session(frame(1200))
        response = client.post(f"/api/model/{session_id}/train", json={**self.BODY, "calibration": "sigmoid"})
        assert response.status_code == 200, response.text
        report = response.json()["calibration"]
        assert report["method"] == "sigmoid" and report["after"]["brier_score"] < report["before"]["brier_score"]

    def test_an_unknown_method_is_a_validation_error(self):
        session_id = _session(frame(600))
        response = client.post(f"/api/model/{session_id}/train", json={**self.BODY, "calibration": "platt"})
        assert response.status_code == 422

    def test_leaving_it_out_changes_nothing(self):
        session_id = _session(frame(600))
        body = client.post(f"/api/model/{session_id}/train", json=self.BODY).json()
        assert body["calibration"]["method"] == "none"

    def test_a_calibrated_model_is_saved_and_scores_new_rows(self):
        session_id = _session(frame(1200))
        client.post(f"/api/model/{session_id}/train", json={**self.BODY, "calibration": "sigmoid"})
        from app.config import get_settings
        from app.services.session_store import session_dir

        directory = session_dir(get_settings().upload_dir, session_id)
        import joblib

        model = joblib.load(model_store.pipeline_path(directory))
        assert isinstance(model, CalibratedClassifierCV)
        assert model.predict_proba(frame(50, seed=9)[["x", "z"]]).shape == (50, 2)

    def test_it_is_one_look_at_the_test_rows_not_two(self):
        session_id = _session(frame(1200))
        body = client.post(f"/api/model/{session_id}/train", json={**self.BODY, "calibration": "sigmoid"}).json()
        assert body["evaluation"]["look_count"] == 1

    def test_the_history_shows_the_run_as_calibrated(self):
        session_id = _session(frame(1200))
        client.post(f"/api/model/{session_id}/train", json=self.BODY)
        client.post(f"/api/model/{session_id}/train", json={**self.BODY, "calibration": "sigmoid"})
        runs = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert [r["calibration"] for r in runs] == ["sigmoid", "none"]

    def test_the_improvement_route_calibrates_its_rebuilt_winner(self):
        session_id = _session(frame(1200))
        response = client.post(
            f"/api/model/{session_id}/improve",
            json={"model_names": [LOGISTIC], "include_deep_diagnostics": False, "budget": 4, "calibration": "sigmoid"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["calibration"]["method"] == "sigmoid"
