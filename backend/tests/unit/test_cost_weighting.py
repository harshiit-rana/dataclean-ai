"""Training with the user's stated mistake costs, not only choosing the cut-off with them.

When the user says a missed case costs N times a false alarm, the last round of the improvement loop trains the
incumbent again with each row weighted by the cost of getting it wrong, and keeps the weighting only when the
expected cost per row at the best cut-off is lower fold by fold, by more than the folds disagree. The weighting lives
in a wrapper inside the pipeline, so the folds, the final fit and the exported script all use it.
"""

from __future__ import annotations

import ast
import io
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app.main import app
from app.services import improvement, model_export, modeling
from app.services.cost_weighting import CostWeighted

LR, RF = "logistic_regression", "random_forest_classifier"
WEIGHTS = {"positive_label": "yes", "cost_of_missing": 8.0, "cost_of_false_alarm": 1.0}
COSTS = {"cost_of_missing": 8.0, "cost_of_false_alarm": 1.0, "positive_label": "yes"}


def classification(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0.8, "yes", "no")})


def regression(n: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "price": 3 * x + rng.normal(0, 1, n)})


def three_answers(n: int = 300, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.select([x < -0.5, x < 0.5], ["a", "b"], "c")})


class TestTheWrapper:
    def test_weighting_the_missed_cases_raises_the_probability_given_to_them(self):
        df = classification()
        X, y = df[["x", "z"]], df["outcome"]
        plain = LogisticRegression(class_weight=None).fit(X, y)
        weighted = CostWeighted(LogisticRegression(class_weight=None), **WEIGHTS).fit(X, y)
        yes = list(weighted.classes_).index("yes")
        assert weighted.predict_proba(X)[:, yes].mean() > plain.predict_proba(X)[:, yes].mean() + 0.05
        assert list(weighted.classes_) == list(plain.classes_)

    def test_equal_costs_change_nothing(self):
        df = classification()
        X, y = df[["x", "z"]], df["outcome"]
        plain = LogisticRegression().fit(X, y)
        even = CostWeighted(LogisticRegression(), positive_label="yes", cost_of_missing=3.0, cost_of_false_alarm=3.0)
        assert np.allclose(even.fit(X, y).predict_proba(X), plain.predict_proba(X), atol=1e-6)

    def test_its_source_is_self_contained(self):
        from app.services import cost_weighting

        namespace: dict = {}
        exec("import numpy as np\nfrom sklearn.base import BaseEstimator, ClassifierMixin\n" + cost_weighting.source(), namespace)
        assert namespace["CostWeighted"](LogisticRegression(), **WEIGHTS).get_params()["cost_of_missing"] == 8.0


class TestThePipeline:
    def test_a_classifier_is_wrapped_and_its_importance_reads_through(self):
        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, model_params={LR: {"C": 0.5, "cost_weighting": WEIGHTS}},
        )
        entry = result["results"][0]
        assert entry["failed"] is None and entry["tuned_params"]["cost_weighting"] == WEIGHTS
        assert entry["feature_importance"] and entry["feature_importance"][0]["feature"].endswith("x")

    def test_it_is_scored_as_a_classifier(self):
        """scikit-learn reads the estimator type from the mixin order; with BaseEstimator first the wrapper was
        not a classifier, the AUC scorer was handed both probability columns, and every fold scored nan."""
        from sklearn.base import is_classifier

        assert is_classifier(CostWeighted(LogisticRegression(), **WEIGHTS))
        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, model_params={LR: {"cost_weighting": WEIGHTS}},
        )
        entry = result["results"][0]
        assert 0.5 < entry["metrics"]["roc_auc"] <= 1
        from app.services import tuning

        df = classification()
        scored = tuning.score_config(
            lambda: modeling.build_pipeline(LR, ["x", "z"], [], model_params={"cost_weighting": WEIGHTS}),
            df[["x", "z"]], df["outcome"], "classification",
        )
        assert scored is not None and 0.5 < scored.score <= 1

    def test_a_forest_s_importance_is_not_read_through_its_template_tree(self):
        """A random forest has an `estimator_` too -- its unfitted template -- and unwrapping anything with one
        emptied every forest's importances. Only the cost-weighting wrapper is read through."""
        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[RF],
            include_deep_diagnostics=False,
        )
        importance = result["results"][0]["feature_importance"]
        assert importance and sum(f["importance"] for f in importance) == pytest.approx(1.0, abs=0.05)

    def test_a_weighted_forest_reads_through_the_wrapper(self):
        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[RF],
            include_deep_diagnostics=False, model_params={RF: {"cost_weighting": WEIGHTS}},
        )
        assert result["results"][0]["feature_importance"]

    def test_the_wrapper_is_built_with_the_model_s_own_settings(self):
        pipeline = modeling.build_pipeline(LR, ["x", "z"], [], model_params={"C": 0.5, "cost_weighting": WEIGHTS})
        model = pipeline.named_steps["model"]
        assert isinstance(model, CostWeighted) and model.estimator.C == 0.5 and model.cost_of_missing == 8.0

    def test_a_regressor_is_never_wrapped(self):
        pipeline = modeling.build_pipeline("linear_regression", ["x"], [], model_params={"cost_weighting": WEIGHTS})
        assert not isinstance(pipeline.named_steps["model"], CostWeighted)


class TestTheExport:
    def test_the_script_carries_the_wrapper_and_reproduces_the_accuracy(self, tmp_path):
        df = classification()
        result = modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, model_params={LR: {"C": 0.5, "cost_weighting": WEIGHTS}},
        )
        script_text = model_export.build_training_script(result, file_name="data.csv")
        ast.parse(script_text)
        assert "class CostWeighted(ClassifierMixin, BaseEstimator):" in script_text
        assert "CostWeighted(LogisticRegression(" in script_text and "cost_of_missing=8.0" in script_text
        assert "cost_weighting" not in script_text.split("class CostWeighted")[0].split("import")[-1]
        df.to_csv(tmp_path / "data.csv", index=False)
        (tmp_path / "train_model.py").write_text(script_text, encoding="utf-8")
        done = subprocess.run([sys.executable, "train_model.py", "data.csv"], capture_output=True, text=True,
                              cwd=tmp_path, timeout=600)
        assert done.returncode == 0, f"the exported script failed:\n{done.stdout}\n{done.stderr}"
        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match and float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)

    def test_an_unweighted_script_has_no_wrapper(self):
        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False,
        )
        assert "CostWeighted" not in model_export.build_training_script(result, file_name="data.csv")


class Controlled:
    """Makes training with the costs lower the expected cost (or not), so the decision is the only thing tested."""

    def __init__(self, monkeypatch, weighted: list[float] | None, plain: list[float] | None = None):
        self.calls: list[dict] = []
        plain = plain if plain is not None else [0.50, 0.52, 0.48, 0.51, 0.49]

        def fold_costs(build, X, y, **kwargs):
            is_weighted = isinstance(build().named_steps["model"], CostWeighted)
            self.calls.append({"weighted": is_weighted, **kwargs})
            return weighted if is_weighted else plain

        monkeypatch.setattr(improvement, "fold_expected_costs", fold_costs)


def improve(df=None, **kwargs):
    return improvement.improve(
        classification() if df is None else df, target_column="outcome", task="classification",
        model_names=[LR, RF], budget=4, **kwargs,
    )


def weighting_round(found):
    return next((r for r in found["rounds"] if r["name"] == "cost_weighting"), None)


class TestTheRound:
    def test_weighting_that_lowers_the_cost_on_every_fold_is_kept(self, monkeypatch):
        Controlled(monkeypatch, weighted=[0.30, 0.31, 0.29, 0.32, 0.30])
        found = improve(stated_costs=COSTS)
        round_ = weighting_round(found)
        assert round_["accepted"] is True and "so the model is trained with them" in round_["outcome"]
        assert round_["detail"]["paired_difference"]["low"] > 0
        assert found["best"]["params"]["cost_weighting"] == WEIGHTS
        # The final model, scored once on the test set, is the weighted one.
        assert isinstance(found["pipeline"].named_steps["model"], CostWeighted)

    def test_weighting_within_the_folds_disagreement_is_not_kept(self, monkeypatch):
        Controlled(monkeypatch, weighted=[0.40, 0.62, 0.45, 0.55, 0.50])
        found = improve(stated_costs=COSTS)
        round_ = weighting_round(found)
        assert round_["accepted"] is False and "the cut-off still uses your costs" in round_["outcome"]
        assert "cost_weighting" not in found["best"]["params"]
        assert not isinstance(found["pipeline"].named_steps["model"], CostWeighted)

    def test_folds_that_cannot_be_compared_keep_the_model_as_it_was(self, monkeypatch):
        Controlled(monkeypatch, weighted=None)
        round_ = weighting_round(improve(stated_costs=COSTS))
        assert round_["accepted"] is False and round_["detail"]["paired_difference"] is None
        assert "could not be compared" in round_["outcome"]

    def test_no_stated_costs_means_no_round(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=[0.1] * 5)
        assert weighting_round(improve()) is None and controlled.calls == []

    def test_a_target_with_three_answers_has_no_round(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=[0.1] * 5)
        found = improve(df=three_answers(), stated_costs=COSTS)
        assert weighting_round(found) is None and controlled.calls == []

    def test_a_regression_has_no_round(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=[0.1] * 5)
        found = improvement.improve(
            regression(), target_column="price", task="regression", model_names=["linear_regression"], budget=4,
            stated_costs=COSTS,
        )
        assert weighting_round(found) is None and controlled.calls == []

    def test_it_compares_on_the_runs_own_grouped_folds(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=None)
        df = classification().assign(shop=lambda d: np.arange(len(d)) % 30)
        improve(df=df, stated_costs=COSTS, split_strategy="group", group_column="shop")
        assert controlled.calls and all(call["groups"] is not None for call in controlled.calls)

    def test_it_compares_on_forward_in_time_folds_for_a_date_split(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=None)
        df = classification().assign(when=pd.date_range("2022-01-01", periods=300, freq="D").astype(str))
        improve(df=df, stated_costs=COSTS, split_strategy="time", time_column="when")
        assert controlled.calls and all(call["ordered"] for call in controlled.calls)

    def test_on_real_data_the_round_runs_and_reports_both_costs(self):
        round_ = weighting_round(improve(stated_costs=COSTS))
        assert round_ is not None
        assert round_["detail"]["weights"] == WEIGHTS
        assert round_["detail"]["expected_cost"]["without"] is not None
        assert round_["detail"]["expected_cost"]["with"] is not None
        assert "weighted by the cost you stated" in round_["what_it_tried"]


class TestTheFoldCosts:
    def test_each_fold_gets_the_cost_of_its_own_mistakes(self):
        df = classification()
        costs = improvement.fold_expected_costs(
            lambda: modeling.build_pipeline(LR, ["x", "z"], []), df[["x", "z"]], df["outcome"],
            positive_label="yes", cost_of_missing=8.0, cost_of_false_alarm=1.0,
        )
        assert costs is not None and len(costs) >= 3
        # The cost of predicting "no" for every row is the share of "yes" rows times 8; a model does better.
        assert np.mean(costs) < (df["outcome"] == "yes").mean() * 8.0


class TestTheRoute:
    def _session(self, client):
        buffer = io.StringIO()
        classification().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return sid

    def test_a_cost_objective_reaches_the_loop_and_a_kept_weighting_is_rebuilt_and_exported(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=[0.30, 0.31, 0.29, 0.32, 0.30])
        client = TestClient(app)
        sid = self._session(client)
        body = client.post(
            f"/api/model/{sid}/improve",
            json={
                "model_names": [LR, RF], "include_deep_diagnostics": False, "budget": 4,
                "operating_point": {"objective": "cost", "cost_of_missing": 8, "cost_of_false_alarm": 1,
                                    "positive_label": "yes"},
            },
        ).json()
        assert controlled.calls and controlled.calls[0]["cost_of_missing"] == 8.0
        chosen = next(r for r in body["results"] if r["model"] == body["best_model"])
        assert chosen["tuned_params"]["cost_weighting"] == WEIGHTS
        script = client.get(f"/api/model/{sid}/script")
        assert script.status_code == 200 and "CostWeighted(" in script.text

    def test_another_objective_states_no_costs(self, monkeypatch):
        controlled = Controlled(monkeypatch, weighted=[0.1] * 5)
        client = TestClient(app)
        sid = self._session(client)
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": [LR], "include_deep_diagnostics": False, "budget": 4,
                  "operating_point": {"objective": "f1"}},
        ).json()
        assert controlled.calls == [] and all(r["name"] != "cost_weighting" for r in body["improvement"]["rounds"])


class TestTheCard:
    def _card(self, model_params):
        from app.services import operating_point
        from app.services.model_card import build_model_card

        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, model_params=model_params,
            point=operating_point.OperatingPoint(
                objective="cost", cost_of_missing=8.0, cost_of_false_alarm=1.0, positive_label="yes"
            ),
        )
        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")

    def test_the_card_says_the_model_was_trained_with_the_costs(self):
        card = self._card({LR: {"cost_weighting": WEIGHTS}})
        assert "The model was also *trained* with these costs" in card

    def test_the_card_says_only_the_cut_off_uses_them(self):
        assert "only the cut-off uses them" in self._card(None)
