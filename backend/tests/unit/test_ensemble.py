"""An averaging ensemble of the strongest models, kept only when it beats them, and exported like any other model.

Two models that make different mistakes can average to fewer than either. The improvement loop averages the incumbent
with the strongest runner-up (each with its searched settings), scores the average on the same folds, and keeps it only
under the gain rule, the fresh-fold check and the chance floor. The ensemble is a registry spec whose members are data,
so the app and the exported script build it from the same values.
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

from app.main import app
from app.services import improvement, model_export, modeling, tuning
from app.services.tuning import SearchResult

LR, RF, GB = "logistic_regression", "random_forest_classifier", "gradient_boosting_classifier"
VOTE, AVERAGE = "soft_voting_classifier", "averaging_regressor"
MEMBERS = [{"model": LR, "params": {"C": 0.3}}, {"model": RF, "params": {"max_depth": 6}}]


def classification(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


def regression(n: int = 300, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "price": 3 * x + np.abs(x) * 2 + rng.normal(0, 1, n)})


class TestTheSpec:
    def test_both_ensembles_are_registered_and_are_never_ordinary_candidates(self):
        assert modeling.MODELS[VOTE].ensemble and modeling.MODELS[AVERAGE].ensemble
        assert modeling.ENSEMBLE_FOR_TASK == {"classification": VOTE, "regression": AVERAGE}
        assert VOTE not in [s.name for s in modeling.models_for_task("classification")]
        assert AVERAGE not in [s.name for s in modeling.models_for_task("regression")]
        assert modeling.MODELS[VOTE].explains_features is False

    def test_it_builds_from_its_members_with_their_settings(self):
        built = modeling.MODELS[VOTE].build({"members": MEMBERS})
        assert built.voting == "soft"
        names = [name for name, _ in built.estimators]
        assert names == [LR, RF]
        assert built.estimators[0][1].C == 0.3 and built.estimators[1][1].max_depth == 6

    def test_the_script_source_builds_the_same_estimator(self):
        spec = modeling.MODELS[VOTE]
        source = spec.constructor_source({"members": MEMBERS})
        ast.parse(source)
        assert "VotingClassifier(estimators=[('logistic_regression', LogisticRegression(" in source
        assert "C=0.3" in source and "max_depth=6" in source and "voting='soft'" in source
        imports = spec.import_lines({"members": MEMBERS})
        assert "from sklearn.ensemble import VotingClassifier" in imports
        assert "from sklearn.linear_model import LogisticRegression" in imports
        assert "from sklearn.ensemble import RandomForestClassifier" in imports
        namespace: dict = {}
        exec(imports, namespace)
        rebuilt = eval(source, namespace)
        assert rebuilt.get_params()["voting"] == "soft"
        assert [n for n, _ in rebuilt.estimators] == [LR, RF]

    def test_its_default_members_let_it_stand_alone(self):
        assert [m["model"] for m in modeling.MODELS[VOTE].params["members"]] == [LR, RF]
        assert modeling.MODELS[VOTE].build() is not None

    def test_it_costs_what_its_members_cost(self):
        assert modeling.MODELS[VOTE].cost_weight >= modeling.MODELS[RF].cost_weight


def train(df, target, task, name, members):
    return modeling.train_candidates(
        df, target_column=target, task=task, model_names=[name],
        include_deep_diagnostics=False, model_params={name: {"members": members}},
    )


class TestTrainedAndExported:
    def test_a_classifier_ensemble_trains_scores_and_has_no_misleading_importance(self):
        result = train(classification(), "outcome", "classification", VOTE, MEMBERS)
        entry = result["results"][0]
        assert result["best_model"] == VOTE and entry["failed"] is None
        assert 0.5 < entry["metrics"]["roc_auc"] <= 1
        assert entry["feature_importance"] == [] and entry["column_importance"] == []

    def test_the_exported_script_reproduces_a_classifier_ensemble(self, tmp_path):
        df = classification()
        result = train(df, "outcome", "classification", VOTE, MEMBERS)
        script_text = model_export.build_training_script(result, file_name="data.csv")
        ast.parse(script_text)
        assert "VotingClassifier(estimators=[" in script_text
        df.to_csv(tmp_path / "data.csv", index=False)
        (tmp_path / "train_model.py").write_text(script_text, encoding="utf-8")
        done = subprocess.run([sys.executable, "train_model.py", "data.csv"], capture_output=True, text=True,
                              cwd=tmp_path, timeout=600)
        assert done.returncode == 0, f"the exported script failed:\n{done.stdout}\n{done.stderr}"
        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match and float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)

    def test_a_regression_ensemble_trains_and_exports(self, tmp_path):
        df = regression()
        members = [{"model": "linear_regression", "params": {}}, {"model": "random_forest_regressor", "params": {"max_depth": 6}}]
        result = train(df, "price", "regression", AVERAGE, members)
        assert result["results"][0]["metrics"]["r2"] > 0.5
        script_text = model_export.build_training_script(result, file_name="data.csv")
        ast.parse(script_text)
        assert "VotingRegressor(estimators=[" in script_text
        df.to_csv(tmp_path / "data.csv", index=False)
        (tmp_path / "train_model.py").write_text(script_text, encoding="utf-8")
        done = subprocess.run([sys.executable, "train_model.py", "data.csv"], capture_output=True, text=True,
                              cwd=tmp_path, timeout=600)
        assert done.returncode == 0, f"the exported script failed:\n{done.stdout}\n{done.stderr}"


class Controlled:
    """Makes the average score better (or not) than the single model, so the decision is the only thing tested."""

    def __init__(self, monkeypatch, better: bool, fresh_better: bool | None = None, fresh: float = 0.9,
                 single_fresh: float = 0.70):
        real = tuning.score_config
        self.ensemble_calls: list[dict] = []

        def score(build, X, y, task, **kwargs):
            model = build().named_steps["model"]
            if type(model).__name__.startswith("Voting"):
                self.ensemble_calls.append(kwargs)
                if kwargs.get("confirmation"):
                    good = better if fresh_better is None else fresh_better
                    return SearchResult(params={}, score=fresh if good else 0.70, std=0.01, metric="roc_auc", fits=5)
                return SearchResult(params={}, score=0.99 if better else 0.01, std=0.001, metric="roc_auc", fits=5)
            if kwargs.get("confirmation"):
                return SearchResult(params={}, score=single_fresh, std=0.01, metric="roc_auc", fits=5)
            return real(build, X, y, task, **kwargs)

        monkeypatch.setattr(tuning, "score_config", score)


def improve(df=None, models=(LR, RF), **kwargs):
    return improvement.improve(
        classification() if df is None else df, target_column="outcome", task="classification",
        model_names=list(models), budget=4, **kwargs,
    )


def ensemble_round(found):
    return next((r for r in found["rounds"] if r["name"] == "ensemble"), None)


class TestTheRound:
    def test_an_average_that_beats_and_holds_becomes_the_model_with_its_members(self, monkeypatch):
        Controlled(monkeypatch, better=True)
        found = improve()
        round_ = ensemble_round(found)
        assert round_["accepted"] is True and "It is the model now." in round_["outcome"]
        assert found["best"]["model"] == VOTE
        members = found["best"]["params"]["members"]
        assert {m["model"] for m in members} == {LR, RF}
        assert all(not k.startswith("model__") for m in members for k in m["params"])
        assert round_["detail"]["confirmation"]["status"] == "confirmed"
        # The final model, scored once on the test set, is the ensemble.
        assert type(found["pipeline"].named_steps["model"]).__name__ == "VotingClassifier"

    def test_an_average_that_is_no_better_leaves_the_single_model(self, monkeypatch):
        Controlled(monkeypatch, better=False)
        found = improve()
        round_ = ensemble_round(found)
        assert round_["accepted"] is False and found["best"]["model"] != VOTE
        assert "the single model stayed" in round_["outcome"]

    def test_an_average_that_does_not_hold_on_fresh_folds_is_refused(self, monkeypatch):
        Controlled(monkeypatch, better=True, fresh_better=False)
        round_ = ensemble_round(improve())
        assert round_["accepted"] is False and round_["detail"]["confirmation"]["status"] == "not_confirmed"

    def test_an_average_that_only_reaches_chance_is_refused(self, monkeypatch):
        # The single model overfits to below chance on fresh folds; the average only brings it up to chance.
        Controlled(monkeypatch, better=True, fresh=0.505, single_fresh=0.40)
        round_ = ensemble_round(improve())
        assert round_["accepted"] is False and round_["detail"]["confirmation"]["status"] == "at_chance"

    def test_one_family_means_no_round(self, monkeypatch):
        Controlled(monkeypatch, better=True)
        assert ensemble_round(improve(models=(LR,))) is None

    def test_it_scores_on_the_runs_own_grouped_folds(self, monkeypatch):
        controlled = Controlled(monkeypatch, better=False)
        df = classification().assign(shop=lambda d: np.arange(len(d)) % 30)
        improvement.improve(
            df, target_column="outcome", task="classification", model_names=[LR, RF], budget=4,
            split_strategy="group", group_column="shop",
        )
        assert controlled.ensemble_calls
        assert all(call.get("groups") is not None for call in controlled.ensemble_calls)

    def test_it_scores_on_forward_in_time_folds_for_a_date_split(self, monkeypatch):
        controlled = Controlled(monkeypatch, better=False)
        df = classification().assign(when=pd.date_range("2022-01-01", periods=300, freq="D").astype(str))
        improvement.improve(
            df, target_column="outcome", task="classification", model_names=[LR, RF], budget=4,
            split_strategy="time", time_column="when",
        )
        assert controlled.ensemble_calls and all(call.get("ordered") for call in controlled.ensemble_calls)

    def test_on_real_data_the_round_runs_and_says_what_it_averaged(self):
        round_ = ensemble_round(improve())
        assert round_ is not None
        assert {m["model"] for m in round_["detail"]["members"]} == {LR, RF}
        assert "Averaged" in round_["what_it_tried"]


class TestTheRebuild:
    def test_an_accepted_ensemble_is_what_the_route_rebuilds_and_exports(self, monkeypatch):
        Controlled(monkeypatch, better=True)
        client = TestClient(app)
        buffer = io.StringIO()
        classification().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": [LR, RF], "include_deep_diagnostics": False, "budget": 4},
        ).json()
        assert body["best_model"] == VOTE
        assert {m["model"] for m in body["results"][0]["tuned_params"]["members"]} == {LR, RF}
        script = client.get(f"/api/model/{sid}/script")
        assert script.status_code == 200 and "VotingClassifier(estimators=[" in script.text


class TestTheCard:
    def test_the_card_names_what_the_ensemble_averages(self):
        from app.services.model_card import build_model_card

        result = train(classification(), "outcome", "classification", VOTE, MEMBERS)
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "| Model chosen | Soft-voting ensemble (averaging logistic regression and random forest) |" in card

    def test_a_single_model_is_named_as_it_was(self):
        from app.services.model_card import build_model_card

        result = modeling.train_candidates(
            classification(), target_column="outcome", task="classification", model_names=[LR], include_deep_diagnostics=False,
        )
        card = build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")
        assert "| Model chosen | Logistic regression |" in card

