"""Products and ratios of numeric inputs, found by the improvement loop and kept only if they help.

A linear model cannot form a ratio or a product, and some relationships only show as one. The recipe is row-wise (so it is
computed before the split and cannot leak), is never proposed up front, and is rebuilt from the pair its id names.
"""

from __future__ import annotations

import io
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import features, improvement, model_export, modeling

LR = "logistic_regression"


def frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """The answer depends on x1 and x2 and, strongly, on their product: a linear model gets the main effects only."""
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    logit = 1.0 * x1 + 1.0 * x2 + 2.5 * x1 * x2
    return pd.DataFrame({
        "x1": x1, "x2": x2, "n0": rng.normal(size=n), "n1": rng.normal(size=n),
        "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "yes", "no"),
    })


def noise(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f"a{i}": rng.normal(size=n) for i in range(4)} | {"outcome": rng.choice(["yes", "no"], n)})


class TestTheRecipe:
    def test_a_product_and_a_ratio_of_two_numeric_columns(self):
        df = pd.DataFrame({"spend": [10.0, 20.0, 30.0], "visits": [2.0, 4.0, 5.0]})
        product = features.interaction_recipe(df, "product", "spend", "visits")
        ratio = features.interaction_recipe(df, "ratio", "spend", "visits")
        assert product.id == "interaction:product:spend x visits" and product.outputs == ["spend__x__visits"]
        assert ratio.id == "interaction:ratio:spend / visits" and ratio.outputs == ["spend__per__visits"]
        assert list(features.apply(df, [product])["spend__x__visits"]) == [20.0, 80.0, 150.0]
        assert list(features.apply(df, [ratio])["spend__per__visits"]) == [5.0, 5.0, 6.0]

    def test_a_zero_denominator_is_blank_never_infinite(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 2.0, 0.0]})
        out = features.apply(df, [features.interaction_recipe(df, "ratio", "a", "b")])["a__per__b"]
        assert out.isna().tolist() == [True, False, True] and not np.isinf(out.fillna(0)).any()

    def test_the_source_frame_is_left_alone(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        before = df.copy()
        features.apply(df, [features.interaction_recipe(df, "product", "a", "b")])
        pd.testing.assert_frame_equal(df, before)

    @pytest.mark.parametrize(
        "df,op,a,b",
        [
            (pd.DataFrame({"a": [1.0, 2.0]}), "product", "a", "a"),  # the same column
            (pd.DataFrame({"a": [1.0, 2.0]}), "product", "a", "missing"),  # not there
            (pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}), "product", "a", "b"),  # not numeric
            (pd.DataFrame({"a": [1.0, 2.0], "b": [5.0, 5.0]}), "product", "a", "b"),  # constant
            (pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 0.0]}), "ratio", "a", "b"),  # nothing to divide by
            (pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]}), "sum", "a", "b"),  # not an operation
        ],
    )
    def test_a_pair_that_cannot_make_one_does_not(self, df, op, a, b):
        assert features.interaction_recipe(df, op, a, b) is None

    def test_it_is_never_proposed_up_front(self):
        assert not [r for r in features.propose(frame(), "outcome") if r.kind == "interaction"]

    def test_its_script_lines_produce_the_same_columns_as_apply(self):
        df = frame()
        recipes = [features.interaction_recipe(df, "product", "x1", "x2"), features.interaction_recipe(df, "ratio", "x1", "x2")]
        zeros = df.assign(x2=lambda d: d.x2.where(d.index % 7 != 0, 0.0))  # some zero denominators
        applied = features.apply(zeros, recipes)
        scripted = zeros.copy()
        exec(features.as_pandas(recipes, frame="scripted"), {"pd": pd}, {"scripted": scripted})
        outputs = [o for r in recipes for o in r.outputs]
        pd.testing.assert_frame_equal(applied[outputs], scripted[outputs])
        assert not np.isinf(scripted[outputs].fillna(0)).any().any()


class TestRebuiltFromItsId:
    def test_the_id_names_the_pair_and_select_rebuilds_it(self):
        df = frame()
        found = features.select(df, "outcome", [], ["interaction:product:x1 x x2", "interaction:ratio:x1 / x2"])
        assert [r.id for r in found] == ["interaction:product:x1 x x2", "interaction:ratio:x1 / x2"]
        assert found[0].evidence == {"op": "product", "a": "x1", "b": "x2"}

    def test_a_stale_or_made_up_id_is_dropped_quietly(self):
        df = frame()
        assert features.select(df, "outcome", [], ["interaction:product:x1 x gone", "interaction:oops", "interaction:sum:x1 x x2"]) == []

    def test_a_pair_that_names_the_target_or_an_excluded_column_cannot_be_smuggled_in(self):
        df = frame().assign(outcome_score=lambda d: (d.outcome == "yes").astype(float))
        assert features.select(df, "outcome_score", [], ["interaction:product:x1 x outcome_score"]) == []
        assert features.select(df, "outcome", ["x2"], ["interaction:product:x1 x x2"]) == []

    def test_a_column_name_that_contains_the_separator_is_found(self):
        df = pd.DataFrame({"price x tax": [1.0, 2.0, 3.0], "visits": [1.0, 5.0, 2.0]})
        found = features.select(df, "outcome", [], ["interaction:product:price x tax x visits"])
        assert len(found) == 1 and found[0].evidence["a"] == "price x tax" and found[0].evidence["b"] == "visits"

    def test_a_recipe_saved_in_a_result_comes_back_through_the_dataclass(self):
        df = frame()
        original = features.interaction_recipe(df, "product", "x1", "x2")
        assert features.Recipe(**original.as_dict()) == original


def improve(df, **kwargs):
    return improvement.improve(df, target_column="outcome", task="classification", model_names=[LR], budget=4, **kwargs)


def combos_round(found):
    return next((r for r in found["rounds"] if r["name"] == "interactions"), None)


class TestInTheLoop:
    def test_a_product_that_helps_a_linear_model_is_kept_with_its_evidence(self):
        found = improve(frame())
        round_ = combos_round(found)
        assert round_["accepted"] is True
        kept = round_["detail"]["kept"]
        assert kept[0]["id"] == "interaction:product:x1 x x2" and kept[0]["gain"] > 0.05
        assert kept[0]["confirmation_score"] > kept[0]["incumbent_confirmation_score"]
        assert "x1__x__x2" in found["features"]["numeric"]
        assert "interaction:product:x1 x x2" in found["best"]["feature_recipes"]
        assert round_["detail"]["tried"] >= 1

    def test_the_model_with_it_scores_better_on_the_training_folds_than_without(self):
        found = improve(frame())
        with_it = combos_round(found)["score"]
        baseline = found["rounds"][0]["score"]
        assert with_it > baseline + 0.05

    def test_no_more_than_the_cap_are_kept_and_no_more_than_the_cap_are_first_tried(self):
        found = improve(frame())
        assert len(combos_round(found)["detail"]["kept"]) <= improvement.MAX_INTERACTIONS_KEPT
        assert improvement.MAX_INTERACTIONS_TRIED == 8

    @pytest.mark.parametrize("seed", range(3))
    def test_on_data_with_no_signal_nothing_is_kept(self, seed):
        round_ = combos_round(improve(noise(seed=seed)))
        assert round_ is not None and round_["accepted"] is False
        assert not round_["detail"].get("kept")

    def test_fewer_than_two_numeric_inputs_that_matter_means_nothing_to_combine(self):
        df = frame()[["x1", "outcome"]]
        assert combos_round(improve(df)) is None  # not even a round: there is only one numeric input

    def test_a_time_limit_that_has_passed_skips_it_and_says_so(self, monkeypatch):
        import sklearn.model_selection as selection

        from app.services import tuning

        class Clock:
            now = 1000.0

            def monotonic(self):
                return self.now

        clock = Clock()
        monkeypatch.setattr(improvement, "time", clock)
        monkeypatch.setattr(tuning, "time", clock)
        real = selection.cross_val_score

        def slow(*args, **kwargs):
            clock.now += 100.0
            return real(*args, **kwargs)

        monkeypatch.setattr(selection, "cross_val_score", slow)
        found = improve(frame(), time_limit_seconds=50)
        assert "interactions" in found["skipped_for_time"]
        assert combos_round(found)["detail"] == {"skipped": "time_limit"}


class TestRebuiltAndExported:
    def session(self):
        client = TestClient(app)
        buffer = io.StringIO()
        frame().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        return client, sid

    def test_the_rebuilt_model_uses_it_and_the_plan_shows_it_ticked_with_its_evidence(self):
        client, sid = self.session()
        body = client.post(
            f"/api/model/{sid}/improve",
            json={"model_names": [LR], "include_deep_diagnostics": False, "budget": 4},
        ).json()
        derived = body["features"]["derived"]
        assert any(r["id"] == "interaction:product:x1 x x2" for r in derived)
        assert "x1__x__x2" in body["features"]["numeric"]

        plan = client.get(f"/api/model/{sid}/plan").json()
        offered = {r["id"]: r for r in plan["suggested_features"]}
        assert "interaction:product:x1 x x2" in plan["feature_recipes"]  # ticked, as the run left it
        assert offered["interaction:product:x1 x x2"]["evidence"] == {"op": "product", "a": "x1", "b": "x2"}
        assert "x1__x__x2" in offered["interaction:product:x1 x x2"]["outputs"]

    def test_the_exported_script_builds_it_and_reproduces_the_score(self, tmp_path):
        df = frame()
        result = modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, feature_recipes_wanted=["interaction:product:x1 x x2"],
        )
        assert "x1__x__x2" in result["features"]["numeric"]
        script_text = model_export.build_training_script(result, file_name="data.csv")
        assert "x1__x__x2" in script_text and "_first * _second" in script_text
        csv = tmp_path / "data.csv"
        df.to_csv(csv, index=False)
        script = tmp_path / "train_model.py"
        script.write_text(script_text, encoding="utf-8")
        done = subprocess.run([sys.executable, str(script), str(csv)], capture_output=True, text=True, cwd=tmp_path, timeout=600)
        assert done.returncode == 0, f"the exported script failed:\n{done.stdout}\n{done.stderr}"
        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match and float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)
