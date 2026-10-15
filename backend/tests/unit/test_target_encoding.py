"""Target encoding for high-cardinality columns, done so it cannot hand the answer to the model.

A postcode or a product code has too many values to one-hot encode and repeats enough to carry signal. Replacing each value
with the average answer for it keeps that signal, but computed naively over the rows the model is scored on it is the
answer itself. The encoding is scikit-learn's `TargetEncoder`, which fits each row's encoding on rows that do not include
it, inside the pipeline and so inside every training fold.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.services import features as feature_recipes
from app.services import model_export, modeling, tuning

LR = "logistic_regression"


def with_city(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """150 cities of ~10 rows each; the answer depends strongly on the city. The common values cover little, so the
    common-values recipe is not offered and target encoding is."""
    rng = np.random.default_rng(seed)
    cities = np.array([f"city_{i:03d}" for i in range(150)])
    lift = rng.normal(0, 1.6, len(cities))
    chosen = rng.integers(0, len(cities), n)
    probability = 1 / (1 + np.exp(-lift[chosen]))
    return pd.DataFrame({
        "city": cities[chosen],
        "noise": rng.normal(size=n),
        "outcome": np.where(rng.random(n) < probability, "yes", "no"),
    })


def random_ids(n: int = 1500, seed: int = 1) -> pd.DataFrame:
    """240 ids of ~6 rows each and an answer that has nothing to do with them."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": rng.choice([f"id_{i:03d}" for i in range(240)], n),
        "outcome": rng.choice(["yes", "no"], n),
    })


class TestWhichColumnsAreOffered:
    def test_a_high_cardinality_column_that_repeats_is_offered(self):
        recipes = feature_recipes.propose(with_city(), "outcome")
        found = [r for r in recipes if r.kind == "target_encoding"]
        assert [r.id for r in found] == ["target_encoding:city"]
        assert found[0].outputs == [] and found[0].evidence["distinct"] == 150
        assert "average answer" in found[0].reason and "grouped split" in found[0].reason

    def test_the_common_values_recipe_wins_where_it_applies_so_a_column_is_encoded_one_way(self):
        rng = np.random.default_rng(2)
        values = rng.choice([f"c{i}" for i in range(60)], 1500, p=np.r_[np.full(10, 0.06), np.full(50, 0.4 / 50)])
        df = pd.DataFrame({"city": values, "outcome": rng.choice(["yes", "no"], 1500)})
        kinds = [r.kind for r in feature_recipes.propose(df, "outcome") if r.column == "city"]
        assert kinds == ["frequent_categories"]

    def test_a_column_where_nearly_every_row_has_its_own_value_is_not_offered(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame({"code": [f"u{i}" for i in range(1500)], "outcome": rng.choice(["yes", "no"], 1500)})
        assert not [r for r in feature_recipes.propose(df, "outcome") if r.kind == "target_encoding"]

    def test_an_ordinary_categorical_numbers_dates_and_prose_are_not_offered(self):
        rng = np.random.default_rng(4)
        n = 1500
        df = pd.DataFrame({
            "plan": rng.choice(["a", "b", "c"], n),
            "amount": rng.normal(size=n),
            "when": pd.date_range("2022-01-01", periods=n, freq="h").astype(str),
            "outcome": rng.choice(["yes", "no"], n),
        })
        assert not [r for r in feature_recipes.propose(df, "outcome") if r.kind == "target_encoding"]

    def test_an_excluded_column_is_never_offered(self):
        assert not [r for r in feature_recipes.propose(with_city(), "outcome", excluded=["city"]) if r.kind == "target_encoding"]


class TestHowItIsClassifiedAndBuilt:
    def recipes(self):
        return feature_recipes.select(with_city(), "outcome", None, ["target_encoding:city"])

    def test_the_column_is_counted_with_the_pipeline_encoded_ones_and_marked_as_target_encoded(self):
        found = modeling.classify_features(with_city(), "outcome", None, self.recipes())
        assert found["frequent"] == ["city"] and found["target_encoded"] == ["city"]
        assert "city" not in {d["column"] for d in found["dropped"]}

    def test_without_the_recipe_the_column_is_dropped_for_having_too_many_values(self):
        found = modeling.classify_features(with_city(), "outcome", None, [])
        assert found["frequent"] == [] and "city" in {d["column"] for d in found["dropped"]}

    def test_the_pipeline_encodes_it_with_scikit_learns_target_encoder(self):
        from sklearn.preprocessing import OneHotEncoder, TargetEncoder

        pipeline = modeling.build_pipeline(LR, ["noise"], [], None, ["city"], target_encoded=["city"])
        steps = {name: step for name, step, _ in pipeline.named_steps["prepare"].transformers}
        assert isinstance(steps["target_encoded"].named_steps["encode"], TargetEncoder)
        assert "frequent" not in steps
        assert not any(isinstance(s, OneHotEncoder) for s in steps["target_encoded"].named_steps.values())

    def test_a_second_high_cardinality_column_can_still_be_one_hot_encoded_beside_it(self):
        pipeline = modeling.build_pipeline(LR, [], [], None, ["city", "region"], target_encoded=["city"])
        steps = {name: cols for name, _, cols in pipeline.named_steps["prepare"].transformers}
        assert steps["target_encoded"] == ["city"] and steps["frequent"] == ["region"]

    def test_a_pipeline_that_is_asked_for_nothing_target_encoded_is_what_it_was(self):
        pipeline = modeling.build_pipeline(LR, ["noise"], [], None, ["city"])
        assert [name for name, _, _ in pipeline.named_steps["prepare"].transformers] == ["numeric", "frequent"]


class TestItDoesNotLeak:
    """The test the issue asks for: a column of random ids gains no fold score from target encoding. It would if the
    encoding were worked out over rows the fold is scored on."""

    def cv_auc(self, df, **kwargs):
        X, y = df.drop(columns="outcome"), df["outcome"]
        build = lambda: modeling.build_pipeline(LR, [], [], None, ["id"], **kwargs)  # noqa: E731
        return tuning.score_config(build, X, y, "classification").score

    def test_random_ids_gain_nothing_from_the_encoding(self):
        assert self.cv_auc(random_ids(), target_encoded=["id"]) < 0.57

    def test_the_test_would_catch_a_leak_an_encoding_over_every_row_scores_far_above_chance(self):
        df = random_ids()
        df["leaky"] = df["id"].map(df.assign(y=(df["outcome"] == "yes").astype(float)).groupby("id")["y"].mean())
        build = lambda: modeling.build_pipeline(LR, ["leaky"], [], None)  # noqa: E731
        leaky = tuning.score_config(build, df[["leaky"]], df["outcome"], "classification").score
        assert leaky > 0.7

    def test_a_column_that_does_carry_the_answer_keeps_its_signal(self):
        df = with_city()
        X, y = df.drop(columns="outcome"), df["outcome"]
        encoded = tuning.score_config(
            lambda: modeling.build_pipeline(LR, ["noise"], [], None, ["city"], target_encoded=["city"]), X, y, "classification"
        ).score
        without = tuning.score_config(lambda: modeling.build_pipeline(LR, ["noise"], []), X[["noise"]], y, "classification").score
        assert encoded > 0.68 and encoded > without + 0.1

    def test_a_value_never_seen_in_training_is_encoded_without_failing(self):
        df = with_city()
        pipeline = modeling.build_pipeline(LR, ["noise"], [], None, ["city"], target_encoded=["city"])
        pipeline.fit(df.drop(columns="outcome"), df["outcome"])
        new = df.drop(columns="outcome").head(3).assign(city="somewhere_new")
        assert pipeline.predict_proba(new).shape == (3, 2)


class TestInATrain:
    def train(self, df, **kwargs):
        return modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, feature_recipes_wanted=["target_encoding:city"], **kwargs,
        )

    def test_the_column_is_used_and_its_importance_is_reported_under_its_own_name(self):
        result = self.train(with_city())
        assert result["features"]["target_encoded"] == ["city"] and result["features"]["declined_recipes"] == []
        assert [r["column"] for r in result["results"][0]["column_importance"]][0] == "city"
        assert result["results"][0]["metrics"]["roc_auc"] > 0.65

    @pytest.mark.parametrize("kwargs", [{"split_strategy": "group", "group_column": "noise_group"}, {"keep_identical_rows_together": True}])
    def test_it_is_declined_where_rows_are_grouped_and_the_result_says_so(self, kwargs):
        df = with_city().assign(noise_group=lambda d: np.arange(len(d)) // 5)
        result = self.train(df, **kwargs)
        assert result["features"]["target_encoded"] == []
        assert [d["id"] for d in result["features"]["declined_recipes"]] == ["target_encoding:city"]
        assert "does not know about the groups" in result["features"]["declined_recipes"][0]["reason"]

    def test_the_cautions_say_it_was_declined(self):
        from app.services import model_card

        df = with_city().assign(noise_group=lambda d: np.arange(len(d)) // 5)
        result = self.train(df, split_strategy="group", group_column="noise_group")
        best = result["results"][0]
        assert any(c["id"] == "declined_target_encoding:city" for c in model_card.caution_items(result, best))


class TestTheExportReproducesIt:
    def result(self):
        return modeling.train_candidates(
            with_city(), target_column="outcome", task="classification", model_names=[LR],
            include_deep_diagnostics=False, feature_recipes_wanted=["target_encoding:city"],
        )

    def test_the_script_builds_the_same_encoder(self):
        script = model_export.build_training_script(self.result())
        ast.parse(script)
        assert "TARGET_ENCODED_FEATURES = ['city']" in script
        assert "from sklearn.preprocessing import StandardScaler, TargetEncoder" in script
        assert "from sklearn.model_selection import StratifiedKFold" in script
        assert 'TargetEncoder(smooth="auto"' in script and "StratifiedKFold(n_splits=5, shuffle=True" in script
        assert f"random_state={modeling.RANDOM_STATE}" in script
        assert "OneHotEncoder" not in script  # nothing else needs it

    def test_a_script_without_it_is_as_it_was(self):
        result = modeling.train_candidates(
            with_city(), target_column="outcome", task="classification", model_names=[LR], include_deep_diagnostics=False,
        )
        script = model_export.build_training_script(result)
        assert "TARGET_ENCODED_FEATURES = []" in script and "TargetEncoder" not in script

    def test_both_encoders_side_by_side_are_exported(self):
        df = with_city().assign(region=lambda d: np.random.default_rng(9).choice([f"r{i}" for i in range(120)], len(d),
                                                                                  p=np.r_[np.full(10, 0.05), np.full(110, 0.5 / 110)]))
        result = modeling.train_candidates(
            df, target_column="outcome", task="classification", model_names=[LR], include_deep_diagnostics=False,
            feature_recipes_wanted=["target_encoding:city", "frequent_categories:region"],
        )
        script = model_export.build_training_script(result)
        ast.parse(script)
        assert "[c for c in FREQUENT_FEATURES if c not in TARGET_ENCODED_FEATURES]" in script
        assert "OneHotEncoder, StandardScaler, TargetEncoder" in script

    def test_the_script_runs_and_reproduces_the_score(self, tmp_path):
        result = self.result()
        csv = tmp_path / "data.csv"
        with_city().to_csv(csv, index=False)
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8")
        done = subprocess.run([sys.executable, str(script), str(csv)], capture_output=True, text=True, cwd=tmp_path, timeout=600)
        assert done.returncode == 0, f"the exported script failed:\n{done.stdout}\n{done.stderr}"
        match = re.search(r"Accuracy:\s+([0-9.]+)", done.stdout)
        assert match and float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)
