"""The exported training script.

The load-bearing test here is `test_the_exported_script_runs_and_reproduces_the_metrics`:
it writes the generated file to disk, runs it in a real subprocess against the same data, and
checks the numbers come back the same. Asserting that the text *contains* the right strings
would pass just as happily on a script that crashes on line one.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.services import features as feature_recipes, model_export, modeling


@pytest.fixture
def classification_data() -> pd.DataFrame:
    rng = np.random.default_rng(31)
    n = 400
    signal = rng.normal(0, 1, n)
    probability = 1 / (1 + np.exp(-2.3 * signal))
    return pd.DataFrame(
        {
            "signal": signal.round(4),
            "noise": rng.normal(0, 1, n).round(4),
            "region": rng.choice(["north", "south"], n),
            "outcome": np.where(rng.random(n) < probability, "yes", "no"),
        }
    )


@pytest.fixture
def regression_data() -> pd.DataFrame:
    rng = np.random.default_rng(32)
    n = 300
    size = rng.uniform(40, 240, n)
    return pd.DataFrame(
        {
            "size": size.round(3),
            "kind": rng.choice(["flat", "house"], n),
            "price": (size * 950 + rng.normal(0, 4000, n)).round(2),
        }
    )


def _train(df: pd.DataFrame, target: str, task: str, models: list[str]) -> dict:
    result = modeling.train_candidates(
        df, target_column=target, task=task, model_names=models,
        include_deep_diagnostics=False,
    )
    result["features"]["excluded"] = []
    return result


class TestScriptStructure:
    def test_the_script_is_valid_python(self, classification_data):
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        script = model_export.build_training_script(result)
        ast.parse(script)  # raises SyntaxError if not

    def test_the_constructor_comes_from_the_registry(self, classification_data):
        """The exported settings must be the settings that actually ran. Held to one source
        by construction -- both are rendered from the same ModelSpec."""
        result = _train(classification_data, "outcome", "classification",
                        ["random_forest_classifier"])
        script = model_export.build_training_script(result)
        spec = modeling.MODELS["random_forest_classifier"]
        assert spec.constructor_source() in script
        assert spec.import_line in script

    def test_every_registered_model_exports_valid_python(self, classification_data, regression_data):
        for name, spec in modeling.MODELS.items():
            data, target = (
                (classification_data, "outcome")
                if spec.task == "classification"
                else (regression_data, "price")
            )
            result = _train(data, target, spec.task, [name])
            ast.parse(model_export.build_training_script(result))

    def test_preprocessing_stays_inside_the_pipeline(self, classification_data):
        """If the script imputed or scaled the dataframe before splitting it would leak, and
        the exported numbers would stop matching the app's."""
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        script = model_export.build_training_script(result)
        tree = ast.parse(script)

        # find the assignment to `pipeline` and confirm the imputer sits within it
        pipeline_source = next(
            ast.get_source_segment(script, node)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "pipeline" for t in node.targets)
        )
        assert "SimpleImputer" in pipeline_source
        assert "StandardScaler" in pipeline_source

    def test_a_time_split_is_reproduced_as_a_time_split(self, classification_data):
        classification_data["joined"] = pd.date_range(
            "2021-01-01", periods=len(classification_data), freq="D"
        ).astype(str)
        result = modeling.train_candidates(
            classification_data, target_column="outcome", task="classification",
            model_names=["logistic_regression"], split_strategy="time", time_column="joined",
            include_deep_diagnostics=False,
        )
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        ast.parse(script)
        assert "train_test_split" not in script
        assert "sort_values" in script and "joined" in script

    def test_columns_left_out_are_documented_with_their_reason(self, classification_data):
        classification_data["row_id"] = [f"R{i}" for i in range(len(classification_data))]
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        script = model_export.build_training_script(result)
        assert "row_id" in script
        assert "too many" in script  # the reason it was dropped, not just the name

    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="registry"):
            model_export.build_training_script({"best_model": "wishful_thinking"})

    def test_the_prediction_snippet_is_valid_python(self, classification_data):
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        ast.parse(model_export.build_prediction_snippet(result))


class TestScriptActuallyRuns:
    """Compiling is not running. These execute the generated file in a real subprocess."""

    def _run(self, tmp_path, df, result, filename="data.csv"):
        csv_path = tmp_path / filename
        df.to_csv(csv_path, index=False)
        script_path = tmp_path / "train_model.py"
        script_path.write_text(
            model_export.build_training_script(result, file_name=filename), encoding="utf-8"
        )
        completed = subprocess.run(
            [sys.executable, str(script_path), str(csv_path)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            timeout=600,
        )
        assert completed.returncode == 0, (
            f"the exported script failed:\n{completed.stdout}\n{completed.stderr}"
        )
        return completed.stdout

    def test_the_exported_script_runs_and_reproduces_the_metrics(self, tmp_path, classification_data):
        """The claim the export makes. If the numbers differ, the app is reporting something
        the user cannot reproduce, and one of the two is lying."""
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        output = self._run(tmp_path, classification_data, result)

        reported = result["results"][0]["metrics"]["accuracy"]
        # The script prints the exact figure alongside classification_report's 2dp table,
        # precisely so this comparison can be made to full precision.
        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, f"no accuracy line in the script's output:\n{output}"
        assert float(match.group(1)) == pytest.approx(reported, abs=1e-4)

    def test_the_regression_script_reproduces_r2(self, tmp_path, regression_data):
        result = _train(regression_data, "price", "regression", ["linear_regression"])
        output = self._run(tmp_path, regression_data, result)

        match = re.search(r"R2:\s+([0-9.\-]+)", output)
        assert match, f"no R2 line in the script's output:\n{output}"
        assert float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["r2"], abs=0.005
        )

    def test_the_script_saves_a_loadable_model(self, tmp_path, classification_data):
        result = _train(classification_data, "outcome", "classification", ["logistic_regression"])
        self._run(tmp_path, classification_data, result)

        saved = tmp_path / "model.joblib"
        assert saved.exists()

        import joblib

        pipeline = joblib.load(saved)
        columns = result["features"]["numeric"] + result["features"]["categorical"]
        predictions = pipeline.predict(classification_data[columns].head(5))
        assert len(predictions) == 5

    def test_the_script_handles_rows_with_no_answer(self, tmp_path, classification_data):
        """Real exports have gaps in the target column. The script drops those rows rather
        than crashing or, worse, inventing labels."""
        classification_data.loc[:19, "outcome"] = None
        result = _train(classification_data.dropna(subset=["outcome"]), "outcome",
                        "classification", ["logistic_regression"])
        self._run(tmp_path, classification_data, result)  # must not raise


class TestGroupedSplitIsExported:
    """A grouped run whose script silently reverted to a random split would print a *higher*
    accuracy than the app reported -- the leak reappearing in the one artifact the user takes
    away to prove the number. Worse than an error, because it looks like success."""

    @staticmethod
    def _repeated_entities(customers: int = 80, per: int = 12) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            {
                "customer_id": np.repeat([f"CUST-{i:03d}" for i in range(customers)], per),
                "account_age_days": np.repeat(
                    rng.integers(100, 9000, customers), per
                ).astype(float),
                "spend": rng.normal(50, 15, customers * per).round(2),
                "outcome": np.repeat(rng.choice(["yes", "no"], customers), per),
            }
        )

    def _result(self, df: pd.DataFrame) -> dict:
        result = modeling.train_candidates(
            df,
            target_column="outcome",
            task="classification",
            model_names=["logistic_regression"],
            split_strategy="group",
            group_column="customer_id",
            include_deep_diagnostics=False,
        )
        result["features"]["excluded"] = []
        return result

    def test_the_script_groups_rather_than_splitting_at_random(self):
        script = model_export.build_training_script(self._result(self._repeated_entities()))
        ast.parse(script)
        assert "GroupShuffleSplit" in script
        assert "train_test_split" not in script
        assert "customer_id" in script

    def test_it_says_why_in_the_script(self):
        """Someone reading the exported file should learn the reason, not just the mechanism.
        A grouped split is the kind of choice that gets 'simplified' away by the next person."""
        script = model_export.build_training_script(self._result(self._repeated_entities()))
        assert "recognising the entity" in script

    def test_the_grouped_script_runs_and_reproduces_the_metrics(self, tmp_path):
        df = self._repeated_entities()
        result = self._result(df)
        output = TestScriptActuallyRuns()._run(tmp_path, df, result)

        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, f"no accuracy line in the script's output:\n{output}"
        assert float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["accuracy"], abs=1e-4
        )


class TestTheCutoffReachesTheScript:
    """A model reported at a cut-off of 0.35 that exports as a script answering at 0.50 is a
    model the export cannot reproduce."""

    def _train_at(self, df, point):
        from app.services.operating_point import OperatingPoint

        result = modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["logistic_regression"], include_deep_diagnostics=False,
            point=OperatingPoint(*point),
        )
        result["features"]["excluded"] = []
        return result

    def test_the_default_leaves_the_script_alone(self, classification_data):
        result = self._train_at(classification_data, ("default",))
        script = model_export.build_training_script(result)
        assert "THRESHOLD" not in script

    def test_the_chosen_line_is_a_declared_constant(self, classification_data):
        result = self._train_at(classification_data, ("cost", 5, 1))
        script = model_export.build_training_script(result)
        ast.parse(script)
        assert f"THRESHOLD = {result['operating_point']['threshold']}" in script
        assert "POSITIVE_LABEL = 'yes'" in script
        assert "a missed case costs 5x a false alarm" in script

    def test_the_script_reproduces_the_recall_at_the_cutoff(self, tmp_path, classification_data):
        result = self._train_at(classification_data, ("min_recall", 1, 1, 0.9))
        runner = TestScriptActuallyRuns()
        output = runner._run(tmp_path, classification_data, result)

        match = re.search(r"recall ([0-9.]+)", output)
        assert match, f"no recall-at-cutoff line in the script's output:\n{output}"
        assert float(match.group(1)) == pytest.approx(
            result["operating_point"]["observed"]["recall"], abs=1e-4
        )

    def test_the_prediction_snippet_thresholds_the_probability(self, classification_data):
        """The joblib file holds the model, not the line. A snippet calling predict() would
        quietly answer at 0.50 and discard the trade-off the model was reported against."""
        result = self._train_at(classification_data, ("cost", 5, 1))
        snippet = model_export.build_prediction_snippet(result)
        ast.parse(snippet)
        assert "predict_proba" in snippet
        assert str(result["operating_point"]["threshold"]) in snippet
        assert "pipeline.predict(" not in snippet


class TestDerivedInputsReachTheScript:
    """A recipe the export cannot rebuild is a model trapped in the app."""

    @pytest.fixture
    def derivable(self) -> pd.DataFrame:
        rng = np.random.default_rng(41)
        n = 400
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        words = ["the", "customer", "called", "about", "billing", "again", "today"]
        return pd.DataFrame(
            {
                "signal": rng.normal(0, 1, n).round(4),
                "when": dates.astype(str),
                "notes": [" ".join(rng.choice(words, size=int(rng.integers(6, 16)))) for _ in range(n)],
                "spend": np.where(rng.random(n) < 0.2, np.nan, rng.normal(50, 10, n).round(2)),
                "outcome": np.where(dates.weekday >= 5, "yes", "no"),
            }
        )

    def _train(self, df, recipes):
        result = modeling.train_candidates(
            df, target_column="outcome", task="classification",
            model_names=["random_forest_classifier"], include_deep_diagnostics=False,
            feature_recipes_wanted=recipes,
        )
        result["features"]["excluded"] = []
        return result

    def test_the_derived_lines_are_in_the_script_with_their_reasons(self, derivable):
        result = self._train(derivable, ["date_parts:when", "text_length:notes", "missing_flag:spend"])
        script = model_export.build_training_script(result)
        ast.parse(script)
        assert "when__weekday" in script
        assert "notes__length" in script
        assert "spend__missing" in script
        assert "Derived inputs" in script
        assert "how long ago" in script  # the reason, not just the mechanics

    def test_the_script_runs_and_reproduces_the_score_with_them(self, tmp_path, derivable):
        result = self._train(derivable, ["date_parts:when"])
        output = TestScriptActuallyRuns()._run(tmp_path, derivable, result)
        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, f"no accuracy line in the script's output:\n{output}"
        assert float(match.group(1)) == pytest.approx(
            result["results"][0]["metrics"]["accuracy"], abs=1e-4
        )

    def test_a_pooled_column_is_encoded_the_same_way_in_the_script(self, derivable):
        derivable["city"] = np.random.default_rng(3).choice(
            [f"c{i}" for i in range(60)], len(derivable),
            p=np.r_[np.full(10, 0.06), np.full(50, 0.4 / 50)],
        )
        result = self._train(derivable, ["frequent_categories:city"])
        assert result["features"]["frequent"] == ["city"]
        script = model_export.build_training_script(result)
        ast.parse(script)
        assert "FREQUENT_FEATURES = ['city']" in script
        assert "infrequent_if_exist" in script
        assert f"max_categories={feature_recipes.MAX_FREQUENT_CATEGORIES}" in script

    def test_nothing_derived_leaves_the_script_as_it_was(self, derivable):
        script = model_export.build_training_script(self._train(derivable, []))
        assert "Derived inputs" not in script
        assert "FREQUENT_FEATURES = []" in script

    def test_the_snippet_says_the_inputs_have_to_be_rebuilt(self, derivable):
        """A caller handed raw rows would be missing half the columns the model expects."""
        result = self._train(derivable, ["date_parts:when"])
        snippet = model_export.build_prediction_snippet(result)
        ast.parse(snippet)
        assert "derived" in snippet
        assert "when__weekday" in snippet
