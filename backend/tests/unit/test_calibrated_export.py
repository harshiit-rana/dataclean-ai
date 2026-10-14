"""A calibrated model has to export as a script that rebuilds it.

The export's central promise is that `train_model.py` reproduces the reported numbers. A model
the app calibrated, exported as a script that fits the plain pipeline, would print different
probabilities -- and a different Brier score and a different answer at 0.50 -- while looking like
success. These run the generated file in a real subprocess for each way the rows can be split,
because the calibration folds follow the split and a script that rebuilt them differently would
fit a different mapping.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from app.services import model_export, modeling
from app.services.operating_point import OperatingPoint

FOREST = "random_forest_classifier"
LOGISTIC = "logistic_regression"


def lopsided(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "x": x.round(4),
            "z": rng.normal(0, 1, n).round(4),
            "region": rng.choice(["north", "south", "east"], n),
            "outcome": np.where(x + rng.normal(0, 1, n) > 0.5, "yes", "no"),
        }
    )


def entities(customers: int = 220, per: int = 7, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.repeat(rng.normal(0, 1, customers), per)
    return pd.DataFrame(
        {
            "customer": np.repeat([f"c{i:03d}" for i in range(customers)], per),
            "x": (base + rng.normal(0, 0.6, customers * per)).round(4),
            "outcome": np.where(base + rng.normal(0, 0.5, customers * per) > 0.3, "yes", "no"),
        }
    )


def dated(n: int = 1400, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "day": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "x": x.round(4),
            "outcome": np.where(x + rng.normal(0, 1, n) > 0.4, "yes", "no"),
        }
    )


def with_copies(seed: int = 5) -> pd.DataFrame:
    base = lopsided(1000, seed)
    return pd.concat([base, base.iloc[:300]], ignore_index=True)


def train(df, method="sigmoid", model=FOREST, **kwargs):
    result = modeling.train_candidates(
        df, target_column="outcome", task="classification", model_names=[model],
        include_deep_diagnostics=False, calibration=method, **kwargs,
    )
    result["features"]["excluded"] = []
    return result


def run(tmp_path, df, result):
    """The generated script, run for real against the same data. Returns its output."""
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    script_path = tmp_path / "train_model.py"
    script_path.write_text(model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script_path), str(csv_path)],
        capture_output=True, text=True, cwd=tmp_path, timeout=900,
    )
    assert completed.returncode == 0, f"the exported script failed:\n{completed.stdout}\n{completed.stderr}"
    return completed.stdout


def printed(output: str, label: str) -> float:
    match = re.search(rf"{label}:\s+([0-9.]+)", output)
    assert match, f"no '{label}' line in the script's output:\n{output}"
    return float(match.group(1))


def chosen(result):
    return next(r for r in result["results"] if r["model"] == result["best_model"])


def reproduces(tmp_path, df, result):
    output = run(tmp_path, df, result)
    metrics = chosen(result)["metrics"]
    assert printed(output, "Accuracy") == pytest.approx(metrics["accuracy"], abs=1e-4)
    assert printed(output, "Brier score") == pytest.approx(metrics["brier_score"], abs=1e-4)
    return output


class TestTheScriptRebuildsTheCalibratedModel:
    def test_sigmoid_reproduces_the_accuracy_and_the_brier_score(self, tmp_path):
        df = lopsided()
        result = train(df)
        assert result["calibration"]["method"] == "sigmoid"
        reproduces(tmp_path, df, result)

    def test_isotonic_reproduces_them_too(self, tmp_path):
        df = lopsided(2000)
        result = train(df, "isotonic")
        assert result["calibration"]["method"] == "isotonic"
        reproduces(tmp_path, df, result)

    def test_a_grouped_split_reproduces_them_with_the_same_folds(self, tmp_path):
        df = entities()
        result = train(df, split_strategy="group", group_column="customer", model=LOGISTIC)
        assert result["calibration"]["method"] == "sigmoid"
        reproduces(tmp_path, df, result)

    def test_a_date_split_reproduces_them_forward_in_time(self, tmp_path):
        df = dated()
        result = train(df, split_strategy="time", time_column="day")
        assert result["calibration"]["method"] == "sigmoid" and result["calibration"]["ensemble"] is True
        reproduces(tmp_path, df, result)

    def test_copies_kept_together_reproduce_them_with_copies_kept_together_in_the_folds(self, tmp_path):
        df = with_copies()
        result = train(df, model=LOGISTIC, keep_identical_rows_together=True)
        assert result["split"]["keep_duplicates_together"] is True
        reproduces(tmp_path, df, result)

    def test_a_sealed_holdout_changes_nothing_about_it(self, tmp_path):
        df = lopsided(2200)
        result = train(df, sealed_holdout=True)
        assert result["split"]["sealed"] is True and result["calibration"]["method"] == "sigmoid"
        reproduces(tmp_path, df, result)

    def test_the_recall_at_a_chosen_cutoff_is_reproduced_on_calibrated_probabilities(self, tmp_path):
        df = lopsided()
        result = train(df, point=OperatingPoint("min_recall", 1, 1, 0.8))
        output = run(tmp_path, df, result)
        match = re.search(r"recall ([0-9.]+)", output)
        assert match, output
        assert float(match.group(1)) == pytest.approx(result["operating_point"]["observed"]["recall"], abs=1e-4)

    def test_the_model_it_saves_is_the_calibrated_one(self, tmp_path):
        df = lopsided()
        result = train(df)
        run(tmp_path, df, result)
        saved = joblib.load(tmp_path / "model.joblib")
        assert isinstance(saved, CalibratedClassifierCV) and saved.method == "sigmoid"
        assert saved.predict_proba(df[["x", "z", "region"]].iloc[:20]).shape == (20, 2)


class TestWhatTheScriptSays:
    def script(self, df=None, **kwargs):
        text = model_export.build_training_script(train(df if df is not None else lopsided(), **kwargs))
        ast.parse(text)
        return text

    def test_the_method_and_the_fold_rule_are_declared(self):
        script = self.script()
        assert "CALIBRATION_METHOD = 'sigmoid'" in script
        assert "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)" in script
        assert "ensemble=False" in script
        assert "TRAINING rows only" in script

    def test_a_grouped_script_uses_whole_group_folds(self):
        script = self.script(entities(), split_strategy="group", group_column="customer", model=LOGISTIC)
        assert "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)" in script
        assert "groups.loc[X_train.index]" in script
        assert "StratifiedKFold" not in script

    def test_a_dated_script_uses_forward_folds_and_the_averaged_form(self):
        script = self.script(dated(), split_strategy="time", time_column="day")
        assert "TimeSeriesSplit(n_splits=5)" in script and "ensemble=True" in script

    def test_a_copies_script_groups_the_folds_by_the_copy_key(self):
        script = self.script(with_copies(), model=LOGISTIC, keep_identical_rows_together=True)
        assert "duplicate_key.loc[X_train.index]" in script and "StratifiedGroupKFold" in script

    def test_it_imports_only_what_it_uses(self):
        """The existing rule for the script, held for every variant: an unused import in a file
        people are meant to read is a small lie about what it does."""
        variants = [
            self.script(),
            self.script(entities(), split_strategy="group", group_column="customer", model=LOGISTIC),
            self.script(dated(), split_strategy="time", time_column="day"),
            self.script(with_copies(), model=LOGISTIC, keep_identical_rows_together=True),
            self.script(method="none"),
        ]
        for text in variants:
            tree = ast.parse(text)
            imported = {
                (alias.asname or alias.name).split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names
            }
            used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
                node.value.id for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            }
            assert not (imported - used), f"unused imports: {imported - used}"

    def test_only_the_splitter_actually_used_is_imported(self):
        script = self.script()
        assert "import StratifiedKFold" in script
        assert "GroupKFold" not in script and "TimeSeriesSplit" not in script

    def test_the_uncalibrated_script_has_no_calibration_in_it(self):
        script = self.script(method="none")
        assert "CalibratedClassifierCV" not in script and "base_pipeline" not in script
        assert "KFold" not in script

    def test_every_two_class_script_prints_the_brier_score_it_can_be_checked_against(self, tmp_path):
        df = lopsided()
        result = train(df, method="none", model=LOGISTIC)
        output = run(tmp_path, df, result)
        assert printed(output, "Brier score") == pytest.approx(chosen(result)["metrics"]["brier_score"], abs=1e-4)

    def test_a_declined_request_exports_as_the_plain_model(self):
        script = self.script(lopsided(500), method="isotonic")  # too few rows: declined
        assert "CalibratedClassifierCV" not in script

    def test_multi_class_and_regression_scripts_are_untouched(self):
        rng = np.random.default_rng(8)
        x = rng.normal(0, 1, 400)
        three = pd.DataFrame({"x": x, "outcome": np.digitize(x + rng.normal(0, 1, 400), [-0.5, 0.5]).astype(str)})
        result = modeling.train_candidates(
            three, target_column="outcome", task="classification", model_names=[LOGISTIC],
            include_deep_diagnostics=False, calibration="sigmoid",
        )
        result["features"]["excluded"] = []
        script = model_export.build_training_script(result)
        assert "Calibrated" not in script and "Brier" not in script and "brier" not in script


class TestThePredictionSnippet:
    def test_it_says_the_probabilities_are_calibrated_and_what_that_does_to_the_default_line(self):
        snippet = model_export.build_prediction_snippet(train(lopsided()))
        ast.parse(snippet)
        assert "calibrated (sigmoid)" in snippet and "fitted on the training rows only" in snippet
        assert "0.50 line" in snippet

    def test_an_uncalibrated_model_has_no_such_note(self):
        assert "calibrated" not in model_export.build_prediction_snippet(train(lopsided(), method="none"))
