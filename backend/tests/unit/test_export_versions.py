"""The exported scripts and the model card say which software produced the numbers.

`train_model.py` promises to reproduce the reported metrics; a year later, with a newer scikit-learn, it might not,
and nothing said why. Both scripts now list the versions used, pin the install line to the same minor series, and
check the installed versions at start-up -- a warning, never a refusal. The card has an Environment section.
"""

from __future__ import annotations

import contextlib
import io
import subprocess
import sys

import numpy as np
import pandas as pd

from app.services import cleaning_export, model_export, modeling, run_metadata

KEYS = ("scikit_learn", "pandas", "numpy", "joblib")


def frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    return pd.DataFrame({"x": x, "z": rng.normal(size=n), "outcome": np.where(x + rng.normal(0, 1, n) > 0, "yes", "no")})


def trained(environment: dict | None = None) -> dict:
    result = modeling.train_candidates(
        frame(), target_column="outcome", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False,
    )
    if environment is not None:
        result["run_metadata"] = {"environment": environment, "seeds": run_metadata.seeds(), "config_hash": "abcd"}
    return result


def run_block(env: dict, keys=KEYS) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(run_metadata.versions_block(env, keys), {})
    return out.getvalue()


class TestTheHelper:
    def test_the_pin_allows_bug_fixes_of_the_same_minor_version(self):
        assert run_metadata.compatible_requirement("scikit-learn", "1.7.2") == "scikit-learn~=1.7.0"
        assert run_metadata.compatible_requirement("pandas", "2.2.0rc1") == "pandas~=2.2.0"

    def test_a_version_it_cannot_read_is_left_unpinned(self):
        assert run_metadata.compatible_requirement("numpy", None) == "numpy"
        assert run_metadata.compatible_requirement("numpy", "dev") == "numpy"

    def test_matching_versions_print_nothing(self):
        assert run_block(run_metadata.environment()) == ""

    def test_different_versions_print_a_note_and_carry_on(self):
        old = {**run_metadata.environment(), "scikit_learn": "0.24.2"}
        printed = run_block(old)
        assert "differ from the ones that produced the reported numbers" in printed
        assert "scikit-learn" in printed and "(the app used 0.24.2)" in printed

    def test_a_library_that_is_not_installed_is_named_not_fatal(self):
        env = {**run_metadata.environment(), "no_such_package_xyz": "1.0.0"}
        printed = run_block(env, keys=("no_such_package_xyz",))
        assert "no_such_package_xyz not installed (the app used 1.0.0)" in printed


class TestTheTrainingScript:
    def test_it_lists_the_run_s_own_versions_and_pins_the_install_line(self):
        env = {"python": "3.11.9", "scikit_learn": "1.5.1", "pandas": "2.1.4", "numpy": "1.26.4", "joblib": "1.4.0"}
        script = model_export.build_training_script(trained(env))
        assert "Produced with Python 3.11.9, scikit-learn 1.5.1, pandas 2.1.4, numpy 1.26.4, joblib 1.4.0." in script
        assert 'pip install "scikit-learn~=1.5.0" "pandas~=2.1.0" "numpy~=1.26.0" "joblib~=1.4.0"' in script
        assert "'scikit-learn': '1.5.1'," in script and "check_versions()" in script

    def test_a_result_from_before_the_record_uses_this_process(self):
        script = model_export.build_training_script(trained())
        assert f"scikit-learn {run_metadata.environment()['scikit_learn']}" in script

    def test_the_script_runs_and_only_warns_when_the_versions_differ(self, tmp_path):
        df = frame()
        result = trained({**run_metadata.environment(), "scikit_learn": "0.24.2"})
        df.to_csv(tmp_path / "data.csv", index=False)
        (tmp_path / "train_model.py").write_text(
            model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8"
        )
        done = subprocess.run([sys.executable, "train_model.py", "data.csv"], capture_output=True, text=True,
                              cwd=tmp_path, timeout=600)
        assert done.returncode == 0, done.stderr
        assert "(the app used 0.24.2)" in done.stdout and "Accuracy:" in done.stdout


class TestTheCleaningScript:
    def test_it_lists_the_versions_and_checks_them(self):
        script = cleaning_export.build_cleaning_script(frame(), [])
        env = run_metadata.environment()
        assert f"Produced with Python {env['python']}, pandas {env['pandas']}, numpy {env['numpy']}." in script
        assert f'"pandas~={".".join(env["pandas"].split(".")[:2])}.0"' in script
        assert "check_versions()" in script
        compile(script, "clean.py", "exec")


class TestTheCard:
    def _card(self, result):
        from app.services.model_card import build_model_card

        return build_model_card(session_id="s", file_name="f.csv", result=result, summary="A.", interpretation="B.")

    def test_the_card_has_an_environment_section(self):
        env = {"python": "3.11.9", "scikit_learn": "1.5.1", "pandas": "2.1.4", "numpy": "1.26.4", "joblib": "1.4.0"}
        card = self._card(trained(env))
        assert "## Environment" in card and "| scikit-learn | 1.5.1 |" in card
        assert "| Configuration | `abcd` |" in card and "| Seeds |" in card

    def test_a_run_from_before_the_record_says_not_recorded(self):
        card = self._card(trained())
        assert "## Environment" in card and "Not recorded" in card
