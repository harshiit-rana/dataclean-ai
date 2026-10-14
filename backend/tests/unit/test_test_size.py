"""How much to hold back, chosen from the file instead of fixed at a fifth.

A fifth of 150 labelled rows is 30 test rows -- exactly the least a verdict can be given on -- and at
a 10% positive rate that is three positives, so "not enough test data" is the answer by construction.
On 200,000 rows a fifth is forty thousand, far more than anything needs, taken from training. The plan
now recommends a fraction with the reason, the user can change it, it is recorded in the split's
definition (so it is part of the test-set ledger's key), and the exported script cuts the same rows.
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
from app.services import evaluation_ledger, ml_readiness, model_export, modeling, validation_advice as advice
from app.models.graph_state import GoalContext

client = TestClient(app)


class TestTheRecommendation:
    def test_an_ordinary_file_keeps_the_usual_fifth(self):
        found = advice.recommend_test_size(1200)
        assert found["fraction"] == 0.2 and found["feasible"] is True
        assert found["test_rows"] == 240 and "enough to judge" in found["reason"]

    def test_a_small_lopsided_file_gets_a_larger_fraction_and_the_reason(self):
        found = advice.recommend_test_size(300, rarest=40)
        assert found["fraction"] == 0.3 and found["feasible"] is True
        assert "Raised to 30%" in found["reason"]
        assert "at least 10" in found["reason"] and "rarer answer" in found["reason"]
        assert found["test_rare_rows"] >= 10

    def test_a_file_at_the_edge_of_judgeable_gets_headroom(self):
        """A fifth of 150 rows is exactly 30: any row short and the verdict is withheld."""
        found = advice.recommend_test_size(150, rarest=75)
        assert found["fraction"] == 0.25 and found["test_rows"] >= 30 * 1.1

    def test_a_large_file_gets_a_smaller_fraction_and_the_reason(self):
        found = advice.recommend_test_size(200_000, rarest=20_000)
        assert found["fraction"] == 0.1 and "Lowered to 10%" in found["reason"]
        assert "better spent on training" in found["reason"]

    def test_a_middling_file_gets_something_between(self):
        assert advice.recommend_test_size(20_000, rarest=2_000)["fraction"] == 0.15

    def test_when_no_fraction_works_it_says_so_and_why(self):
        found = advice.recommend_test_size(150, rarest=15)
        assert found["feasible"] is False and found["fraction"] == advice.MAX_FRACTION
        assert "No fraction works" in found["reason"]
        assert "10 of the rarer answer" in found["reason"] and "more labelled rows" in found["reason"]

    def test_a_file_too_small_to_train_on_as_well_is_infeasible_too(self):
        found = advice.recommend_test_size(60)
        assert found["feasible"] is False and "too little to train on" in found["reason"]

    def test_nothing_labelled_is_infeasible(self):
        assert advice.recommend_test_size(0)["feasible"] is False

    @pytest.mark.parametrize("rows,rare", [(150, 75), (400, 60), (900, 90), (5000, 500), (80_000, 4000), (1200, None)])
    def test_the_fraction_stays_in_range_and_in_five_percent_steps(self, rows, rare):
        fraction = advice.recommend_test_size(rows, rare)["fraction"]
        assert advice.MIN_FRACTION <= fraction <= advice.MAX_FRACTION
        assert round(fraction * 20) == pytest.approx(fraction * 20)

    @pytest.mark.parametrize("rows,rare", [(300, 40), (450, 50), (1000, 100), (2000, 150)])
    def test_whenever_it_says_feasible_the_test_rows_can_be_judged(self, rows, rare):
        found = advice.recommend_test_size(rows, rare)
        if found["feasible"]:
            assert found["test_rows"] >= modeling.MIN_TEST_ROWS_TO_JUDGE
            assert found["test_rare_rows"] >= modeling.MIN_TEST_ROWS_PER_CLASS_TO_JUDGE

    def test_it_is_a_function_of_the_counts_only(self):
        """Not of which rows carry which answer, so the rows that end up held back cannot move it."""
        rng = np.random.default_rng(0)
        answers = pd.Series(np.where(rng.random(500) < 0.2, "yes", "no"))
        shuffled = answers.sample(frac=1, random_state=1).reset_index(drop=True)
        assert advice.recommend_for(answers, "classification") == advice.recommend_for(shuffled, "classification")

    def test_recommend_for_counts_the_rarer_answer_for_two_answers_only(self):
        two = pd.Series(["a"] * 90 + ["b"] * 10)
        three = pd.Series(["a"] * 90 + ["b"] * 5 + ["c"] * 5)
        numbers = pd.Series(np.arange(100.0))
        assert advice.recommend_for(two, "classification")["test_rare_rows"] is not None
        assert advice.recommend_for(three, "classification")["test_rare_rows"] is None
        assert advice.recommend_for(numbers, "regression")["test_rare_rows"] is None
        assert advice.recommend_for(pd.Series([None, None]), "classification") is None

    def test_the_thresholds_it_uses_are_the_verdicts_own(self):
        """If the verdict's bar moves, this follows it; they are the same numbers."""
        found = advice.recommend_test_size(150, rarest=75)
        assert str(modeling.MIN_TEST_ROWS_TO_JUDGE) in advice.recommend_test_size(200_000)["reason"]
        assert found["feasible"]


def frame(n: int = 900, seed: int = 0, rate: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    return pd.DataFrame({"x": x, "z": rng.normal(0, 1, n), "y": np.where(x + rng.normal(0, 1, n) > np.quantile(x, 1 - rate), "yes", "no")})


def train(df, **kwargs):
    return modeling.train_candidates(
        df, target_column="y", task="classification", model_names=["logistic_regression"],
        include_deep_diagnostics=False, **kwargs,
    )


class TestTheSplitUsesIt:
    def test_the_chosen_fraction_is_what_is_held_back(self):
        result = train(frame(1000), test_size=0.3)
        assert result["split"]["test_rows"] == 300 and result["split"]["train_rows"] == 700
        assert result["split"]["requested_test_size"] == 0.3

    def test_leaving_it_out_is_the_usual_fifth(self):
        result = train(frame(1000))
        assert result["split"]["requested_test_size"] == modeling.DEFAULT_TEST_SIZE
        assert result["split"]["test_rows"] == 200

    def test_a_date_split_cuts_where_the_fraction_says(self):
        rng = np.random.default_rng(2)
        n = 1000
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"day": pd.date_range("2021-01-01", periods=n).astype(str), "x": x,
                           "y": np.where(x + rng.normal(0, 1, n) > 0.3, "yes", "no")})
        result = train(df, split_strategy="time", time_column="day", test_size=0.3)
        assert result["split"]["test_rows"] == pytest.approx(300, abs=2)

    def test_a_grouped_split_uses_it(self):
        rng = np.random.default_rng(3)
        base = np.repeat(rng.normal(size=200), 6)
        df = pd.DataFrame({"customer": np.repeat(np.arange(200), 6), "x": base + rng.normal(0, 0.5, 1200),
                           "y": np.where(base + rng.normal(0, 0.5, 1200) > 0.2, "yes", "no")})
        result = train(df, split_strategy="group", group_column="customer", test_size=0.3)
        assert result["split"]["test_groups"] == pytest.approx(60, abs=3)

    def test_a_sealed_holdout_keeps_its_fixed_parts_and_ignores_it(self):
        result = train(frame(1500), sealed_holdout=True, test_size=0.3)
        assert result["split"]["sealed"] is True
        assert result["split"]["requested_test_size"] == modeling.SEALED_DEV_SIZE

    def test_the_fraction_is_part_of_the_ledger_key(self):
        a, b, default = train(frame(), test_size=0.3), train(frame(), test_size=0.25), train(frame())
        keys = {evaluation_ledger.split_key(r["split"]) for r in (a, b, default)}
        assert len(keys) == 3

    def test_omitting_it_keeps_the_key_it_always_had(self):
        explicit = train(frame(), test_size=modeling.DEFAULT_TEST_SIZE)
        assert evaluation_ledger.split_key(explicit["split"]) == evaluation_ledger.split_key(train(frame())["split"])

    def test_the_history_records_it(self):
        from app.services import model_store

        summary = model_store.summarise({**train(frame(), test_size=0.25), "run_id": "r"})
        assert summary["split"]["requested_test_size"] == 0.25


def session(df: pd.DataFrame) -> str:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id


BODY = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


class TestOverTheApi:
    def test_the_plan_recommends_a_fraction_with_the_reason(self):
        rng = np.random.default_rng(9)
        exact = pd.DataFrame({"x": rng.normal(size=300), "y": rng.permutation(["yes"] * 42 + ["no"] * 258)})
        plan = client.get(f"/api/model/{session(exact)}/plan").json()
        rec = plan["recommended_test_size"]
        assert rec["fraction"] > 0.2 and "Raised" in rec["reason"] and rec["feasible"] is True

    def test_the_plan_for_a_big_clean_file_recommends_less(self):
        plan = client.get(f"/api/model/{session(frame(60_000, rate=0.4))}/plan").json()
        assert plan["recommended_test_size"]["fraction"] == 0.1

    def test_train_accepts_it_and_holds_that_much_back(self):
        sid = session(frame(1000))
        body = client.post(f"/api/model/{sid}/train", json={**BODY, "test_size": 0.3}).json()
        assert body["split"]["test_rows"] == 300

    def test_a_silly_fraction_is_refused(self):
        sid = session(frame(400))
        assert client.post(f"/api/model/{sid}/train", json={**BODY, "test_size": 0.9}).status_code == 422
        assert client.post(f"/api/model/{sid}/train", json={**BODY, "test_size": 0.01}).status_code == 422

    def test_the_improvement_route_holds_back_the_same_fraction(self):
        sid = session(frame(800))
        body = client.post(f"/api/model/{sid}/improve", json={**BODY, "budget": 4, "test_size": 0.3}).json()
        assert body["split"]["test_rows"] == 240

    def test_the_run_history_shows_it(self):
        sid = session(frame(800))
        client.post(f"/api/model/{sid}/train", json={**BODY, "test_size": 0.25})
        runs = client.get(f"/api/model/{sid}/history").json()["runs"]
        assert runs[0]["split"]["requested_test_size"] == 0.25


class TestTheReadinessVerdict:
    GOAL = GoalContext(use_case="ml_training", expected_output="x", target_column="y", task_type="auto")

    def test_a_file_no_fraction_can_serve_says_so_before_anything_is_trained(self):
        small = frame(150, rate=0.1)
        report = ml_readiness.assess_readiness(small, self.GOAL)
        assert any("No fraction works" in w for w in report["warnings"])

    def test_an_ordinary_file_gets_no_such_warning(self):
        report = ml_readiness.assess_readiness(frame(1200), self.GOAL)
        assert not any("No fraction works" in w for w in report["warnings"])

    def test_it_is_a_warning_not_a_block_because_the_model_can_still_be_trained(self):
        report = ml_readiness.assess_readiness(frame(150, rate=0.1), self.GOAL)
        assert not any("No fraction works" in b for b in report["blockers"])


class TestTheExportedScript:
    def run(self, tmp_path, df, result):
        (tmp_path / "data.csv").write_text(df.to_csv(index=False), encoding="utf-8")
        script = tmp_path / "train_model.py"
        script.write_text(model_export.build_training_script(result, file_name="data.csv"), encoding="utf-8")
        done = subprocess.run([sys.executable, str(script), str(tmp_path / "data.csv")],
                              capture_output=True, text=True, cwd=tmp_path, timeout=600)
        assert done.returncode == 0, done.stderr
        return done.stdout

    def reproduces(self, tmp_path, df, **kwargs):
        result = train(df, **kwargs)
        result["features"]["excluded"] = []
        output = self.run(tmp_path, df, result)
        match = re.search(r"Accuracy:\s+([0-9.]+)", output)
        assert match, output
        assert float(match.group(1)) == pytest.approx(result["results"][0]["metrics"]["accuracy"], abs=1e-4)
        return result

    def test_a_non_default_fraction_is_cut_the_same_way_in_the_script(self, tmp_path):
        result = self.reproduces(tmp_path, frame(800), test_size=0.3)
        assert "test_size=0.3" in model_export.build_training_script(result)

    def test_a_date_split_with_a_non_default_fraction_reproduces(self, tmp_path):
        rng = np.random.default_rng(5)
        n = 800
        x = rng.normal(0, 1, n)
        df = pd.DataFrame({"day": pd.date_range("2021-01-01", periods=n).astype(str), "x": x,
                           "y": np.where(x + rng.normal(0, 1, n) > 0.3, "yes", "no")})
        self.reproduces(tmp_path, df, split_strategy="time", time_column="day", test_size=0.25)

    def test_the_default_script_is_unchanged(self):
        assert "test_size=0.2" in model_export.build_training_script(
            {**train(frame(600)), "features": {**train(frame(600))["features"], "excluded": []}}
        )
