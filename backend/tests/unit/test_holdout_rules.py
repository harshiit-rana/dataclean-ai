"""What the user is told as the held-out rows get used (decision 0001, section 3).

Counting looks changes nothing unless the count reaches the user and qualifies the number it is
about. A score revealed for the fourth time on the same rows is not an independent estimate and
must not be presented as one -- in the headline, where it gets quoted, not only in a panel.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_card, model_store
from app.services.session_store import session_dir

client = TestClient(app)
RANDOM = {"strategy": "random"}
FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def _record(directory, n: int, split=None):
    for i in range(n):
        ledger.record_look(
            directory, version=None, split=split or RANDOM, route="train", run_id=str(i), metrics={}
        )


class TestTheStatus:
    def test_the_first_look_is_independent_and_says_nothing(self, tmp_path):
        status = ledger.evaluation_status(tmp_path, RANDOM)
        assert status["status"] == "independent"
        assert status["look_count"] == 1
        assert status["message"] == "" and status["qualification"] == ""

    def test_the_second_look_is_reused_and_states_the_count(self, tmp_path):
        _record(tmp_path, 1)
        status = ledger.evaluation_status(tmp_path, RANDOM)
        assert status["status"] == "reused" and status["look_count"] == 2
        assert "2 times" in status["message"]
        assert "no longer an independent estimate" in status["message"]

    def test_the_third_look_is_still_only_reused(self, tmp_path):
        _record(tmp_path, 2)
        assert ledger.evaluation_status(tmp_path, RANDOM)["status"] == "reused"

    def test_the_fourth_look_is_exhausted(self, tmp_path):
        _record(tmp_path, 3)
        status = ledger.evaluation_status(tmp_path, RANDOM)
        assert status["status"] == "exhausted" and status["look_count"] == 4
        assert "optimistic" in status["message"]
        assert "not an independent one" in status["message"]
        assert "Use this on a new file" in status["message"]

    def test_the_limit_is_the_last_look_that_is_merely_reused(self, tmp_path):
        assert ledger.evaluation_status(tmp_path, RANDOM)["limit"] == 3

    def test_further_looks_stay_exhausted(self, tmp_path):
        _record(tmp_path, 9)
        status = ledger.evaluation_status(tmp_path, RANDOM)
        assert status["status"] == "exhausted" and "10 times" in status["message"]

    def test_a_different_split_rule_starts_its_own_count(self, tmp_path):
        _record(tmp_path, 5, RANDOM)
        status = ledger.evaluation_status(tmp_path, {"strategy": "time", "time_column": "d"})
        assert status["status"] == "independent" and status["look_count"] == 1

    def test_but_it_says_the_rows_overlap_the_ones_already_seen(self, tmp_path):
        """A new rule is a different held-out set, not a clean slate."""
        _record(tmp_path, 2, RANDOM)
        status = ledger.evaluation_status(tmp_path, {"strategy": "time", "time_column": "d"})
        assert status["looks_on_other_splits"] == 2
        assert "2 models" in status["message"] and "overlap" in status["message"]

    def test_new_data_evaluations_do_not_advance_it(self, tmp_path):
        ledger.record(
            tmp_path, event=ledger.NEW_DATA, split=RANDOM, route="apply_answers",
            run_id="a", version=None, metrics={},
        )
        assert ledger.evaluation_status(tmp_path, RANDOM)["status"] == "independent"

    def test_the_wording_is_the_text_in_the_decision_record(self, tmp_path):
        """The ADR quotes these messages verbatim; they must not drift apart."""
        adr = (
            Path(__file__).resolve().parents[3] / "docs" / "decisions" / "0001-evaluation-protocol.md"
        ).read_text(encoding="utf-8")
        flat = re.sub(r"\s+", " ", adr.replace("**", ""))

        _record(tmp_path, 1)
        reused = ledger.evaluation_status(tmp_path, RANDOM)["qualification"]
        _record(tmp_path, 2)
        exhausted = ledger.evaluation_status(tmp_path, RANDOM)["qualification"]

        for message, count in ((reused, "N times"), (exhausted, "N times")):
            template = re.sub(r"\d+ times", "N times", message)
            template = template.replace("'Use this on a new file'", "'Use this on a new file'")
            assert template in flat, f"the ADR no longer contains: {template}"
            assert count in template


class TestTheHeadlineIsQualified:
    ASSESSMENT = {"verdict": "useful", "headline": "This model separates the two outcomes."}

    def test_an_independent_look_leaves_the_headline_alone(self):
        evaluation = {"status": "independent", "qualification": ""}
        assert ledger.qualify(self.ASSESSMENT, evaluation) == self.ASSESSMENT

    @pytest.mark.parametrize("status", ["reused", "exhausted"])
    def test_a_reused_or_exhausted_look_qualifies_it(self, status):
        evaluation = {"status": status, "qualification": "These rows have been used 3 times."}
        qualified = ledger.qualify(self.ASSESSMENT, evaluation)
        assert qualified["headline"].startswith("This model separates the two outcomes.")
        assert qualified["headline"].endswith("These rows have been used 3 times.")
        assert qualified["qualified_by_reuse"] is True

    def test_the_verdict_itself_is_never_changed(self):
        """Saying how much to trust a number is not a re-judging of the number."""
        evaluation = {"status": "exhausted", "qualification": "Optimistic."}
        assert ledger.qualify(self.ASSESSMENT, evaluation)["verdict"] == "useful"

    def test_the_input_is_not_mutated(self):
        original = dict(self.ASSESSMENT)
        ledger.qualify(original, {"status": "reused", "qualification": "x"})
        assert original == self.ASSESSMENT

    def test_a_missing_assessment_or_evaluation_is_survived(self):
        assert ledger.qualify(None, {"status": "reused", "qualification": "x"}) is None
        assert ledger.qualify(self.ASSESSMENT, None) == self.ASSESSMENT


def _session() -> tuple[str, object]:
    rng = np.random.default_rng(17)
    n = 300
    signal = rng.normal(0, 1, n)
    buffer = io.StringIO()
    pd.DataFrame(
        {
            "signal": signal.round(4),
            "noise": rng.normal(0, 1, n).round(4),
            "joined": pd.date_range("2021-01-01", periods=n, freq="D").astype(str),
            "outcome": np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"),
        }
    ).to_csv(buffer, index=False)
    session_id = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome",
            "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id, session_dir(get_settings().upload_dir, session_id)


def _train(session_id, **extra):
    response = client.post(f"/api/model/{session_id}/train", json={**FAST, **extra})
    assert response.status_code == 200, response.text
    return response.json()


class TestOverTheApi:
    def test_stepping_through_the_looks(self):
        session_id, _ = _session()

        first = _train(session_id)
        assert first["evaluation"]["status"] == "independent"
        assert first["evaluation"]["look_count"] == 1
        assert first["assessment"].get("qualified_by_reuse") is None  # no warning on look one

        second = _train(session_id)
        assert second["evaluation"]["status"] == "reused"
        assert "2 times" in second["evaluation"]["message"]
        assert second["assessment"]["qualified_by_reuse"] is True

        third = _train(session_id)
        assert third["evaluation"]["status"] == "reused"

        fourth = _train(session_id)
        assert fourth["evaluation"]["status"] == "exhausted"
        assert fourth["evaluation"]["look_count"] == 4
        # The qualification is in the headline itself, verbatim.
        assert fourth["evaluation"]["qualification"] in fourth["assessment"]["headline"]
        assert "optimistic" in fourth["assessment"]["headline"]

    def test_the_verdict_is_the_same_on_every_look(self):
        """The headline is qualified; the judgement of the number is not."""
        session_id, _ = _session()
        verdicts = {_train(session_id)["assessment"]["verdict"] for _ in range(4)}
        assert len(verdicts) == 1

    def test_changing_the_split_resets_the_status_for_the_new_split_only(self):
        session_id, _ = _session()
        for _ in range(4):
            _train(session_id)
        timed = _train(session_id, split_strategy="time", time_column="joined")
        assert timed["evaluation"]["status"] == "independent"
        assert timed["evaluation"]["looks_on_other_splits"] == 4
        assert "overlap" in timed["evaluation"]["message"]
        assert _train(session_id)["evaluation"]["look_count"] == 5  # the random count carried on

    def test_the_improvement_loop_counts_like_a_train(self):
        session_id, _ = _session()
        _train(session_id)
        improved = client.post(
            f"/api/model/{session_id}/improve", json={**FAST, "budget": 4}
        ).json()
        assert improved["evaluation"]["look_count"] == 2
        assert improved["evaluation"]["status"] == "reused"

    def test_the_stored_result_keeps_what_the_run_was_told(self):
        session_id, _ = _session()
        _train(session_id)
        second = _train(session_id)
        stored = client.get(f"/api/model/{session_id}/result").json()
        assert stored["evaluation"] == second["evaluation"]
        assert stored["assessment"]["headline"] == second["assessment"]["headline"]

    def test_reading_the_result_again_does_not_advance_the_count(self):
        session_id, _ = _session()
        _train(session_id)
        for _ in range(3):
            assert client.get(f"/api/model/{session_id}/result").json()["evaluation"]["look_count"] == 1

    def test_the_run_history_records_the_status_at_the_time_of_the_run(self):
        session_id, _ = _session()
        for _ in range(3):
            _train(session_id)
        runs = client.get(f"/api/model/{session_id}/history").json()["runs"]  # newest first
        assert [(r["evaluation"]["look_count"], r["evaluation"]["status"]) for r in runs] == [
            (3, "reused"), (2, "reused"), (1, "independent"),
        ]

    def test_the_model_card_records_the_status(self):
        session_id, _ = _session()
        _train(session_id)
        _train(session_id)
        card = client.get(f"/api/model/{session_id}/card").json()["markdown"]
        assert "Times these held-out rows have been used | 2 (reused)" in card
        assert "no longer an independent estimate" in card


class TestOlderRunsAreNotCalledIndependent:
    def test_a_result_from_before_the_ledger_says_not_recorded(self):
        assert model_card._looks_cell({"split": {}}) == "not recorded"

    def test_a_history_entry_without_a_status_has_none(self):
        entry = model_store.summarise({"results": [], "split": {}})
        assert entry["evaluation"] is None
