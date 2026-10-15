"""Newly labelled rows as a fresh, independent evaluation.

The cleanest holdout there is is data that did not exist when the model was built. "Use this on a
new file" already scores the model on any real answers in the upload; these pin that the score is
recorded as evidence about the model, shown beside the original score, never counted against the
session's held-out rows, never counted twice for one file, and never computed on rows the model
was built from.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services.session_store import session_dir

client = TestClient(app)


class FakeGraph:
    async def aget_state(self, _config):
        return SimpleNamespace(values={"action_log": []}, next=())


@pytest.fixture(autouse=True)
def graph_slot():
    had = hasattr(app.state, "graph")
    previous = getattr(app.state, "graph", None)
    app.state.graph = FakeGraph()
    yield
    if had:
        app.state.graph = previous
    else:
        del app.state.graph


def frame(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    signal = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "signal": signal.round(4),
            "spend": rng.normal(50, 10, n).round(2),
            "plan": rng.choice(["basic", "pro"], n),
            "churned": np.where(rng.random(n) < 1 / (1 + np.exp(-2.5 * signal)), "yes", "no"),
        }
    )


TRAINING = frame(300, 1)


def csv(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def session() -> tuple[str, object]:
    session_id = client.post(
        "/api/upload", files={"file": ("train.csv", io.BytesIO(csv(TRAINING)), "text/csv")}
    ).json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "churned",
            "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    response = client.post(
        f"/api/model/{session_id}/train",
        json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
    )
    assert response.status_code == 200, response.text
    return session_id, session_dir(get_settings().upload_dir, session_id)


def apply(session_id: str, df: pd.DataFrame, name: str = "march.csv"):
    return client.post(
        f"/api/apply/{session_id}", files={"file": (name, io.BytesIO(csv(df)), "text/csv")}
    )


def entries(directory):
    return [e for e in ledger.looks(directory) if e["event"] == ledger.NEW_DATA]


class TestWhenItCounts:
    def test_enough_labelled_rows_record_an_independent_evaluation(self):
        session_id, directory = session()
        body = apply(session_id, frame(200, 11)).json()

        assert body["answers"]["now"] is not None
        assert body["answers"]["counted_as_independent"] is True
        (entry,) = entries(directory)
        assert entry["route"] == "apply_answers" and entry["run_id"] == body["apply_id"]
        assert entry["extra"]["rows"] == 200

    def test_too_few_labelled_rows_record_nothing_and_say_why(self):
        session_id, directory = session()
        body = apply(session_id, frame(12, 11)).json()
        assert body["answers"]["now"] is None
        assert "isn't enough" in body["answers"]["headline"]
        assert not body["answers"].get("counted_as_independent")
        assert entries(directory) == []

    def test_a_file_with_no_answers_records_nothing(self):
        session_id, directory = session()
        no_answers = frame(80, 12).drop(columns=["churned"])
        body = apply(session_id, no_answers).json()
        assert body["answers"] is None
        assert entries(directory) == []


class TestTheAuditReportHearsAboutIt:
    def test_a_counted_check_on_new_data_marks_a_written_report_stale(self):
        from app.graph.nodes.audit_agent import is_report_stale

        session_id, directory = session()
        (directory / "audit_report.md").write_text("# report", encoding="utf-8")
        assert not is_report_stale(directory)
        assert apply(session_id, frame(200, 11)).json()["answers"]["counted_as_independent"] is True
        assert is_report_stale(directory)

    def test_a_file_with_no_answers_leaves_it_alone(self):
        from app.graph.nodes.audit_agent import is_report_stale

        session_id, directory = session()
        (directory / "audit_report.md").write_text("# report", encoding="utf-8")
        apply(session_id, frame(200, 11).drop(columns="churned"))
        assert not is_report_stale(directory)


class TestTheSameFileTwice:
    def test_the_second_upload_is_recognised_and_not_counted(self):
        session_id, directory = session()
        new = frame(200, 11)

        first = apply(session_id, new).json()
        second = apply(session_id, new).json()

        assert first["answers"]["counted_as_independent"] is True
        assert second["answers"]["counted_as_independent"] is False
        assert second["answers"]["already_counted"] is True
        assert len(entries(directory)) == 1
        # It is still scored and still answered; it just is not new evidence.
        assert second["answers"]["now"] == first["answers"]["now"]

    def test_a_different_file_is_a_second_piece_of_evidence(self):
        session_id, directory = session()
        apply(session_id, frame(200, 11))
        apply(session_id, frame(200, 12))
        assert len(entries(directory)) == 2

    def test_renaming_the_file_does_not_make_it_new(self):
        session_id, directory = session()
        new = frame(200, 11)
        apply(session_id, new, name="march.csv")
        again = apply(session_id, new, name="march_final_v2.csv").json()
        assert again["answers"]["already_counted"] is True
        assert len(entries(directory)) == 1


class TestRowsTheModelWasBuiltFrom:
    def test_rows_that_were_in_the_original_file_are_left_out_of_the_check(self):
        """Re-exports repeat earlier rows. Scoring on a row the model was trained or tested on
        is not new evidence."""
        session_id, directory = session()
        mixed = pd.concat([TRAINING.head(100), frame(100, 21)], ignore_index=True)

        body = apply(session_id, mixed).json()

        assert body["answers"]["overlap_excluded"] == 100
        assert body["answers"]["rows"] == 100
        assert any("100 labelled rows in this file were already" in n for n in body["notes"])
        assert entries(directory)[0]["extra"]["overlap_excluded"] == 100

    def test_a_revised_label_does_not_make_an_old_row_new(self):
        session_id, _ = session()
        revised = TRAINING.head(100).assign(
            churned=lambda d: d["churned"].map({"yes": "no", "no": "yes"})
        )
        body = apply(session_id, pd.concat([revised, frame(100, 22)], ignore_index=True)).json()
        assert body["answers"]["overlap_excluded"] == 100

    def test_a_file_made_only_of_old_rows_has_nothing_to_check_and_records_nothing(self):
        session_id, directory = session()
        body = apply(session_id, TRAINING.head(150)).json()

        assert body["answers"]["now"] is None
        assert "was already in the file this model was built from" in body["answers"]["headline"]
        assert body["answers"]["overlap_excluded"] == 150
        assert entries(directory) == []
        assert body["predictions"]["count"] == 150  # predictions are still made for every row

    def test_the_overlapping_rows_are_still_predicted(self):
        session_id, _ = session()
        mixed = pd.concat([TRAINING.head(60), frame(100, 23)], ignore_index=True)
        assert apply(session_id, mixed).json()["predictions"]["count"] == 160


class TestShownBesideTheOriginalScore:
    def test_the_result_names_the_latest_check_and_how_many_there_have_been(self):
        session_id, _ = session()
        apply(session_id, frame(200, 11))
        second = apply(session_id, frame(150, 12)).json()

        result = client.get(f"/api/model/{session_id}/result").json()
        new_data = result["evaluation"]["new_data"]
        assert new_data["count"] == 2
        assert new_data["latest"]["rows"] == 150
        assert new_data["latest"]["metric"] == "roc_auc"
        assert new_data["latest"]["value"] == pytest.approx(second["answers"]["now"], abs=1e-6)
        assert new_data["latest"]["reported"] is not None

    def test_it_does_not_replace_or_alter_the_original_evaluation(self):
        session_id, _ = session()
        before = client.get(f"/api/model/{session_id}/result").json()
        apply(session_id, frame(200, 11))
        after = client.get(f"/api/model/{session_id}/result").json()

        assert after["results"] == before["results"]
        assert after["assessment"] == before["assessment"]
        assert after["evaluation"]["status"] == before["evaluation"]["status"]
        assert after["evaluation"]["look_count"] == before["evaluation"]["look_count"]

    def test_there_is_nothing_to_show_before_any_file_with_answers(self):
        session_id, _ = session()
        assert client.get(f"/api/model/{session_id}/result").json()["evaluation"]["new_data"] is None

    def test_a_new_data_check_does_not_use_up_the_held_out_rows(self):
        """Applying is not a look at the session's held-out rows."""
        session_id, directory = session()
        apply(session_id, frame(200, 11))
        apply(session_id, frame(200, 12))
        retrained = client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        ).json()
        assert retrained["evaluation"]["look_count"] == 2  # two trains, not four events

    def test_a_retrained_model_starts_with_no_new_data_checks_of_its_own(self):
        """An evaluation of the previous model says nothing about this one."""
        session_id, directory = session()
        apply(session_id, frame(200, 11))
        client.post(
            f"/api/model/{session_id}/train",
            json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False},
        )
        assert client.get(f"/api/model/{session_id}/result").json()["evaluation"]["new_data"] is None
        assert len(entries(directory)) == 1  # the record itself is kept

    def test_the_new_rows_are_never_mixed_into_the_session_data(self):
        session_id, directory = session()
        original_before = (directory / "original.csv").read_bytes()
        apply(session_id, frame(200, 11))
        assert (directory / "original.csv").read_bytes() == original_before
        assert not (directory / "cleaned.csv").exists()
