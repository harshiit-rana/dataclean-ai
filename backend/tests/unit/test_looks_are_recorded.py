"""Every route that reveals held-out performance writes one look to the ledger -- and nothing else does.

The ledger (decision 0001) is only worth having if its count is right: one per successful run,
none for a refused or failed request, none for re-reading a stored result. Counting too few makes
a used holdout look fresh; counting too many cries wolf until the warning is ignored.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.services import evaluation_ledger as ledger
from app.services import model_store
from app.services.session_store import session_dir

client = TestClient(app)

RANDOM = {"strategy": "random"}
FAST = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def _csv(n: int = 300, *, dates: bool = False, labelled: int | None = None) -> bytes:
    rng = np.random.default_rng(17)
    signal = rng.normal(0, 1, n)
    outcome = list(np.where(rng.random(n) < 1 / (1 + np.exp(-2.4 * signal)), "yes", "no"))
    if labelled is not None:
        outcome = outcome[:labelled] + [None] * (n - labelled)
    data = {
        "signal": signal.round(4),
        "noise": rng.normal(0, 1, n).round(4),
        "region": rng.choice(["north", "south"], n),
        "outcome": outcome,
    }
    if dates:
        data["joined"] = pd.date_range("2021-01-01", periods=n, freq="D").astype(str)
    buffer = io.StringIO()
    pd.DataFrame(data).to_csv(buffer, index=False)
    return buffer.getvalue().encode()


def _session(payload: bytes | None = None) -> tuple[str, object]:
    response = client.post(
        "/api/upload", files={"file": ("d.csv", io.BytesIO(payload or _csv()), "text/csv")}
    )
    session_id = response.json()["session_id"]
    goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome",
            "task_type": "auto"}
    assert client.post(f"/api/goal/{session_id}", json=goal).status_code == 200
    return session_id, session_dir(get_settings().upload_dir, session_id)


def _count(directory, split=None) -> int:
    return ledger.look_count(directory, None, split or RANDOM)


class TestTrain:
    def test_one_successful_train_is_one_look(self):
        session_id, directory = _session()
        assert _count(directory) == 0
        response = client.post(f"/api/model/{session_id}/train", json=FAST)
        assert response.status_code == 200, response.text
        assert _count(directory) == 1

    def test_each_further_train_is_a_further_look(self):
        session_id, directory = _session()
        for expected in (1, 2, 3):
            assert client.post(f"/api/model/{session_id}/train", json=FAST).status_code == 200
            assert _count(directory) == expected

    def test_the_look_names_the_route_the_run_the_data_and_the_score(self):
        session_id, directory = _session()
        body = client.post(f"/api/model/{session_id}/train", json=FAST).json()

        (entry,) = ledger.looks(directory)
        assert entry["event"] == ledger.LOOK and entry["route"] == "train"
        assert entry["run_id"] == body["run_id"]
        assert entry["metrics"]["roc_auc"] == body["results"][0]["metrics"]["roc_auc"]
        # Tied to the exact data: the version and the full content hash, as the model path
        # reads it.
        assert entry["dataset_version"] == 1
        assert entry["dataset_fingerprint"] == model_store.full_fingerprint(
            pd.read_csv(directory / "original.csv")
        )
        assert entry["split_key"] == ledger.split_key(body["split"])

    def test_the_same_run_id_is_in_the_history_and_the_ledger(self):
        session_id, directory = _session()
        body = client.post(f"/api/model/{session_id}/train", json=FAST).json()
        history = client.get(f"/api/model/{session_id}/history").json()["runs"]
        assert history[0]["run_id"] == body["run_id"] == ledger.looks(directory)[0]["run_id"]


class TestImprove:
    def test_an_improvement_is_one_look_not_one_per_round(self):
        """The search scores its rounds on training folds. Only the final rebuild opens the
        held-out rows."""
        session_id, directory = _session()
        response = client.post(
            f"/api/model/{session_id}/improve", json={**FAST, "budget": 4}
        )
        assert response.status_code == 200, response.text
        assert len(response.json()["improvement"]["rounds"]) >= 2  # several rounds ran
        assert _count(directory) == 1
        assert ledger.looks(directory)[0]["route"] == "improve"

    def test_the_streamed_improvement_is_one_look(self):
        session_id, directory = _session()
        response = client.post(
            f"/api/model/{session_id}/improve/stream", json={**FAST, "budget": 4}
        )
        events = [
            json.loads(line[len("data: "):])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert events[-1]["type"] == "result"
        assert _count(directory) == 1
        assert ledger.looks(directory)[0]["route"] == "improve_stream"

    def test_a_train_then_an_improvement_are_two_looks(self):
        session_id, directory = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        client.post(f"/api/model/{session_id}/improve", json={**FAST, "budget": 4})
        assert _count(directory) == 2


class TestWhatIsNotALook:
    def test_a_refused_train_records_nothing(self):
        session_id, directory = _session(_csv(400, labelled=40))
        response = client.post(f"/api/model/{session_id}/train", json=FAST)
        assert response.status_code >= 400
        assert ledger.looks(directory) == []
        assert not (directory / model_store.LEDGER_FILENAME).exists()

    def test_a_refused_improvement_records_nothing(self):
        session_id, directory = _session(_csv(400, labelled=40))
        assert client.post(f"/api/model/{session_id}/improve", json=FAST).status_code >= 400
        assert client.post(f"/api/model/{session_id}/improve/stream", json=FAST).status_code >= 400
        assert ledger.looks(directory) == []

    def test_a_train_that_fails_records_nothing(self):
        session_id, directory = _session()
        response = client.post(
            f"/api/model/{session_id}/train",
            json={**FAST, "excluded_columns": ["signal", "noise", "region"]},
        )
        assert response.status_code == 422
        assert ledger.looks(directory) == []

    def test_reading_a_stored_result_records_nothing(self):
        session_id, directory = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        before = len(ledger.looks(directory))

        for path in ("result", "history", "card", "script", "download", "plan", "correlations"):
            client.get(f"/api/model/{session_id}/{path}")
        client.get(f"/api/readiness/{session_id}")

        assert len(ledger.looks(directory)) == before == 1

    def test_the_improvement_that_keeps_it_all_still_counts_once_after_a_refresh(self):
        """A refresh re-reads; it does not re-run."""
        session_id, directory = _session()
        client.post(f"/api/model/{session_id}/improve", json={**FAST, "budget": 4})
        client.get(f"/api/model/{session_id}/result")
        client.get(f"/api/model/{session_id}/result")
        assert _count(directory) == 1


class TestCountsAreKeyedBySplitRule:
    def test_a_time_split_counts_separately_from_a_random_one(self):
        session_id, directory = _session(_csv(dates=True))
        client.post(f"/api/model/{session_id}/train", json=FAST)
        client.post(f"/api/model/{session_id}/train", json=FAST)
        time_split = {"strategy": "time", "time_column": "joined"}
        client.post(
            f"/api/model/{session_id}/train",
            json={**FAST, "split_strategy": "time", "time_column": "joined"},
        )
        assert _count(directory, RANDOM) == 2
        assert _count(directory, time_split) == 1

    def test_changing_the_data_does_not_reset_the_count(self):
        """Cleaning after a look is the commonest way to select on the test set by hand."""
        from app.services.session_store import record_dataset_version_in

        session_id, directory = _session()
        client.post(f"/api/model/{session_id}/train", json=FAST)
        frame = pd.read_csv(directory / "original.csv").iloc[:-5]
        frame.to_csv(directory / "cleaned.csv", index=False)
        record_dataset_version_in(directory, "cleaning_run")
        model_store.clear(directory)  # what an undo or a chat instruction does

        client.post(f"/api/model/{session_id}/train", json=FAST)

        assert _count(directory) == 2
        versions = [e["dataset_version"] for e in ledger.looks(directory)]
        assert versions == [1, 2]
        assert ledger.look_count(directory, 1, RANDOM) == 1
