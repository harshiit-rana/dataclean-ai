"""What changed between two versions of a session's data, and which fixes did it.

Each version keeps the applied fixes behind it, so it can be rebuilt from the upload by replay. Replay keeps each row's
position in the upload through every drop, so two versions line up row by row and the diff can say which rows went,
which cells changed, and which fixes were applied or undone in between.
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
from app.services import model_store, replay, session_store, version_diff

client = TestClient(app)


def upload() -> pd.DataFrame:
    return pd.DataFrame({
        "spend": [10.0, np.nan, 30.0, 30.0, np.nan, 50.0],
        "plan": ["a", "b", "c", "c", "a", "b"],
        "note": ["x", "y", "z", "z", "x", "y"],
    })


FILL = {"action_id": "a1", "operation": "fill_median", "column_name": "spend", "operation_params": {}, "status": "applied"}
DEDUPE = {"action_id": "a2", "operation": "drop_duplicate_rows", "column_name": "DATASET", "operation_params": {},
          "status": "applied"}
DROP_NOTE = {"action_id": "a3", "operation": "drop_column", "column_name": "note", "operation_params": {},
             "status": "applied"}


def version(number: int, actions: list[dict] | None, event: str = "cleaning_run") -> dict:
    frame = replay.freeze(upload(), actions or [])[1] if number > 1 else upload()
    entry = {"version": number, "fingerprint": model_store.full_fingerprint(frame), "rows": len(frame),
             "columns": frame.shape[1], "at": "t", "event": "upload" if number == 1 else event}
    if actions is not None:
        entry["actions"] = actions
    return entry


@pytest.fixture
def directory(tmp_path):
    upload().to_csv(tmp_path / "original.csv", index=False)
    return tmp_path


class TestTheDiff:
    def test_a_fill_changes_cells_but_no_rows(self, directory):
        versions = [version(1, []), version(2, [FILL])]
        found = version_diff.diff(directory, versions, 1, 2)
        assert found["cells"] == {"spend": 2}
        assert found["rows"]["removed"] == 0 and found["rows"]["change"] == 0
        assert found["actions"]["applied"][0]["action_id"] == "a1" and found["actions"]["undone"] == []
        assert found["rebuilt"] == {"from": "upload", "to": "replayed"}

    def test_dropped_rows_are_counted_and_the_rest_still_line_up(self, directory):
        versions = [version(1, []), version(2, [FILL, DEDUPE])]
        found = version_diff.diff(directory, versions, 1, 2)
        assert found["rows"]["removed"] == 1 and found["rows"]["change"] == -1
        assert found["cells"] == {"spend": 2}  # the duplicate's removal does not shift every later row

    def test_a_dropped_column_is_named(self, directory):
        versions = [version(1, []), version(2, [DROP_NOTE])]
        found = version_diff.diff(directory, versions, 1, 2)
        assert found["columns"] == {"added": [], "removed": ["note"]} and found["cells"] == {}

    def test_an_undo_between_versions_is_named(self, directory):
        versions = [version(1, []), version(2, [FILL, DEDUPE]), version(3, [FILL], event="undo:a2")]
        found = version_diff.diff(directory, versions, 2, 3)
        assert [a["action_id"] for a in found["actions"]["undone"]] == ["a2"] and found["actions"]["applied"] == []
        assert found["rows"]["added"] == 1

    def test_a_version_with_no_snapshot_gives_counts_only(self, directory):
        versions = [version(1, []), version(2, None), version(3, [FILL, DEDUPE])]
        found = version_diff.diff(directory, versions, 2, 3)
        assert found["rebuilt"]["from"] == "unavailable" and found["cells"] is None and found["columns"] is None
        assert "cannot be rebuilt" in found["notes"][0]

    def test_the_latest_version_without_a_snapshot_is_read_from_disk_but_not_lined_up(self, directory):
        replay.freeze(upload(), [FILL])[1].to_csv(directory / "cleaned.csv", index=False)
        versions = [version(1, []), version(2, None)]
        found = version_diff.diff(directory, versions, 1, 2)
        assert found["rebuilt"]["to"] == "current_file" and found["columns"] is not None and found["cells"] is None

    def test_a_replay_that_does_not_match_its_fingerprint_says_so(self, directory):
        wrong = {**version(2, [FILL]), "fingerprint": "0" * 64}
        found = version_diff.diff(directory, [version(1, []), wrong], 1, 2)
        assert found["rebuilt"]["to"] == "replayed_unverified" and "may not be exact" in found["notes"][0]

    def test_a_large_diff_gives_the_summary_only(self, directory, monkeypatch):
        monkeypatch.setattr(version_diff, "MAX_DIFF_CELLS", 3)
        found = version_diff.diff(directory, [version(1, []), version(2, [FILL])], 1, 2)
        assert found["cells"] is None and found["rows"]["removed"] == 0 and "Too large" in found["notes"][0]

    def test_an_unknown_version_is_an_error(self, directory):
        with pytest.raises(version_diff.VersionNotFoundError):
            version_diff.diff(directory, [version(1, [])], 1, 9)


class TestTheSnapshot:
    def test_a_recorded_version_keeps_the_applied_fixes(self, directory):
        (directory / "meta.json").write_text(json.dumps({"session_id": "s"}))
        session_store.record_dataset_version_in(directory, "upload", upload(), [])
        cleaned = replay.freeze(upload(), [FILL])[1]
        rejected = {**DEDUPE, "status": "rejected"}
        entry = session_store.record_dataset_version_in(directory, "cleaning_run", cleaned, [FILL, rejected])
        assert [a["action_id"] for a in entry["actions"]] == ["a1"]


class TestOverHttp:
    def test_the_diff_endpoint(self):
        buffer = io.StringIO()
        upload().to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("u.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        directory = session_store.session_dir(get_settings().upload_dir, sid)
        cleaned = replay.freeze(upload(), [FILL, DEDUPE])[1]
        cleaned.to_csv(directory / "cleaned.csv", index=False)
        session_store.record_dataset_version_in(directory, "cleaning_run", cleaned, [FILL, DEDUPE])

        body = client.get(f"/api/session/{sid}/versions/diff", params={"from": 1, "to": 2}).json()
        assert body["rows"]["removed"] == 1 and body["cells"] == {"spend": 2}
        assert [a["operation"] for a in body["actions"]["applied"]] == ["fill_median", "drop_duplicate_rows"]
        missing = client.get(f"/api/session/{sid}/versions/diff", params={"from": 1, "to": 7})
        assert missing.status_code == 404 and missing.json()["code"] == "version_not_found"

    def test_an_unknown_session_is_not_found(self):
        assert client.get("/api/session/nope/versions/diff", params={"from": 1, "to": 2}).status_code == 404
