"""The run history is paged and says how much is older, instead of being cut to 20 entries silently."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_store

client = TestClient(app)
LR = {"model_names": ["logistic_regression"], "include_deep_diagnostics": False}


def entries(n: int) -> list[dict]:
    return [{"run_id": f"r{i}"} for i in range(n)]  # newest first


class TestThePage:
    def test_the_first_page_and_what_is_older(self):
        page = model_store.history_page(entries(5), limit=2)
        assert [e["run_id"] for e in page["runs"]] == ["r0", "r1"]
        assert page["total"] == 5 and page["older_count"] == 3 and page["next_before"] == "r1"

    def test_the_next_page_follows_on(self):
        page = model_store.history_page(entries(5), limit=2, before="r1")
        assert [e["run_id"] for e in page["runs"]] == ["r2", "r3"] and page["older_count"] == 1

    def test_the_last_page_has_no_next(self):
        page = model_store.history_page(entries(5), limit=2, before="r3")
        assert [e["run_id"] for e in page["runs"]] == ["r4"]
        assert page["older_count"] == 0 and page["next_before"] is None

    def test_an_unknown_before_starts_from_the_newest(self):
        assert model_store.history_page(entries(3), limit=2, before="gone")["runs"][0]["run_id"] == "r0"

    def test_an_empty_history_is_an_empty_page(self):
        assert model_store.history_page([], limit=2) == {"runs": [], "total": 0, "older_count": 0, "next_before": None}

    def test_the_cap_is_far_past_a_working_session(self):
        assert model_store.MAX_HISTORY_ENTRIES >= 500


class TestOverHttp:
    def session_with_three_runs(self) -> tuple[str, list[str]]:
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        df = pd.DataFrame({"x": x, "z": rng.normal(size=300), "outcome": np.where(x + rng.normal(0, 1, 300) > 0, "yes", "no")})
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        sid = client.post(
            "/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}
        ).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "outcome", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        ids = [client.post(f"/api/model/{sid}/train", json={**LR, "test_size": size}).json()["run_id"]
               for size in (0.2, 0.25, 0.3)]
        return sid, ids[::-1]  # newest first

    def test_pages_cover_the_whole_history_and_the_edge_keeps_its_comparison(self):
        sid, newest_first = self.session_with_three_runs()
        first = client.get(f"/api/model/{sid}/history", params={"limit": 2}).json()
        assert [r["run_id"] for r in first["runs"]] == newest_first[:2]
        assert first["older_count"] == 1 and first["next_before"] == newest_first[1] and first["total"] == 3
        # The run on the page's edge is compared with the run before it, which is on the next page.
        assert newest_first[1] in first["comparisons"]
        rest = client.get(f"/api/model/{sid}/history", params={"limit": 2, "before": first["next_before"]}).json()
        assert [r["run_id"] for r in rest["runs"]] == newest_first[2:]
        assert rest["older_count"] == 0 and rest["next_before"] is None

    def test_the_default_page_is_the_newest_twenty(self):
        sid, newest_first = self.session_with_three_runs()
        body = client.get(f"/api/model/{sid}/history").json()
        assert [r["run_id"] for r in body["runs"]] == newest_first and body["older_count"] == 0
