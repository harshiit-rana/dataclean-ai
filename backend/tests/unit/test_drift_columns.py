"""The drift check judges each column on its own filled-in values, and notices a column going blank.

It refused to judge a new file under 30 rows, but a column that is 95% blank in a 500-row file was
judged on its 25 values with the same thresholds as a full one. And a column that stops being filled in
had not "drifted" at all as far as the check could tell, though it plainly has.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services import drift


def frame(n: int, seed: int = 0, blank_share: float = 0.0, column: str = "spend") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = rng.normal(100, 15, n)
    values[rng.random(n) < blank_share] = np.nan
    return pd.DataFrame({column: values, "plan": rng.choice(["basic", "plus"], n)})


def column(report: dict, name: str) -> dict:
    return next(c for c in report["columns"] if c["column"] == name)


class TestEachColumnOnItsOwnValues:
    def test_a_mostly_blank_column_is_not_judged_and_says_on_how_many_values(self):
        reference = frame(2000, blank_share=0.97)
        new = frame(500, seed=1, blank_share=0.97)
        found = column(drift.compare(reference, new, ["spend"], ["plan"]), "spend")
        assert found["status"] == "not_judged"
        assert found["new_values"] < drift.MIN_VALUES_TO_JUDGE
        assert f"{found['new_values']:,} here" in found["note"] and "at least 30" in found["note"]

    def test_too_few_in_the_training_data_is_not_judged_either(self):
        found = column(drift.compare(frame(400, blank_share=0.97), frame(400, seed=1, blank_share=0.97), ["spend"], []), "spend")
        assert found["status"] == "not_judged"

    def test_a_full_column_reports_the_counts_it_was_judged_on(self):
        found = column(drift.compare(frame(1000), frame(300, seed=1), ["spend"], ["plan"]), "spend")
        assert found["training_values"] == 1000 and found["new_values"] == 300 and found["status"] == "stable"

    def test_categories_are_counted_too(self):
        found = column(drift.compare(frame(1000), frame(300, seed=1), ["spend"], ["plan"]), "plan")
        assert found["training_values"] == 1000 and found["new_values"] == 300


class TestAColumnGoingBlank:
    def test_a_blank_share_moving_from_two_to_forty_percent_is_reported_as_changed(self):
        report = drift.compare(frame(2000, blank_share=0.02), frame(500, seed=1, blank_share=0.40), ["spend"], ["plan"])
        found = column(report, "spend")
        assert found["blank_status"] == "different" and found["status"] == "different"
        assert found["training_blank_share"] < 0.05 and found["new_blank_share"] > 0.35
        assert report["verdict"] == "different"

    def test_a_smaller_rise_is_a_shift(self):
        found = column(drift.compare(frame(3000, blank_share=0.02), frame(1000, seed=1, blank_share=0.17), ["spend"], []), "spend")
        assert found["blank_status"] == "shifted"

    def test_a_column_that_stops_being_filled_in_is_a_finding_even_though_its_values_cannot_be_compared(self):
        found = column(drift.compare(frame(2000, blank_share=0.02), frame(500, seed=1, blank_share=0.97), ["spend"], []), "spend")
        assert found["status"] == "different" and "filled-in values" in found["note"]

    def test_the_same_blank_share_is_not_a_change(self):
        found = column(drift.compare(frame(2000, blank_share=0.2), frame(500, seed=1, blank_share=0.2), ["spend"], []), "spend")
        assert found["blank_status"] == "stable"

    def test_a_wobble_on_a_small_file_is_not_called_a_change(self):
        """Twelve points on 40 rows is well within chance."""
        found = column(drift.compare(frame(2000, blank_share=0.10), frame(40, seed=1, blank_share=0.22), ["spend"], []), "spend")
        assert found["blank_status"] == "stable"


class TestTheOverallVerdict:
    def test_columns_that_could_not_be_judged_are_left_out_of_it_and_named_in_the_headline(self):
        reference = frame(2000, blank_share=0.97).assign(other=np.random.default_rng(3).normal(size=2000))
        new = frame(500, seed=1, blank_share=0.97).assign(other=np.random.default_rng(4).normal(size=500))
        report = drift.compare(reference, new, ["spend", "other"], [])
        assert report["verdict"] == "similar"
        assert "'spend' had too few filled-in values to compare" in report["headline"]
        assert "could be compared" in report["headline"]

    def test_a_file_where_everything_was_judged_keeps_the_old_headline(self):
        report = drift.compare(frame(1000), frame(300, seed=1), ["spend"], ["plan"])
        assert report["headline"] == "Every input looks like the data the model learned from."


class TestDerivedInputs:
    def test_a_derived_input_inherits_its_sources_blankness(self):
        """A length is never blank, so without its source it would pass unnoticed."""
        rng = np.random.default_rng(5)
        ref_notes = pd.Series(np.where(rng.random(2000) < 0.02, None, "a note"))
        new_notes = pd.Series(np.where(rng.random(500) < 0.6, None, "a note"))
        reference = pd.DataFrame({"notes__length": ref_notes.fillna("").str.len()})
        new = pd.DataFrame({"notes__length": new_notes.fillna("").str.len()})
        without = column(drift.compare(reference, new, ["notes__length"], []), "notes__length")
        with_source = column(
            drift.compare(reference, new, ["notes__length"], [], blanks_from={"notes__length": (ref_notes, new_notes)}),
            "notes__length",
        )
        assert without["blank_status"] == "stable"
        assert with_source["blank_status"] == "different" and with_source["status"] == "different"

    def test_over_the_api_a_note_that_stops_being_written_is_caught_through_its_length(self):
        """End to end: train with the length of a free-text column as an input, then apply a file
        where most notes are blank. The length is 0, never blank; its source tells the truth."""
        import io

        from fastapi.testclient import TestClient
        from types import SimpleNamespace

        from app.main import app

        class FakeGraph:
            async def aget_state(self, _config):
                return SimpleNamespace(values={"action_log": []}, next=())

        def rows(n, seed, blank):
            rng = np.random.default_rng(seed)
            signal = rng.normal(size=n)
            vocabulary = [f"w{i}" for i in range(60)]
            notes = np.array([" ".join(rng.choice(vocabulary, int(k))) for k in rng.integers(3, 30, n)], dtype=object)
            notes[rng.random(n) < blank] = None
            return pd.DataFrame({"signal": signal.round(4), "notes": notes,
                                 "churned": np.where(signal + rng.normal(size=n) > 0, "yes", "no")})

        client = TestClient(app)
        had, previous = hasattr(app.state, "graph"), getattr(app.state, "graph", None)
        app.state.graph = FakeGraph()
        try:
            buffer = io.StringIO()
            rows(400, 1, 0.02).to_csv(buffer, index=False)
            sid = client.post("/api/upload", files={"file": ("t.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
            client.post(f"/api/goal/{sid}", json={"use_case": "ml_training", "expected_output": "x", "target_column": "churned", "task_type": "auto"})
            plan = client.get(f"/api/model/{sid}/plan").json()
            length = next(r for r in plan["suggested_features"] if r["column"] == "notes" and any("length" in o for o in r["outputs"]))
            trained = client.post(f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False, "feature_recipes": [length["id"]]})
            assert trained.status_code == 200, trained.text

            new = io.StringIO()
            rows(300, 2, 0.6).drop(columns="churned").to_csv(new, index=False)
            applied = client.post(f"/api/apply/{sid}", files={"file": ("new.csv", io.BytesIO(new.getvalue().encode()), "text/csv")})
            assert applied.status_code == 200, applied.text
            report = applied.json()["drift"]
            derived = [c for c in report["columns"] if c["column"] in length["outputs"]]
            assert derived and all(c["blank_status"] == "different" for c in derived)
            assert report["verdict"] == "different"
        finally:
            if had:
                app.state.graph = previous
            else:
                del app.state.graph
