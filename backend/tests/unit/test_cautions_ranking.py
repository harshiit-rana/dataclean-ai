"""The cautions are ranked, and the page and the card read the same list.

The card computed its cautions in the order they happened to be found. The results page showed only
some of them, scattered, so the most important one could be below the fold. `caution_items` ranks
them by a documented severity; the card prints that list, and the results page shows its top three.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_card
from app.services.model_card import SEVERITY_ORDER, build_model_card, caution_items

BEST = {"diagnostics": {"per_class": [{"label": "rare", "support": 3, "precision": 0.5, "recall": 0.5, "f1": 0.5}]}}


def result(**overrides) -> dict:
    base = {
        "task": "classification",
        "best_model": "m",
        "split": {"strategy": "random", "train_rows": 80, "test_rows": 60, "test_rows_seen_in_training": 9},
        "features": {"excluded": ["id"], "derived": [{"column": "when"}]},
        "evaluation": {
            "status": "reused", "message": "These rows have now been used 4 times.",
            "whole_file_statistics": [{"operation": "fill_median", "column": "a"}],
        },
        "results": [{"model": "m", "label": "M", "metrics": {}, "diagnostics": BEST["diagnostics"]}],
        "operating_point": {"objective": "recall", "reachable": False, "description": "it catches 99%"},
        "improvement": {"rounds": [{"accepted": True}], "cv_score": 0.8, "stopped_because": "goal_out_of_reach", "goal": 0.99},
    }
    base.update(overrides)
    return copy.deepcopy(base)


class TestTheRanking:
    def test_the_most_severe_come_first_and_ties_keep_the_order_found(self):
        items = caution_items(result(), BEST)
        severities = [i["severity"] for i in items]
        assert severities == sorted(severities, key=SEVERITY_ORDER.get)
        assert items[0]["id"] == "test_rows_are_copies" and items[0]["severity"] == "high"
        assert [i["id"] for i in items if i["severity"] == "medium"] == [
            "holdout_reuse", "whole_file_statistics", "small_test_set", "thin_class_rare",
            "cutoff_unreachable", "goal_out_of_reach",
        ]

    def test_every_caution_has_an_id_a_known_severity_and_text(self):
        items = caution_items(result(), BEST)
        assert len({i["id"] for i in items}) == len(items) >= 8
        assert all(i["severity"] in SEVERITY_ORDER and i["text"].strip() for i in items)

    def test_an_exhausted_holdout_outranks_a_merely_reused_one(self):
        reused = {i["id"]: i["severity"] for i in caution_items(result(), BEST)}
        exhausted = result()
        exhausted["evaluation"]["status"] = "exhausted"
        worn = {i["id"]: i["severity"] for i in caution_items(exhausted, BEST)}
        assert reused["holdout_reuse"] == "medium" and worn["holdout_reuse"] == "high"

    def test_a_clean_result_has_none(self):
        clean = result(
            split={"strategy": "time", "test_rows": 500}, features={}, evaluation={}, operating_point=None,
            improvement=None,
        )
        assert caution_items(clean, {"diagnostics": {}}) == []


class TestTheCardPrintsTheSameList:
    def test_the_card_has_the_cautions_in_the_ranked_order_with_the_same_words(self):
        res = result()
        card = build_model_card(session_id="s", file_name="f.csv", result=res, summary="A.", interpretation="B.")
        tail = card.split("## Before you rely on this", 1)[1]
        positions = [tail.index(item["text"]) for item in caution_items(res, BEST)]
        assert positions == sorted(positions)

    def test_the_plain_text_helper_agrees(self):
        res = result()
        assert model_card._deterministic_cautions(res, BEST) == [i["text"] for i in caution_items(res, BEST)]


class TestServedWithTheResult:
    def test_a_trained_result_carries_its_ranked_cautions_and_the_card_agrees(self):
        client = TestClient(app)
        rng = np.random.default_rng(9)
        x = rng.normal(size=60)
        frame = pd.DataFrame({"x": x.round(4), "y": np.where(x + rng.normal(size=60) > 0, "yes", "no")})
        buffer = io.StringIO()
        frame.to_csv(buffer, index=False)
        sid = client.post("/api/upload", files={"file": ("d.csv", io.BytesIO(buffer.getvalue().encode()), "text/csv")}).json()["session_id"]
        goal = {"use_case": "ml_training", "expected_output": "x", "target_column": "y", "task_type": "auto"}
        assert client.post(f"/api/goal/{sid}", json=goal).status_code == 200
        trained = client.post(
            f"/api/model/{sid}/train", json={"model_names": ["logistic_regression"], "include_deep_diagnostics": False}
        )
        assert trained.status_code == 200, trained.text
        cautions = trained.json()["cautions"]
        assert cautions and cautions == client.get(f"/api/model/{sid}/result").json()["cautions"]
        assert [c["severity"] for c in cautions] == sorted((c["severity"] for c in cautions), key=SEVERITY_ORDER.get)
        # The small test set is among them: this file is tiny.
        assert any(c["id"] == "small_test_set" for c in cautions)
