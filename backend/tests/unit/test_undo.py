"""Undo via replay. PRD US-10."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.undo import ActionNotFoundError, ActionNotUndoableError, undo_action


@pytest.fixture
def session(tmp_path):
    original = tmp_path / "original.csv"
    pd.DataFrame(
        {
            "salary": [100.0, None, 300.0, 1_000_000.0, 200.0],
            "dept": ["Eng", "eng", "Sales", "SALES", "Eng"],
        }
    ).to_csv(original, index=False)
    return original, tmp_path / "cleaned.csv"


def _log() -> list[dict]:
    return [
        {
            "action_id": "a1",
            "card_id": "c1",
            "column_name": "salary",
            "operation": "fill_median",
            "operation_params": {"column": "salary"},
            "status": "applied",
        },
        {
            "action_id": "a2",
            "card_id": "c2",
            "column_name": "dept",
            "operation": "standardize_case",
            "operation_params": {"column": "dept", "case": "lower"},
            "status": "applied",
        },
    ]


def test_undo_reverts_only_the_named_action(session):
    original, cleaned = session
    updated = undo_action(
        original_path=original, cleaned_path=cleaned, action_log=_log(), action_id="a1"
    )

    by_id = {e["action_id"]: e for e in updated}
    assert by_id["a1"]["status"] == "undone"
    assert by_id["a2"]["status"] == "applied"

    df = pd.read_csv(cleaned)
    assert df["salary"].isna().sum() == 1  # the fill was reverted
    assert set(df["dept"]) == {"eng", "sales"}  # the case fix still applied


def test_undo_replays_parameters_faithfully(session):
    """The replay must honour operation_params -- dropping case="lower" would silently
    produce different data than the user approved."""
    original, cleaned = session
    undo_action(original_path=original, cleaned_path=cleaned, action_log=_log(), action_id="a1")
    df = pd.read_csv(cleaned)
    assert "eng" in df["dept"].tolist()
    assert "Eng" not in df["dept"].tolist()


def test_undoing_the_only_action_restores_the_original(session):
    original, cleaned = session
    log = [_log()[0]]
    undo_action(original_path=original, cleaned_path=cleaned, action_log=log, action_id="a1")

    pd.testing.assert_frame_equal(pd.read_csv(cleaned), pd.read_csv(original))


def test_undo_is_repeatable(session):
    original, cleaned = session
    log = undo_action(
        original_path=original, cleaned_path=cleaned, action_log=_log(), action_id="a1"
    )
    log = undo_action(original_path=original, cleaned_path=cleaned, action_log=log, action_id="a2")

    assert all(e["status"] == "undone" for e in log)
    pd.testing.assert_frame_equal(pd.read_csv(cleaned), pd.read_csv(original))


def test_undo_never_touches_the_original_file(session):
    original, cleaned = session
    before = original.read_bytes()
    undo_action(original_path=original, cleaned_path=cleaned, action_log=_log(), action_id="a1")
    assert original.read_bytes() == before


def test_unknown_action_id_raises(session):
    original, cleaned = session
    with pytest.raises(ActionNotFoundError):
        undo_action(
            original_path=original, cleaned_path=cleaned, action_log=_log(), action_id="nope"
        )


def test_cannot_undo_something_that_was_never_applied(session):
    original, cleaned = session
    log = _log()
    log[0]["status"] = "rejected"
    with pytest.raises(ActionNotUndoableError):
        undo_action(original_path=original, cleaned_path=cleaned, action_log=log, action_id="a1")


def test_dependent_action_that_no_longer_replays_is_marked_undone(session):
    """If removing one fix makes a later one impossible, the log must say so rather than
    claim a change that isn't in the data."""
    original, cleaned = session
    log = _log()
    log.append(
        {
            "action_id": "a3",
            "card_id": "c3",
            "column_name": "does_not_exist",
            "operation": "fill_median",
            "operation_params": {"column": "does_not_exist"},
            "status": "applied",
        }
    )
    updated = undo_action(
        original_path=original, cleaned_path=cleaned, action_log=log, action_id="a1"
    )

    a3 = next(e for e in updated if e["action_id"] == "a3")
    assert a3["status"] == "undone"
    # failure_reason, not rejection_reason: the user didn't decide this, the replay did.
    assert "no longer applies" in a3["failure_reason"]
    assert a3.get("rejection_reason") is None
