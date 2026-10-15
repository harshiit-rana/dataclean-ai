"""What changed between two versions of a session's data.

Runs from before a cleaning change are labelled as such, but the label says nothing about *what* changed -- rows
removed, columns dropped, values rewritten -- which is what a reader needs to interpret a score that moved across it.

A version is rebuilt by replaying, from the upload, the fixes that were applied when it was recorded (each version
keeps that snapshot). Replay keeps each row's position in the upload as its label, through every row it drops, so two
rebuilt versions line up row by row even when rows were removed between them: which rows went is known, and cells
are compared only on rows both have. A rebuild is checked against the version's fingerprint and the diff says when
it does not match. Above `MAX_DIFF_CELLS` cells the per-column counts are skipped and only the summary is given.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Above this many cells compared, the per-column counts are left out: the summary is still exact, and a request
# for it must not take minutes.
MAX_DIFF_CELLS = 5_000_000


class VersionNotFoundError(Exception):
    pass


def _current_path(directory: Path) -> Path:
    cleaned = directory / "cleaned.csv"
    return cleaned if cleaned.exists() else directory / "original.csv"


def rebuild(directory: Path, version: dict, versions: list[dict]) -> tuple[pd.DataFrame | None, str]:
    """The data as it was at `version`, and how it was obtained: "upload", "replayed", "replayed_unverified"
    (the replay's fingerprint does not match the one recorded), "current_file" (the latest version with no
    snapshot, read from disk, so its rows cannot be lined up with another version's), or "unavailable"."""
    from app.services import model_store, replay

    original = pd.read_csv(directory / "original.csv")
    if version.get("event") == "upload" or version.get("version") == 1:
        return original, "upload"
    actions = version.get("actions")
    if actions is not None:
        _, frame = replay.freeze(original, actions)
        matches = model_store.full_fingerprint(frame) == version.get("fingerprint")
        return frame, "replayed" if matches else "replayed_unverified"
    if versions and version.get("version") == versions[-1].get("version"):
        return pd.read_csv(_current_path(directory)), "current_file"
    return None, "unavailable"


def _summary(version: dict) -> dict:
    return {k: version.get(k) for k in ("version", "rows", "columns", "event", "at")}


def _actions_between(older: dict, newer: dict) -> dict | None:
    if older.get("actions") is None or newer.get("actions") is None:
        return None
    before = {a.get("action_id"): a for a in older["actions"]}
    after = {a.get("action_id"): a for a in newer["actions"]}

    def brief(entry: dict) -> dict:
        return {k: entry.get(k) for k in ("action_id", "operation", "column_name", "source")}

    return {
        "applied": [brief(a) for i, a in after.items() if i not in before],
        "undone": [brief(a) for i, a in before.items() if i not in after],
    }


def _changed_cells(old: pd.DataFrame, new: pd.DataFrame, rows: pd.Index, columns: list[str]) -> dict[str, int]:
    """How many cells differ per column on the rows both have. Blanks equal blanks; a value and its string
    form are different only if they print differently, so a column re-read as another type is not every cell."""
    found: dict[str, int] = {}
    for column in columns:
        a, b = old.loc[rows, column], new.loc[rows, column]
        both_blank = a.isna().to_numpy() & b.isna().to_numpy()
        same = (a.astype(str).to_numpy() == b.astype(str).to_numpy()) | both_blank
        changed = int((~same).sum())
        if changed:
            found[column] = changed
    return found


def diff(directory: Path, versions: list[dict], from_version: int, to_version: int) -> dict:
    """What changed from one version of the data to another."""
    by_number = {v.get("version"): v for v in versions}
    if from_version not in by_number or to_version not in by_number:
        raise VersionNotFoundError(f"No version {from_version if from_version not in by_number else to_version}.")
    older, newer = by_number[from_version], by_number[to_version]

    old_frame, old_how = rebuild(directory, older, versions)
    new_frame, new_how = rebuild(directory, newer, versions)
    result: dict = {
        "from": _summary(older),
        "to": _summary(newer),
        "rebuilt": {"from": old_how, "to": new_how},
        "rows": {"from": older.get("rows"), "to": newer.get("rows"),
                 "change": (newer.get("rows") or 0) - (older.get("rows") or 0)},
        "actions": _actions_between(older, newer),
        "columns": None,
        "cells": None,
        "notes": [],
    }
    if old_frame is None or new_frame is None:
        result["notes"].append(
            "One of these versions was recorded before versions kept the fixes behind them, so it cannot be "
            "rebuilt; only the row and column counts are known."
        )
        return result

    result["columns"] = {
        "added": [c for c in new_frame.columns if c not in old_frame.columns],
        "removed": [c for c in old_frame.columns if c not in new_frame.columns],
    }
    if "unverified" in old_how or "unverified" in new_how:
        result["notes"].append(
            "Replaying the recorded fixes did not reproduce one of these versions exactly, so the counts below are "
            "close but may not be exact."
        )
    if "current_file" in (old_how, new_how):
        result["notes"].append(
            "The latest version has no record of the fixes behind it, so its rows cannot be lined up with the "
            "other version's; only the counts and columns are compared."
        )
        return result

    common_rows = old_frame.index.intersection(new_frame.index)
    result["rows"].update({
        "removed": int(len(old_frame.index.difference(new_frame.index))),
        "added": int(len(new_frame.index.difference(old_frame.index))),
    })
    shared = [c for c in old_frame.columns if c in new_frame.columns]
    if len(common_rows) * max(len(shared), 1) > MAX_DIFF_CELLS:
        result["notes"].append(
            f"Too large to compare cell by cell ({len(common_rows):,} rows x {len(shared)} columns); the row and "
            f"column changes above are exact."
        )
        return result
    result["cells"] = _changed_cells(old_frame, new_frame, common_rows, shared)
    return result
