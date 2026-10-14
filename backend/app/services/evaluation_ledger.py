"""A durable record of every time held-out performance was revealed.

The protocol is decided in docs/decisions/0001-evaluation-protocol.md. This module is its
storage: one entry per look, never truncated, kept when the data changes underneath a model.

A person who retrains, reads the score, changes something and retrains is selecting on the test
set by hand, and nothing automatic can stop them. What can be done is to count the looks and say
what they cost -- and a count derived from the run history would undercount, because the history
keeps only the newest twenty runs and is not keyed by how the data was split.

Three kinds of entry, because they mean different things:

- `look`: a train or an improvement run scored its chosen model on the held-out rows.
- `new_data_evaluation`: the model was scored on newly uploaded labelled rows (apply). Those are
  not the session's held-out rows and are never counted against them.
- `final_evaluation`: the sealed final holdout was opened (once).

Writing here is bookkeeping about a result that already exists, so it never raises: failing a
finished training run because its record could not be written would lose the result.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.services.model_store import LEDGER_FILENAME, _jsonable

logger = logging.getLogger(__name__)

# The status thresholds (applied in #7), with the reasoning from the ADR. If a person keeps the
# best of k looks, the expected inflation is the expected maximum of k draws of the score's own
# noise: 0.56 standard errors at k=2, 0.85 at 3, 1.03 at 4, 1.16 at 5. A holdout is called
# exhausted when that expected optimism reaches one standard error -- the size of effect the
# verdict already treats as noise -- which is the fourth look. A worst case (it assumes every
# look was used to pick the best), which is the right direction to err for a claim about
# independence.
INDEPENDENT_LOOKS = 1
EXHAUSTED_AT_LOOK = 4

LOOK = "look"
NEW_DATA = "new_data_evaluation"
FINAL = "final_evaluation"

# Serialises appends within this process. Two tabs training at once record two looks, which is
# right -- both revealed a score -- and the lock is what stops one overwriting the other.
_LOCK = threading.Lock()


def split_key(split: dict | None) -> str:
    """Identifies the rows a look was taken on, by the rule that held them out.

    Two runs share a count only if they hold out the same rows by the same rule: strategy, the
    time or group column that strategy uses, the *requested* test fraction and the seed, plus
    the sealed-holdout size and the duplicate-grouping choice once those exist. The requested
    fraction, not the fraction a date split ended up with after moving its cut to a timestamp
    boundary -- that varies with the data, and a key that moved with it would reset the count.

    Changing `modeling.RANDOM_STATE` or `DEFAULT_TEST_SIZE` changes every key. Old looks then no
    longer apply, which is correct (different rows are held out) and should be called out in
    whatever change does it.
    """
    from app.services import modeling

    split = split or {}
    strategy = split.get("strategy") or "random"
    fields = {
        "strategy": strategy,
        "time_column": split.get("time_column") if strategy == "time" else None,
        "group_column": split.get("group_column") if strategy == "group" else None,
        "test_size": split.get("requested_test_size", modeling.DEFAULT_TEST_SIZE),
        "seed": split.get("seed", modeling.RANDOM_STATE),
        "final_fraction": split.get("final_fraction") or 0,
        "keep_duplicates_together": bool(split.get("keep_duplicates_together")),
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _path(directory: Path) -> Path:
    return directory / LEDGER_FILENAME


def _read(directory: Path) -> list[dict]:
    path = _path(directory)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Test-set ledger at %s could not be read", path, exc_info=True)
        return []
    return data if isinstance(data, list) else []


def looks(directory: Path) -> list[dict]:
    """Every entry, oldest first."""
    return _read(directory)


def record(
    directory: Path,
    *,
    event: str,
    split: dict | None,
    route: str,
    run_id: str | None = None,
    version: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
) -> Optional[dict]:
    """Appends one entry. Returns it, or None if it could not be written.

    `version` is the session's current dataset version (`session_store.current_dataset_version`):
    its number and its full content hash are stored, not the display prefix, so a look can be
    tied to the exact data it was taken on.
    """
    try:
        entry = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "route": route,
            "run_id": run_id,
            "split_key": split_key(split),
            "split": _readable(split),
            "dataset_version": (version or {}).get("version"),
            "dataset_fingerprint": (version or {}).get("fingerprint"),
            "metrics": _jsonable(metrics or {}),
            **({"extra": _jsonable(extra)} if extra else {}),
        }
        with _LOCK:
            existing = _read_for_append(directory)
            existing.append(entry)
            _write(directory, existing)
        return entry
    except Exception:  # noqa: BLE001 -- bookkeeping must never fail the result it describes
        logger.warning("Could not record a %s in the test-set ledger", event, exc_info=True)
        return None


def record_look(
    directory: Path,
    *,
    version: dict | None,
    split: dict | None,
    route: str,
    run_id: str | None,
    metrics: dict | None,
) -> Optional[dict]:
    """One look at the held-out rows, by a route that revealed their score."""
    return record(
        directory, event=LOOK, split=split, route=route, run_id=run_id, version=version,
        metrics=metrics,
    )


def look_count(directory: Path, version: int | None, split: dict | None) -> int:
    """How many times the held-out rows for this split rule have been looked at.

    `version=None` counts across every dataset version of the session, which is what the
    status shown to the user uses: cleaning after a look is the commonest way to select on the
    test set by hand, so a new version must not reset the count (see the ADR, §2). Pass a
    number to ask about one version alone.

    Counts only `look` entries. Evaluations on new data and the sealed final holdout are
    different events and never count against the held-out rows.
    """
    key = split_key(split)
    return sum(
        1
        for entry in _read(directory)
        if entry.get("event") == LOOK
        and entry.get("split_key") == key
        and (version is None or entry.get("dataset_version") == version)
    )


def looks_on_other_splits(directory: Path, split: dict | None) -> int:
    """Looks taken under a different split rule. Those rows overlap this split's, so the count
    for a new rule is not a clean slate and is reported as such."""
    key = split_key(split)
    return sum(
        1
        for entry in _read(directory)
        if entry.get("event") == LOOK and entry.get("split_key") != key
    )


def _readable(split: dict | None) -> dict:
    split = split or {}
    keep = (
        "strategy", "time_column", "group_column", "requested_test_size", "seed",
        "final_fraction", "keep_duplicates_together",
    )
    return {k: split.get(k) for k in keep if split.get(k) is not None}


def _read_for_append(directory: Path) -> list[dict]:
    """The existing entries, for appending to. An unreadable ledger is set aside rather than
    overwritten: starting a new one over it would silently destroy the record this file exists
    to keep."""
    path = _path(directory)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except (OSError, ValueError):
        pass
    aside = path.with_name(path.name + ".unreadable")
    try:
        os.replace(path, aside)
        logger.warning("Test-set ledger was unreadable; kept as %s and started a new one", aside)
    except OSError:
        logger.warning("Test-set ledger was unreadable and could not be set aside", exc_info=True)
    return []


def _write(directory: Path, entries: list[dict[str, Any]]) -> None:
    path = _path(directory)
    temporary = path.with_name(path.name + ".tmp")
    # default=str: a metric of some type nobody anticipated is better recorded as text than
    # the whole look being lost -- the count is the point of this file, not the metric.
    temporary.write_text(json.dumps(entries, default=str), encoding="utf-8")
    os.replace(temporary, path)
