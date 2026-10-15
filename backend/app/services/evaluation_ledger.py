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
    # A different embargo holds out different rows. Added only when there is one, so every key a
    # run without a gap ever had is unchanged.
    if strategy == "time" and split.get("gap_days"):
        fields["gap_days"] = int(split["gap_days"])
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


INDEPENDENT = "independent"
REUSED = "reused"
EXHAUSTED = "exhausted"


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def evaluation_status(directory: Path, split: dict | None) -> dict:
    """What the user is told about the look that is about to be recorded.

    Computed from the ledger *before* the current look is written, then the look is recorded,
    so the first run of a split reads `independent` and the second `reused`. `look_count` is
    the number this look will be, counting itself -- "these rows have now been used N times".

    The wording is fixed by decision 0001, section 3, and is text rather than a template the
    caller fills in: the ADR quotes it verbatim and a test holds the two together. Nothing is
    ever blocked here; a limit would make small files unusable and only teach people to work
    around it.

    `looks_on_other_splits` is looks taken under a different split rule. Changing the rule
    starts a new count -- it is a different held-out set -- but those rows overlap, so the
    count is not a clean slate and the message says so.
    """
    prior = look_count(directory, None, split)
    this = prior + 1
    other = looks_on_other_splits(directory, split)

    if this <= INDEPENDENT_LOOKS:
        status, qualification = INDEPENDENT, ""
    elif this < EXHAUSTED_AT_LOOK:
        status = REUSED
        qualification = (
            f"These held-out rows have now been used to evaluate a model {_times(this)} in "
            "this session. If you changed anything because of an earlier score, this one is "
            "no longer an independent estimate."
        )
    else:
        status = EXHAUSTED
        qualification = (
            f"These held-out rows have been used {_times(this)}, and the choices made between "
            "runs were informed by them, so this score is optimistic. Treat it as a "
            "development figure, not an independent one. For an independent estimate, use the "
            "model on new labelled rows ('Use this on a new file')."
        )

    message = qualification
    if other:
        overlap = (
            f"This session has also evaluated {other} model{'s' if other != 1 else ''} under a "
            "different split rule. Those held-out rows overlap these, so this is not a clean "
            "slate."
        )
        message = f"{qualification} {overlap}".strip()

    return {
        "look_count": this,
        # The last look that is still only "reused"; the next one is exhausted.
        "limit": EXHAUSTED_AT_LOOK - 1,
        "status": status,
        "message": message,
        "qualification": qualification,
        "looks_on_other_splits": other,
    }


def qualify(assessment: dict | None, evaluation: dict | None) -> dict | None:
    """The verdict with its headline qualified by how the held-out rows have been used.

    The headline has to carry it: a verdict that reads "this model works" with the caveat in a
    panel somewhere else is a verdict that gets quoted without the caveat. Qualified exactly
    when the status is not `independent`; the verdict itself is never changed, because
    loosening or tightening it is a separate decision from saying how much to trust the number
    it was made from.
    """
    if not assessment or not evaluation or evaluation.get("status") == INDEPENDENT:
        return assessment
    qualification = evaluation.get("qualification") or ""
    if not qualification:
        return assessment
    return {
        **assessment,
        "headline": f"{assessment.get('headline', '')} {qualification}".strip(),
        "qualified_by_reuse": True,
    }


def find_final(directory: Path, version: dict | None, split: dict | None) -> Optional[dict]:
    """The time the sealed final holdout was opened for this data version and split rule, if it
    has been. The sealed rows can be opened once per (version, split): opening them again for a
    second model would be a second look at rows the first result may already have informed."""
    key = split_key(split)
    number = (version or {}).get("version")
    for entry in _read(directory):
        if (
            entry.get("event") == FINAL
            and entry.get("split_key") == key
            and entry.get("dataset_version") == number
        ):
            return entry
    return None


def final_openings(directory: Path, split: dict | None) -> int:
    """How many times the sealed holdout has been opened under this split rule, on any version."""
    key = split_key(split)
    return sum(1 for e in _read(directory) if e.get("event") == FINAL and e.get("split_key") == key)


def final_summary(directory: Path, run_id: str | None) -> dict | None:
    """The sealed final evaluation of the model with this run id, or None if it has not been
    opened. Looked up when a result is read, as the new-data checks are."""
    for entry in reversed(_read(directory)):
        if entry.get("event") == FINAL and entry.get("run_id") == run_id:
            extra = entry.get("extra") or {}
            return {
                "at": entry.get("at"),
                "rows": extra.get("rows"),
                "metrics": entry.get("metrics") or {},
                "baseline": extra.get("baseline"),
                "metrics_intervals": extra.get("metrics_intervals"),
                "assessment": extra.get("assessment"),
                "primary_metric": extra.get("primary_metric"),
                "rows_seen_in_training": extra.get("rows_seen_in_training", 0),
            }
    return None


def new_data_evaluations(directory: Path, run_id: str | None) -> list[dict]:
    """Evaluations on newly uploaded labelled rows that were made on the model with this run id.

    Tied to the model rather than the session: retraining makes a new model, and an evaluation
    of the previous one says nothing about it.
    """
    return [
        entry
        for entry in _read(directory)
        if entry.get("event") == NEW_DATA
        and (run_id is None or (entry.get("extra") or {}).get("model_run_id") == run_id)
    ]


def already_evaluated(directory: Path, upload_fingerprint: str | None, run_id: str | None) -> bool:
    """Whether this exact file has already been counted as a new-data evaluation of this model.

    The same file uploaded twice is one piece of evidence, not two: counting it again would let
    the number of 'independent' evaluations be inflated by clicking.
    """
    if not upload_fingerprint:
        return False
    return any(
        (entry.get("extra") or {}).get("upload_fingerprint") == upload_fingerprint
        for entry in new_data_evaluations(directory, run_id)
    )


def new_data_summary(directory: Path, run_id: str | None) -> dict | None:
    """How the model has done on rows that did not exist when it was built, or None.

    Shown beside the held-out score, never in place of it (decision 0001, section 4): the new
    rows may define the label differently, so this is evidence about the model, not a
    replacement for the original evaluation.
    """
    entries = new_data_evaluations(directory, run_id)
    if not entries:
        return None
    latest = entries[-1]
    extra = latest.get("extra") or {}
    metric = extra.get("metric")
    return {
        "count": len(entries),
        "latest": {
            "at": latest.get("at"),
            "apply_id": extra.get("apply_id"),
            "rows": extra.get("rows"),
            "metric": metric,
            "value": (latest.get("metrics") or {}).get(metric),
            "reported": extra.get("reported"),
            "overlap_excluded": extra.get("overlap_excluded", 0),
        },
    }


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
