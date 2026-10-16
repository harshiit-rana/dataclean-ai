"""A journal of an improvement search's expensive calls, so an interrupted search resumes instead of starting over
(#104, docs/decisions/0004-platform-architecture.md).

The loop (`improvement.improve`) is one long function whose rounds share state -- the incumbent, the inputs, the
settings each family's search found -- so checkpointing that state round by round would mean rewriting it. It is
also deterministic: fixed seeds, the same split, the same folds. So the journal records the result of every
cross-validation, settings search and fold-importance measurement as it finishes, and a resumed search runs the
loop again from the top with the journal in replay: each call whose arguments match the next recorded entry returns
that entry's result at once, without fitting anything. The rounds that finished before the interruption therefore
take seconds, the loop makes exactly the decisions it made the first time, and fitting starts again where it
stopped. The final result is the one an uninterrupted run gives.

The safety rule is the one #61 was rejected over: a cached score whose key misses something is silently wrong. Here
the key is everything the call was given -- the estimator's full parameters, the rows and values of the training
data (hashed), the target, and every option -- and entries are matched in order, so the first call that differs in
any way switches replay off for the rest of the search and everything after it is computed afresh. A journal for
different data is never replayed at all: its header carries the data's fingerprint.

The journal is JSON in the job's folder, not a pickle: reading it back must never be able to run code, and the app
unpickles nothing but its own fitted models (a test holds it to that). Results are encoded with their types -- a
search's result, tuples, numpy scalars -- so a replayed result is the same value the first run had.
"""

from __future__ import annotations

import hashlib
import logging
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

JOURNAL_FILE = "journal.json"
# Arguments whose value is a moment in time, not an input: only whether one was given is part of the key.
_MOMENTS = {"deadline"}


def _digest(value: Any) -> str:
    """A stable description of one argument, hashed."""
    if isinstance(value, (pd.DataFrame, pd.Series)):
        shape = f"{type(value).__name__}{value.shape}"
        columns = list(value.columns) if isinstance(value, pd.DataFrame) else [value.name]
        content = int(pd.util.hash_pandas_object(value, index=True).sum())
        return f"{shape}|{columns}|{content}"
    if callable(value) and not isinstance(value, type):
        # A `build` factory: what it builds, with every parameter, nested ones included.
        try:
            built = value()
            params = built.get_params(deep=True) if hasattr(built, "get_params") else {}
        except Exception:  # noqa: BLE001 -- an unbuildable factory just cannot be matched
            return f"unbuildable:{id(value)}"
        return f"{type(built).__name__}|" + "|".join(
            f"{key}={_describe(val)}" for key, val in sorted(params.items())
        )
    return _describe(value)


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python value it holds, so a replayed value describes and compares the same."""
    return value.item() if hasattr(value, "item") and getattr(value, "shape", None) == () else value


def encode(value: Any) -> Any:
    from dataclasses import fields, is_dataclass

    value = _plain(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {"__dataclass__": type(value).__name__,
                "fields": {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
    if isinstance(value, tuple):
        return {"__tuple__": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {"__dict__": [[encode(k), encode(v)] for k, v in value.items()]}
    return value


def decode(value: Any) -> Any:
    from app.services import tuning

    if isinstance(value, list):
        return [decode(v) for v in value]
    if isinstance(value, dict):
        if "__tuple__" in value:
            return tuple(decode(v) for v in value["__tuple__"])
        if "__dict__" in value:
            return {decode(k): decode(v) for k, v in value["__dict__"]}
        if value.get("__dataclass__") == "SearchResult":
            return tuning.SearchResult(**{k: decode(v) for k, v in value["fields"].items()})
        raise ValueError(f"Unknown journal entry {sorted(value)}")
    return value


def _describe(value: Any) -> str:
    value = _plain(value)
    if hasattr(value, "get_params") and not isinstance(value, type):
        return type(value).__name__
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_describe(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{_describe(v)}" for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))) + "}"
    return repr(value)


def signature(kind: str, args: tuple, kwargs: dict) -> str:
    parts = [kind, *(_digest(a) for a in args)]
    for key in sorted(kwargs):
        value = kwargs[key]
        parts.append(f"{key}={'set' if value is not None else 'none'}" if key in _MOMENTS else f"{key}={_digest(value)}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


class Journal:
    """Records calls, or replays a recorded run's calls while they match. See the module docstring."""

    def __init__(self, path: Path, *, fingerprint: str, previous: Path | None = None):
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.entries: list[tuple[str, Any]] = []
        self.replay: list[tuple[str, Any]] = []
        self.replayed = 0
        self.diverged_at: int | None = None
        # The job this one carries on from, when it is a resumption; set by the caller.
        self.resumed_from: str | None = None
        if previous is not None and previous.exists():
            try:
                stored = json.loads(previous.read_text(encoding="utf-8"))
                if stored.get("fingerprint") == fingerprint:
                    self.replay = [(key, decode(result)) for key, result in stored.get("entries") or []]
                else:
                    logger.info("Not replaying %s: it was recorded on different data", previous)
            except Exception:  # noqa: BLE001 -- an unreadable journal means starting over, not failing
                logger.warning("Could not read the search journal %s; starting over", previous, exc_info=True)

    @property
    def replaying(self) -> bool:
        return self.diverged_at is None and len(self.entries) < len(self.replay)

    def call(self, kind: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        key = signature(kind, args, kwargs)
        position = len(self.entries)
        if self.replaying:
            recorded_key, recorded = self.replay[position]
            if recorded_key == key:
                self.entries.append((key, recorded))
                self.replayed += 1
                return recorded
            self.diverged_at = position
            logger.info("Search journal diverged at call %d (%s); computing from here", position, kind)
        result = function(*args, **kwargs)
        self.entries.append((key, result))
        self._save()
        return result

    def _save(self) -> None:
        temporary = self.path.with_name(self.path.name + ".tmp")
        try:
            temporary.write_text(json.dumps({
                "fingerprint": self.fingerprint,
                "entries": [[key, encode(result)] for key, result in self.entries],
            }), encoding="utf-8")
            temporary.replace(self.path)
        except Exception:  # noqa: BLE001 -- a journal that cannot be written costs a resume, not the search
            logger.warning("Could not write the search journal %s", self.path, exc_info=True)

    def summary(self) -> dict:
        return {"reused": self.replayed, "recorded": len(self.replay), "diverged_at": self.diverged_at}
