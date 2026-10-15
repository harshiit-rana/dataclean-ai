"""What exactly produced a run: the software, the seeds, the request, the data version and how long it took.

A run history entry used to answer "what was the score" but not "what made it". Two runs that differ in an
unrecorded way -- a scikit-learn upgrade, a different improvement budget, a changed default -- looked identical, and
none of them could be reproduced or audited from the record. `describe` gathers everything a re-run needs, and
`config_hash` reduces the parts that decide the result to one value, so "same configuration?" is an equality test.

Everything here is small (a few hundred bytes) and nothing here may fail a run: the caller treats a failure to
describe as a missing record, never as a failed training.
"""

from __future__ import annotations

import functools
import hashlib
import json
import platform
from importlib import metadata
from typing import Any

# The libraries whose versions change what a model is. joblib reads and writes the saved pipeline.
LIBRARIES = {"scikit-learn": "scikit_learn", "pandas": "pandas", "numpy": "numpy", "joblib": "joblib"}


@functools.lru_cache(maxsize=1)
def environment() -> dict[str, str | None]:
    """The Python and library versions of this process. Read once: they cannot change while it runs."""
    found: dict[str, str | None] = {"python": platform.python_version()}
    for name, key in LIBRARIES.items():
        try:
            found[key] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[key] = None
    return found


def seeds() -> dict[str, int]:
    """Every seed a run depends on. Fixed in code, which is why they were never written down -- and why a change
    to one would make two runs differ with nothing in the record to say so."""
    from app.services import modeling, tuning

    return {
        # The train/test split, the models' own randomness and the seed check's first seed.
        "split": modeling.RANDOM_STATE,
        "model": modeling.RANDOM_STATE,
        # The settings search's folds and the sample of settings it tries.
        "search": 42,
        # The fresh folds a search winner is confirmed on.
        "confirmation": tuning.CONFIRMATION_SEED,
    }


def constants() -> dict[str, Any]:
    """Defaults in code that decide the result when a request leaves them out. A change to one changes the hash."""
    from app.services import modeling, tuning

    return {
        "default_test_size": modeling.DEFAULT_TEST_SIZE,
        "cv_folds": tuning.CV_FOLDS,
        "cv_repeats": tuning.CV_REPEATS,
        "default_budget": tuning.DEFAULT_BUDGET,
    }


def kind_of(route: str) -> str:
    """"train" or "improve": the streamed and the blocking search are the same experiment."""
    return "improve" if route.startswith("improve") else route


def config_hash(request: dict | None, dataset_version: dict | None, *, route: str, cost_limit: int | None = None) -> str:
    """One value for everything that decides the result: what was asked for, on which data, under which defaults
    and seeds. Two runs with the same hash on the same software should give the same numbers.

    The data is named by its content fingerprint, not its version number, so an undo back to an earlier state
    hashes like that state. The software versions are left out on purpose: "same configuration, different
    scikit-learn" is the case a reproduction check (#72) has to be able to name, not hide. The processing limit
    is in: it decides which models are tried.
    """
    payload = {
        "kind": kind_of(route),
        "cost_limit": cost_limit or None,
        "request": request or {},
        "data": (dataset_version or {}).get("fingerprint"),
        "constants": constants(),
        "seeds": seeds(),
    }
    text = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def selection(result: dict) -> dict | None:
    """How the model was chosen, with each candidate's fold scores."""
    chosen = result.get("selection")
    if not chosen:
        return None
    return {
        "method": chosen.get("method"),
        "metric": chosen.get("metric"),
        "rule": chosen.get("rule"),
        "scores": {name: score.get("score") for name, score in (chosen.get("scores") or {}).items()},
    }


def describe(
    *, request: dict | None, route: str, dataset_version: dict | None, runtime_seconds: float | None, result: dict,
    cost_limit: int | None = None,
) -> dict:
    """The run's metadata, as kept on the saved result and copied into its history entry."""
    version = (
        {k: dataset_version.get(k) for k in ("version", "fingerprint", "rows", "columns", "event")}
        if dataset_version
        else None
    )
    return {
        "environment": environment(),
        "seeds": seeds(),
        "request": request,
        "route": route,
        "cost_limit": cost_limit or None,
        "config_hash": config_hash(request, dataset_version, route=route, cost_limit=cost_limit),
        "runtime_seconds": None if runtime_seconds is None else round(float(runtime_seconds), 2),
        "dataset_version": version,
        "selection": selection(result),
    }
