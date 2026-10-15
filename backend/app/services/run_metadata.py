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


def pip_name(key: str) -> str:
    """The package name pip installs for an environment key ("scikit_learn" -> "scikit-learn")."""
    return next((name for name, k in LIBRARIES.items() if k == key), key)


def compatible_requirement(name: str, version: str | None) -> str:
    """A pin that allows bug-fix releases of the same minor version (`~=1.7.0` means >=1.7.0 and <1.8).

    Exact pins fail to install on a Python the old wheel was never built for; an open requirement installs
    whatever is newest, which is how a script stops reproducing. The minor series is the middle ground.
    """
    parts = (version or "").split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return name
    return f"{name}~={parts[0]}.{parts[1]}.0"


def install_line(env: dict, keys: tuple[str, ...]) -> str:
    """`pip install ...` with each library pinned to the minor series that produced the result."""
    return "pip install " + " ".join(f'"{compatible_requirement(pip_name(k), env.get(k))}"' for k in keys)


def versions_block(env: dict, keys: tuple[str, ...]) -> str:
    """Source for an exported script: the versions that produced its result, and a check that only warns.

    Printed rather than raised: a newer library usually gives the same numbers, and refusing to run would
    turn "might differ slightly" into "cannot be used". The check reads the installed versions the same way
    `environment` does, so the names match.
    """
    used = {"python": env.get("python")} | {pip_name(k): env.get(k) for k in keys}
    lines = [f"    {name!r}: {version!r}," for name, version in used.items() if version]
    return (
        "# --- The versions that produced the reported numbers ------------------------------------\n"
        "# A different version usually gives the same result, but not always; the check below says so\n"
        "# when they differ, and carries on either way.\n"
        "VERSIONS_USED = {\n" + "\n".join(lines) + "\n}\n"
        "\n"
        "\n"
        "def check_versions():\n"
        "    import platform\n"
        "    from importlib import metadata\n"
        "\n"
        "    different = []\n"
        "    for name, used in VERSIONS_USED.items():\n"
        "        try:\n"
        "            installed = platform.python_version() if name == 'python' else metadata.version(name)\n"
        "        except metadata.PackageNotFoundError:\n"
        "            installed = 'not installed'\n"
        "        if installed != used:\n"
        "            different.append(f'{name} {installed} (the app used {used})')\n"
        "    if different:\n"
        "        print('Note: these versions differ from the ones that produced the reported numbers, '\n"
        "              'so the results may differ slightly:')\n"
        "        for line in different:\n"
        "            print('  ' + line)\n"
        "\n"
        "\n"
        "check_versions()\n"
    )


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


# What a run keeps of each fix it was trained after: enough to write clean.py from it again.
ACTION_FIELDS = ("action_id", "operation", "column_name", "operation_params", "status", "source")


def applied_actions(action_log: list[dict] | None) -> list[dict] | None:
    """The fixes applied when the run was trained, in order. None when the log could not be read (not the same
    as a session with no fixes, which is an empty list)."""
    if action_log is None:
        return None
    return [
        {k: entry.get(k) for k in ACTION_FIELDS if k in entry}
        for entry in action_log
        if entry.get("status") == "applied"
    ]


def describe(
    *, request: dict | None, route: str, dataset_version: dict | None, runtime_seconds: float | None, result: dict,
    cost_limit: int | None = None, action_log: list[dict] | None = None,
) -> dict:
    """The run's metadata, as kept on the saved result and copied into its history entry.

    `actions` is a snapshot of the fixes applied when it was trained, so its clean.py can be written from the log
    as it stood then, whatever undo or chat does to the log afterwards."""
    actions = applied_actions(action_log)
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
        "actions": actions,
        "applied_action_ids": None if actions is None else [a.get("action_id") for a in actions],
    }
