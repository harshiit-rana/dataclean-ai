"""Does a recorded run reproduce? Re-running its request and comparing the numbers, with the likely causes when not.

Recording a run's configuration (#63) is half of reproducibility; the other half is showing it. A reproduction re-runs
the recorded request on the same data and compares every metric of the chosen model within `TOLERANCE`. Every model
here has a fixed seed, so on the same data, code and libraries the numbers should match to the last digit. When they
do not, the recorded environment, configuration and stopping reason say why, as far as they can.
"""

from __future__ import annotations

from datetime import datetime, timezone

from app.services import run_metadata

TOLERANCE = 1e-6


def _best(result: dict) -> dict:
    return next((r for r in result.get("results", []) if r.get("model") == result.get("best_model")), {}) or {}


def causes(entry: dict, result: dict, *, config_hash_now: str | None) -> list[str]:
    """What differs between when the run was made and now that could change its numbers."""
    found = []
    then, now = entry.get("environment") or {}, run_metadata.environment()
    labels = {"python": "Python", "scikit_learn": "scikit-learn", "pandas": "pandas", "numpy": "numpy",
              "joblib": "joblib"}
    for key, label in labels.items():
        if then.get(key) and now.get(key) and then[key] != now[key]:
            found.append(f"{label} was {then[key]} when the run was made and is {now[key]} now.")
    if not then:
        found.append("The run did not record its software versions, so a change there cannot be ruled out.")
    if entry.get("config_hash") and config_hash_now and entry["config_hash"] != config_hash_now:
        found.append("The app's defaults, seeds or processing limit have changed since the run was made.")
    request = entry.get("request") or {}
    if request.get("time_limit_seconds") or entry.get("stopped_because") == "time_limit":
        found.append(
            "A time limit stopped the search, and how far a search gets in a given time depends on how busy the "
            "machine is."
        )
    if entry.get("best_model") != result.get("best_model"):
        found.append(f"A different model was chosen ({entry.get('best_model')} then, {result.get('best_model')} now).")
    return found


def compare(entry: dict, result: dict, *, config_hash_now: str | None = None) -> dict:
    """The recorded run against its re-run: whether every shared metric matches within `TOLERANCE`, each
    difference, and the likely causes when they do not."""
    recorded, now = entry.get("metrics") or {}, _best(result).get("metrics") or {}
    differences = []
    for key in [k for k in recorded if k in now]:
        a, b = recorded[key], now[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            differences.append({"metric": key, "recorded": a, "now": b, "difference": round(float(b) - float(a), 9)})
    compared = bool(differences)
    reproduced = (
        compared
        and entry.get("best_model") == result.get("best_model")
        and all(abs(d["difference"]) <= TOLERANCE for d in differences)
    )
    likely = [] if reproduced else causes(entry, result, config_hash_now=config_hash_now)
    if not reproduced and not likely:
        likely = ["Nothing recorded explains it: some step is not fully deterministic on this machine."]
    if reproduced:
        sentence = "Re-running the recorded configuration on the same data gave the same numbers."
    elif not compared:
        sentence = "The re-run's scores could not be compared with the recorded ones."
    else:
        worst = max(differences, key=lambda d: abs(d["difference"]))
        sentence = (
            f"Re-running the recorded configuration gave different numbers: {worst['metric']} was "
            f"{worst['recorded']:.4f} and is {worst['now']:.4f}."
        )
    return {
        "run_id": entry.get("run_id"),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reproduced": reproduced,
        "tolerance": TOLERANCE,
        "differences": differences,
        "likely_causes": likely,
        "summary": sentence,
    }
