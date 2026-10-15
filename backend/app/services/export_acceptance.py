"""Does the export reproduce the app's model? Run it and compare, for this session's model, not a fixture.

`train_model.py` promises to rebuild the model the app trained, and the test suite checks that on fixtures. Nothing
checked it for the *user's* model, so a session's own edge case -- an unusual type, a derived input on a sparse column
-- could produce an export that quietly disagrees with the app. `verify` runs the exported `clean.py` and
`train_model.py` on the session's upload in a separate Python, with a time limit, and compares what the rebuilt model
says about each held-out row with what the app's model said (kept per run since #28): the same answers, and the same
probabilities within `TOLERANCE`. The outcome is saved as `acceptance.json` beside the model and goes in the bundle.

It never raises and never blocks a download: a failure to run is reported as a result like any other.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from app.services import cleaning_export, model_export, model_store, predict_export

TOLERANCE = 1e-9
TIMEOUT_SECONDS = 600
FILENAME = "acceptance.json"
FIRST_DIFFERENCES = 5


def _run(arguments: list[str], folder: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", *arguments], cwd=folder, capture_output=True, text=True,
                          timeout=timeout)


def _compare(app: dict, exported: pd.DataFrame) -> dict:
    """The app's held-out predictions against the export's, row by row."""
    ids = pd.Index(exported["row_id"])
    position = ids.get_indexer(app["row_ids"])
    missing = int((position < 0).sum())
    if missing:
        return {"status": "failed", "rows_compared": 0,
                "message": f"{missing} held-out rows of the app's model are not in the export's held-out rows: "
                           f"the two split the data differently."}
    rows = exported.iloc[position].reset_index(drop=True)
    differences = []
    called_app = np.asarray(app["called"]).astype(str)
    called_export = rows["called"].astype(str).to_numpy()
    worst = 0.0
    scores = app.get("scores")
    label = str(app.get("positive_label") or "")
    column = f"proba__{label}"
    for i in range(len(rows)):
        problem = None
        if called_app[i] != called_export[i]:
            problem = f"answered {called_export[i]!r} where the app answered {called_app[i]!r}"
        if scores is not None and column in rows.columns:
            gap = abs(float(scores[i]) - float(rows.loc[i, column]))
            worst = max(worst, gap)
            if problem is None and gap > TOLERANCE:
                problem = f"P({label}) {float(rows.loc[i, column]):.12f} where the app had {float(scores[i]):.12f}"
        if problem:
            differences.append({"row": int(app["row_ids"][i]) + 1, "difference": problem})
    passed = not differences
    return {
        "status": "passed" if passed else "failed",
        "rows_compared": int(len(rows)),
        "rows_differing": len(differences),
        "max_probability_difference": worst if scores is not None else None,
        "first_differences": differences[:FIRST_DIFFERENCES],
        "message": (
            f"The exported scripts rebuilt the model and it agrees with the app's on all {len(rows)} held-out rows."
            if passed
            else f"The rebuilt model disagrees with the app's on {len(differences)} of {len(rows)} held-out rows."
        ),
    }


NEVER_SEEN = "__a value never seen in training__"


def probe_frame(schema: dict) -> tuple[pd.DataFrame, list[str]]:
    """Rows built to break a deployment, from the schema: a typical row, then for each input column the same row
    with that column blank, with a category never seen in training, with a number written as text or not a number,
    with a date that is not a date. Returns the rows and what each one probes."""
    columns = schema.get("columns") or []
    typical: dict = {}
    for column in columns:
        if column["kind"] == "numeric":
            typical[column["name"]] = ((column.get("quantiles") or {}).get("0.5")
                                       if column.get("quantiles") else (column.get("range") or {}).get("min", 0))
        elif column["kind"] == "date":
            typical[column["name"]] = (column.get("range") or {}).get("min", "2020-01-01")
        elif column["kind"] == "text":
            typical[column["name"]] = "some text"
        else:
            typical[column["name"]] = next(iter(column.get("categories") or {"": 0}))
    rows, labels = [dict(typical)], ["a typical row"]

    def variant(name: str, value, label: str) -> None:
        rows.append({**typical, name: value})
        labels.append(label)

    for column in columns:
        name = column["name"]
        variant(name, None, f"{name} blank")
        if column["kind"] == "numeric":
            variant(name, f" {typical[name]} ", f"{name} written as text")
            variant(name, "n/a", f"{name} not a number")
        elif column["kind"] == "date":
            variant(name, "not a date", f"{name} not a date")
        elif column["kind"] == "categorical":
            variant(name, NEVER_SEEN, f"{name} a value never seen in training")
    return pd.DataFrame(rows), labels


def _probe(directory: Path, result: dict, actions: list[dict], folder: Path, timeout: int) -> dict | None:
    """The probe rows through the app's path and the exported predict.py: the outputs must agree row for row."""
    schema = result.get("input_schema")
    model_file = model_store.pipeline_path(directory)
    if not schema or model_file is None:
        return None
    original = pd.read_csv(directory / "original.csv")
    cleaned = directory / "cleaned.csv"
    reference = pd.read_csv(cleaned) if cleaned.exists() else original
    steps = predict_export.frozen_steps(original, actions)
    numeric = predict_export.reference_numeric(reference, result)
    bundle = folder / "bundle"
    for name, text in predict_export.runtime_files().items():
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_text(text, encoding="utf-8")
    (bundle / "predict.py").write_text(
        predict_export.build_predict_script(result, steps, numeric, file_name="probes.csv"), encoding="utf-8"
    )
    shutil.copy(model_file, bundle / model_store.PIPELINE_FILENAME)
    sidecar = model_file.parent / model_store.MODEL_SIDECAR_FILENAME
    if sidecar.exists():
        shutil.copy(sidecar, bundle / model_store.MODEL_SIDECAR_FILENAME)
    (bundle / model_store.SCHEMA_FILENAME).write_text(json.dumps(schema), encoding="utf-8")

    probes, labels = probe_frame(schema)
    probes.to_csv(bundle / "probes.csv", index=False)
    new = pd.read_csv(bundle / "probes.csv")
    pipeline, _ = model_store.load_pipeline(directory)
    app_out = predict_export.predict_in_app(new, steps, result, reference, pipeline, schema)
    done = _run(["predict.py", "probes.csv", "out.csv"], bundle, timeout)
    if isinstance(app_out, str) or done.returncode != 0:
        same = isinstance(app_out, str) and done.returncode != 0 and app_out in done.stderr
        return {"rows": len(labels), "agreed": len(labels) if same else 0,
                "disagreements": [] if same else [{"probe": "the whole file", "app": str(app_out)[:300],
                                                   "export": done.stderr[-300:]}]}
    buffer = io.StringIO()
    app_out.to_csv(buffer, index=False)
    ours = pd.read_csv(io.StringIO(buffer.getvalue()), dtype=str, keep_default_na=False)
    theirs = pd.read_csv(bundle / "out.csv", dtype=str, keep_default_na=False)
    disagreements = []
    for i, label in enumerate(labels):
        if i >= len(theirs) or ours.iloc[i].to_dict() != theirs.iloc[i].to_dict():
            disagreements.append({"probe": label, "app": ours.iloc[i].to_dict(),
                                  "export": theirs.iloc[i].to_dict() if i < len(theirs) else None})
    return {"rows": len(labels), "agreed": len(labels) - len(disagreements),
            "disagreements": disagreements[:FIRST_DIFFERENCES]}


def verify(directory: Path, result: dict, actions: list[dict] | None, *, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Rebuild the model from the export and compare it with the app's; save and return the outcome."""
    outcome: dict = {"run_id": result.get("run_id"), "tolerance": TOLERANCE, "timeout_seconds": timeout}
    folder = Path(tempfile.mkdtemp(prefix="dataclean-acceptance-"))
    try:
        app_predictions = model_store.load_run_predictions(directory, result.get("run_id") or "")
        if app_predictions is None:
            outcome.update(status="not_checked",
                           message="The app's model did not keep its held-out predictions, so there is nothing to "
                                   "compare the export with.")
            return outcome
        original = pd.read_csv(directory / "original.csv")
        shutil.copy(directory / "original.csv", folder / "original.csv")
        applied = [a for a in (actions or []) if a.get("status") == "applied"]
        if applied:
            (folder / "clean.py").write_text(
                cleaning_export.build_cleaning_script(original, applied, file_name="original.csv"), encoding="utf-8"
            )
            cleaned = _run(["clean.py", "original.csv", "cleaned.csv"], folder, timeout)
            if cleaned.returncode != 0:
                outcome.update(status="failed", message="clean.py did not run.", error=cleaned.stderr[-2000:])
                return outcome
        else:
            shutil.copy(folder / "original.csv", folder / "cleaned.csv")
        (folder / "train_model.py").write_text(
            model_export.build_training_script(result, file_name="cleaned.csv"), encoding="utf-8"
        )
        trained = _run(["train_model.py", "cleaned.csv", "--held-out-predictions", "held_out.csv"], folder, timeout)
        if trained.returncode != 0:
            outcome.update(status="failed", message="train_model.py did not run.", error=trained.stderr[-2000:])
            return outcome
        outcome.update(_compare(app_predictions, pd.read_csv(folder / "held_out.csv")))
        # Then the inputs that break real deployments, through the app and through predict.py.
        probes = _probe(directory, result, actions or [], folder, timeout)
        if probes is not None:
            outcome["probes"] = probes
            if probes["disagreements"]:
                outcome["status"] = "failed"
                outcome["message"] += (
                    f" On {len(probes['disagreements'])} of {probes['rows']} probe rows (blanks, unseen values, "
                    f"wrong types) the exported predict.py and the app disagree."
                )
            else:
                outcome["message"] += (
                    f" On all {probes['rows']} probe rows -- blanks, values never seen in training, wrong types -- "
                    f"predict.py and the app agree."
                )
        return outcome
    except subprocess.TimeoutExpired:
        outcome.update(status="error", message=f"The export took longer than {timeout} seconds to rebuild, so it "
                                               f"was stopped. The download is unaffected.")
        return outcome
    except Exception as exc:  # noqa: BLE001 -- a check that cannot run is a result, never a failed export
        outcome.update(status="error", message=f"The check could not run: {exc}")
        return outcome
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        outcome["at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(directory, result, outcome)


def _save(directory: Path, result: dict, outcome: dict) -> None:
    try:
        run_id = result.get("run_id")
        if run_id and model_store.run_dir(directory, run_id).is_dir():
            (model_store.run_dir(directory, run_id) / FILENAME).write_text(json.dumps(outcome, indent=2),
                                                                           encoding="utf-8")
    except OSError:
        pass


def load(directory: Path) -> dict | None:
    """The last acceptance check of the model in use, or None."""
    run_id = model_store.current_run_id(directory)
    path = model_store.run_dir(directory, run_id) / FILENAME if run_id else None
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
