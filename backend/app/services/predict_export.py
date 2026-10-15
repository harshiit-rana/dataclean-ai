"""predict.py: the app's "use this on a new file", as a script that ships with the model.

Using an exported model used to mean running clean.py, rebuilding the derived inputs by hand, selecting the columns in
the right order and applying the cut-off -- every step a place for the export to behave differently from the app.
`predict.py` does all of it with the app's own code: the modules that clean, check, prepare and score a new file are
copied unchanged into an `app/services/` folder beside it (`runtime_files`), so the import paths -- and the pickled
model's references to the app's clipping and cost-weighting steps -- resolve as they do in the app. Only what is
specific to this model is written into the script: the frozen cleaning steps with every learned value, the model's
inputs and cut-off, and which category columns the training file held as numbers.

An acceptance test runs predict.py on a file and compares its output with the app's predictions file for the same
file, byte for byte.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict

import pandas as pd

from app.services import clipping, cost_weighting, features, operations, replay, scoring, validation

# The modules predict.py runs, copied verbatim. They import only pandas, numpy, scikit-learn and each other.
RUNTIME_MODULES = {
    "operations": operations,
    "features": features,
    "replay": replay,
    "scoring": scoring,
    "validation": validation,
    "clipping": clipping,
    "cost_weighting": cost_weighting,
}

_PACKAGE_DOC = '''"""The parts of DataClean AI that predict.py runs, copied from the app unchanged.

Keeping them as they are is the point: predict.py cleans, checks and scores a new file with exactly the code the app
uses, and model.joblib finds the app's own clipping and cost-weighting steps where it expects them.
"""
'''

_MODELING_STUB = '''"""Not shipped: the app's training code. scoring.py imports it for checks predict.py never runs."""


class ModelingError(Exception):
    pass
'''


def runtime_files() -> dict[str, str]:
    """Archive path -> text for the `app/` folder that sits beside predict.py."""
    files = {"app/__init__.py": _PACKAGE_DOC, "app/services/__init__.py": "", "app/services/modeling.py": _MODELING_STUB}
    for name, module in RUNTIME_MODULES.items():
        files[f"app/services/{name}.py"] = inspect.getsource(module)
    return files


def _plain(value):
    """Python literals only (numpy scalars and arrays become floats, ints and lists), so repr() writes valid source."""
    import numpy as np

    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    return value


def frozen_steps(original: pd.DataFrame, actions: list[dict]) -> list[dict]:
    """The applied fixes with every value they learned from the session's data, as the app freezes them."""
    steps, _ = replay.freeze(original, actions)
    return [_plain(asdict(step)) for step in steps]


def reference_numeric(reference: pd.DataFrame, result: dict) -> list[str]:
    """The category inputs the training file held as numbers: the app keeps those as numbers, the rest as text."""
    _, categorical = scoring.input_columns(result)
    return [c for c in categorical if c in reference.columns and pd.api.types.is_numeric_dtype(reference[c])]


def _result_for_scoring(result: dict) -> dict:
    """The part of a result that preparing and scoring a new file reads."""
    feats = result.get("features") or {}
    point = result.get("operating_point") or {}
    return _plain({
        "task": result.get("task"),
        "target_column": result.get("target_column"),
        "features": {k: feats.get(k, []) for k in ("numeric", "categorical", "frequent", "derived")},
        "operating_point": {k: point.get(k) for k in ("positive_label", "threshold")} if point else None,
    })


_TEMPLATE = '''"""Predicts with the model DataClean AI trained on __FILE__, on a new file, the way the app does.

    python predict.py new_rows.csv predictions.csv

It runs the same steps as the app's "use this on a new file", with the app's own code -- the app/ folder beside this
file, copied from it unchanged:

1. The approved cleaning, with every value it learned from the training data written in below (STEPS).
2. A check against schema.json: a missing column stops it; anything else is noted on the row it is in.
3. The derived inputs and the model's inputs, built as the app builds them.
4. model.joblib, checked against model.json first, answering at the cut-off the model was reported at.

Every input row is written back with its prediction, or the reason it has none (dataclean_note), and what the schema
check found in it (dataclean_validation). Needs pandas, scikit-learn and joblib: the versions are in model.json.
"""

import hashlib
import json
import sys
from io import StringIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from app.services import replay, scoring, validation  # noqa: E402

nan, inf = float("nan"), float("inf")  # how the literals below write a blank or an unbounded value

TARGET = __TARGET__
# What scoring reads from the model's result: its inputs, derived inputs and cut-off.
RESULT = __RESULT__
# The approved fixes in order, with what each learned from the training data.
STEPS = __STEPS__
# Category inputs the training file held as numbers; every other category input is compared as text.
REFERENCE_NUMERIC = __REFERENCE__


def read(path):
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(StringIO(text))


def load_model():
    model_file = HERE / "model.joblib"
    sidecar = HERE / "model.json"
    if sidecar.exists():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        if recorded.get("sha256") != hashlib.sha256(model_file.read_bytes()).hexdigest():
            sys.exit("model.joblib is not the file that was saved: its checksum does not match model.json. "
                     "It was not loaded.")
    return joblib.load(model_file)


def main(source, destination):
    new = read(source)
    applied = replay.apply(new, [replay.Step(**step) for step in STEPS], target_column=TARGET)
    schema = json.loads((HERE / "schema.json").read_text(encoding="utf-8"))
    report, row_notes = validation.validate(schema, applied.frame, TARGET)
    if report["blocking"]:
        sys.exit(validation.describe_blocking(report))
    reference = pd.DataFrame({column: pd.Series([0.0]) for column in REFERENCE_NUMERIC})
    X, notes = scoring.prepare_inputs(applied.frame, RESULT, reference=reference)
    predicted = scoring.predict(load_model(), X, RESULT) if len(X) else pd.DataFrame(index=X.index)
    output = scoring.assemble_output(new, applied.dropped, predicted, row_notes)
    output.to_csv(destination, index=False)
    for note in notes:
        print(note)
    print(f"Wrote {len(output)} rows to {destination}: {len(predicted)} predicted, "
          f"{report['rows_with_findings']} with something to check.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "new_rows.csv", sys.argv[2] if len(sys.argv) > 2 else "predictions.csv")
'''


def predict_in_app(new: pd.DataFrame, steps: list[dict], result: dict, reference: pd.DataFrame, pipeline, schema: dict):
    """What the app writes for `new`, step for step as predict.py does it, in this process with the app's own model.
    Returns the output frame, or the refusal message when the file cannot be scored."""
    applied = replay.apply(new, [replay.Step(**step) for step in steps], target_column=result.get("target_column"))
    report, row_notes = validation.validate(schema, applied.frame, result.get("target_column"))
    if report["blocking"]:
        return validation.describe_blocking(report)
    X, _ = scoring.prepare_inputs(applied.frame, result, reference=reference)
    predicted = scoring.predict(pipeline, X, result) if len(X) else pd.DataFrame(index=X.index)
    return scoring.assemble_output(new, applied.dropped, predicted, row_notes)


def build_predict_script(result: dict, steps: list[dict], reference: list[str], *, file_name: str) -> str:
    """predict.py for this model."""
    return (
        _TEMPLATE.replace("__FILE__", file_name)
        .replace("__TARGET__", repr(result.get("target_column")))
        .replace("__RESULT__", repr(_result_for_scoring(result)))
        .replace("__STEPS__", repr(steps))
        .replace("__REFERENCE__", repr(reference))
    )
