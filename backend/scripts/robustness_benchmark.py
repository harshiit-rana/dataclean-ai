"""What happens to a model trained on a clean file when the new file it is applied to is messy.  (issue #128)

The idea is Jenga's (Schelter et al., "JENGA - A Framework to Study the Impact of Data Errors on the Predictions of
Machine Learning Models", EDBT 2021): train on clean data, corrupt only the data the model is later asked about, and
measure how far its score falls. The app's own route for a new file is used -- `validation.validate` against the
model's input schema, `scoring.prepare_inputs`, then the fitted pipeline -- so the benchmark also records whether the
app *warned* about the mess, and whether it failed outright.

For each dataset (the cleaning benchmark's six, through CSV as uploads) and seed (one official fold each), the plain
train is fitted once on the clean training rows. Its test rows are then scored clean, and once per kind of mess, with
20% of the rows of up to three input columns corrupted:

- `missing_values`: blanked;
- `outliers`: multiplied by 10, 100 or 1000;
- `type_mismatch`: replaced by "?", "unknown" or "--";
- `inconsistent_casing`: written in capitals or title case;
- `stray_whitespace`: padded with spaces;
- `unseen_categories`: replaced by a value the column never had.

Per kind: the mean fall in score (AUC, log loss or RMSE, as in the accuracy benchmark; a positive fall is worse), how
often the app's checks named every corrupted column, and how often scoring failed.

Usage (from `backend/`):
    python scripts/robustness_benchmark.py
    python scripts/robustness_benchmark.py --datasets credit-g --seeds 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "benchmark-key-not-real")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import accuracy_benchmark as accuracy  # noqa: E402
import cleaning_benchmark as cleaning  # noqa: E402
import openml_tasks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services import modeling, scoring, validation  # noqa: E402

KINDS = ["missing_values", "outliers", "type_mismatch", "inconsistent_casing", "stray_whitespace", "unseen_categories"]
SHARE = 0.20
COLUMNS_PER_KIND = 3


def corrupt(frame: pd.DataFrame, inputs: list[str], kind: str, seed: int) -> tuple[pd.DataFrame, list[str]]:
    """`frame` with one kind of mess in up to three of the model's input columns, and which columns."""
    rng = np.random.default_rng(seed)
    numeric = [c for c in cleaning._numeric_candidates(frame, "") if c in inputs]
    text = [c for c in frame.columns if c in inputs and frame[c].dtype == object]
    if kind in ("inconsistent_casing",):
        text = [c for c in text if c in cleaning._text_candidates(frame, "", with_letters=True)]
    pool = {"missing_values": numeric + text, "outliers": numeric, "type_mismatch": numeric,
            "inconsistent_casing": text, "stray_whitespace": text, "unseen_categories": text}[kind]
    if not pool:
        return frame, []
    columns = list(rng.choice(pool, size=min(COLUMNS_PER_KIND, len(pool)), replace=False))
    out = frame.copy()
    for column in columns:
        rows = cleaning._some(rng, len(out), SHARE)
        if kind == "missing_values":
            out[column] = out[column].astype(object) if out[column].dtype == object else out[column].astype(float)
            out.loc[rows, column] = np.nan
        elif kind == "outliers":
            out[column] = out[column].astype(float)
            out.loc[rows, column] = out.loc[rows, column] * rng.choice([10, 100, 1000], len(rows))
        elif kind == "type_mismatch":
            out[column] = out[column].astype(object)
            out.loc[rows, column] = rng.choice(cleaning.TEXT_FOR_NUMBER, len(rows))
        elif kind == "inconsistent_casing":
            values = out.loc[rows, column].astype(str)
            out.loc[rows, column] = [v.upper() if v.upper() != v else v.title() for v in values]
        elif kind == "stray_whitespace":
            out.loc[rows, column] = [f" {v} " for v in out.loc[rows, column].astype(str)]
        else:
            out.loc[rows, column] = "never-seen-before"
    return openml_tasks.as_uploaded(out), [str(c) for c in columns]


def _score(pipeline, result: dict, frame: pd.DataFrame, reference: pd.DataFrame, target: str, kind: str,
           labels) -> float:
    X, _ = scoring.prepare_inputs(frame, result, reference=reference)
    if kind == "regression":
        prediction = accuracy.Prediction(values=np.asarray(pipeline.predict(X), dtype=float))
    else:
        prediction = accuracy.Prediction(proba=np.asarray(pipeline.predict_proba(X)), classes=list(pipeline.classes_))
    return accuracy.score(kind, frame[target], prediction, labels)[accuracy.METRICS[kind]]


def run_one(loaded: openml_tasks.LoadedTask, seed: int) -> dict:
    target, kind = loaded.target, loaded.task.kind
    train_rows, test_rows = loaded.folds[seed % len(loaded.folds)]
    train = loaded.frame.iloc[train_rows].reset_index(drop=True)
    test = loaded.frame.iloc[test_rows].reset_index(drop=True)
    task = "regression" if kind == "regression" else "classification"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = modeling.train_candidates(
            train, target_column=target, task=task, model_names=[s.name for s in modeling.models_for_task(task)],
            include_deep_diagnostics=False, cost_limit=get_settings().training_cost_limit or None,
        )
    pipeline = next(r for r in result["results"] if r["model"] == result["best_model"])["pipeline"]
    numeric, categorical = scoring.input_columns(result)
    inputs = numeric + categorical
    labels = None if kind == "regression" else sorted(loaded.frame[target].dropna().unique().tolist(), key=str)
    metric = accuracy.METRICS[kind]
    clean_score = _score(pipeline, result, test, train, target, kind, labels)
    out: dict = {"metric": metric, "clean": clean_score, "kinds": {}}
    for mess in KINDS:
        messy, columns = corrupt(test, inputs, mess, seed)
        if not columns:
            continue
        entry: dict = {"columns": columns}
        report, _ = validation.validate(result["input_schema"], messy, target)
        named = {f["column"] for f in report.get("findings", [])}
        entry["warned"] = all(c in named for c in columns)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = _score(pipeline, result, messy, train, target, kind, labels)
            fall = (clean_score - value) if accuracy.HIGHER_IS_BETTER[metric] else (value - clean_score)
            entry.update({"score": value, "fall": fall})
        except Exception as exc:  # noqa: BLE001 -- a failure is the finding
            entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["kinds"][mess] = entry
    return out


def run(datasets: list[str], seeds: int, cache: Path, progress: Path | None = None) -> dict:
    """Every dataset at every seed. With `progress`, the results so far are written there after each seed, and an
    interrupted run with the same settings carries on from them (see `accuracy_benchmark.interrupted_run`)."""
    results: dict = {"python": platform.python_version(), "seeds": seeds, "share": SHARE, "datasets": {}}
    earlier = accuracy.interrupted_run(progress, results, ("seeds", "share"))
    if earlier:
        results["datasets"] = {k: v for k, v in earlier.get("datasets", {}).items() if k in datasets}
        done = sum(len(e["runs"]) for e in results["datasets"].values())
        print(f"Picking up {done} finished run(s) from {progress}.", flush=True)
    for name in datasets:
        loaded = openml_tasks.load(openml_tasks.select([name])[0], cache)
        entry: dict = results["datasets"].setdefault(name, {"kind": loaded.task.kind, "runs": []})
        for seed in range(len(entry["runs"]), seeds):
            one = run_one(loaded, seed)
            entry["runs"].append(one)
            print(f"{name} seed {seed}: {one['metric']} clean {one['clean']:.4f}; " + ", ".join(
                f"{k} {'FAILED' if 'error' in v else format(v['fall'], '+.4f')}{'' if v['warned'] else ' (no warning)'}"
                for k, v in one["kinds"].items()), flush=True)
            accuracy.save_progress(progress, results)
    results["summary"] = summarise(results)
    return results


def summarise(results: dict) -> dict:
    per_kind: dict[str, dict] = {}
    for name, entry in results["datasets"].items():
        for one in entry["runs"]:
            # Falls are relative to the clean score's size, so AUC, log loss and RMSE can be averaged.
            scale = max(abs(one["clean"]), 1e-9)
            for mess, v in one["kinds"].items():
                k = per_kind.setdefault(mess, {"runs": 0, "warned": 0, "failed": 0, "relative_falls": []})
                k["runs"] += 1
                k["warned"] += int(v["warned"])
                if "error" in v:
                    k["failed"] += 1
                else:
                    k["relative_falls"].append(v["fall"] / scale)
    return {mess: {"runs": k["runs"], "warned": k["warned"], "failed": k["failed"],
                   "mean_relative_fall": round(float(np.mean(k["relative_falls"])), 4) if k["relative_falls"] else None,
                   # A log loss near zero turns an ordinary fall into a fall of hundreds of times the clean score, and
                   # one such file carries the mean (car's 0.002 did): the median says what a typical file loses.
                   "median_relative_fall": round(float(np.median(k["relative_falls"])), 4) if k["relative_falls"] else None}
            for mess, k in per_kind.items()}


def markdown(results: dict) -> str:
    lines = [f"{len(results['datasets'])} datasets, {results['seeds']} seed(s), {results['share']:.0%} of the rows of up "
             f"to {COLUMNS_PER_KIND} input columns corrupted per kind.", "",
             "| Mess in the new file | Runs | Fall in score (share of the clean score), median | mean | App warned "
             "| Failed |",
             "|---|---:|---:|---:|---:|---:|"]
    for mess, k in results["summary"].items():
        fall = {s: "" if k.get(f"{s}_relative_fall") is None else f"{k[f'{s}_relative_fall']:+.1%}"
                for s in ("median", "mean")}
        lines.append(f"| {mess} | {k['runs']} | {fall['median']} | {fall['mean']} | {k['warned']} of {k['runs']} "
                     f"| {k['failed']} |")
    lines += ["", "| Dataset | Metric | Clean | " + " | ".join(KINDS) + " |", "|---|---|---:|" + "---:|" * len(KINDS)]
    for name, entry in results["datasets"].items():
        runs = entry["runs"]
        cells = []
        for mess in KINDS:
            scores = [r["kinds"][mess]["score"] for r in runs if mess in r["kinds"] and "score" in r["kinds"][mess]]
            failed = any("error" in r["kinds"].get(mess, {}) for r in runs)
            cells.append("failed" if failed else (f"{np.mean(scores):.4f}" if scores else "n/a"))
        lines.append(f"| {name} | {runs[0]['metric']} | {np.mean([r['clean'] for r in runs]):.4f} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=cleaning.DATASETS)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--cache", type=Path, default=openml_tasks.DEFAULT_CACHE)
    parser.add_argument("--out", default="robustness_benchmark.json")
    parser.add_argument("--markdown", help="also write the tables to this file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    logging.getLogger("app").setLevel(logging.ERROR)
    # The output file doubles as the progress file, so running the same command again carries on.
    results = run(args.datasets, args.seeds, args.cache, progress=Path(args.out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = markdown(results)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
