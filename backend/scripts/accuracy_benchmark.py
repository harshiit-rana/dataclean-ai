"""How accurate DataClean AI's models are next to other systems, on public AutoML Benchmark tasks.  (issue #124)

Every system trains on the same official OpenML training fold and is scored on the same test fold
(`openml_tasks.py` has the tasks and why they were chosen). Metrics are AMLB's: AUC for a two-class target,
log loss for more classes, RMSE for a numeric target.

Systems:
- `constant`: class shares or the mean, from the training fold. The floor every system must clear.
- `random_forest`: scikit-learn's random forest, 500 trees, otherwise defaults. The other reference point.
- `hist_gradient_boosting`: scikit-learn's histogram gradient boosting with defaults, the strong plain model.
- `dataclean_train`: the app's plain train (`modeling.train_candidates` with every ordinary model for the task).
- `dataclean_improve`: the app's improvement loop with the time budget, then the rebuild the route makes.
- `flaml`: FLAML, an AutoML library, with the same time budget. Skipped, and said so, when it is not installed
  (`pip install -r requirements-benchmark.txt`).

The two DataClean AI systems hold back part of the training fold for their own test, as the app does; the
configuration they choose is then refitted on the whole training fold before it sees the test fold, as FLAML
refits its choice by default, so no system trains on fewer rows than another.

A fold where a system fails counts as the constant predictor's score, as in AMLB, and the failure is listed.

The summary, for each system: the normalised score (0 is the constant predictor, 1 the random forest; the same idea
as AMLB's scaled score), its average rank across tasks, and the improvement loop's wins, ties and losses against
every other system. A tie is a difference within one standard error of the per-fold differences. Development and
held-out tasks are summarised apart: a change to the app's defaults is judged on development tasks, and the
held-out tasks say whether it carries over.

Usage (from `backend/`):
    python scripts/accuracy_benchmark.py                                  # every task, 2 folds, 120 s budgets
    python scripts/accuracy_benchmark.py --tasks credit-g boston --folds 1 --budget 20
    python scripts/accuracy_benchmark.py --systems constant random_forest dataclean_train
    python scripts/accuracy_benchmark.py --baseline last.json --tolerance 0.1   # flag drops against a run
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ANTHROPIC_API_KEY", "benchmark-key-not-real")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402

import openml_tasks  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services import features as feature_recipes  # noqa: E402
from app.services import improvement, modeling  # noqa: E402

METRICS = {"binary": "auc", "multiclass": "log_loss", "regression": "rmse"}
HIGHER_IS_BETTER = {"auc": True, "log_loss": False, "rmse": False}
SYSTEMS = ["constant", "random_forest", "hist_gradient_boosting", "dataclean_train", "dataclean_improve", "flaml"]
FOCUS = "dataclean_improve"  # the system the wins and losses are counted for
IMPROVE_BUDGET = 24  # the improvement route's default number of settings tried per search
EPS = 1e-15


class Prediction:
    """What a system hands back for the test rows: class probabilities with their labels, or numbers."""

    def __init__(self, *, proba: np.ndarray | None = None, classes: list | None = None, values: np.ndarray | None = None):
        self.proba, self.classes, self.values = proba, classes, values


# --- scoring ----------------------------------------------------------------------------------------------------


def aligned_proba(prediction: Prediction, labels: list) -> np.ndarray:
    """The probabilities with one column per label in `labels`, a label the system never saw getting none, clipped
    away from 0 and 1 and renormalised so the log loss of a confident miss is large rather than infinite."""
    assert prediction.proba is not None and prediction.classes is not None
    # Matched as text, so a system that turned the labels 1 and 2 into "1" and "2" is matched up all the same.
    columns = {str(c): i for i, c in enumerate(prediction.classes)}
    out = np.zeros((prediction.proba.shape[0], len(labels)))
    for j, label in enumerate(labels):
        if str(label) in columns:
            out[:, j] = prediction.proba[:, columns[str(label)]]
    out = np.clip(out, EPS, 1 - EPS)
    return out / out.sum(axis=1, keepdims=True)


def score(kind: str, y_true: pd.Series, prediction: Prediction, labels: list | None = None) -> dict:
    """AMLB's metric for the kind of target, one plain-language companion (accuracy or R²) and, for classes, the
    Brier score: the mean squared gap between the probabilities and what happened, so a model that says 80% should
    be right about 80% of the time to score well. For two classes it is the usual 0-1 form on the later label; for
    more, the sum over classes (0-2)."""
    if kind == "regression":
        assert prediction.values is not None
        values = np.asarray(prediction.values, dtype=float)
        return {
            "rmse": float(math.sqrt(mean_squared_error(y_true, values))),
            "r2": float(r2_score(y_true, values)),
        }
    labels = labels if labels is not None else sorted(set(y_true) | set(prediction.classes or []), key=str)
    proba = aligned_proba(prediction, labels)
    # Compared as text: scikit-learn cannot tell what kind of target an object array of numbers is.
    names = [str(label) for label in labels]
    truth = np.asarray([str(v) for v in y_true])
    predicted = np.asarray(names)[proba.argmax(axis=1)]
    out = {"accuracy": float(accuracy_score(truth, predicted))}
    happened = (truth[:, None] == np.asarray(names)[None, :]).astype(float)
    squared = (proba - happened) ** 2
    out["brier"] = float(squared[:, -1].mean() if kind == "binary" else squared.sum(axis=1).mean())
    if kind == "binary":
        out["auc"] = float(roc_auc_score(truth == names[-1], proba[:, -1]))
    else:
        out["log_loss"] = float(log_loss(truth, proba, labels=names))
    return out


# --- the systems --------------------------------------------------------------------------------------------------


def _split_xy(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return frame.drop(columns=[target]), frame[target]


def _encoded(X: pd.DataFrame) -> tuple[ColumnTransformer, list[bool]]:
    """Medians for blank numbers, an ordinal code for text (unseen and blank text both -1): the least a
    scikit-learn model needs to run, so the reference systems are plain."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    text = [c for c in X.columns if c not in numeric]
    transformer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
            ("text", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), text),
        ],
        verbose_feature_names_out=False,
    )
    categorical = [False] * len(numeric) + [X[c].nunique() < 250 for c in text]
    return transformer, categorical


def _as_text(X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    for c in out.columns:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].astype(object).where(out[c].notna(), np.nan)
    return out


def _predict(model, X_test: pd.DataFrame, kind: str) -> Prediction:
    if kind == "regression":
        return Prediction(values=np.asarray(model.predict(X_test), dtype=float))
    return Prediction(proba=np.asarray(model.predict_proba(X_test)), classes=list(model.classes_))


def run_constant(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, budget: float) -> Prediction:
    y = train[target]
    if kind == "regression":
        return Prediction(values=np.full(len(test), float(y.mean())))
    shares = y.value_counts(normalize=True)
    classes = list(shares.index)
    return Prediction(proba=np.tile(shares.to_numpy(), (len(test), 1)), classes=classes)


def _plain_sklearn(estimator_for: Callable[[list[bool]], object]):
    def run(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, budget: float) -> Prediction:
        X, y = _split_xy(train, target)
        X = _as_text(X)
        transformer, categorical = _encoded(X)
        model = Pipeline([("encode", transformer), ("model", estimator_for(categorical, kind))])
        model.fit(X, y)
        return _predict(model, _as_text(test.drop(columns=[target])), kind)

    return run


run_random_forest = _plain_sklearn(
    lambda categorical, kind: (RandomForestRegressor if kind == "regression" else RandomForestClassifier)(
        n_estimators=500, n_jobs=-1, random_state=0
    )
)
run_hist_gradient_boosting = _plain_sklearn(
    lambda categorical, kind: (HistGradientBoostingRegressor if kind == "regression" else HistGradientBoostingClassifier)(
        categorical_features=categorical if any(categorical) else None, random_state=0
    )
)


def _refit_on_whole_fold(pipeline, train: pd.DataFrame, test: pd.DataFrame, target: str, recipes_wanted: list[str],
                         withheld: list[str], kind: str) -> Prediction:
    """The configuration the app chose, refitted on every row of the training fold and asked about the test fold,
    with the same derived inputs added to both."""
    recipes = feature_recipes.select(train, target, withheld, recipes_wanted)
    train_x = feature_recipes.apply(train, recipes)
    test_x = feature_recipes.apply(test, recipes)
    columns = list(getattr(pipeline, "feature_names_in_", [c for c in train_x.columns if c != target]))
    refitted = clone(pipeline)
    refitted.fit(train_x[columns], train_x[target])
    return _predict(refitted, test_x[columns], kind)


def _cost_limit() -> int | None:
    return get_settings().training_cost_limit or None


def run_dataclean_train(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, budget: float) -> Prediction:
    task = "regression" if kind == "regression" else "classification"
    result = modeling.train_candidates(
        train, target_column=target, task=task, model_names=[s.name for s in modeling.models_for_task(task)],
        include_deep_diagnostics=False, cost_limit=_cost_limit(),
    )
    best = next(r for r in result["results"] if r["model"] == result["best_model"])
    return _refit_on_whole_fold(best["pipeline"], train, test, target, [], [], kind)


def run_dataclean_improve(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, budget: float) -> Prediction:
    """The loop, then the rebuild exactly as routes_model makes it: the winner's settings, without the inputs the
    loop removed, with the derived inputs it settled on."""
    task = "regression" if kind == "regression" else "classification"
    found = improvement.improve(
        train, target_column=target, task=task, budget=IMPROVE_BUDGET, time_limit_seconds=budget,
        cost_limit=_cost_limit(),
    )
    best = found["best"]
    withheld = [r["column"] for r in best.get("removed_columns", [])]
    recipes = best.get("feature_recipes", [])
    rebuilt = modeling.train_candidates(
        train, target_column=target, task=task, model_names=[best["model"]], excluded_columns=withheld,
        include_deep_diagnostics=False, model_params={best["model"]: dict(best["params"])},
        feature_recipes_wanted=recipes, cost_limit=_cost_limit(),
    )
    chosen = next(r for r in rebuilt["results"] if r["model"] == rebuilt["best_model"])
    return _refit_on_whole_fold(chosen["pipeline"], train, test, target, recipes, withheld, kind)


def flaml_available() -> bool:
    try:
        import flaml.automl  # noqa: F401
    except ImportError:
        return False
    return True


def run_flaml(train: pd.DataFrame, test: pd.DataFrame, target: str, kind: str, budget: float) -> Prediction:
    from flaml import AutoML

    def prepared(X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c in out.columns:
            if not pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].astype("category")
        return out

    X, y = _split_xy(train, target)
    X_test = test.drop(columns=[target])
    # One set of categories for both, so a category FLAML never saw in training is a known blank, not an error.
    both = prepared(pd.concat([X, X_test]))
    X, X_test = both.iloc[: len(X)], both.iloc[len(X):]
    automl = AutoML()
    automl.fit(
        X, y, task="regression" if kind == "regression" else "classification", time_budget=budget,
        metric={"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[kind],
        seed=0, verbose=0,
    )
    if kind == "regression":
        return Prediction(values=np.asarray(automl.predict(X_test), dtype=float))
    return Prediction(proba=np.asarray(automl.predict_proba(X_test)), classes=list(automl.classes_))


RUNNERS: dict[str, Callable[..., Prediction]] = {
    "constant": run_constant,
    "random_forest": run_random_forest,
    "hist_gradient_boosting": run_hist_gradient_boosting,
    "dataclean_train": run_dataclean_train,
    "dataclean_improve": run_dataclean_improve,
    "flaml": run_flaml,
}


# --- running ------------------------------------------------------------------------------------------------------


def run_fold(loaded: openml_tasks.LoadedTask, fold: int, systems: list[str], budget: float) -> dict:
    train_rows, test_rows = loaded.folds[fold]
    train = loaded.frame.iloc[train_rows].reset_index(drop=True)
    test = loaded.frame.iloc[test_rows].reset_index(drop=True)
    kind, target = loaded.task.kind, loaded.target
    labels = None if kind == "regression" else sorted(loaded.frame[target].dropna().unique().tolist(), key=str)
    out: dict = {}
    # The worker processes joblib starts inherit the environment, not the filter below: without both, one run wrote
    # 23 MB of the same warning to the log.
    os.environ["PYTHONWARNINGS"] = "ignore"
    for name in systems:
        started = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prediction = RUNNERS[name](train, test, target, kind, budget)
            out[name] = {**score(kind, test[target], prediction, labels), "seconds": round(time.perf_counter() - started, 1)}
        except Exception as exc:  # noqa: BLE001 -- one system failing must not stop the others
            out[name] = {"error": f"{type(exc).__name__}: {exc}"[:500], "seconds": round(time.perf_counter() - started, 1)}
    return out


def interrupted_run(progress: Path | None, current: dict, settings: tuple[str, ...]) -> dict | None:
    """What an interrupted run with the same settings left in `progress`, if it did. A run of hours on a laptop can
    be stopped by sleep or by the system reclaiming memory, and starting over is the expensive way to recover. A
    finished run has its summary in the file and is not picked up: the same command run again after a change has to
    measure the change, not repeat the last answer."""
    if progress is None or not Path(progress).exists():
        return None
    earlier = json.loads(Path(progress).read_text(encoding="utf-8"))
    if "summary" in earlier or any(earlier.get(k) != current[k] for k in settings):
        return None
    return earlier


def save_progress(progress: Path | None, results: dict) -> None:
    """The results so far, written whole or not at all: a run stopped halfway through a write must not leave a file
    the next run cannot read."""
    if progress is None:
        return
    Path(progress).parent.mkdir(parents=True, exist_ok=True)
    partial = Path(f"{progress}.partial")
    partial.write_text(json.dumps(results, indent=2), encoding="utf-8")
    partial.replace(progress)


def run(tasks: list[openml_tasks.Task], folds: int, systems: list[str], budget: float, cache: Path,
        suite: str = "amlb", progress: Path | None = None) -> dict:
    """Every task's folds for every system. With `progress`, the results so far are written there after each task,
    and the finished tasks of an interrupted run with the same folds, budget and systems are kept, not run again."""
    results: dict = {
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "folds": folds,
        "budget_seconds": budget,
        "systems": systems,
        "suite": suite,
        "tasks": {},
    }
    earlier = interrupted_run(progress, results, ("folds", "budget_seconds", "systems", "suite"))
    if earlier:
        results["tasks"] = earlier.get("tasks", {})
        print(f"Picking up {len(results['tasks'])} finished task(s) from {progress}.", flush=True)
    for task in tasks:
        if str(task.id) in results["tasks"]:
            continue
        loaded = openml_tasks.load(task, cache)
        entry = {"name": task.name, "kind": task.kind, "held_out": task.held_out, "metric": METRICS[task.kind],
                 "rows": int(len(loaded.frame)), "columns": int(loaded.frame.shape[1] - 1), "folds": []}
        for fold in range(min(folds, len(loaded.folds))):
            scores = run_fold(loaded, fold, systems, budget)
            entry["folds"].append(scores)
            metric = METRICS[task.kind]
            print(f"{task.name} fold {fold}: " + ", ".join(
                f"{s} {v[metric]:.4f} ({v['seconds']:.0f}s)" if metric in v else f"{s} FAILED ({v['error'][:80]})"
                for s, v in scores.items()), flush=True)
        results["tasks"][str(task.id)] = entry
        save_progress(progress, results)
    results["summary"] = summarise(results)
    return results


# --- summarising --------------------------------------------------------------------------------------------------


def fold_values(entry: dict, system: str) -> list[float] | None:
    """The system's metric on each fold, a failed fold taking the constant predictor's score (AMLB's rule). None
    when the system was not run on this task."""
    metric = entry["metric"]
    values = []
    for fold in entry["folds"]:
        if system not in fold:
            return None
        got = fold[system].get(metric)
        if got is None or (isinstance(got, float) and math.isnan(got)):
            got = (fold.get("constant") or {}).get(metric)
        if got is None:
            return None
        values.append(float(got))
    return values or None


def normalised(value: float, constant: float, reference: float, higher_better: bool) -> float | None:
    """0 at the constant predictor, 1 at the random forest, above 1 when better than the forest."""
    gap = (reference - constant) if higher_better else (constant - reference)
    if abs(gap) < 1e-12:
        return None
    return ((value - constant) if higher_better else (constant - value)) / gap


def compare(a: list[float], b: list[float], higher_better: bool) -> str:
    """'win', 'tie' or 'loss' for `a` against `b` on paired folds: a tie when the mean difference is within one
    standard error of the per-fold differences (with one fold, within a hair)."""
    diffs = np.asarray(a) - np.asarray(b)
    if not higher_better:
        diffs = -diffs
    mean = float(diffs.mean())
    se = float(diffs.std(ddof=1) / math.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    if abs(mean) <= max(se, 1e-9):
        return "tie"
    return "win" if mean > 0 else "loss"


def summarise(results: dict) -> dict:
    systems = results["systems"]
    groups = {"development": [], "held_out": [], "all": []}
    for task_id, entry in results["tasks"].items():
        groups["held_out" if entry["held_out"] else "development"].append(task_id)
        groups["all"].append(task_id)

    summary: dict = {}
    for group, ids in groups.items():
        normal: dict[str, list[float]] = {s: [] for s in systems}
        ranks: dict[str, list[float]] = {s: [] for s in systems}
        record: dict[str, dict[str, int]] = {s: {"win": 0, "tie": 0, "loss": 0} for s in systems if s != FOCUS}
        failures: dict[str, int] = {s: 0 for s in systems}
        for task_id in ids:
            entry = results["tasks"][task_id]
            higher = HIGHER_IS_BETTER[entry["metric"]]
            means = {}
            for s in systems:
                values = fold_values(entry, s)
                failures[s] += sum(1 for fold in entry["folds"] if "error" in fold.get(s, {}))
                if values is not None:
                    means[s] = float(np.mean(values))
            if "constant" in means and "random_forest" in means:
                for s, m in means.items():
                    n = normalised(m, means["constant"], means["random_forest"], higher)
                    if n is not None:
                        normal[s].append(n)
            ordered = sorted(means, key=lambda s: -means[s] if higher else means[s])
            for s in means:
                tied = [o for o in ordered if math.isclose(means[o], means[s], rel_tol=0, abs_tol=1e-12)]
                ranks[s].append(float(np.mean([ordered.index(o) + 1 for o in tied])))
            focus = fold_values(entry, FOCUS)
            if focus is not None:
                for s in record:
                    other = fold_values(entry, s)
                    if other is not None:
                        record[s][compare(focus, other, higher)] += 1
        summary[group] = {
            "tasks": len(ids),
            "normalised_score": {s: round(float(np.mean(v)), 4) if v else None for s, v in normal.items()},
            # The mean can be carried by one task: where the forest barely beats the constant predictor (cmc), the
            # scale is a tiny gap and small differences become scores of +10 or -5. The median is not moved by it.
            "median_normalised_score": {s: round(float(np.median(v)), 4) if v else None for s, v in normal.items()},
            "average_rank": {s: round(float(np.mean(v)), 2) if v else None for s, v in ranks.items()},
            f"{FOCUS}_against": record if FOCUS in systems else {},
            "failed_folds": failures,
            "brier": mean_brier(results, ids),
            "significance": significance(results, ids),
        }
    return summary


def mean_brier(results: dict, ids: list[str]) -> dict[str, float | None]:
    """Each system's Brier score averaged over the classification tasks' folds: a rough calibration check, lower is
    better. Failed folds are left out."""
    out: dict[str, float | None] = {}
    for system in results["systems"]:
        values = [fold[system]["brier"] for task_id in ids for fold in results["tasks"][task_id]["folds"]
                  if "brier" in fold.get(system, {})]
        out[system] = round(float(np.mean(values)), 4) if values else None
    return out


# Two-tailed Nemenyi critical values at the 5% level for k = 2..10 systems (Demsar, JMLR 2006, table 5a).
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def significance(results: dict, ids: list[str]) -> dict | None:
    """Whether the systems differ beyond chance across tasks, the way systems are compared over many datasets
    (Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 2006):

    - the Friedman test on the systems' ranks per task, in Iman and Davenport's F form, whose p-value says whether
      any system differs from any other;
    - the Nemenyi critical difference: two systems whose average ranks differ by less are not told apart at 5%;
    - Wilcoxon signed-rank tests of the improvement loop against each other system on the per-task normalised
      scores, Holm-corrected for the number of comparisons. More sensitive than the ranks, as it uses how much one
      system beat another and not only which was ahead.

    Only tasks where every system has a score count. None with fewer than two such tasks or three systems."""
    from scipy import stats

    systems = results["systems"]
    complete = []
    for task_id in ids:
        entry = results["tasks"][task_id]
        means = {}
        for s in systems:
            values = fold_values(entry, s)
            if values is None:
                break
            means[s] = float(np.mean(values))
        else:
            sign = 1.0 if HIGHER_IS_BETTER[entry["metric"]] else -1.0
            # To 12 significant figures, so two systems that fit the same model (the loop that kept the defaults
            # and the plain train) tie, as they do in the summary's ranks, instead of differing in the 15th digit.
            complete.append((task_id, {s: sign * float(f"{m:.12g}") for s, m in means.items()}))
    n, k = len(complete), len(systems)
    if n < 2 or k < 3:
        return None
    rank_rows = np.array([stats.rankdata([-m[s] for s in systems]) for _, m in complete])
    average = rank_rows.mean(axis=0)
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(average ** 2) - k * (k + 1) ** 2 / 4)
    denominator = n * (k - 1) - chi2
    if denominator <= 0:  # every task ranked the systems the same way
        f_stat, p_value = float("inf"), 0.0
    else:
        f_stat = (n - 1) * chi2 / denominator
        p_value = float(stats.f.sf(f_stat, k - 1, (k - 1) * (n - 1)))
    q = NEMENYI_Q05.get(k)
    critical_difference = None if q is None else float(q * math.sqrt(k * (k + 1) / (6 * n)))

    pairwise = {}
    if FOCUS in systems:
        normal = {s: per_task_normalised(results, s) for s in systems}
        raw = {}
        for s in systems:
            if s == FOCUS:
                continue
            shared = [t for t, _ in complete if t in normal[FOCUS] and t in normal[s]]
            diffs = np.array([normal[FOCUS][t] - normal[s][t] for t in shared])
            median = float(np.median(diffs)) if len(diffs) else 0.0
            if len(diffs) < 2 or np.allclose(diffs, 0):
                raw[s] = (1.0, median, len(diffs))
            else:
                raw[s] = (float(stats.wilcoxon(diffs).pvalue), median, len(diffs))
        # Holm: the i-th smallest p-value times the number of comparisons still in play, never below the one before.
        ordered = sorted(raw, key=lambda s: raw[s][0])
        running = 0.0
        for i, s in enumerate(ordered):
            running = max(running, min(1.0, raw[s][0] * (len(ordered) - i)))
            pairwise[s] = {"p_holm": round(running, 4), "median_difference": round(raw[s][1], 4), "tasks": raw[s][2]}

    return {
        "tasks": n,
        "average_rank": {s: round(float(r), 3) for s, r in zip(systems, average)},
        "friedman_f": None if math.isinf(f_stat) else round(float(f_stat), 3),
        "friedman_p": round(p_value, 4),
        "critical_difference": None if critical_difference is None else round(critical_difference, 3),
        f"{FOCUS}_wilcoxon": pairwise,
    }


def cd_runs(average_rank: dict[str, float], critical_difference: float) -> list[list[str]]:
    """The groups a critical-difference diagram joins with a bar: maximal runs of systems, in rank order, whose
    average ranks all lie within the critical difference of each other. Systems in one group are not told apart."""
    systems = sorted(average_rank, key=average_rank.get)
    runs: list[tuple[int, int]] = []
    for i in range(len(systems)):
        j = i
        while j + 1 < len(systems) and average_rank[systems[j + 1]] - average_rank[systems[i]] < critical_difference:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in runs):
            runs.append((i, j))
    return [systems[i:j + 1] for i, j in runs]


def cd_diagram(average_rank: dict[str, float], critical_difference: float, title: str) -> str:
    """A critical-difference diagram as SVG: the systems on an axis of average rank (1, the best, on the left), and
    a bar under every group `cd_runs` finds. Drawn in the page's text colour, so it reads in light and dark themes."""
    systems = sorted(average_rank, key=average_rank.get)
    k = len(systems)
    width, left, right, axis_y = 720, 170, 550, 70
    scale = (right - left) / max(k - 1, 1)

    def x(rank: float) -> float:
        return round(left + (rank - 1) * scale, 1)

    half = (k + 1) // 2
    height = axis_y + 50 + 22 * half
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" font-family="sans-serif" '
        f'font-size="13" role="img" aria-label="{title}">',
        "<style>.ink{stroke:currentColor;fill:none}.txt{fill:currentColor}</style>",
        f'<text class="txt" x="{width / 2}" y="16" text-anchor="middle" font-weight="bold">{title}</text>',
        f'<line class="ink" x1="{left}" y1="{axis_y}" x2="{right}" y2="{axis_y}"/>',
    ]
    for r in range(1, k + 1):
        parts.append(f'<line class="ink" x1="{x(r)}" y1="{axis_y - 5}" x2="{x(r)}" y2="{axis_y}"/>')
        parts.append(f'<text class="txt" x="{x(r)}" y="{axis_y - 9}" text-anchor="middle">{r}</text>')
    cd_end = x(1 + critical_difference)
    parts.append(f'<line class="ink" stroke-width="2" x1="{left}" y1="34" x2="{cd_end}" y2="34"/>')
    parts.append(f'<text class="txt" x="{cd_end + 6}" y="38">CD = {critical_difference:.2f}</text>')
    for i, s in enumerate(systems):
        on_left = i < half
        row = i if on_left else k - 1 - i
        y = axis_y + 44 + 22 * row
        end = 12 if on_left else width - 12
        parts.append(f'<polyline class="ink" points="{x(average_rank[s])},{axis_y} {x(average_rank[s])},{y} {end},{y}"/>')
        parts.append(f'<text class="txt" x="{end}" y="{y - 4}" text-anchor="{"start" if on_left else "end"}">'
                     f"{s} ({average_rank[s]:.2f})</text>")
    for n_bar, group in enumerate(cd_runs(average_rank, critical_difference)):
        y = axis_y + 12 + 7 * n_bar
        parts.append(f'<line class="ink" stroke-width="3" x1="{x(average_rank[group[0]]) - 3}" y1="{y}" '
                     f'x2="{x(average_rank[group[-1]]) + 3}" y2="{y}"/>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def per_task_normalised(results: dict, system: str) -> dict[str, float]:
    out = {}
    for task_id, entry in results["tasks"].items():
        values, constant, reference = (fold_values(entry, s) for s in (system, "constant", "random_forest"))
        if values and constant and reference:
            n = normalised(float(np.mean(values)), float(np.mean(constant)), float(np.mean(reference)),
                           HIGHER_IS_BETTER[entry["metric"]])
            if n is not None:
                out[task_id] = n
    return out


def regressions(results: dict, baseline: dict, tolerance: float) -> list[str]:
    """Every task where a DataClean AI system's normalised score fell by more than `tolerance` against the baseline
    run. Normalised scores, not raw metrics, so a change in the reference systems' luck is not counted twice."""
    found = []
    for system in ("dataclean_train", "dataclean_improve"):
        now, before = per_task_normalised(results, system), per_task_normalised(baseline, system)
        for task_id, value in now.items():
            if task_id in before and value < before[task_id] - tolerance:
                name = results["tasks"][task_id]["name"]
                found.append(f"{system} on {name}: normalised score {value:.3f} against {before[task_id]:.3f}")
    return found


def significance_lines(sig: dict | None) -> list[str]:
    if not sig:
        return []
    verdict = ("the systems differ beyond chance" if sig["friedman_p"] < 0.05
               else "no difference between the systems beyond chance")
    lines = ["", f"Friedman test over {sig['tasks']} tasks: p = {sig['friedman_p']:.4f} ({verdict})."]
    if sig["critical_difference"] is not None:
        groups = cd_runs(sig["average_rank"], sig["critical_difference"])
        lines[-1] += f" Nemenyi critical difference at 5%: {sig['critical_difference']:.2f} ranks."
        if groups:
            lines += ["", "Not told apart (average ranks within the critical difference): "
                      + "; ".join(", ".join(g) for g in groups) + "."]
    wilcoxon = sig.get(f"{FOCUS}_wilcoxon") or {}
    if wilcoxon:
        lines += ["", f"{FOCUS} against each system, Wilcoxon signed-rank on normalised scores, Holm-corrected:", ""]
        for s, w in wilcoxon.items():
            side = "ahead" if w["median_difference"] > 0 else "behind" if w["median_difference"] < 0 else "level"
            lines.append(f"- {s}: median {side} by {abs(w['median_difference']):.3f}, p = {w['p_holm']:.4f}"
                         + (" (significant)" if w["p_holm"] < 0.05 else ""))
    return lines


def markdown(results: dict) -> str:
    """The summary as tables, for the docs and the CI job's page."""
    systems = results["systems"]
    lines = [
        f"{len(results['tasks'])} tasks, {results['folds']} fold(s) each, {results['budget_seconds']:g} s budget, "
        f"{results['cpus']} CPUs, Python {results['python']}, scikit-learn {results['scikit_learn']}.",
        "",
    ]
    for group in ("development", "held_out", "all"):
        block = results["summary"].get(group)
        if not block or not block["tasks"]:
            continue
        lines += [f"### {group.replace('_', '-').capitalize()} tasks ({block['tasks']})", "",
                  "| System | Normalised score, mean | median | Average rank | Brier (classes) | Failed folds |",
                  "|---|---:|---:|---:|---:|---:|"]
        brier = block.get("brier") or {}
        for s in systems:
            n, r, b = block["normalised_score"].get(s), block["average_rank"].get(s), brier.get(s)
            med = (block.get("median_normalised_score") or {}).get(s)
            lines.append(f"| {s} | {'' if n is None else f'{n:.3f}'} | {'' if med is None else f'{med:.3f}'} | "
                         f"{'' if r is None else f'{r:.2f}'} | "
                         f"{'' if b is None else f'{b:.4f}'} | {block['failed_folds'].get(s, 0)} |")
        against = block.get(f"{FOCUS}_against") or {}
        if against:
            lines += ["", f"{FOCUS} against each system (win / tie / loss):", ""]
            lines += [f"- {s}: {v['win']} / {v['tie']} / {v['loss']}" for s, v in against.items()]
        lines += significance_lines(block.get("significance"))
        lines.append("")
    lines += ["### Per task", "", "| Task | Metric | " + " | ".join(systems) + " |",
              "|---|---|" + "---:|" * len(systems)]
    for entry in results["tasks"].values():
        cells = []
        for s in systems:
            values = fold_values(entry, s)
            cells.append("" if values is None else f"{np.mean(values):.4f}")
        flag = " (held out)" if entry["held_out"] else ""
        lines.append(f"| {entry['name']}{flag} | {entry['metric']} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def merge(parts: list[dict]) -> dict:
    """Several runs over different tasks (the CI job runs development and held-out tasks side by side) as one, with
    the summary recomputed over all of them. The runs must agree on folds, budget and systems."""
    first = parts[0]
    for other in parts[1:]:
        for key in ("folds", "budget_seconds", "systems"):
            if other[key] != first[key]:
                raise ValueError(f"The runs differ in {key}: {first[key]!r} and {other[key]!r}.")
    merged = {k: v for k, v in first.items() if k not in ("tasks", "summary")}
    merged["tasks"] = {task_id: entry for part in parts for task_id, entry in part["tasks"].items()}
    merged["summary"] = summarise(merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=sorted(openml_tasks.SUITES), default="amlb",
                        help="amlb (the default 24 tasks) or tabarena (28 more, all held out)")
    parser.add_argument("--tasks", nargs="+", help="task names or OpenML task ids (default: all)")
    parser.add_argument("--held-out", choices=["only", "exclude"], help="run only, or leave out, the held-out tasks")
    parser.add_argument("--folds", type=int, default=2, help="official folds per task, from the first (max 10)")
    parser.add_argument("--budget", type=float, default=120.0, help="seconds for the improvement loop and FLAML")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    parser.add_argument("--cache", type=Path, default=openml_tasks.DEFAULT_CACHE)
    parser.add_argument("--out", default="accuracy_benchmark.json")
    parser.add_argument("--markdown", help="also write the summary tables to this file")
    parser.add_argument("--baseline", help="an earlier run's JSON to compare with")
    parser.add_argument("--tolerance", type=float, default=0.1, help="largest allowed fall in normalised score")
    parser.add_argument("--merge", nargs="+", help="combine these runs' JSON files instead of running anything")
    parser.add_argument("--cd-svg", help="also draw the critical-difference diagram for all tasks to this SVG file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    for noisy in ("app", "flaml", "flaml.automl"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    if args.merge:
        results = merge([json.loads(Path(p).read_text(encoding="utf-8")) for p in args.merge])
    else:
        systems = list(args.systems)
        if "flaml" in systems and not flaml_available():
            print("FLAML is not installed (pip install -r requirements-benchmark.txt); skipping it.")
            systems.remove("flaml")
        held = {"only": True, "exclude": False}.get(args.held_out or "")
        tasks = openml_tasks.select(args.tasks, held_out=held, suite=args.suite)
        # The output file doubles as the progress file, so running the same command again carries on.
        results = run(tasks, args.folds, systems, args.budget, args.cache, args.suite, progress=Path(args.out))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    report = markdown(results)
    print(report)
    if args.markdown:
        Path(args.markdown).write_text(report, encoding="utf-8")
    sig = results["summary"]["all"].get("significance")
    if args.cd_svg and sig and sig["critical_difference"] is not None:
        title = f"Average rank over {sig['tasks']} tasks (lower is better)"
        Path(args.cd_svg).write_text(cd_diagram(sig["average_rank"], sig["critical_difference"], title),
                                     encoding="utf-8")
    print(f"Wrote {args.out}")
    if args.baseline and Path(args.baseline).exists():
        worse = regressions(results, json.loads(Path(args.baseline).read_text(encoding="utf-8")), args.tolerance)
        for line in worse:
            print(f"WORSE: {line}")
        if worse:
            sys.exit(1)
        print(f"No task more than {args.tolerance} worse in normalised score than the baseline.")


if __name__ == "__main__":
    main()
