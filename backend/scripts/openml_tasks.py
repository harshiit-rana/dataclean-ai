"""Public tasks from the OpenML AutoML Benchmark, with their official folds, for the accuracy and cleaning
benchmarks (#124, #125).

AMLB (Gijsbers et al., "AMLB: an AutoML Benchmark", JMLR 2024) is the reference set AutoML systems are compared on:
71 classification and 33 regression tasks on OpenML (studies 271 and 269), each with fixed ten-fold cross-validation
splits, so every system trains and is scored on exactly the same rows. The tasks below are the ones small enough for
a weekly CI run (up to 15,000 rows and 120 columns), chosen to cover two-class, many-class and numeric targets,
categorical columns and blanks.

They are split in two. **Development** tasks are the ones a change to the app's defaults may be judged on.
**Held-out** tasks are never used to choose a default: they are there to show whether a change that helped on the
development tasks helps on tasks it was not chosen on. Moving a task from one list to the other defeats that.

Each file is fetched once from openml.org and kept in the cache folder (`--cache`, by default
`backend/.benchmark_cache`, which git ignores), then read back the way an upload would be: written to CSV and read
with `pandas.read_csv`, so text columns are text and not pandas categories, as they are in the app.
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

OPENML_API = "https://www.openml.org/api/v1/json"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / ".benchmark_cache"


@dataclass(frozen=True)
class Task:
    id: int
    name: str
    kind: str  # "binary", "multiclass" or "regression"
    held_out: bool = False


TASKS: list[Task] = [
    # Development: two classes
    Task(146818, "Australian", "binary"),
    Task(359955, "blood-transfusion-service-center", "binary"),
    Task(168757, "credit-g", "binary"),
    Task(359962, "kc1", "binary"),
    Task(168350, "phoneme", "binary"),
    Task(359968, "churn", "binary"),
    # Development: more than two classes
    Task(359954, "eucalyptus", "multiclass"),
    Task(190146, "vehicle", "multiclass"),
    Task(359959, "cmc", "multiclass"),
    Task(359960, "car", "multiclass"),
    # Development: numeric targets
    Task(359950, "boston", "regression"),
    Task(167210, "Moneyball", "regression"),
    Task(359944, "abalone", "regression"),
    Task(359935, "wine_quality", "regression"),
    # Held out: never used to choose a default
    Task(359956, "qsar-biodeg", "binary", held_out=True),
    Task(359958, "pc4", "binary", held_out=True),
    Task(190137, "ozone-level-8hr", "binary", held_out=True),
    Task(359965, "kr-vs-kp", "binary", held_out=True),
    Task(168784, "steel-plates-fault", "multiclass", held_out=True),
    Task(359963, "segment", "multiclass", held_out=True),
    Task(2073, "yeast", "multiclass", held_out=True),
    Task(359932, "socmob", "regression", held_out=True),
    Task(359951, "house_prices_nominal", "regression", held_out=True),
    Task(359933, "space_ga", "regression", held_out=True),
]


# TabArena (Erickson et al., "TabArena: A Living Benchmark for Machine Learning on Tabular Data", NeurIPS 2025), the
# OpenML suite "tabarena-v0.1" (study 457): 51 real-world datasets curated for tabular ML, most of them in no earlier
# benchmark. These are the 28 within the same limits as above (up to 15,000 rows and 120 columns), less the five whose
# data is already here from AMLB (blood-transfusion, credit-g, qsar-biodeg, churn, wine_quality). Run them with
# `--suite tabarena`. Every one is held out: nothing in the app has ever been chosen on them.
TABARENA: list[Task] = [
    Task(363629, "diabetes", "binary", held_out=True),
    Task(363671, "Fitness_Club", "binary", held_out=True),
    Task(363682, "Is-this-a-good-customer", "binary", held_out=True),
    Task(363684, "Marketing_Campaign", "binary", held_out=True),
    Task(363674, "hazelnut-spread-contaminant-detection", "binary", held_out=True),
    Task(363700, "seismic-bumps", "binary", held_out=True),
    Task(363694, "polish_companies_bankruptcy", "binary", held_out=True),
    Task(363706, "taiwanese_bankruptcy_prediction", "binary", held_out=True),
    Task(363689, "NATICUSdroid", "binary", held_out=True),
    Task(363624, "coil2000_insurance_policies", "binary", held_out=True),
    Task(363619, "Bank_Customer_Churn", "binary", held_out=True),
    Task(363676, "heloc", "binary", held_out=True),
    Task(363712, "jm1", "binary", held_out=True),
    Task(363632, "E-CommereShippingData", "binary", held_out=True),
    Task(363691, "online_shoppers_intention", "binary", held_out=True),
    Task(363681, "in_vehicle_coupon_recommendation", "binary", held_out=True),
    Task(363614, "anneal", "multiclass", held_out=True),
    Task(363685, "maternal_health_risk", "multiclass", held_out=True),
    Task(363707, "website_phishing", "multiclass", held_out=True),
    Task(363711, "MIC", "multiclass", held_out=True),
    Task(363702, "splice", "multiclass", held_out=True),
    Task(363704, "students_dropout_and_academic_success", "multiclass", held_out=True),
    Task(363698, "QSAR_fish_toxicity", "regression", held_out=True),
    Task(363625, "concrete_compressive_strength", "regression", held_out=True),
    Task(363675, "healthcare_insurance_expenses", "regression", held_out=True),
    Task(363612, "airfoil_self_noise", "regression", held_out=True),
    Task(363615, "Another-Dataset-on-used-Fiat-500", "regression", held_out=True),
    Task(363686, "miami_housing", "regression", held_out=True),
]

SUITES = {"amlb": TASKS, "tabarena": TABARENA}


@dataclass
class LoadedTask:
    task: Task
    frame: pd.DataFrame
    target: str
    folds: list[tuple[list[int], list[int]]] = field(default_factory=list)  # (train rows, test rows) per fold

    @property
    def sklearn_task(self) -> str:
        return "regression" if self.task.kind == "regression" else "classification"


def by_id(task_id: int) -> Task:
    for task in TASKS + TABARENA:
        if task.id == task_id:
            return task
    raise KeyError(f"Task {task_id} is not one of the benchmark's tasks.")


def select(names_or_ids: list[str] | None = None, *, held_out: bool | None = None, suite: str = "amlb") -> list[Task]:
    """The suite's tasks asked for by name or id (all when none are named), optionally only one of the two sets."""
    tasks = SUITES[suite]
    chosen = tasks
    if names_or_ids:
        wanted = {str(n).lower() for n in names_or_ids}
        chosen = [t for t in tasks if str(t.id) in wanted or t.name.lower() in wanted]
        unknown = wanted - {str(t.id) for t in chosen} - {t.name.lower() for t in chosen}
        if unknown:
            raise KeyError(f"Not benchmark tasks: {', '.join(sorted(unknown))}")
    if held_out is not None:
        chosen = [t for t in chosen if t.held_out == held_out]
    return chosen


def parse_splits(text: str, repeat: int = 0) -> list[tuple[list[int], list[int]]]:
    """OpenML's split file (ARFF: `TRAIN|TEST, rowid, repeat, fold` per line) as (train rows, test rows) per fold
    of one repeat, in fold order."""
    folds: dict[int, tuple[list[int], list[int]]] = {}
    in_data = False
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("%"):
            continue
        if not in_data:
            in_data = line.lower() == "@data"
            continue
        kind, rowid, rep, fold = (part.strip().strip("'\"") for part in line.split(","))
        if int(float(rep)) != repeat:
            continue
        train, test = folds.setdefault(int(float(fold)), ([], []))
        (train if kind.upper() == "TRAIN" else test).append(int(float(rowid)))
    return [folds[f] for f in sorted(folds)]


def _fetch(url: str, attempts: int = 4) -> bytes:
    """The bytes at `url`, trying again after 5, 10 and 20 seconds: openml.org answers "too many connections" or 503
    at busy times, and one refusal should not fail a weekly run."""
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "dataclean-ai-benchmark"})
            with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 -- fixed openml.org URLs
                body = response.read()
            if body.lstrip()[:1] in (b"<",) and b"Error was encountered" in body[:500]:
                raise urllib.error.URLError("openml.org returned an error page")
            return body
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(5 * 2 ** attempt)
    raise AssertionError("unreachable")


def _cached(path: Path, url: str) -> bytes:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_fetch(url))
    return path.read_bytes()


def as_uploaded(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame as the app would see it after an upload: through CSV and back."""
    return pd.read_csv(io.StringIO(frame.to_csv(index=False)))


def load(task: Task, cache: Path = DEFAULT_CACHE) -> LoadedTask:
    """The task's data, target column and official folds, from the cache or from openml.org."""
    from sklearn.datasets import fetch_openml

    cache = Path(cache)
    meta = json.loads(_cached(cache / f"task_{task.id}.json", f"{OPENML_API}/task/{task.id}"))["task"]
    inputs = {item["name"]: item for item in meta["input"]}
    source = inputs["source_data"]["data_set"]
    splits_url = inputs["estimation_procedure"]["estimation_procedure"]["data_splits_url"]
    splits = _cached(cache / f"task_{task.id}_splits.arff", splits_url).decode("utf-8")

    data_csv = cache / f"data_{source['data_set_id']}.csv"
    if not data_csv.exists():
        bunch = fetch_openml(data_id=int(source["data_set_id"]), as_frame=True, data_home=str(cache / "openml"))
        bunch.frame.to_csv(data_csv, index=False)
    frame = pd.read_csv(data_csv)
    return LoadedTask(task=task, frame=frame, target=source["target_feature"], folds=parse_splits(splits))
