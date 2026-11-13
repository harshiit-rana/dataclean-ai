# The cleaning benchmark (#125)

## What it does

The method is REIN's ([Abdelaal et al., 2023](https://arxiv.org/abs/2302.04702)) and CleanML's (Li et al., 2019).
Start from clean public data, inject problems whose every cell is known, and score the tool against that truth. The
data are six of the [accuracy benchmark's](README.md) tasks (credit-g, churn, vehicle, abalone, car, house_prices_nominal), read through CSV as
an upload is. Between them they have columns for every kind of problem.

| Problem | Injected as |
|---|---|
| Missing values | 10% of a complete number column, and of a complete text column, blanked |
| Outliers | 2% of a number column multiplied by 10, 100 or 1000 (a unit or decimal-point mistake) |
| Numbers stored as text | 3% of a number column replaced by `?`, `unknown` or `--` |
| Inconsistent capitals | 10% of a text column's values in capitals or title case |
| Stray spaces | 10% of a text column's values padded |
| Mixed date formats | 20% of an added ISO date column written day-first |
| Duplicates | 5% of the rows repeated |

**Detection.** The profiler's findings on the dirty file are compared with its findings on the clean one. An injected
problem counts as found when its column gets a finding of that kind that the clean file did not have, or one that
grew (by more than a point for a share). Injected outliers count as found when at least half of them sit outside the
profiler's own range. Every other new finding is a false alarm. What the profiler already says about the clean file
is reported separately and not counted: blanks in a public dataset are real blanks, and a value far out in the tail
is an outlier whether or not it is a mistake.

**Repair.** The app's plain train is trained three times on one official fold (a different one for each seed): on
the clean file, on the dirty file, and on the dirty file after the fix each finding implies. The fixes are learned on
the training rows and replayed on the test rows, as the app replays a recipe. The fixes are a fixed mapping from
finding to operation, not the explanation agent's choice, so this measures the detectors and the operations, not the
LLM:

| Finding | Fix |
|---|---|
| Numbers stored as text | `cast_to_numeric` |
| Stray spaces | `trim_whitespace` |
| Inconsistent capitals | `unify_case_variants` (merges the variants only, #129) |
| Mixed date formats | `standardize_dates` |
| Outliers | `cap_outliers` |
| Missing values | `fill_median` for numbers, `fill_mode` for text |
| Duplicates | `drop_duplicate_rows`, on the training rows only |

They run in that order: casts first, so later steps see numbers, and outliers before blanks, so the range is learned
from the values actually written. Filling blanks with the median first squeezes the range and caps ordinary values;
the real-error benchmark showed it capping 512 correct values in one column.

Blanks are information destroyed, not hidden. Filling them with a median or the commonest value cannot bring it back,
so on a file whose answer depends on every column, most of the loss is beyond any fix.

```bash
cd backend
python scripts/cleaning_benchmark.py --no-repair   # detection only: seconds
python scripts/cleaning_benchmark.py               # with repair: about half an hour
```

## Results (13 November 2026)

Six datasets, three seeds each (18 runs, about 32 minutes on a 16-core laptop). The first 17 runs were made on
4 November; the last one was made on 13 November with the environment rebuilt from the same requirements files.

**Detection.** The profiler found every problem that was injected (recall 1.0 for every kind), with 2 false alarms
(overall precision 0.982):

| Problem | Injected | Found | Recall | False alarms | Precision |
|---|---:|---:|---:|---:|---:|
| Missing values | 27 | 27 | 1.0 | 0 | 1.0 |
| Outliers | 15 | 15 | 1.0 | 2 | 0.882 |
| Numbers stored as text | 15 | 15 | 1.0 | 0 | 1.0 |
| Inconsistent capitals | 9 | 9 | 1.0 | 0 | 1.0 |
| Stray spaces | 9 | 9 | 1.0 | 0 | 1.0 |
| Mixed date formats | 18 | 18 | 1.0 | 0 | 1.0 |
| Duplicates | 18 | 18 | 1.0 | 0 | 1.0 |

99% of the injected outlier cells fell outside the profiler's own range. Both false alarms were outliers in
house_prices_nominal's `GarageYrBlt`, a column nothing was injected into, which got an outlier finding on the dirty
file that the clean file did not have (or a larger one). The cause has not been traced. On the clean files the profiler already reports 57 missing-value,
174 outlier and 18 type findings; they are real features of public data and are not counted either way.

**Repair** (the mean over three seeds; the share won back is averaged over the seeds where there was a measurable
loss):

| Dataset | Metric | Clean | Dirty | Fixed | Lost score won back |
|---|---|---:|---:|---:|---:|
| credit-g | AUC | 0.7808 | 0.7873 | 0.7932 | no measurable loss |
| churn | AUC | 0.9102 | 0.9108 | 0.9007 | 27% (one seed of three) |
| vehicle | log loss | 0.4600 | 0.5348 | 0.4904 | 60% |
| abalone | RMSE | 2.3172 | 2.2398 | 2.2155 | no measurable loss |
| car | log loss | 0.0067 | 0.1956 | 0.2028 | -2% |
| house_prices_nominal | RMSE | 23,885 | 25,189 | 24,562 | 57% |

**What it says.**

- **Finding the problems is solved for these kinds.** Every injected problem was found, and 2 of its 113
  findings were false alarms.
- **On credit-g, churn and abalone the injected problems cost nothing measurable.** The dirty file scored *better*
  than the clean one in all three seeds of credit-g, in two of churn and in two of abalone (by up to 0.12 RMSE). A
  dirty file cannot hold more information than the clean one, so differences of this size between two trains are
  noise: which model wins and how it splits move the score by about 0.01-0.02 AUC or 0.1 RMSE. The fixed scores
  differ from the dirty ones by no more than that, so these rows say nothing either way about the fixes.
- **Where the problems did cost something, the fixes won back about 60% of it** (vehicle 86%, 10% and 83% by seed;
  house_prices_nominal 79% and 34%, the third seed losing nothing measurable).
- **car shows the limit.** Its class is a fixed rule over its six attributes, so 10% of one attribute blanked loses
  information that no fill can recover: the most common value is a guess, and the rule needs the real one. The other
  problems on car (capitals, spaces, dates, duplicates) are undone exactly by their fixes, so what remains of the loss
  is the blanks, as the note above predicts.
