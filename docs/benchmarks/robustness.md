# The robustness benchmark (#128)

A model is trained once and then applied to new files, which may be messier than the one it learned from. The idea
is Jenga's (Schelter et al., "JENGA - A Framework to Study the Impact of Data Errors on the Predictions of Machine
Learning Models", EDBT 2021): train on clean data, corrupt only the data the model is later asked about, and measure
how far its score falls.

The new rows go through the app's own route for a new file:
1. `validation.validate` checks them against the model's input schema;
2. `scoring.prepare_inputs` coerces them;
3. the fitted pipeline scores them.

So the benchmark also records whether the app **warned** about the mess, and whether scoring **failed**.

For each of six OpenML tasks (the cleaning benchmark's datasets) and three seeds (one official fold each), the plain
train is fitted once on the clean training rows. Its test rows are then scored clean, and once for each kind of mess,
with 20% of the rows of up to three input columns corrupted:

| Mess | Corruption |
|---|---|
| `missing_values` | blanked |
| `outliers` | multiplied by 10, 100 or 1000 |
| `type_mismatch` | replaced by `?`, `unknown` or `--` |
| `inconsistent_casing` | written in capitals or title case |
| `stray_whitespace` | padded with spaces |
| `unseen_categories` | replaced by a value the column never had |

The fall in score is given as a share of the clean score, so AUC, log loss and RMSE can be put together; positive is
worse. Both the median and the mean over runs are reported, because one file with a clean score near zero carries
the mean.

```bash
cd backend
python scripts/robustness_benchmark.py           # under ten minutes
```

## Results (13 November 2026)

Six datasets, three seeds each, under ten minutes on a 16-core laptop. A kind of mess is skipped where the model has
no column for it (churn and vehicle have no text inputs).

| Mess in the new file | Runs | Fall in score, median | mean | App warned | Scoring failed |
|---|---:|---:|---:|---:|---:|
| Missing values | 18 | +2.6% | +3996% | 17 of 18 | 0 |
| Outliers | 15 | +27.2% | +22736% | 15 of 15 | 0 |
| Numbers stored as text | 15 | +1.8% | +15.7% | 15 of 15 | 0 |
| Inconsistent capitals | 9 | +0.8% | +3025% | 9 of 9 | 0 |
| Stray spaces | 12 | +0.7% | +2974% | 12 of 12 | 0 |
| Unseen categories | 12 | +0.7% | +2974% | 12 of 12 | 0 |

The means are carried by car, whose clean log loss is close to zero (0.002 to 0.016), so any fall there is hundreds
of times the clean score. The median is the typical file. The scores, averaged over seeds:

| Dataset | Metric | Clean | Missing | Outliers | As text | Capitals | Spaces | Unseen |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| credit-g | AUC | 0.7913 | 0.7960 | 0.7005 | 0.7884 | 0.7890 | 0.7999 | 0.7999 |
| churn | AUC | 0.9092 | 0.9070 | 0.9005 | 0.9079 | n/a | n/a | n/a |
| vehicle | log loss | 0.4688 | 0.6422 | 12.7801 | 0.6335 | n/a | n/a | n/a |
| abalone | RMSE | 2.3119 | 2.9887 | 2586.5630 | 3.2067 | n/a | 2.3222 | 2.3222 |
| car | log loss | 0.0081 | 0.9625 | n/a | n/a | 0.4969 | 0.5197 | 0.5197 |
| house_prices_nominal | RMSE | 25,424 | 25,475 | 28,766 | 26,768 | 25,529 | 25,671 | 25,671 |

**What it says.**

- **Scoring never failed, and the app named every corrupted column in 80 of 81 runs.** In the other one it named
  two of three blanked house_prices_nominal columns (`Foundation`, `RoofStyle`) but not `FireplaceQu`, which is blank
  in 47% of the data already, so a blank there is within what the model was trained on. The score did not fall.
- **Outliers do the most damage, and the app scores them anyway.** A few values multiplied by 10 to 1000 sent
  abalone's error from 2.3 to about 2,600 rings and vehicle's log loss from 0.47 to 12.8. The app warned every time
  but still returned the predictions, and the scoring step does not clip new values to the range seen in training.
  Whether it should clip, refuse or only warn is a product decision (#133).

  **Changed in #133 (18 November 2026).** The model that blew up was linear on every seed (linear regression on
  abalone, logistic regression on vehicle), which extrapolates without limit. A value beyond the training range by
  more than half its width is now reported apart as *far outside the training range*, and the row's note says its
  prediction is not reliable. Measured on the six tasks' test folds, that line caught 87% of values multiplied by 10
  to 1000 (most of the rest were zeros, which stay zero) and 2 of 47,466 real values. The predictions are still
  written, and nothing is clipped: in the real-error benchmark none of the 112 flagged extremes was an error, so
  changing them silently would be wrong more often than right.
- **Stray spaces count as categories the model never saw.** The spaces and unseen-category columns are identical to
  the last digit: the scoring step does not trim text, so `" red "` is as unknown to the model as a made-up value,
  and the encoder ignores it. The same holds for capitals (`"RED"`). On car, where every input is a category, that
  costs a lot (log loss 0.008 to about 0.5).

  **Fixed in #134 (18 November 2026).** When scoring, a value that matches exactly one training category apart from
  spaces or capitals is scored as that category, and the check reports it as matched rather than unseen. Rerun on the
  same seeds, padded and re-cased values cost nothing on car, credit-g and abalone (car's log loss stays 0.0081, the
  clean score). On house_prices_nominal the extra RMSE fell from about 250 to 20-40: there, some values match no
  category or more than one, and those are still left alone. A value that matches none, or several, is still treated
  as unseen, because guessing between two real categories would change the answer silently.
- **Blanks and numbers stored as text cost what imputation costs**: small on most files, large on car (whose class
  is a fixed rule over its six attributes) and on abalone and vehicle (where 20% of three numeric inputs were lost).
