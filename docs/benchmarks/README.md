# Benchmarks

The scripts in `backend/scripts/` measure the app against things outside it. None of them runs on a push: a timing
or a score is evidence about a change, not a pass or a fail for it.

| Script | Measures | Data | CI, weekly |
|---|---|---|---|
| `benchmark.py` (#107) | How long profiling, a train and the improvement loop take | Synthetic, 1k–100k rows | `Benchmark` |
| `accuracy_benchmark.py` (#124) | How accurate the models are next to other systems, and how well calibrated | 24 OpenML AutoML Benchmark tasks | `Accuracy benchmark` |
| `cleaning_benchmark.py` (#125) | How many known problems the profiler finds, and what fixing them wins back | 6 of those tasks, with injected problems | `Data benchmarks` |
| `real_errors_benchmark.py` (#127) | The same, cell by cell, on errors people actually made | The Raha/Baran datasets | `Data benchmarks` |
| `robustness_benchmark.py` (#128) | What a messy new file does to a model trained on a clean one | 6 OpenML tasks | `Data benchmarks` |

The data benchmarks have their own pages: [cleaning](cleaning.md), [real errors](real-errors.md),
[robustness](robustness.md).

## The accuracy benchmark

### What it does

The tasks come from the [OpenML AutoML Benchmark](https://openml.github.io/automlbenchmark/) (AMLB; Gijsbers et al.,
JMLR 2024), the reference set AutoML systems are compared on. The 24 used here are the AMLB tasks with up to 15,000
rows and 120 columns, chosen to cover two-class, many-class and numeric targets, text columns and blanks (the list,
and why, is in `openml_tasks.py`). Every system trains on the same official training fold and is scored on the same
test fold, with AMLB's metrics: AUC for two classes, log loss for more, RMSE for numbers.

| System | What it is |
|---|---|
| `constant` | Class shares or the mean of the training fold. The floor. |
| `random_forest` | scikit-learn's random forest, 500 trees. The second reference point. |
| `hist_gradient_boosting` | scikit-learn's histogram gradient boosting, defaults. The strong plain model. |
| `dataclean_train` | The app's plain train, every ordinary model for the task. |
| `dataclean_improve` | The app's improvement loop with the time budget, then the rebuild the route makes. |
| `flaml` | [FLAML](https://github.com/microsoft/FLAML), an AutoML library, with the same time budget. |

The app's two systems hold back part of the training fold for their own test, as the app does. The configuration
they choose is then refitted on the whole training fold, as FLAML refits its choice, so no system trains on fewer
rows. A fold where a system fails counts as the constant predictor's score, as in AMLB, and is listed.

### How the systems are compared

- **Normalised score**: 0 is the constant predictor and 1 the random forest; above 1 is better than the forest. The
  same idea as AMLB's scaled score, and what makes AUC, log loss and RMSE comparable across tasks.
- **Average rank** across tasks, 1 the best.
- **Wins, ties and losses** of the improvement loop against every other system, task by task. A tie is a difference
  within one standard error of the per-fold differences.
- **Is any of it more than chance?** The way systems are compared over many datasets (Demšar, "Statistical
  Comparisons of Classifiers over Multiple Data Sets", JMLR 2006):
  - the **Friedman test** on the ranks, whose p-value says whether the systems differ at all;
  - the **Nemenyi critical difference**: two systems whose average ranks differ by less are not told apart. The
    **critical-difference diagram** draws this, with a bar joining systems that are not told apart
    (`--cd-svg diagram.svg`);
  - **Wilcoxon signed-rank tests** of the improvement loop against each other system on the normalised scores,
    Holm-corrected for the number of comparisons. These use how much one system beat another, not only which was
    ahead, so they can see a consistent small difference that the ranks miss.
- **Brier score** for the classification tasks: the mean squared gap between the predicted probabilities and what
  happened. AUC only asks whether the rows are in the right order; the Brier score asks whether "80%" comes true about
  80% of the time, which matters wherever the app shows a probability or chooses a cut-off from one. Lower is better.

### Development and held-out tasks

Fourteen tasks are **development** tasks: a change to the app's defaults (a threshold, a search space, a new model
family under [ADR 0003](../decisions/0003-model-registry-criteria.md)) may be judged on them. Ten are **held out** and
are never used to choose anything. They show whether a change that helped on the development tasks also helps on
tasks it was not chosen on. Tuning on them, or moving a task between the lists, turns the benchmark into a
measurement of how well the defaults were fitted to it.

A second suite, 28 tasks from **TabArena** (Erickson et al., NeurIPS 2025; OpenML study 457), repeats no AMLB
dataset and is all held out. Its results are reported, never used to choose a default.

### Running it

```bash
cd backend
pip install -r requirements-benchmark.txt          # FLAML; without it the other systems still run
python scripts/accuracy_benchmark.py               # all 24 tasks, 2 folds, 120 s budgets: about 4 hours
python scripts/accuracy_benchmark.py --tasks credit-g boston --folds 1 --budget 30   # a few minutes
python scripts/accuracy_benchmark.py --held-out exclude --markdown dev.md            # development tasks only
python scripts/accuracy_benchmark.py --suite tabarena                              # the 28 TabArena tasks, all held out
python scripts/accuracy_benchmark.py --baseline last.json   # exits non-zero if the app fell > 0.1 on any task
python scripts/accuracy_benchmark.py --merge a.json b.json --cd-svg diagram.svg      # combine runs, draw the diagram
```

The first run downloads the tasks from openml.org into `backend/.benchmark_cache/`. Run nothing else heavy at the
same time, and keep the machine from sleeping: the improvement loop and FLAML work to a wall-clock budget, so a busy
or sleeping machine gives them less search.

The long benchmarks (this one, cleaning and robustness) save after every task or seed. If a run is stopped, the same
command carries on from where it stopped. A run that finished is not picked up: the same command starts again, so a
before-and-after comparison measures the change.

The weekly workflow runs the development and held-out tasks as two parallel jobs, merges them, compares with the last
successful run, and puts the tables on the run's page.

### First results (4 November 2026)

All 24 tasks, the first two official folds of each, 120-second budgets, on a 16-thread Windows laptop (Python 3.13,
scikit-learn 1.9). FLAML and the improvement loop work to a wall-clock budget, so their scores move a little from
run to run (FLAML scored 0.836 and 0.871 AUC on the same credit-g fold in two runs). Two folds whose timings showed
the laptop had gone into standby mid-search were run again.

![Critical-difference diagram: FLAML 2.21, the app 2.94, random forest 3.46, gradient boosting 3.62, constant 5.83; a bar joins all but the constant](accuracy-cd.svg)

**What it says.**

- **The app is level with a well-set-up random forest, and a little behind FLAML.** Median normalised score 1.00
  (the forest is 1 by definition) against FLAML's 1.04. Average rank 2.94 against FLAML's 2.21, ahead of the forest
  (3.46) and plain gradient boosting (3.62). The Friedman test says the systems differ, but the critical difference
  is 1.54 ranks, so FLAML, the app, the forest and gradient boosting are **not told apart**: only the constant
  predictor is. The Wilcoxon tests agree: the app is ahead of the constant predictor beyond chance (p = 0.0005),
  and behind FLAML by a median of 0.017 that is not (p = 0.17).
- **The mean normalised score is misleading here, so read the median.** On cmc the forest is barely better than
  predicting the class shares (log loss 1.050 against 1.069), so a difference of a few hundredths becomes a
  normalised score of +10 for FLAML and -5 for the app, and that one task carries the mean (FLAML 1.46, the app
  0.78).
- **The improvement loop never changed the answer.** On all 24 tasks it ended with the plain train's model and
  settings. At 120 seconds it rarely gets past its second round. Run to the end it still keeps nothing on small
  files, and [#131](https://github.com/harshiit-rana/dataclean-ai/issues/131) asked whether its bar was the wrong
  one: the spread of the fold scores rather than of the paired, fold-by-fold differences. Measured on the 14
  development tasks (18 November 2026), the paired differences were barely tighter (median 0.71 of the fold spread
  for other settings of the same model, 1.0 for another model family), and a rule built on them accepted the same 3
  of 28 real changes as the current one. On blood-transfusion, the rejected settings won 8 of 15 folds by a mean of
  0.0035 AUC: a coin flip. So the bar is not what holds the loop back. With the current model families there is
  little left to gain on these tasks, and the gap to FLAML comes from model families the registry does not have
  yet (deferred under #52). The rule was kept.
- **Where the app falls behind is probabilities on many-class targets.** Its log loss is well behind FLAML's on
  eucalyptus, cmc, steel-plates-fault, yeast and segment, and on cmc worse than predicting the class shares. Its
  Brier score is 0.187 against FLAML's 0.168 and the forest's 0.179. The app chooses and tunes models for many
  classes by accuracy, never by the quality of the probabilities. That was
  [#132](https://github.com/harshiit-rana/dataclean-ai/issues/132), and since 19 November 2026 they are chosen and
  tuned by log loss. Re-measured on the four development many-class tasks (eucalyptus, vehicle, cmc, car; both
  official folds, plain train), the chosen model's mean test log loss went from 0.629 to 0.539. The best of the
  registry's models on each fold would have scored 0.537, and FLAML 0.506. The held-out tasks were not looked at.
  The rest of the gap is model families, as in #131.
- **Where it is ahead is numeric targets.** Best of all six on boston, Moneyball and socmob, and close behind FLAML
  on wine_quality and space_ga.
- **Held-out tasks.** On the ten tasks nothing was ever chosen on, the app's median is 1.00 and its rank 3.30,
  against FLAML's 1.03 and 1.90. FLAML won 8 of the 10. The development tasks do not flatter the app much: its
  median there is 1.02.
- **No system failed a fold.**

#### All 24 tasks

| System | Normalised score, mean | median | Average rank | Brier (classes) | Failed folds |
|---|---:|---:|---:|---:|---:|
| constant | 0.000 | 0.000 | 5.83 | 0.3993 | 0 |
| random_forest | 1.000 | 1.000 | 3.46 | 0.1790 | 0 |
| hist_gradient_boosting | 0.828 | 0.997 | 3.62 | 0.1915 | 0 |
| dataclean_train | 0.775 | 1.002 | 2.94 | 0.1870 | 0 |
| dataclean_improve | 0.775 | 1.002 | 2.94 | 0.1870 | 0 |
| flaml | 1.464 | 1.037 | 2.21 | 0.1675 | 0 |

dataclean_improve against each system (win / tie / loss):

- constant: 23 / 1 / 0
- random_forest: 10 / 6 / 8
- hist_gradient_boosting: 10 / 10 / 4
- dataclean_train: 0 / 24 / 0
- flaml: 6 / 6 / 12

Friedman test over 24 tasks: p = 0.0000 (the systems differ beyond chance). Nemenyi critical difference at 5%: 1.54 ranks.

Not told apart (average ranks within the critical difference): flaml, dataclean_train, dataclean_improve, random_forest, hist_gradient_boosting.

dataclean_improve against each system, Wilcoxon signed-rank on normalised scores, Holm-corrected:

- constant: median ahead by 1.002, p = 0.0005 (significant)
- flaml: median behind by 0.017, p = 0.1700
- hist_gradient_boosting: median ahead by 0.017, p = 0.2376
- random_forest: median ahead by 0.002, p = 1.0000
- dataclean_train: median level by 0.000, p = 1.0000

#### Per task (mean of the two folds)

| Task | Metric | constant | random_forest | hist_gradient_boosting | dataclean_train | dataclean_improve | flaml |
|---|---|---:|---:|---:|---:|---:|---:|
| Australian | auc | 0.5000 | 0.9312 | 0.9283 | 0.9266 | 0.9266 | 0.9249 |
| blood-transfusion-service-center | auc | 0.5000 | 0.7337 | 0.7388 | 0.8060 | 0.8060 | 0.8007 |
| credit-g | auc | 0.5000 | 0.8149 | 0.7781 | 0.7995 | 0.7995 | 0.8171 |
| kc1 | auc | 0.5000 | 0.8372 | 0.7944 | 0.8155 | 0.8155 | 0.8413 |
| phoneme | auc | 0.5000 | 0.9695 | 0.9606 | 0.9761 | 0.9761 | 0.9697 |
| churn | auc | 0.5000 | 0.8836 | 0.8900 | 0.8900 | 0.8900 | 0.8854 |
| eucalyptus | log_loss | 1.5738 | 0.7141 | 0.9993 | 0.9031 | 0.9031 | 0.6828 |
| vehicle | log_loss | 1.3856 | 0.4976 | 0.7317 | 0.4533 | 0.4533 | 0.4634 |
| cmc | log_loss | 1.0686 | 1.0495 | 1.0954 | 1.1587 | 1.1587 | 0.8770 |
| car | log_loss | 0.8324 | 0.1344 | 0.0034 | 0.0019 | 0.0019 | 0.0026 |
| boston | rmse | 8.9161 | 2.6826 | 2.4905 | 2.1885 | 2.1885 | 2.2696 |
| Moneyball | rmse | 90.2017 | 24.9530 | 23.1912 | 21.0632 | 21.0632 | 22.4057 |
| abalone | rmse | 3.2304 | 2.1899 | 2.1507 | 2.2770 | 2.2770 | 2.1573 |
| pc4 (held out) | auc | 0.5000 | 0.9260 | 0.9388 | 0.9237 | 0.9237 | 0.9332 |
| ozone-level-8hr (held out) | auc | 0.5000 | 0.9491 | 0.9354 | 0.9546 | 0.9546 | 0.9342 |
| kr-vs-kp (held out) | auc | 0.5000 | 0.9989 | 0.9993 | 0.9996 | 0.9996 | 0.9999 |
| steel-plates-fault (held out) | log_loss | 1.6678 | 0.5436 | 0.6246 | 0.6561 | 0.6561 | 0.4904 |
| segment (held out) | log_loss | 1.9459 | 0.1084 | 0.1226 | 0.1226 | 0.1226 | 0.0800 |
| wine_quality | rmse | 0.8564 | 0.6041 | 0.6492 | 0.5881 | 0.5881 | 0.5832 |
| qsar-biodeg (held out) | auc | 0.5000 | 0.9512 | 0.9567 | 0.9394 | 0.9394 | 0.9460 |
| yeast (held out) | log_loss | 1.7368 | 1.3600 | 1.8169 | 1.4944 | 1.4944 | 1.0413 |
| socmob (held out) | rmse | 28.5785 | 13.1368 | 10.7685 | 8.1315 | 8.1315 | 8.5083 |
| house_prices_nominal (held out) | rmse | 84926.0698 | 24270.3491 | 23951.4621 | 24179.1288 | 24179.1288 | 21284.6949 |
| space_ga (held out) | rmse | 0.1952 | 0.1157 | 0.1078 | 0.1078 | 0.1078 | 0.1057 |

### TabArena results (19 November 2026)

The 28 TabArena tasks, the first two official folds of each, 120-second budgets, on a GitHub runner (4 CPUs,
Python 3.12, scikit-learn 1.9.1), after #132. Every task is held out. The machine is slower than the laptop above,
and the app's plain train alone took 11 to 78 seconds (median 25) of the budget, so the improvement loop had little search time.
Do not line these numbers up against the AMLB run one for one.

![Critical-difference diagram: FLAML 1.68, the app's improvement loop 3.07, its plain train 3.11, gradient boosting 3.61, random forest 3.64, constant 5.89](tabarena-cd.svg)

| System | Normalised score, mean | median | Average rank | Brier (classes) |
|---|---:|---:|---:|---:|
| constant | 0.000 | 0.000 | 5.89 | 0.2581 |
| random_forest | 1.000 | 1.000 | 3.64 | 0.1272 |
| hist_gradient_boosting | 0.833 | 0.998 | 3.61 | 0.1254 |
| dataclean_train | 0.961 | 1.015 | 3.11 | 0.1525 |
| dataclean_improve | 0.964 | 1.015 | 3.07 | 0.1512 |
| flaml | 1.069 | 1.036 | 1.68 | 0.1205 |

No system failed a fold.

**What it says.**

- **The app is behind FLAML, and here that is more than chance.** Median normalised score 1.015 against FLAML's
  1.036, and it beat FLAML on 1 task, tied 8 and lost 19. The paired Wilcoxon test puts it behind by a median of
  0.035, p = 0.0005 after Holm correction (on AMLB the gap was 0.017, p = 0.17). The Nemenyi grouping does not
  separate the two (critical difference 1.43 ranks, gap 1.39), but it compares all six systems at once and is
  conservative; the paired test of these two is the sharper one.
- **It is level with the forest and gradient boosting.** Median 1.015 against 1.000 and 0.998, average rank 3.07
  against 3.64 and 3.61. Neither difference is more than chance (Wilcoxon p = 0.58 for both). On numbers the
  app is ahead of both on three of the six regression tasks, and behind FLAML on all six.
- **Its probabilities are the weakest of the four real systems.** Brier score over the classification tasks 0.151,
  against the forest's 0.127, gradient boosting's 0.125 and FLAML's 0.121. On AMLB the gap to the forest was
  smaller (0.187 against 0.179).
- **#132 did not close the gap for many classes on these tasks.** There is no TabArena run from before #132, so this
  says where the app stands, not what #132 changed. On the six many-class tasks it is best of all six systems on
  anneal and ahead of the forest on splice and students_dropout, but behind the forest and FLAML on
  maternal_health_risk, website_phishing and MIC. Mean log loss over the six: the app 0.409, the forest 0.383,
  FLAML 0.322.
- **On MIC the app is worse than predicting the class shares.** Log loss 0.784 on both folds against 0.709 and 0.706
  for the constant predictor, and a Brier score of 0.333 against its 0.289, while its accuracy matches the forest's.
  Its normalised score there is -0.67. Every model in the registry scored worse than chance in its cross-validation there, and
  the model selection did not say so. This is the held-out form of what cmc showed on AMLB, and is
  [#140](https://github.com/harshiit-rana/dataclean-ai/issues/140). Gradient boosting does worse still on MIC
  (1.157), which is why its mean normalised score is 0.833: read the median.
- **The improvement loop changed the result once.** On the second fold of website_phishing it took the plain
  train's log loss from 0.470 to 0.374. On every other fold it ended with the plain train's model. The summary still
  counts the two as 28 ties: with one fold changed and one not, the mean difference is within one standard error
  of the fold differences, which is the benchmark's tie rule. The results file does not
  say which round was kept.

#### Per task (mean of the two folds)


| Task | Metric | constant | random_forest | hist_gradient_boosting | dataclean_train | dataclean_improve | flaml |
|---|---|---:|---:|---:|---:|---:|---:|
| diabetes (held out) | auc | 0.5000 | 0.8227 | 0.8027 | 0.8320 | 0.8320 | 0.8242 |
| Fitness_Club (held out) | auc | 0.5000 | 0.7804 | 0.7737 | 0.8058 | 0.8058 | 0.8075 |
| Is-this-a-good-customer (held out) | auc | 0.5000 | 0.7191 | 0.7012 | 0.7324 | 0.7324 | 0.7235 |
| Marketing_Campaign (held out) | auc | 0.5000 | 0.8693 | 0.8959 | 0.8838 | 0.8838 | 0.9053 |
| hazelnut-spread-contaminant-detection (held out) | auc | 0.5000 | 0.9573 | 0.9703 | 0.9703 | 0.9703 | 0.9733 |
| seismic-bumps (held out) | auc | 0.5000 | 0.7556 | 0.7225 | 0.7499 | 0.7499 | 0.7609 |
| polish_companies_bankruptcy (held out) | auc | 0.5000 | 0.9013 | 0.9475 | 0.9475 | 0.9475 | 0.9450 |
| taiwanese_bankruptcy_prediction (held out) | auc | 0.5000 | 0.9387 | 0.9377 | 0.9300 | 0.9300 | 0.9248 |
| NATICUSdroid (held out) | auc | 0.5000 | 0.9793 | 0.9850 | 0.9850 | 0.9850 | 0.9830 |
| coil2000_insurance_policies (held out) | auc | 0.5000 | 0.6854 | 0.7284 | 0.7343 | 0.7343 | 0.7573 |
| Bank_Customer_Churn (held out) | auc | 0.5000 | 0.8549 | 0.8616 | 0.8553 | 0.8553 | 0.8749 |
| heloc (held out) | auc | 0.5000 | 0.7956 | 0.7891 | 0.7950 | 0.7950 | 0.7980 |
| jm1 (held out) | auc | 0.5000 | 0.7528 | 0.7411 | 0.7266 | 0.7266 | 0.7504 |
| E-CommereShippingData (held out) | auc | 0.5000 | 0.7394 | 0.7322 | 0.7390 | 0.7390 | 0.7407 |
| online_shoppers_intention (held out) | auc | 0.5000 | 0.9251 | 0.9268 | 0.9277 | 0.9277 | 0.9313 |
| in_vehicle_coupon_recommendation (held out) | auc | 0.5000 | 0.8094 | 0.8279 | 0.8269 | 0.8269 | 0.8364 |
| anneal (held out) | log_loss | 0.8251 | 0.0481 | 0.0496 | 0.0336 | 0.0336 | 0.0370 |
| maternal_health_risk (held out) | log_loss | 1.0853 | 0.4123 | 0.3965 | 0.5022 | 0.5022 | 0.3975 |
| website_phishing (held out) | log_loss | 0.9013 | 0.3083 | 0.3095 | 0.3955 | 0.3471 | 0.2425 |
| MIC (held out) | log_loss | 0.7075 | 0.5932 | 1.1571 | 0.7842 | 0.7842 | 0.5791 |
| splice (held out) | log_loss | 1.0258 | 0.3165 | 0.1711 | 0.1803 | 0.1803 | 0.1090 |
| students_dropout_and_academic_success (held out) | log_loss | 1.0198 | 0.6183 | 0.6657 | 0.6061 | 0.6061 | 0.5667 |
| QSAR_fish_toxicity (held out) | rmse | 1.4446 | 0.8704 | 0.9018 | 0.9070 | 0.9070 | 0.8581 |
| concrete_compressive_strength (held out) | rmse | 16.7697 | 5.4113 | 4.5612 | 4.8421 | 4.8421 | 4.2136 |
| healthcare_insurance_expenses (held out) | rmse | 12131.4077 | 4976.7774 | 4962.3126 | 4919.3060 | 4919.3060 | 4715.6102 |
| airfoil_self_noise (held out) | rmse | 6.7678 | 1.8202 | 1.9046 | 1.5539 | 1.5539 | 1.3467 |
| Another-Dataset-on-used-Fiat-500 (held out) | rmse | 1930.1094 | 738.3202 | 762.4539 | 743.7459 | 743.7459 | 739.8138 |
| miami_housing (held out) | rmse | 320712.3226 | 97339.8131 | 92227.5242 | 91855.6187 | 91855.6187 | 83191.2196 |

## What was not done, and why

**MLE-bench** ([OpenAI, 2024](https://openai.com/index/mle-bench/)) scores AI agents on 75 Kaggle competitions by
how often they reach a medal. It was considered and not adopted. It measures an agent that writes its own code end
to end, which DataClean AI deliberately is not: it chooses from a fixed registry and never runs generated code. Many
of its competitions are images, text or audio, and need GPUs and downloads of tens to hundreds of gigabytes. The
tabular part of what it measures is covered more cheaply, and more fairly, by the AMLB tasks above.

**Published AMLB results.** Comparing with the scores AutoGluon, H2O and others reached in AMLB's own runs would
mean comparing different hardware and time budgets. FLAML is run here instead, on the same machine and budget as
the app, which is a fair comparison.

**The LLM's choice of fix.** The cleaning benchmarks use a fixed fix for each finding. Scoring the explanation
agent's choices would need an API key and paid calls on every run, and would measure a model that changes under the
app. It can be added as an option if the fixed mapping and the agent are ever seen to disagree.
