# 0002. Should the verdict's multiple-candidate correction be relaxed now that models are chosen on training folds?

- **Status:** Recommendation only, 15 October 2026. The decision is the owner's (issue #35); nothing in
  the app has been changed.
- **Recommendation:** keep the correction as it is.
- **Reproduce the numbers:** `backend/.venv/Scripts/python.exe backend/scripts/simulate_verdict_correction.py --runs 30 --weak-runs 12`
  (about 14 minutes; the defaults, 18 and 6, finish in under ten and are too few to settle a close question).

## The question

`modeling._auc_noise_threshold` sets the AUC a two-answer model must clear before it is called better
than guessing: `0.5 + z * SE`, with `SE = sqrt((n1 + n2 + 1) / (12 * n1 * n2))` and `z` rising with the
number of candidates trained (`_Z_BY_CANDIDATE_COUNT`: 1.645 for one, 2.13 for three). The rise was there
because the app used to report the best of several draws on the test set, and the maximum of several noisy
estimates runs high.

Since `ec21948` the winner is chosen on cross-validated training folds. The reported test AUC is then one
draw, which would make the rise unnecessary: it would only make the verdict stricter than the 5% false-alarm
rate it was built for. `ec21948` left it in place on purpose, because loosening a guard deserves evidence.
This is that evidence.

## What was measured

`train_candidates` was run for real, with all three classifiers (the production candidate count), on
generated data at four test-set sizes (30, 60, 120 and 500 rows, with the usual 20% hold-out, so 150 to 2,500
rows in all). Each run was then re-scored under two rules by calling `assess_result` again with a different
`model_count`:

- **current:** the rule as shipped (three candidates, z = 2.13);
- **relaxed:** one draw (z = 1.645) whatever the number of candidates.

Two scenarios. **Noise:** the target is a coin flip unrelated to every column, so any "marginal" or "useful" call
is a false positive. **Weak signal:** the target depends weakly on one column (population AUC about 0.63), so a
"marginal" or "useful" call is a correct detection, which is what the stricter rule costs. Balanced answers,
five numeric columns and one category. The seed check and the best-single-column baseline were switched off for
speed; neither touches the verdict.

## Results

30 noise and 12 weak-signal datasets per size, seed 7.

### Pure noise: every marginal or useful call is a false positive

| Test rows | Rule | Runs | Guessing | Marginal | Useful | False-positive rate (95% range) |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | current | 30 | 29 | 0 | 1 | 3% (1%-17%) |
| 30 | relaxed | 30 | 28 | 0 | 2 | 7% (2%-21%) |
| 60 | current | 30 | 29 | 1 | 0 | 3% (1%-17%) |
| 60 | relaxed | 30 | 28 | 2 | 0 | 7% (2%-21%) |
| 120 | current | 30 | 30 | 0 | 0 | 0% (0%-11%) |
| 120 | relaxed | 30 | 28 | 2 | 0 | 7% (2%-21%) |
| 500 | current | 30 | 28 | 2 | 0 | 7% (2%-21%) |
| 500 | relaxed | 30 | 23 | 7 | 0 | 23% (12%-41%) |
| **All four sizes** | current | 120 | 116 | 3 | 1 | **3%** (1%-8%) |
| **All four sizes** | relaxed | 120 | 107 | 11 | 2 | **11%** (6%-18%) |

No test set in the simulation was too small to judge (balanced answers, so every size cleared the 30-row and
10-per-answer minimums).

### Weak real signal: marginal or useful is a correct detection

| Test rows | Rule | Runs | Guessing | Marginal | Useful | Detection rate (95% range) |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | current | 12 | 10 | 0 | 2 | 17% (5%-45%) |
| 30 | relaxed | 12 | 9 | 1 | 2 | 25% (9%-53%) |
| 60 | current | 12 | 9 | 1 | 2 | 25% (9%-53%) |
| 60 | relaxed | 12 | 9 | 1 | 2 | 25% (9%-53%) |
| 120 | current | 12 | 4 | 8 | 0 | 67% (39%-86%) |
| 120 | relaxed | 12 | 3 | 9 | 0 | 75% (47%-91%) |
| 500 | current | 12 | 0 | 12 | 0 | 100% (76%-100%) |
| 500 | relaxed | 12 | 0 | 12 | 0 | 100% (76%-100%) |
| **All four sizes** | current | 48 | | | | **52%** (38%-66%) |
| **All four sizes** | relaxed | 48 | | | | **56%** (42%-70%) |

### Is the chosen model's test AUC a single draw? (pure noise, same runs)

| Test rows | Mean AUC, chosen | Mean AUC, all candidates | 95th percentile, chosen | Bar, relaxed | Bar, current |
| --- | --- | --- | --- | --- | --- |
| 30 | 0.503 | 0.505 | 0.685 | 0.677 | 0.729 |
| 60 | 0.494 | 0.503 | 0.635 | 0.624 | 0.660 |
| 120 | 0.504 | 0.500 | 0.580 | 0.587 | 0.613 |
| 500 | 0.504 | 0.500 | 0.554 | 0.543 | 0.555 |

The chosen model's mean AUC is no higher than the average candidate's, which is what a single draw looks like and
not what the best of three test draws would look like.

### A larger check on the one cell that looked wrong

The relaxed rule's 23% at 500 test rows is far from the 5% a single draw should give, so that cell was repeated
with 120 further datasets (`scratchpad`, not committed): one model alone cleared the relaxed bar in **5.0%** of runs
(spread of its AUC 0.024, against 0.026 from the formula), and the three-model run in **7.5%** (9 of 120; mean AUC
0.497). The 23% was mostly chance, and the three-model tail may be slightly heavier than one draw, but 9 of 120 is
inside what 5% produces.

## What it shows

1. **The premise holds.** Choosing on training folds leaves the test AUC close to a single draw: mean unmoved,
   spread near the formula's.
2. **The correction is therefore stricter than it needs to be**, which is what the issue suspected. On noise the
   current rule's pooled false-positive rate is about 3%, against the 1.7% that z = 2.13 aims at and the 5% that z = 1.645
   aims at; the relaxed rule's is about 11% in these runs, higher than its 5% aim, though the largest part of that excess
   is the 500-row cell that the larger check explained as mostly chance.
3. **What relaxing buys is small.** On the weak signal it found 2 more of 48 datasets (56% against 52%), both of them
   at 30 to 120 test rows, where a verdict is least reliable anyway. At 500 rows both rules find everything.
4. **The project's benchmark** ("on data with no signal, 23 of 25 runs are correctly flagged and none are called useful")
   holds under both rules at 60 test rows (29 and 28 of 30 flagged, none useful). At 30 test rows neither holds it:
   each rule calls one or two noise runs "useful", see below.

## Recommendation

**Keep the correction.** The two rules differ by a few points in either direction, the extra detections come from the sizes
where a verdict is weakest, and the cost is a false-positive rate that reaches about one in ten on noise in these runs. The
guard exists to stop a tool whose job is to say "this learned nothing" from saying "this works"; when the evidence for
loosening it is two datasets in 48, the safer error is the one already made.

The trade-off, stated plainly: keeping it means a real but weak signal on 30 to 120 test rows is called "no better than
guessing" slightly more often (about 4 points in these runs); relaxing it means pure noise is called "marginal" more often
(about 8 points).

If the owner wants it relaxed anyway, the change is one constant table and the cost is the benchmark above; a milder
option is to relax it only from 120 test rows up, where the simulation shows the detection gain concentrated, though one
cell of 12 datasets is thin evidence for a threshold.

No follow-up issue is linked because no change is recommended.

## A different finding, worth its own issue if the owner agrees

At 30 test rows the noise bar is 0.729 under the current rule and 0.677 under the relaxed one. Under the current rule
that is above the 0.70 cut-off for "useful", so any run that clears the bar is called **useful**, never merely marginal,
and a lucky noise run reads as the best verdict; under the relaxed rule the "marginal" band is only 0.02 wide. That is how
each rule came to call one or two noise runs useful at 30 rows. The verdict logic checks `auc < noise_floor` first and
`auc < 0.70` second, so on small test sets the middle band is missing or thin. Not changed here; it would be a small,
separate change (for example, never call a result "useful" on fewer than about 60 test rows) and a decision of its own.

## Limits of this evidence

- Few datasets per cell (30 noise, 12 weak): the intervals are wide and the differences between rules are inside them
  at most single sizes. The pooled rows are the readable ones.
- One data-generating process: balanced answers, five numeric columns and one category, one effect size. Lopsided
  targets, many columns and strongly correlated columns were not tried.
- Three candidates, the production count. With fewer, the current rule is closer to the relaxed one.
- Two-answer targets only; the many-answer rule (`ovr_noise_floor`) uses the same candidate-count table and was not
  simulated here.

## What would change this

- More runs showing the relaxed rule holding near 5% on noise at every size (`--runs 100`), which would make the
  detection gain free.
- A real weak-signal data set where the stricter rule demonstrably hides a model people would want to see.
- The owner preferring the aim of the rule (5% false alarms) to the aim of the benchmark ("none called useful").
