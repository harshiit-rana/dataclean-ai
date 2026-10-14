# DataClean AI — what this is, what's built, what's next

Last updated: 14 October 2026 · 72 commits · 980 backend tests, 211 frontend tests

---

## What the project is

An open-source tool for the gap between *"I have a messy CSV"* and *"I have a model I can
defend."*

You upload a CSV and say what you're trying to do with it — including, if you know it, the
column you're ultimately trying to predict. The tool profiles the file, explains what's wrong
with it in plain English, and proposes a fix for each problem. **You approve or reject every
fix individually.** Only what you approved runs. Then, if the cleaned data can support it, it
builds a model, tells you honestly whether that model is worth anything, and hands you a
script that reproduces it.

Two things make it different from a cleaning script with an LLM bolted on:

**Nothing happens without your say-so.** The AI proposes; you decide. Each suggestion arrives
as a card with four parts — what was found, what it proposes, why, and what the tradeoff is.
When you decline one, your reason is recorded verbatim and printed in the audit report,
attributed to you, never paraphrased and never sent back to the model.

**The app is allowed to say no.** This is the thread running through everything. It will tell
you the data can't support the model you asked for. It will tell you the model it just built
learned nothing. It will tell you your accuracy target is out of reach and that the problem is
your columns, not your settings. A tool that always hands back a confident answer is less
useful than one that can say it hasn't got one.

---

## Architecture

| Layer | Choice | Why |
| --- | --- | --- |
| Orchestration | LangGraph `StateGraph` with a native `interrupt()` at human approval | The pause for approval *is* the workflow, not an afterthought |
| Persistence | Redis 8 via `AsyncRedisSaver`, appendonly, named volume | A session survives a container restart mid-review |
| API | FastAPI, POST-based SSE for streaming | Cards stream as they're written; `EventSource` can't POST |
| Frontend | React 18 + Vite + Tailwind + Recharts | — |
| LLM | Anthropic or Groq, one config switch | Groq is far cheaper; Anthropic is better verified for structured output |
| Deploy | Docker Compose, three services | Clone → `docker compose up` → working instance |

**Roughly 15,500 lines of backend, 8,400 of frontend, 12,400 of backend tests** (comments and
blank lines counted).

Two architectural decisions worth knowing about:

- **Graph state is JSON-native.** `GraphState` holds `list[dict]`, not nested Pydantic models,
  because nested models get serialised into checkpoint envelopes that don't reliably
  reconstruct — which breaks resuming a session.
- **The LLM never writes code.** It *selects* an operation from a fixed registry of 18 and
  supplies parameters. Nothing calls `exec`. This is why there's no sandbox: there's nothing
  to sandbox.

---

## What's built

### The cleaning path

**Upload and goal.** CSV up to 50 MB, non-UTF-8 detected and converted. You state your use
case, what the cleaned data should look like, which columns matter most, how aggressive to be
— and optionally the column you're predicting.

**Profiling.** Seven issue types detected: missing values, duplicates, outliers (IQR),
type mismatches, inconsistent formats, stray whitespace, high cardinality. Ranked by relevance to your stated
goal.

**Suggestion cards.** One per issue, written by the LLM and streamed token by token. Found /
Proposed action / Reasoning / Tradeoff. Goal-aware — a card for a churn model reads differently
from the same issue in a reporting dataset.

Two things the cards used to get wrong, both the profiler's fault rather than the model's: every
casing issue was described with the same canned pair (*"e.g. 'USA' vs 'usa'"*), which the LLM
dutifully quoted on a card about plan tiers; and outlier cards quoted the detection fences as if
they were the data — *"between -19.32 and 63.15"* for monthly spend. The issue description now
carries the real colliding spellings from the column, and outliers are described by the values
found (*"1 unusually high value (the largest is 250.00), while the other 8 sit between 10.00 and
13.00"*), never by the threshold that found them.

Stray spaces are a seventh kind of problem. The churn sample's `plan` column held `' basic '`
beside `basic`, `Basic` and `BASIC`; the casing check caught the capitals, and lower-casing them
left `' basic '` and `basic` as two unrelated categories. The card now says what fixing it buys:
*"Trimming them merges 12 different spellings into 9, and ignoring capitals as well leaves 3."*

**Approval, then execution.** The graph pauses at `human_approval` and checkpoints to Redis.
Your decisions resume it. Eighteen whitelisted operations do the work. Your original file is
never modified — there's a test asserting it's byte-identical after a run.

**Undo.** Any applied fix can be reversed. The action log is rewritten in place and the audit
report regenerated.

**Conversational cleaning.** A plain-English box: *"drop rows where salary is under 30000"*,
*"remove the duplicate rows"*. Same whitelist, same audit trail.

**Audit report.** Markdown and PDF. Every change with before/after statistics, every declined
change with your reason verbatim, every issue found. Bundled into a zip with the cleaned CSV.

### Cleaning towards a model

Naming a target column changes what the tool does, not just how it words things:

- **Leakage detection.** Columns that predict the target too well to be legitimate — including
  *missingness* leakage, where the fact a value is blank gives the answer away. Normalised
  mutual information with a deliberately high threshold, because a warning that fires on
  ordinary correlation teaches people to ignore it.
- **Target safety, enforced in code.** A fix that would impute the column you're predicting is
  impossible, not discouraged. The guard runs when the card is *built*, so you approve the
  corrected operation — substituting it silently at execution time would make approval
  meaningless.
- **Readiness verdict.** Ready / usable-with-caveats / blocked, recomputed from the file as it
  currently stands, so it stays correct after an undo or a chat edit.
- **Contradictory labels.** Rows with identical inputs and different answers. No statistical
  check finds these and no model survives them — they're a hard ceiling on accuracy.
- **Repeated entities, and the split that fixes them.** A column whose values recur across
  rows means a random split can put the same customer in both training and test, so the model
  scores well by recognising the customer rather than learning the pattern. The app detected
  this for a long time and could only advise treating the numbers as optimistic; it now offers
  a grouped split on exactly the columns it flags. Grouping is applied all the way down --
  `GroupShuffleSplit` for train/test, `GroupKFold` for cross-validation and for every fold the
  improvement loop searches on, and the grouping column withheld from the inputs. Missing any
  one of those just moves the leak.

  On a test file with **no learnable pattern at all** -- the label a coin flip per customer,
  one column a per-customer constant -- a random split reports AUC 0.63 and calls it
  *"marginal"*. The grouped split reports 0.50 and says *no better than guessing*. The old
  behaviour invented a signal out of nothing.

### Business rules

Everything else in the app reasons from the distribution of the data, which makes it
structurally blind to the largest category of real-world error. `age = -5` is not an outlier,
it is impossible. A 150% discount sits comfortably inside the interquartile range of a column
that runs 0 to 200 because half the rows are wrong. A delivery date before its order date is
two perfectly ordinary dates. The IQR check will flag a legitimate premium salary and walk
straight past all three.

Four kinds of rule: a floor and/or ceiling on a column, one column ordered against another
(dates or numbers), and `price * quantity == total` with a tolerance — relative *and* absolute,
because money is rounded and an exact comparison fires on nearly every row of every real file.

**A rule is data, never code.** A frozen dataclass and a match statement; no `eval`, no
expression strings. Rules arrive from proposals, saved sessions and API requests — all
untrusted input, and untrusted input that can express arbitrary computation is a hole no
amount of validation closes. The same reasoning that keeps the cleaning path on a whitelist.

**The app proposes; nothing applies itself.** Proposals are guesses from column names and
values, each carrying the evidence behind it — not *"we think age can't be negative"* but
*"age holds 3 negative values"*. They are also filtered to rules that would find something in
this file, because a list padded with plausible rules that catch nothing is how people learn
to approve without reading. On clean data it proposes nothing at all. `balance`, `profit` and
`margin` are left alone, which is why the non-negative list is a list of known-safe words
rather than a guess at what sounds quantity-like.

Rule findings are `Check`s, so the provenance panel, the readiness summary and the audit report
picked them up for free — that was the point of doing provenance first. They run outside the
target-dependent block, because a broken rule is a fact about the data rather than about a
model, and a session that never named a target still needs to hear it.

A violation is a finding, never a fix. It doesn't block a model and nothing is dropped: which
rows are wrong, and whether the rule itself is the error, is a decision. Past 30% of rows the
app says so outright — a constraint a third of the data disagrees with was probably mis-stated.

### Decision provenance

Every verdict shows its working. "Ready" is a claim about what *isn't* wrong, and a reader
can't confirm that by looking at the data — so each check records what it concluded, the
numbers behind it, and what would change the answer.

Three outcomes matter, not two. **Passed** and **failed** are obvious; the one that earns this
its place is **not_run**, with the reason. `assess_readiness` used to return early when the
target was missing, so six later checks silently never happened and the report was
indistinguishable from one where they had all passed. Now nothing returns early: a check that
can't run records what stopped it.

Deliberately a flat list with a few "blocked by" edges rather than a graph — the checks aren't
causally chained, they're independent conditions that all have to hold, and arrows between them
would encode a dependency that mostly isn't there.

Nothing is stored. It's derived on every read, the same as the verdict it explains, because the
moment evidence is persisted an undo turns it into a lie. `blockers` and `warnings` are now
derived views over the checks, so a new check reaches the summary, the panel and the audit
report for free. The panel leads with what failed or couldn't run and collapses the passes
behind a count — twelve green ticks on open is compliance theatre.

### The model path

**Plan first.** The decisions that matter happen before any fitting — which columns are
inputs, how the data splits, what gets tried. Leaky columns and row identifiers arrive
pre-unticked with the reason attached. A time-based split is offered when there are dates,
because a random split lets a model learn from the future to predict the past.

**Training.** Six models across classification and regression. All preprocessing lives *inside*
the sklearn `Pipeline`, so `fit` sees only the training fold — there's a test asserting the
fitted imputer's median came from training rows alone.

**An honest verdict.** A model that learned nothing still reports 70% accuracy on a 70/30
target. So results are judged against what luck alone could produce on a test set of that size,
and against always guessing the same answer. On data with no signal, 23 of 25 runs are
correctly flagged and none are called useful.

A regression is judged the same way. It used to be fixed R² cut-offs (under 0.05 is guessing,
under 0.3 marginal) for 35 test rows and 35,000 alike, so a lucky 0.35 on a few dozen rows was
"useful". The verdict now stands on the lower end of the 95% R² interval from the bootstrap over
the test rows: the interval includes 0 means no better than guessing, a lower bound under 0.3
means marginal, and only a range that clears 0.3 throughout is useful. The headline prints the
range, so the same R² reads differently on 40 rows and on 4,000. On regression targets with no
signal, 25 of 25 runs with 40 test rows are called guessing, and a clearly learnable problem is
still useful. A result with no interval (an older one) keeps the fixed cut-offs.

There is a fourth verdict before those three: **not enough test data to judge**. Below 30 test
rows, or 10 in the rarer class, no verdict is given at all — and that includes "this model does
not work", which is just as much a claim about the test set as "this model works" is. The
distinction matters because the two failures look identical from the outside and call for
opposite responses: a bad model needs different columns, an unjudgeable one needs more rows.

**Inputs built from the columns, not just chosen among them.** Tuning can only move a score
as far as the columns allow, and the app was throwing columns away: a date was dropped
wholesale for having too many distinct values, and free text was misfiled as an identifier.
Four recipes now turn columns the model couldn't use into ones it can: **date parts** (year, month,
weekday, days-since), **text length** in characters and words, a **missingness flag**, and
**frequent-value pooling** that keeps a high-cardinality column by giving its common values an
input each and pooling the rest rather than discarding the column.

Each is a declared piece of data, never code, so `train_model.py` reproduces it line for line
— a recipe the export can't rebuild is a model trapped in the app, and there is a test that
runs the generated script and checks the score matches. The first three are row-wise and
stateless, so deriving them before the split cannot leak; the fourth learns which values are
common, so it lives inside the pipeline on the training fold like every other learned
statistic. And nothing applies itself: proposals arrive with the evidence behind them, off
until ticked, and re-proposed server-side rather than trusted from the request — so a recipe
on a column withheld for leaking the answer can't smuggle it back in through a missing flag.

**Two wrong reasons fixed on the way.** A date column is row-unique, so the identifier rule
claimed it and pre-ticked it for removal — which meant the recipe that would make it usable
was never even offered. Free text was claimed the same way; the RSNA run dropped a column of
radiology reports because *"nearly every row has a different value, so this identifies rows
rather than describing them"*, which for a paragraph of clinical prose is precisely backwards.
Neither is an identifier any more, and `classify_features` now names them for what they are
with what to do about them attached.

**The loop searches the columns too.** A fourth round tries the derived inputs the opposite
way round from how the run started — all of them if none were on, none if some were — with the
leading model and settings, judged on the same training folds and subject to the same
"beat the fold spread" rule. *"The derived inputs did not help"* and *"they did"* are both
findings. On a test file whose answer depends only on the weekday of a date column, the run
without them is correctly called **no better than guessing** and the run with them is
**useful**.

**Extreme values, tamed where they cannot leak.** Cleaning no longer proposes capping outliers
when a target is named — it computes the limit over every row, test rows included — so the model
plan offers the same job inside the pipeline. Ticking a column adds an `IQRClipper` step before
the imputer and scaler: it learns each column's limits (1.5 × the interquartile range) from the
training fold only, pulls values beyond them back to the limit, and loses no row. It is offered
only where some value lies past 3 × the range — a bell-shaped column puts a few values past 1.5
by chance, and a list padded with columns that have nothing to clip teaches people to tick
without reading — described by the values found, off until ticked, and it survives "Improve it"
and a page refresh. The exported script carries the app's own class source rather than a second
copy, and a test runs it and compares the score. Because the class is not scikit-learn's, a
`model.joblib` with this step needs the class to be defined before loading; the model card and
the script say so.

**You watch it work.** The search used to be a single blocking request, and the page showed a
paragraph of explanation next to a spinner for anything from twenty seconds to a few minutes.
It now streams: a line as each round *starts* (the settings search alone can outlast anyone's
patience with a still screen), each round as it finishes — rejected ones included — and the
result. The search saves its own result in the worker, so a closed tab or a dropped connection
doesn't lose it; the page says so and a refresh finds it. A comment line every fifteen seconds
keeps proxies from closing a connection that looks idle.

Doing this found a defect in every stream the app has. The client library's default closes a
request when the tab is hidden and *sends it again* when the tab comes back. Every stream here
is a POST that does something — start profiling, apply approved fixes, run a search — so
switching tabs and back re-sent it. All three now keep the one request open.

And "Improve it" after a page refresh used to fall back to the plan's defaults, because the
request that trained the model lived only in component state — which quietly turned a grouped
run into a random one. It now rebuilds the request from the saved result.

**What the churn sample actually showed** (30 September 2026). This section used to open by
saying the churn sample proved the ceiling was the columns, and that `signup_date` was the
missing signal. Running it end to end through the API said otherwise, and found three things:

- *"Improve it" crashed whenever the settings search kept a result.* The route re-added the
  `model__` pipeline prefix before handing the settings to the constructor, which rejected
  them. The tests passed because on their data the search never kept anything — the settings
  dict was empty and every check over it was vacuous. Now fixed, with tests that force a kept
  setting and fail on the old code.
- *The file already has `tenure_months`,* so `signup_date`'s parts add nothing — and with the
  derived inputs pre-ticked, the default train got slightly worse (AUC 0.825 → 0.799). The
  fourth round measured it correctly (about −0.005, inside the fold noise). Derived inputs are now
  **off until ticked**: an exclusion is pre-ticked because it guards against harm, but an extra
  input is a change, and changes are the user's to opt into. The loop tries them either way.
- *`last_name` was the model's second most important input.* With about a hundred rows per
  surname, a churn-rate spread of 0.21–0.35 across twelve of them is what chance produces, and
  the importance chart — which sums the coefficients of every category — made chance look like
  a finding. Columns holding a person's name or contact details are now suggested for exclusion
  with the reason attached (a closed list of words, so `product_name` is left alone). Leaving
  them out *raised* AUC to 0.842, and the settings gain the loop had kept shrank to within the
  noise — some of what it had "found" was fitting the surnames.

**Where the cut-off sits.** A two-class model produces a probability, and somewhere a line is
drawn; every precision and recall figure is measured at that line, and by default it sits at
0.50 because that is scikit-learn's number. The threshold sweep showed the trade for a long
time without letting anyone act on it. Now the user states what matters — *a missed case
costs 5× a false alarm*, *catch at least 80%*, *be right at least 70% of the time when
flagging*, or just *balance the two* — and which answer counts as "a case", and the app moves
the line.

The user never picks the number, and that is the design. Choosing a cut-off by eye from the
sweep, which is drawn over the test rows, is selecting on the test set in miniature — on a
small test set it is enough to manufacture a recall figure that will not survive new data. So
the line is chosen on out-of-fold predictions over the *training* rows (grouped folds when
the split is grouped, the same folds the improvement loop searches on) and the test set is
opened afterwards to report what that line actually did. Both numbers are kept: what the
folds promised and what the test set delivered, next to the default at 0.50. A floor that no
cut-off can clear is said outright rather than quietly approximated. The line reaches the run
history, the model card and `train_model.py` as a declared constant — and the card explains
that `model.joblib` doesn't know about it, because `predict()` answers at 0.50.

**Thirteen diagnostics**, as interactive charts with a table view behind every one: ROC,
precision-recall, confusion matrix, per-class breakdown, calibration, threshold trade-off,
class balance, feature importance, permutation importance, cross-validation spread, learning
curve, correlation matrix — and for regression, predicted-vs-actual, residuals, residual
distribution, error by range.

**One metric across the charts.** The cross-validation spread was scored in AUC for a two-class
target while the learning curve and the permutation importance beside it were scored in accuracy,
so on a lopsided target the curve could sit flat while the AUC improved and the two read as
disagreeing about one model. All three now use the primary metric (AUC for two classes, accuracy
for more, R² for a number), fixed from the training rows, and every chart and the card name it.

**Calibration, in numbers.** The calibration chart showed whether "70% likely" happens 70% of the
time but gave no figure, so two runs could not be compared on it, and the cut-off and the exported
probabilities both depend on it. Two-class results now carry the **Brier score** (mean squared
distance between each probability and what happened), the **expected calibration error** (the
count-weighted mean of |predicted - observed| over the chart's own ten bins) and the Brier score of
always predicting the training base rate, which is what the first has to be read against. They sit
in the held-out table and the card with their bootstrap ranges, in the chart's description, and in
the run history, are marked lower-is-better, and are deliberately not coloured pass or fail: the
error is noisy on a small test set. Multi-class and regression results are unchanged.

**Calibration as an action.** When the error is past five points *and* clearly past what chance
alone would show on that many test rows (on a few hundred rows even honest probabilities score
several points, so the bar is relative), the results page offers to calibrate: sigmoid, or isotonic
where there are at least a thousand training rows and a hundred of the rarer answer. It is fitted
with `CalibratedClassifierCV` on cross-validation folds of the training rows only, through the same
fold rule as everything else done to them (whole groups, copies together, forward in time), then
the held-out rows are scored once more with it and the page and card show before and after: Brier
score, calibration error, AUC, and recall and precision at the default line, which move because 0.50
now sits on honest probabilities. It is one model with one mapping (`ensemble=False`), so a sigmoid
leaves the AUC unchanged to the last digit and a test asserts it; a forward-in-time split cannot use
that form, so it averages one copy per fold and says so. A method the data cannot support is
declined with the reason and the model left alone. The cut-off is chosen on calibrated
probabilities, calibration appears in the run history, and applying it re-runs training, which
counts as another look at the test rows. A test with every held-out answer flipped shows the fitted
calibrator and the cut-off do not move.

The cut-off is chosen on calibrated probabilities. With calibration on, the out-of-fold scores the
line is chosen on come from a calibrated model, with the calibration fitted inside each fold, and
the line is applied to the calibrated model, so the precision and recall in the table are what the
delivered model does; the card says the line is on calibrated probabilities. Choosing the line
before calibrating would not have been a small error: on a lopsided target a line of 0.46 promised
81% recall and delivered 55% once the probabilities were honest, where the line chosen on the
calibrated probabilities (0.22) delivered what it promised. The cost is nested folds, outer x
(inner + 1) fits instead of one per outer fold -- 30 rather than 5 -- which measured about six
times the uncalibrated cut-off step for a random forest (roughly 2-3 s to 14-17 s at 2,000-10,000
rows) and under a second for logistic regression. No bound was agreed, so none is enforced beyond
a test that pins the fit count; the resource guard planned for Milestone 3 is where a ceiling
belongs. The cheaper alternative, fitting the mapping once on the pooled out-of-fold scores and
applying it to them, costs no extra fits but lets the mapping see the rows it is scoring, which
flatters an isotonic fit.

The exported `train_model.py` rebuilds the calibrated model: the same method, the same
`ensemble` setting, and the same folds listed explicitly (stratified, whole groups, copies together,
or forward in time), so it reproduces the reported accuracy and Brier score to four places; the
script now prints the Brier score of every two-class model, and the prediction snippet says the
probabilities are calibrated and what that does to the default line. Tests run the generated file
in a subprocess for a plain, a grouped, a dated, a copy-keeping and a sealed run, for both methods.

**The simplest model the folds cannot separate from the best.** The winner used to be the raw
highest cross-validated score, so a forest that beat logistic regression by 0.002, with the folds
disagreeing by 0.03, "won" although the evidence does not tell them apart. It is now the
one-standard-error rule: among candidates whose mean is within one standard error (the fold spread
over the square root of the number of folds) of the best, the simplest is kept, "simplest" being
a written `complexity_rank` on each model rather than the order of a list. When the rule changed
the answer the page and the card say which model scored highest and why the simpler one was kept;
when it did not, they say nothing. The improvement loop's starting family uses the same rule, so
a train and a search cannot begin from different models on the same evidence. It reads fold scores
only, and a test flips every held-out answer to show the choice does not move.

**Saying when two models cannot be told apart.** One winner in a table reads as a ranking whether
or not the evidence separates the models. Every candidate is cross-validated on the same folds, so
`score_config` now returns each fold's score and the chosen model is compared with its closest
rival -- the raw highest scorer when the rule kept a simpler model, otherwise the second-highest --
fold by fold. The interval on the average gap uses the Nadeau-Bengio correction, because folds
share training rows and the plain paired interval calls differences real far too often. When it
includes zero the page (under the comparison table) and the card say the two are within the noise
of each other, with the gap and its 95% range, and why one was kept; when it excludes zero they say
nothing. If the rule kept a simpler model against a lead the folds do show consistently, the note
says that lead is not noise instead of calling it indistinguishable. With three folds (large files)
the interval is very wide, which is the honest answer.

**How far a score could move.** "AUC 0.78" on 60 rows and on 6,000 used to look identical. The
chosen model's headline held-out metrics now carry a 95% interval from a percentile bootstrap over
the test rows — `0.78 (0.71–0.84)` — resampled within each class so a rare class cannot vanish
from a resample, and by whole groups for a grouped split, where drawing a customer's rows one at a
time would claim certainty the data does not hold. It is one `predict` and arithmetic over
resample weights, so it costs about 0.04 s on 500 rows and 0.3 s on 20,000; below the size at which
a verdict can be given at all it prints nothing. The primary metric's interval goes into the run
history, and the sealed final evaluation carries its own. A test shows that the bootstrap's point
estimates equal `evaluate`'s to the last digit, and that on a problem with a known AUC a 95%
interval contains it close to 95% of the time.

**A change between two runs is only coloured when it is real.** The run history painted any
difference above 0.0005 green or red, so +0.01 on a few hundred test rows -- well inside what
those rows could produce by chance -- read as "this helped". Each run now keeps the predictions
it made on the test rows (`run_predictions/<run>.npz`, removed when the run falls off the
history), and two runs trained on the same data under the same split are compared row for row: a
paired bootstrap scores both on the same resampled rows, which cancels what they share and is far
more sensitive than comparing two separate intervals. A difference whose 95% range excludes zero
is coloured and shows that range; one that straddles zero says "within the noise". Runs that
cannot be paired (an older run, a different split) fall back to comparing their stored intervals,
and a run with neither gets no verdict and no colour rather than the old fixed threshold.

**Every number says what kind it is.** The app shows three kinds of score that are not
interchangeable — fold scores on the training rows (*exploratory*), the one-time score on rows
nothing was chosen against (*held-out evaluation*), and a score on newly uploaded rows (*new
data*) — and used to print them alike. Each now carries a label, and the held-out one carries
how many times those rows have been used. The table comparing models leads with the fold
scores and spreads that actually chose between them, where it used to print every candidate's
test score and mark one "chosen", which read as though the highest test score had won; the
others' held-out scores are behind a button that says why choosing by them would bring back
the problem. The model card makes the same split.

**An improvement loop** (newest). Searches model settings and the other model types, optionally
until a target you name. Built so it cannot fool itself: the test set is split off once and
never touched during the search, every round competes on cross-validated folds of the training
data, and a round is only accepted if it beats the incumbent by more than the folds disagree
with each other. It stops on its own and says which of three reasons — target reached, target
out of reach, or the rounds stopped paying for themselves — and it shows the rounds it
*rejected*, because "tuning this data didn't help" means the ceiling is your columns.

**A record of what you tried.** Results used to overwrite a single file, so retraining
discarded the previous run — which made the question people actually ask unanswerable: *did the
thing I just changed help?* Every run is now kept with its settings, its headline number and
its verdict, and the table shows the change between consecutive runs, because the subtraction
is the finding.

Runs survive a cleaning change rather than being wiped by it. Comparing across one is often the
most interesting comparison available — it just has to be labelled, so each run carries a
fingerprint of the data it trained on and anything older is marked *trained before the data
changed* and excluded from the arithmetic. It is a record, not a restore point.

**Artifacts you can leave with.** `train_model.py` that rebuilds the model from scratch
(tuned settings included), the fitted `model.joblib`, and a written model card that says what
not to trust. A test runs the generated script in a subprocess and checks it reproduces the
reported metrics.

### Where test information could reach a decision — an audit

Prompted by a roadmap review that made the rule explicit: *the final test set must never be used
to choose anything*. Every decision the model path makes was traced to the rows it sees.

**Fixed:**

- **The Train button chose its winner on the test set.** It fitted every candidate, scored each
  on the held-out rows and kept the highest — so the reported number was the best of several
  draws on those rows. The improvement loop had always avoided this; the plain train had not.
  Candidates are now compared on cross-validated folds of the training rows and the test set is
  opened once, for the winner. On a fixture where the honest run chose the random forest,
  flipping every test-set answer made the old code choose logistic regression instead; it now
  chooses the same model either way, and that is a test. The verdict's extra strictness for
  "best of several" is deliberately left in place: loosening a guard is its own decision, not a
  side effect of fixing what it guarded against.
- **A date-based split shuffled every fold inside the training rows.** The final test set was
  correctly the most recent rows, but model choice, tuning, the cut-off, the cross-validation
  spread and the learning curve all trained on later rows and validated on earlier ones —
  rewarding whatever exploits the future, the very thing the split exists to rule out. All of
  them now use forward-in-time folds.
- **Copies of training rows in the test set went uncounted.** The contradictory-labels check
  finds identical inputs with different answers; identical rows with the *same* answer on both
  sides of the split — questions the model has already seen answered — were invisible. They are
  now counted, shown on the results page and cautioned in the model card.
- **Copies of training rows in the test set can now be prevented, not just counted.** The plan
  offers "keep identical rows together" when exact copies exist (the model's inputs and the
  answer), pre-ticked because it guards against harm, with the count as evidence. Ticked, all
  copies of a row go to one side of a random split — stratified so the rarer class still reaches
  both — and the count of test rows seen in training is zero. It changes nothing when there are no
  copies, applies only to the random split (the others already keep their own units whole), is
  recorded as its own split rule in the test-set ledger, and the exported script carries the app's
  own splitting function so it cuts the same rows. The folds inside the training rows keep copies
  together too — model choice, tuning, the improvement loop, the cut-off and the diagnostics all
  receive the copy ids — because a row and its copy straddling a fold is the same leak one level
  down. Those grouped folds are not stratified (the known limit below). Near-duplicates (one cell
  different) are not addressed.
- **`drop_outlier_rows` wasn't warned about** alongside the other fixes that read a statistic
  from the whole file before any split.
- **The app proposed fixes that leaked the test set into the training rows.** Filling blanks
  with a column's median, and capping or dropping outliers, compute their value over every row
  before any split. They were warned about but offered and, once approved, baked in — leaving
  the pipeline's own training-only imputer with nothing to do. With a target named, a card for a
  model input now comes out with no executable fix and says why ("Left for the model": the
  pipeline fills gaps from the training rows only); the prompt tells the model not to propose
  them and the code enforces it either way. Chat can still run them, with the warning. Sessions
  with no target are unchanged, and so is the target column's own handling.
- **A date-based split could cut a day in half.** The cut was a row count, so on daily data, or
  records loaded in batches, it usually fell mid-day: the model trained on part of a day and was
  tested on the rest, which is not predicting the future. The cut now moves to the nearest point
  where the timestamp changes (the larger training side on a tie), the result reports the
  fraction actually held out, `train_model.py` reproduces it, and a column where one timestamp
  holds more than half the rows is refused with the count rather than approximated. Forward-in-
  time folds inside the training rows still cut by row count; that is a separate issue (#21).
- **The cross-validation spread and the learning curve were scored with the test rows inside
  their folds.** They sit on the results page as the independent check on the test score, and
  were computed over training and test rows joined together. Both now use the training rows
  only; permutation importance still reads the test rows, because it explains the evaluated
  model and chooses nothing. On a small file the training-only spread is wider, which is the
  honest number. The learning curve was also the one fold-producing call that never received
  the grouping column, so a grouped run drew it from folds that scattered each customer; it
  does now. A test flips every test answer and asserts neither moves — under the old code the
  spread went from 0.013 to 0.054.
- **The plan's leakage suggestions read the test rows' answers.** Which columns to pre-untick as
  giving away the answer was measured over every labelled row. It is now measured on the training
  rows of the default split, and the plan says so; a file too small to split falls back to every
  row and says that instead. A test flips every held-out answer, on a file where doing so hides
  the leak from a whole-file check, and asserts the suggestions do not move.

**Known, and left as they are:**

- **A fix a user types in chat still runs from a whole-file statistic**, with the warning. That
  is deliberate: the app no longer *proposes* them, but an instruction is the user's call.
  Sessions that approved such fixes before this change carry the small leak, and so does a model
  trained after a chat fill: every run now lists, on its result, the applied whole-file fixes on
  columns it uses (undone ones and unused columns are left out), the card cautions with the
  operation and column named, and the results page says so in a line. Reading the action log
  adds a checkpoint read to each train, and a log that cannot be read is an empty one — training
  never depends on Redis for this.
- **The profiling-stage leakage cards still read the whole file.** The cards raised while
  cleaning, before any split exists, warn about a column that predicts the answer near-perfectly.
  They propose nothing that drops a column on its own, and no split exists yet to measure
  against, so they are left as a data-quality statement. The model plan's suggested exclusions —
  the ones that actually pre-untick a column — are the part that was fixed.
- **You can still look at the test set as often as you like — but the app now counts, and says
  so.** "Change a column, retrain, look" is selection on the test set by hand, and nothing
  automatic can stop a person. Every successful train and improvement is recorded in a ledger
  (per split rule, across data versions, never truncated). The first look is an independent
  estimate; the second and third carry a note that if anything was changed because of an earlier
  score this one is no longer independent; from the fourth the verdict's headline itself says
  the score is optimistic and points to new labelled rows. Nothing is blocked — a strict limit
  would make small files unusable. New labelled rows are the cleanest fresh holdout there is, so
  "Use this on a new file" records a check on a file's real answers as an independent evaluation
  of that model — only when there are enough labelled rows to judge, never for the same file
  twice, and never on rows the model was built from (matched on their inputs, so a revised label
  does not make an old row new). It is shown beside the original score, never in place of it, and
  does not use up the held-out rows.

  **A holdout sealed before the first look** is the answer when the held-out rows have been used
  up. Opt-in at plan time, and offered only when both held-out parts are big enough to be judged
  (on the churn sample 852 / 183 / 183; on a 300-row file with a 20% rarer class each part would
  hold 9 of the rarer class, so it is withheld with the reason). Training, development and sealed
  rows are 70 / 15 / 15, with every guarantee the ordinary split gives — whole entities, forward in
  time, copies together — holding between all three. The sealed rows are fixed by the data and the
  split rule alone, so nothing done while iterating can move them; only `finalise` scores them,
  once per data version, and a second model asking for the same rows is handed the stored result
  with the reason. A seal taken after looks says so. The exported script reproduces all three
  partitions.

### Using a session again

A session used to be single-use. Next month's export meant uploading it and approving every
fix again, and a trained model could only be used by writing Python against the downloaded
joblib. Now a finished session takes a new file — **Use this on a new file** from the export
or model page — and does everything it can with it:

- **Cleans it the same way.** The action log already describes the cleaning completely; undo
  depends on that. The same log runs against the new file.
- **With the statistics this session learned, not the new file's own.** `fill_median` on the
  new file fills with the *original's* median, and the outlier limits are the original's.
  Filling February's gaps with February's median would clean the new rows differently from the
  rows the model learned from. Those values are not stored: replaying the log over
  `original.csv` reads each one off the data as it stood at that step, and they are handed to
  the operation through an argument that neither the LLM nor chat can set.
- **Predicts every row** that survives, at the cut-off the model was reported at rather than
  `predict()`'s 0.50, with the derived inputs rebuilt and the columns aligned.
- **Accounts for every row.** The predictions file has every uploaded row, in order, with its
  prediction or the step that removed it. A row that silently vanishes breaks anyone joining
  predictions back onto their own data.
- **Says whether the new rows look like the training data.** PSI per input against the
  training deciles, the share of values outside the training range, the share of categories
  the model never saw. Below 30 rows it declines to judge. A model scores whatever it is given
  and never notices when it is extrapolating; this is the app saying so for it.
- **Checks itself when it can.** If the new file carries real answers, it reports how the model
  did on them against how it did when built — with the same too-few-rows rule as the verdict.

**And outside the app, `clean.py`.** `train_model.py` let someone rebuild the model without
the app; the cleaning behind its training data had no equivalent. The export now includes the
session's cleaning as a script, one commented block per applied fix, with every learned value
written in as a literal. Run on the original upload it reproduces the cleaned file; run on a
later export it cleans it exactly as the app's "use this on a new file" does. A test runs it in
a subprocess over a log that exercises all eighteen operations and compares the output with
the app's, byte for byte, on both the original and a changed file.

A column the new file lacks is skipped with the reason rather than failing everything else; the
target being absent is expected, and said to be. A missing model input is refused by name, all
of them at once.

Tried on the churn sample, trained on the first 900 rows and applied to the remaining 318:
every row came back, 316 predicted and 2 dropped as duplicates with that reason, drift was
judged stable, and the model scored 0.785 AUC on the later rows against 0.749 when built.

### Cross-cutting

- **Dataset versions.** The session's data is rewritten in place — by the cleaning run, by undo,
  by a chat instruction — so nothing stable existed to attach a model, a test-set look or an
  export to. Each state now has a numbered version in the session's meta sidecar, with the full
  content hash, the row and column counts, when, and what produced it. A step that changes
  nothing adds nothing, and an undo that lands on an old state is recorded as a new version, so
  the history stays chronological. `GET /api/session/{id}` returns them.
- **Session access tokens.** Knowing a session id isn't enough to read its data. Issued at
  upload, only the hash stored, required on every session-scoped request. Off in development.
- **Retention.** Files deleted 24 hours after last use, swept hourly in-process.
- **Light and dark**, following your OS until you use the toggle.
- **A reload doesn't redo work.** Reloading the Review page used to re-profile the file and
  rewrite every suggestion card; it now replays the saved cards, and a session that has
  already been cleaned points you at the cleaning page instead.
- **Reproducible installs.** Dependencies are bounded above as well as below, at the versions
  the tests pass against. ydata-profiling was removed: its output was never read, and on a
  current install it could not even be imported.
- **Degrade, don't crash.** A failed LLM call, an operation that can't run, a PDF that won't
  render — none of these end a session that already touched your data.

---

## Tested against something it was never designed for

Run against the RSNA Knee Abnormality Detection competition data (7 August 2026) — deliberately
hostile input: 4,407 studies, twelve simultaneous targets, free-text radiology reports in five
languages, and the actual signal in 570 GB of DICOM images. The tool cannot enter this
competition, and the useful question was never whether it could. It was **what the tool claims
when it is out of its depth.**

It claimed too much. Three defects, all now fixed and pinned by tests:

- **The readiness verdict didn't bind.** `assess_readiness` returned `blocked` — 4,349 of 4,407
  rows had no label — and the training route, which never asked, fitted on the 58 that remained
  and reported the result as **useful**. The app's central claim is that it will say no; it said
  no and then proceeded anyway.
- **"Enough rows" counted rows it could not learn from.** It reported *"4,407 rows, comfortably
  above the 50 needed"* directly above a check reporting 4,349 rows with no answer. Both read
  the same frame; only one described what would be trained on.
- **The noise floor was capped at 0.75.** `min(0.75, ...)` sounds cautious and is the exact
  opposite: the formula raises the bar precisely when evidence is thinnest, so a ceiling only
  ever fires on the smallest test sets and *lowers* it there. On twelve test rows it pulled the
  threshold from 0.79 down to 0.75, and AUC 0.89 sailed through as a real finding.

Two data-quality findings in the competition data itself, both inside five minutes:
`Fluid_Sensitive` and `Fat_Suppression` are documented as independent sequence properties and
are **byte-identical across all 24,371 series**; and a de-identification find-and-replace turned
`1.` into `intact` in 87 reports, so *"a 1.3 cm bone cyst"* reads *"a intact3 cm bone cyst"*.

What the tool genuinely cannot do here, stated plainly: twelve targets at once (one target per
session), images, and free text as a *feature* — the `Report` column was dropped as an
"identifier" because nearly every value is distinct, which reaches the right action for the
wrong reason and discards the most informative column in the file.

## Deliberately not built

- **The `conversational_loop` graph node** stays an unwired stub. Chat is served by
  `POST /api/chat` instead: the graph has already completed by the time you're chatting, and
  re-entering it per message would re-run profiling for nothing.
- **ChromaDB and Great Expectations.** Both appear in the original tech-stack table, but no
  agent spec or workflow node ever gives them a job.

---

## What's next

**The work is tracked as GitHub issues.** The roadmap issue
([#1](https://github.com/harshiit-rana/dataclean-ai/issues/1)) lists 117 scoped issues across
seven workstreams and five milestones -- Evaluation Integrity, Statistical Reliability, AutoML
Improvements, Reproducibility, Production Readiness -- with priorities, sizes, dependencies and
a recommended order.

**Milestone 1, Evaluation Integrity, is built** (23 issues): the evaluation protocol is decided in
[`docs/decisions/0001-evaluation-protocol.md`](docs/decisions/0001-evaluation-protocol.md) — what
counts as a look at the test set, how looks are counted and what happens as they accumulate
(warn, never block), and where an honest fresh holdout comes from — and implemented: dataset
versions, the look ledger, the reuse rules and the labels that show them, a sealed final holdout,
new data as a fresh evaluation, copies and timestamps kept on one side of every split, leakage
checks and diagnostics off the test rows, whole-file fixes no longer proposed for a model input
(with training-only clipping instead), a property suite that flips every held-out answer, and a
whole-session HTTP test. Statistical Reliability — confidence intervals first — is next.

The direction below is unchanged; the issues are its concrete form. The end state is an
*honest* AutoML — not a button that hands back a leaderboard score, but a system that takes a
raw file to a working model, and an honest answer about whether it works, while showing its
working at every step. Roughly in order of how much they'd be worth:

- **More recipes, same contract.** Interaction and ratio features; target encoding done
  leakage-safely inside the fold; automatic feature *selection* judged the way every other
  round is. Each one still has to be a declared recipe the export can rebuild.
- **Ensembling and stacking**, so "best of six" becomes "best combination of six" — with the
  same rule that a combination has to beat the incumbent by more than the folds disagree.
- **Calibration for more than two answers, and inside the improvement search.** Calibration is
  an action now (below), but only for a two-class target, and the search itself still ranks
  uncalibrated models.
- **Free text as a real feature.** Length is a proxy. Embeddings would make the RSNA `Report`
  column usable rather than merely correctly described.
- **Several targets in one session.** The RSNA data had twelve; the app does one.
- **Time-series awareness** beyond a date-ordered split: lags, rolling windows, and the
  honesty about horizon that goes with them.
- **A job queue**, so a long search survives a restart and the retention sweep can leave the
  app process. This is also what unblocks multi-worker deployment.
- **Accounts and a session history**, which today is a deliberate scope choice rather than an
  oversight.
- **Deployment artifacts beyond a joblib**: a scoring endpoint, an image, a batch script —
  and eventually the whole pipeline as a scheduled, re-runnable object you can point at next
  month's file.

The constraint that governs all of it: **every automated step has to be explainable,
approvable and reproducible outside the app.** A recipe the export can't rebuild is a model
trapped here; the only score worth printing is one nothing was selected against; and any
capability that would let the tool sound more confident than its evidence supports doesn't get
built.

---

## Known limitations

- **Column importance favours columns with many categories.** Coefficients are summed across a
  column's one-hot categories, so twelve small noise coefficients add up to something that
  looks important — that is how `last_name` ranked second on the churn sample. Permutation
  importance (shown when the deeper diagnostics run) does not have this bias and is the one to
  trust when the two disagree.

- **The improvement loop is slow by nature.** It fits a model many times over. Sized to run
  behind a button — twelve draws per search, families more than 0.05 behind the leader skipped —
  but it's still tens of seconds on a few thousand rows, and the derived-inputs round adds a
  further cross-validated fit on top. It now streams, so the wait is visible, but it isn't
  shorter.
- **A search can't be cancelled.** Closing the page stops listening, not searching: the worker
  thread runs to the end and saves its result. That is deliberate — it is what keeps a closed
  tab from losing a minute of fitting — but a second search started meanwhile runs alongside
  it, and whichever finishes last is the one saved.
- **A derived input is only as stable as its source column.** A date that starts arriving in
  a different format, or a text field that stops being filled in, changes the inputs a model
  was fitted on. When a session is applied to a new file the drift check compares the derived
  inputs along with the rest, so a shift shows up there -- but only when someone runs it.
- **No account system.** Sessions are ephemeral and access tokens are per-session. This is a
  deliberate scope choice, not an oversight — but it means there's no "my past sessions" view.
- **Single-process.** Fine for the intended use; a multi-worker deployment would need the
  retention sweep moved out of the app process.
- **A grouped split can't also be stratified.** Keeping an entity whole and balancing the
  classes across folds are competing demands, and splitting the entity is the one that
  inflates the score — so grouping wins. On a lopsided target with few entities this can leave
  a fold thin on the rare class; the fold spread shown next to the score is where that shows
  up.
