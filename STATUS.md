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

**A grouped split can be on more than one column.** Some entities are named by several columns
together (a store within a region, a patient at a site), and grouping on either alone puts the same
entity on both sides. The plan form now builds a key a column at a time (the entity-shaped columns
first, but any column can be added, up to five), the request carries `group_columns`, and the rows with
the same combination of values stay on one side of every split and every fold. Every column in the key
is withheld from the inputs, a blank in any of them is refused with the column named, the answer
itself cannot be part of the key, and a single-column request behaves exactly as it always did (a
test checks identical metrics). The key is a column named after its parts ("store + region") so the
result, the card and the error messages read naturally, `split.group_columns` lists the parts, a sealed
holdout is re-split from them when it is opened, and the exported script copies the app's key function
so it builds exactly the same groups. A composite key is a departure from a single-column
recommendation, and is recorded as one.

**A date split can leave a gap.** When the answer is defined over a future window ("cancelled within
30 days"), a row just before the cut has an answer partly decided by what happens in the test period,
so a model trained on it has been shown something about that period. The plan offers an optional gap in
days for a date split: training stops at least that long before the first test row, at the train/test
cut and inside every forward-in-time fold (model choice, tuning, the improvement loop, the cut-off,
calibration, the cross-validated spread, the learning curve), the rows inside it are dropped and
counted on the result, the card says how many, `train_model.py` reproduces it (and the same gap in a
calibrated model's folds), and it is part of the test-set ledger's key only when there is one. A gap
longer than the data leaves nothing to train on and is refused with the numbers. With a sealed holdout
the gap sits between training and development, which is what keeps the sealed rows' later period safe.

**The way to split is recommended, with the reason, and can be overruled.** The form used to default
to a random split even when the app had already found a column repeating across rows or a clear time
order, and anyone who does not know what splitting means took the default and got a score the model
could not repeat. The plan now carries a recommendation in three rules, in order. A column that looks
like one entity repeating (the readiness panel's own test, not re-tuned) gets a grouped split on the
most repeated one, with the evidence ("'customer_id' repeats in 1,180 of 1,218 rows ... a random split
would put the same customer_id on both sides"); a name that does not look like an identifier gets a
medium confidence and a caution that a category such as a region should not be grouped on. Otherwise a
date column that is not a date of birth or a deadline, together with either a goal stated in terms of
the future ("forecast", "next month") or an answer that drifts between the earliest and latest third of
the rows (a chi-square or Welch test, with an effect-size floor, read only from the training rows of
the default split), gets a split by date. Otherwise a random split, stratified for a classification,
because "no repeated entities or time order found". The form pre-selects it and shows the reasons and
the confidence; choosing differently is allowed, said on the form, and recorded on the run
(`split.overridden`, and `split_overridden` in the history), judged on the server from the split the
run actually used.

**How much to hold back is chosen from the file, not fixed at a fifth.** A fifth of 150 labelled rows
is 30 test rows, exactly the least a verdict can be given on, and at a 10% positive rate three
positives, so "not enough test data" was the answer by construction; on 200,000 rows a fifth is forty
thousand rows taken from training for nothing. The plan now recommends a fraction between 10% and 30%
from the number of labelled rows and, for two answers, how many carry the rarer one: enough that the
test rows meet the verdict's own bars (30 rows, and 10 of the rarer answer) with a little headroom,
less for larger files, with the reason in a sentence ("Raised to 30%: 20% would leave about 8 of the
rarer answer to test on, and at least 10 are needed..."). The form pre-fills it and shows what was
chosen if the user overrides it; the fraction is recorded in the split's definition, so it is part of
the test-set ledger's key and the run history, and `train_model.py` cuts the same rows. When no
fraction works (too few labelled rows to judge on and train on at once) the plan says so in an amber
box and the readiness panel carries a warning before anything is trained. The recommendation is a
function of the counts only, never of which rows end up held back. A sealed holdout keeps its fixed
70 / 15 / 15 parts and ignores it.

**Training.** Six models across classification and regression. All preprocessing lives *inside*
the sklearn `Pipeline`, so `fit` sees only the training fold — there's a test asserting the
fitted imputer's median came from training rows alone.

**An honest verdict.** A model that learned nothing still reports 70% accuracy on a 70/30
target. So results are judged against what luck alone could produce on a test set of that size,
and against always guessing the same answer. On data with no signal, 23 of 25 runs are
correctly flagged and none are called useful.

A target with more than two answers is judged on a chance-adjusted AUC too. It used to compare accuracy
with the commonest answer plus a binomial margin, which ignores how well the model ranks every answer.
Results now report the macro one-vs-rest AUC (`roc_auc_ovr_macro`, left out when an answer is missing
from the test rows or the model gives no probabilities), and the verdict compares it with what luck
reaches on this many test rows, read off the data by shuffling the test answers (no refits; the
probabilities stay put, so any skill is destroyed and the class counts are kept) and taking the tail the
multiple-candidates correction asks for. The accuracy against the baseline stays as a second line, and
a result without the AUC falls back to the accuracy rule and says why. Measured: on four-answer noise with
48 test rows, 23 of 25 runs with two candidates are called guessing and none useful (the old accuracy rule
got 24 of 25 with one, so noise is no better and no worse); on a weak but real signal the old rule called
20 of 20 files "no better than guessing" and the new one finds all of them (marginal or useful). Two
answers are untouched.

A regression is judged the same way. It used to be fixed R² cut-offs (under 0.05 is guessing,
under 0.3 marginal) for 35 test rows and 35,000 alike, so a lucky 0.35 on a few dozen rows was
"useful". The verdict now stands on the lower end of the 95% R² interval from the bootstrap over
the test rows: the interval includes 0 means no better than guessing, a lower bound under 0.3
means marginal, and only a range that clears 0.3 throughout is useful. The headline prints the
range, so the same R² reads differently on 40 rows and on 4,000. On regression targets with no
signal, 25 of 25 runs with 40 test rows are called guessing, and a clearly learnable problem is
still useful. A result with no interval (an older one) keeps the fixed cut-offs.

The "no model at all" row used to be one cell wide: accuracy for always answering the commonest
answer, and a dash under precision, recall, F1 and AUC. On a lopsided target that guesser's F1 is
0, which flatters every model, since anything beats zero. A classification result now has two
baseline rows, each with a value under every column the models have: the common-answer guesser,
scored through scikit-learn's `DummyClassifier("prior")` and `evaluate` exactly as a model is (it is
the figure the verdict compares accuracy with, unchanged), and a guesser that answers at random in
proportion to how common each answer is, whose expected scores are computed from the training and
test shares rather than sampled, so the row is the same every time and gives precision, recall and F1
a real floor. Both are fitted on the training rows; the test rows are only what they are scored on.
The page, the sealed final table and the card show both.

A regression gets the same treatment: the training average and the training median as two
baselines, each scored on R², MAE and RMSE. On a skewed target (spend, price) a few huge values drag
the average away from where most rows sit, so the median is the stronger naive guess for the typical
miss and beating only the average flatters a model. Every model also reports `mae_vs_baseline`, its
MAE as a fraction of the median guess's MAE, which gives an absolute error the scale it lacks: 0.79
says its typical miss is 79% the size of just guessing the middle. It is in the comparison table, the
sealed table and the card's glossary, and absent where the target has no spread around its median.

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

The delivered precision and recall carry 95% ranges from a bootstrap over the test rows at the
chosen line (class-stratified, or whole groups for a grouped split; a resample in which a rate is
undefined is dropped rather than counted as 0). The card's "over-promised" message used to appear
for any gap over ten points, which on forty positives is inside what other rows would give; it now
appears only when the promised recall is outside the delivered range, says so when the gap is
inside it, and says nothing for a result with no range. The model card's table prints the same
ranges.

**Thirteen diagnostics**, as interactive charts with a table view behind every one: ROC,
precision-recall, confusion matrix, per-class breakdown, calibration, threshold trade-off,
class balance, feature importance, permutation importance, cross-validation spread, learning
curve, correlation matrix — and for regression, predicted-vs-actual, residuals, residual
distribution, error by range.

**The learning curve is read against its own noise.** The card called a train/unseen gap over 0.15
"memorizing" and a rise over 0.05 "still climbing" whatever the folds' disagreement, so on small data
noise could produce either statement. Each point now carries the folds' spread (drawn as a band on the
chart), a statement is made only when the difference is past the old size *and* more than twice that
spread, "flattened out" is said only when the spread is small enough to rule out a real rise, and
otherwise the curve is called inconclusive -- with a note that three folds give only a rough spread. The
reading is worked out once on the server, so the page and the card say the same thing.

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

**Checking the choice against other seeds.** Random forests and boosting are fitted with seed 42
everywhere, and a result that holds only for that seed is not a result. When one of them is chosen, it
is cross-validated again on the same training folds with four more seeds and the spread of its fold
score is set against its lead over the runner-up: when the spread is at least the lead, the page and the
card say which of the two comes out ahead depends on the seed, not on the data; otherwise a quiet line
says it was checked. Logistic and ridge regression are marked as having no randomness (they take a seed
but their fits are deterministic) and report "not applicable"; above 20,000 training rows the check is
skipped and says why. It reads the training folds only, and a case in the integrity suite flips every
held-out answer to show it does not move. The price is four more cross-validations of the chosen model.

**What the chosen model adds over its best single input.** A model that scores well may be doing little
more than one strong column. The simplest model for the task (logistic or ridge regression) is
cross-validated on each input column alone, on the training rows and the same folds the candidates were
compared on, and the best is stored as `baseline.best_single_column` with its column, fold score and a
paired fold-by-fold difference against the chosen model. The comparison table shows it as a row and the
page and card say whether the chosen model's lead is more than noise or whether most of what it does may
come from that one column. It is skipped, with the reason shown, for a single input column, more than 40
input columns or more than 50,000 training rows. The simplest model is linear, so a column whose effect is
a U-shape under-scores here; the sentence says "on its own", not "is all the model uses". A case in the
integrity suite flips every held-out answer to show it does not move. The price is one cross-validation per
input column of a fast model.

**Why this model?** The model page and the model card now answer, in order: how the model was chosen (the
metric, the folds, repeats, and the one-standard-error rule when it applied), why each alternative was not
(its fold score against the chosen model's, whether it was within one standard error, and for the closest
rival whether the folds can tell the two apart), what constrained the choice (the split and whether the folds
respected it, copies kept together, a sealed holdout, withheld columns, a cut-off, calibration, and whether
the app's recommended split was overridden) and what the improvement search tried and rejected. It is
`why_this_model.explain(result)`, a pure function of the stored result, attached whenever a result is served
and called again by the card, so the two cannot differ. No sentence comes from a language model, and each
carries the result fields it rests on with their values. A choice that is not clearly better than its rival
(or a model picked by default order, or on the test set in an old result) is a warning and leads the panel.
Results saved before it existed get one computed on read.

**How these numbers were obtained (model card).** Every card now has a section before the results saying how
the rows were divided (and whether that is the split the app recommended), how the model was chosen, how often
the held-out rows had been used when it was scored, and, where they apply, the sealed final holdout and whether
it was opened, checks on new data, calibration, the cut-off, and overlaps such as test rows that are copies of
training rows or fixes worked out over every row. It ends with what the numbers support (with the plausible
range of the main score) and what they do not (a later period under a random or grouped split, data collected
differently, an independent estimate once the holdout has been reused). It is assembled in code from the result
and its evaluation record, so no figure in it passes through the language model, and the card route now reads
the result with its later evidence so the new-data and final-holdout lines are present. A card is written once
and kept, so those lines are as of the day it was first requested.

**Before you rely on this, under the verdict.** The cautions the model card computes (a reused or exhausted
holdout, fixes worked out over every row, test rows that are copies of training rows, a small test set, thin
classes, a random split, withheld or derived columns, an unreachable cut-off, a searched model) are now
`model_card.caution_items`, each with an id and a severity, ranked most severe first. The model page shows the
top three in a box right under the verdict headline with a count of the rest and "Show all", in the card's own
words, and the card prints the same list in the same order. The order is a judgement, written down beside
`SEVERITY_ORDER`: *high* means the headline number is wrong in a known direction (copies in the test set, an
exhausted holdout), *medium* that the scores are thin or the model is not what was asked for, *low* is context
every reader should have. No caution in the box comes from the language model. The two notes the verdict card
used to print itself (copies, whole-file statistics) are now shown only for a result served without the list.

**One way of saying "cannot judge".** The verdict on a small test set, the drift check on a small file, the
check against a new file's real answers, and a cut-off that could not be chosen each used to decline in their
own words and styling, and the cut-off simply vanished. They now share a block (`not_judged.not_judged`:
`{status: "not_judged", what, have, need, unit, sentence}`) and one sentence pattern, "Not enough {what} to
judge: {have} rows, at least {need} needed." (or "Cannot judge: {reason}." when the obstacle is not a count),
shown by one component in the same sky treatment, which is deliberately not a failure colour. The cut-off
card now appears when a cut-off was asked for and could not be chosen, giving the reason (no probabilities,
too few of the rarer answer or too few groups to make two folds, too few rows to fold by date), as
`operating_point_not_judged` on the result; `operating_point` itself stays empty so nothing that reads the
threshold changes. The existing headline still follows the sentence as the explanation.

**A refused or failed training offers a next step.** The model page used to show the server's message in a box
and nothing else. The errors a page can act on now carry a stable `code` (in the `{error, detail}` body, in the
streamed error event, and on `ModelingError`): `data_blocked` links to the failing readiness checks,
`too_few_labelled_rows` explains and links to the readiness checks, `no_usable_columns` lists why each column was
left out and opens the column chooser, `target_missing` / `target_unusable` send the user back to choose a
column to predict, `split_unusable` (a date or grouping column that cannot be used as asked) opens the split
chooser, and `no_model_trained` opens the column and model chooser. A code the page does not know, or none, shows
the same message box as before, and an error without a code has exactly the body it always had. The codes are
listed in the `TrainingFailedError` docstring and must not change meaning.

**The audit report mentions the model.** A session that trained a model now gets a short "Model built from this
data" section in the audit report, before the changes: the model chosen, rows trained on and held back, its main
score with the plausible range, the verdict, and the same lines the model card prints under "How these numbers
were obtained" (how the rows were divided, how the model was chosen, how often the held-out rows were used, the
sealed holdout, checks on new data), with a pointer to `model_card.md` in the export bundle. It is read from the
stored result and the evaluation record, so no figure passes through the language model, and it is absent when no
model was trained. Anything that changes the model's evaluation (training again, opening the sealed holdout, a
counted check on new data) marks a written report stale, so the next export rebuilds it; a report that was never
written is left alone.

**Readiness checks say where to go.** The checks that find something now carry `actions`, places to navigate
to and never something applied: a column that gives away the answer offers "Leave '<column>' out of the model"
(up to three), a column that repeats across rows offers "Split by '<column>'", and a missing or unnamed target
offers "Choose a column to predict". The panel shows them as buttons on the failing check only. The first two
open the model page with the choice already set up in the form (a column left out, a grouped split on that
column) through `exclude`, `split` and `group` in the address; the user still reviews the form and presses
train, and a column the plan does not offer is ignored.

**Repeated folds when a small file is compared.** With a few hundred training rows one 5-fold
partition can rank two close candidates one way and another partition the other way round. Below
2,000 training rows, choosing the model, tuning it and the fold spread shown beside the score now
repeat the folds three times with different shuffles (15 fits a candidate instead of 5). Measured on
two logistic regressions that each see one of two equally informative columns -- so only the partition
decides the winner -- the share of partitions agreeing on the winner rose from 0.905 to 0.96 over ten
files, and on the close calls from about 0.76 to 0.90; with candidates further apart every partition
agreed either way. The one-standard-error rule still takes its error over the folds of one partition,
and the paired comparison uses the correction for repeated folds, so repeats are not counted as more
independent evidence than they are. Grouped and date splits, the cut-off and calibration are unchanged:
the last two need every row validated exactly once. The price is three times the fits on small files.

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

**A time limit on the search (#47).** "Try to do better" takes an optional "spend at most N minutes". The
deadline (a monotonic clock, counted from when the search starts) is checked before each round, before each
runner-up family in the third round, and between the settings inside a search, so the search returns within the
limit plus the time one setting takes to score. The settings search is now an explicit loop over the same draws
`RandomizedSearchCV` used (a test holds it to the same winner, score, spread and fit count), so a search cut short
reports the best of the settings it reached, a prefix of the sequence a full search would have drawn. Rounds that
did not run are listed in the trail marked "Not run: time limit", and the run's stop reason is `time_limit`. A goal
that was met still wins over the clock. The final refit and the single scoring on the held-out rows come after the
limit, so the response takes a little longer than the limit; the limit bounds the search, not the report. The
integrity suite checks that a limit which bites stops at the same place whatever the held-out answers are.

**A seam for search strategies (#48).** The settings search sits behind a small interface (`tuning.SearchStrategy`:
`run(build, space, X, y, *, splitter, metric, budget, deadline, groups)`), with random search, the loop that was
already there, as `RandomSearch`; `tuning.search` settles what every strategy needs the same way and hands the rest
over. Nothing changed for the user: a test holds it to the previous output. The strategy's name travels with the
result (`SearchResult.strategy`), into each settings round of the trail, the trail's footer and the run history, so a
second strategy can be told apart from the first. It was kept to what random search and successive halving both
need, with no hooks for strategies that do not exist.

**Successive halving (#49).** A second way to search settings, `search_strategy: "halving"` on the improve request
(random stays the default; choosing automatically is #50). It scores many settings on a few folds, the best third on
more, and the best third of those on all of them, and spends about a third of random search's fits for the same
budget while looking at twice as many settings. The resource is cross-validation *folds*, not training rows (the
issue suggested rows): the folds are cut once by the same splitter random search uses, so grouped and forward-in-time
folds hold at every rung without cutting the data a second time, and the survivor ends with a score on every fold,
which is the only score the improvement loop compares with the incumbent. The trail lists each rung (settings, folds,
best score, how many went on). The trade-off, stated on the class: early rungs rank on a few noisy folds, so a setting
that only shows its worth across all of them can be dropped before it gets there. With fewer than six settings or three
folds there is nothing to halve over, and it searches at random and says so. It checks the deadline between settings;
if the time limit hits, the leader is finished on the remaining folds so what is reported is still a full score.

**The search strategy is chosen for you (#50).** "Try to do better" no longer needs the user to know which search
suits their file. `tuning.choose_strategy(model, training_rows, budget)` is a pure rule: halving when the model has at
least 12 possible settings, the budget is at least 12 and there are at least 5,000 training rows (below those, fits
are cheap and scoring fewer folds saves little and only adds noise); random search otherwise. It is decided for each
settings search, so a small linear model and a large forest in the same run can use different ones, and the trail says
which was used and why in a sentence for each search. `search_strategy` on the improve request overrides it
("random" or "halving"); an override is respected, recorded as the user's, and a halving search that finds nothing
to halve says it fell back. The run history shows the strategies used. The three thresholds are a first guess to be
revisited with benchmark data on real files; moving them changes which strategy a run uses, never what either does.

**A search winner is checked on fresh folds (#51).** The settings search returns the best of several noisy fold
scores, so that score is biased upward and a setting that is no better can clear the gain rule by being the luckiest.
A winner that clears the rule is now re-scored, together with the incumbent it would replace, on folds the search did
not select on (a reshuffle with another seed; under a date split or ungrouped-regression groups, which cannot be
reshuffled, one more fold, a weaker guarantee that the code says so), and has to clear the gain rule there too. Both are
re-scored so the comparison is like for like, which costs two cross-validations a round whatever the budget. The trail
shows the score on the folds it was picked on beside the fresh-fold score and what it would replace; a winner that did
not hold says so and the incumbent stays; the check is skipped, and the winner refused, if the time limit has passed.
Measured on pure noise (`backend/scripts/measure_winners_curse.py`), the settings round accepting a setting: logistic
regression 0 of 25 seeds before and after (its existing gain rule already holds); gradient boosting on 80 rows 9 of 25
before and 8 of 25 after. The second does not reach the "at most 1 of 25" the issue hoped for, and the reason is not luck:
every accepted setting scores exactly 0.500 AUC on fresh folds (a model that predicts the same for everyone) while the
default, which overfits the noise, scores 0.29 to 0.49, so replacing it is a genuine improvement to chance. The check
refused the one case that was luck. Stopping a loop from "improving" a worse-than-chance default to chance would need a
rule about chance, not about fold luck; that is a separate decision and is not made here.

**A processing limit on one request (#62).** Deep diagnostics were already skipped above 20,000 rows, but plain
training, model choice and the improvement loop had no limit on rows times candidates times folds, so a large upload
could keep the single API process busy for as long as it took. `modeling.estimate_cost` puts a number on a request in
"row-fits", with each family weighted by how dear a row is to fit (linear 1, boosting 25, forest 40), and
`Settings.training_cost_limit` (`TRAINING_COST_LIMIT`, default 30 million, 0 for none) is the most a request may be
estimated to cost; a file of about a hundred thousand rows passes unreduced. Over it, in order: model choice is made on
a sample of the training rows (whole groups for a grouped split, answers in proportion for a classification; never
for a date split, whose folds are cut by timestamps), then the dearest family is skipped, repeatedly, then the request
is refused with `too_large` and how many rows would fit. The final fit is still on every training row, the test set
takes no part, and whatever was reduced is stated in the result's `resource_limits`, in the model card and in the
cautions box. The improvement loop never samples, so it skips families and trims the search budget instead. Within the
limit nothing changes. A real 200,000-row run lives in `tests/slow` (`pytest -m slow`), left out of the default run: with all three
families it was estimated at 42 million, compared on 98,181 of the 160,000 training rows to come to 30 million, and took
46 seconds on a multi-core laptop. The weights err towards caution (a forest fits in parallel), and the limit is one
setting if a deployment wants it looser.

**A round that asks whether each input earns its place (#57).** Every usable column is an input, and a weak or noisy one
adds variance, most of all on a small file. After the other rounds, the loop ranks the inputs by *fold importance*: for each
validation fold the incumbent, fitted on that fold's training rows, is scored with one input's values shuffled, and the
drop averaged over the folds is that input's importance. It never touches the held-out test rows, nor any row the model was
fitted on, which is what makes an importance of zero or below mean something. The inputs that did not matter (at most half
of them) are removed, the model is re-scored, and the smaller set is kept only if it clears the same gain rule as every
other round; if not, the lowest-ranked third of them is tried, because correlated inputs share their importance. The trail
lists each removed input with its importance and spread; the removed inputs are withheld from the rebuilt model and so from
the exported script, shown on the model card under "Columns not used" with their own reason (not as the user's exclusions),
and kept apart from the user's exclusions in the result. With fewer than four inputs there is nothing to rank. The round is
skipped, and says so, when the time limit has passed.

**Target encoding for columns with too many values (#58).** A postcode or product code has too many values to one-hot
encode and was either dropped or reduced to its commonest values. A column that repeats (about two rows a value or more)
and is not already offered as "keep the common values" can now be replaced by the average answer for each value, offered
in the derived-inputs list with its reason and ticked like any other proposal. It is scikit-learn's `TargetEncoder` inside
the pipeline, which encodes each training row from the other internal folds so a row's own answer is never in its own input,
all within the training fold; validation and test rows are encoded with the full training fit, and a value never seen in
training is encoded as the overall average. A column of random ids gains nothing from it (a test), where an encoding worked
out over every row scores far above chance (so the test can tell). Its internal folds do not know about groups, so it is
declined for a grouped split and for identical rows kept together, and the result and the cautions say so. The improvement
loop's derived-inputs round tries it along with the other recipes under the gain rule, and the exported script builds the
same encoder with the same folds (a test runs the script and compares its accuracy).

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
  them now use forward-in-time folds. Those folds used to cut at row positions, so rows sharing a
  timestamp (daily data, records loaded in batches) could straddle a fold boundary exactly as they
  once did at the train/test cut; they now use a splitter that is handed the timestamps and moves
  every boundary to the nearest change of timestamp, so no timestamp is on both sides of any fold. With
  distinct timestamps it reproduces `TimeSeriesSplit` exactly. The exported script carries the same
  splitter (copied by source) for a calibrated date-split model.
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
  down. Those grouped folds are stratified as far as the grouping allows (see the known limit
  below). Near-duplicates (one cell different) are not addressed.
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
  the model never saw. Below 30 rows it declines to judge, and each column is judged on its own
  filled-in values too: a column that is 95% blank in a 500-row file has 25 values to go on, so it is
  reported as not judged, with the counts, and left out of the verdict (the headline names it). How
  often a column is blank is compared as well -- a column that stops being filled in has drifted as
  surely as one whose values moved -- and a derived input such as a text length, which is never blank,
  is judged on its source column's blankness. With thirty inputs a few cross the conventional PSI 0.1
  by chance on a moderately sized file, so a fresh sample from the very same population came back
  "shifted" in 15 of 20 tries; each column now also has to clear the PSI that chance reaches for
  samples of its sizes (simulated from the training distribution, at a 5% error rate spread across all
  the columns compared), and the same twenty samples now come back "similar" in 19, while a real shift
  in one column is still caught. The convention stays as a floor on how much a column has to move, the
  page shows both numbers, and the headline says how many inputs moved more than chance explains. A model scores whatever it is given and never notices
  when it is extrapolating; this is the app saying so for it.
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

**Is the multiple-candidate correction still needed? (decision pending, #35).** The verdict raises the AUC bar
with the number of candidates trained, which compensated for reporting the best of several test-set draws. A
simulation (`backend/scripts/simulate_verdict_correction.py`) shows that choosing on training folds leaves the test
AUC close to a single draw, but that relaxing the correction would find only about 2 more of 48 weak-signal datasets
while raising the pure-noise false-positive rate from about 3% to about 11% in those runs. The recommendation, in
[`docs/decisions/0002-multiple-candidate-correction.md`](docs/decisions/0002-multiple-candidate-correction.md), is to
keep it; nothing has been changed, and the decision is the owner's. The note also records that on 30 test rows a lucky
noise run can be called "useful" because the noise bar sits above the 0.70 cut-off, which is a separate question.

**What a model family must meet to join the registry (decision pending, #52).** A recommendation in
[`docs/decisions/0003-model-registry-criteria.md`](docs/decisions/0003-model-registry-criteria.md): eight criteria (it
fits the pipeline, reproduces from the exported script, is reproducible, respects the folds, is affordable, has a small
curated search space, can be explained, and earns its place on evidence after the verdict's candidate correction is
counted) and the current decisions under them: Extra Trees accepted (#53), gradient-boosting libraries deferred because
`HistGradientBoosting*` covers the same family without a new dependency, and SVM, k-NN, Naive Bayes and neural networks
not planned. It is the owner's decision; #53 and #56 wait for it.

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
- **A grouped split is stratified only as far as the grouping allows.** Keeping an entity whole
  and balancing the classes across folds are competing demands, and splitting the entity is the
  one that inflates the score — so grouping wins. Classification folds inside a grouped split (and
  the cross-validated spread and the learning curve) now use `StratifiedGroupKFold`, which balances
  the answers within that constraint: on forty customers and a 10% positive rate the positive rate
  varies less across folds than it did with the plain `GroupKFold`, and no fold lacks the rarer
  answer. But when the rare answer is concentrated in a few large groups no assignment can balance
  it, and a fold can still be thin; the fold spread shown next to the score is where that shows
  up. Regression still uses `GroupKFold`.
