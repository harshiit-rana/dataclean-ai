# DataClean AI — what this is, what's built, what's next

Last updated: 30 September 2026 · 43 commits · 543 backend tests, 104 frontend tests

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

**Roughly 8,300 lines of backend, 6,000 of frontend, 3,800 of backend tests.**

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

**Profiling.** Six issue types detected: missing values, duplicates, outliers (IQR),
type mismatches, inconsistent formats, high cardinality. Ranked by relevance to your stated
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

### Cross-cutting

- **Session access tokens.** Knowing a session id isn't enough to read its data. Issued at
  upload, only the hash stored, required on every session-scoped request. Off in development.
- **Retention.** Files deleted 24 hours after last use, swept hourly in-process.
- **Light and dark**, following your OS until you use the toggle.
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

### 1. Streaming the improvement loop

It already reports progress round by round internally (`on_progress`), but the endpoint is a
single blocking POST. A search takes tens of seconds to minutes; watching it work would be
better than a paragraph of explanation next to a spinner. The fourth round made this more
pressing rather than less: the loop now has one more thing to try.

### Further out

Direction rather than a queue. The end state is an *honest* AutoML — not a button that hands
back a leaderboard score, but a system that walks from a raw file to a defensible model while
showing its working at every step. Roughly in order of how much they'd be worth:

- **More recipes, same contract.** Interaction and ratio features; target encoding done
  leakage-safely inside the fold; automatic feature *selection* judged the way every other
  round is. Each one still has to be a declared recipe the export can rebuild.
- **Ensembling and stacking**, so "best of six" becomes "best combination of six" — with the
  same rule that a combination has to beat the incumbent by more than the folds disagree.
- **Calibration as an action, not a chart.** The calibration curve is already shown; being
  able to *fix* a badly calibrated model is the same shape of win as moving the cut-off was.
- **Free text as a real feature.** Length is a proxy. Embeddings would make the RSNA `Report`
  column usable rather than merely correctly described.
- **Several targets in one session.** The RSNA data had twelve; the app does one.
- **Time-series awareness** beyond a date-ordered split: lags, rolling windows, and the
  honesty about horizon that goes with them.
- **Drift detection**, so a model can say *"the rows you are scoring no longer look like the
  rows I was trained on"* — the natural extension of a tool built to admit what it can't
  support.
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
  further cross-validated fit on top.
- **A derived input is only as stable as its source column.** A date that starts arriving in
  a different format, or a text field that stops being filled in, silently changes the inputs
  a model was fitted on. The model card says so; nothing detects it yet, which is what the
  drift item above is for.
- **No account system.** Sessions are ephemeral and access tokens are per-session. This is a
  deliberate scope choice, not an oversight — but it means there's no "my past sessions" view.
- **Single-process.** Fine for the intended use; a multi-worker deployment would need the
  retention sweep moved out of the app process.
- **A grouped split can't also be stratified.** Keeping an entity whole and balancing the
  classes across folds are competing demands, and splitting the entity is the one that
  inflates the score — so grouping wins. On a lopsided target with few entities this can leave
  a fold thin on the rare class; the fold spread shown next to the score is where that shows
  up.
