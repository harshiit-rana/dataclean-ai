# DataClean AI — what this is, what's built, what's next

Last updated: 6 August 2026 · 32 commits · 350 backend tests, 47 frontend tests

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
- **Repeated entities.** A column whose values recur across rows means a random split can put
  the same customer in both training and test, which inflates every score.

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

## Deliberately not built

- **The `conversational_loop` graph node** stays an unwired stub. Chat is served by
  `POST /api/chat` instead: the graph has already completed by the time you're chatting, and
  re-entering it per message would re-run profiling for nothing.
- **ChromaDB and Great Expectations.** Both appear in the original tech-stack table, but no
  agent spec or workflow node ever gives them a job.

---

## What's next

### 1. Business rules — the validation stage's first real tenant

The check framework now exists (see *Decision provenance* above), and business rules are what
it was built to hold. The tool detects statistical problems but not semantic ones: `age = -5`
isn't an outlier, it's impossible, and IQR bounds will miss it entirely while flagging
legitimate premium salaries. Planned: a small rule language (`age >= 0`, `discount <= 100`,
`order_date <= delivery_date`, `price * qty == total`), rules proposed by the AI but approved
like every other suggestion, and cross-column arithmetic with a tolerance so floating point
doesn't fire constantly.

### 2. Feature engineering — the big one

Tuning is the small lever, and the live run proved it: on the churn sample the settings search
found nothing worth keeping, which means **the ceiling is the columns, not the settings.**

The app currently leaves real signal on the table. In that same run `signup_date` was dropped
entirely — 870 distinct values, "too many to turn into model inputs" — when *tenure derived
from signup date* is obviously predictive of churn. Planned as a fourth round in the
improvement loop, scored the same honest way:

- **Date parts** — days since, year, month, day-of-week. Turns a dropped date column into four
  usable ones.
- **Frequency encoding** for high-cardinality categoricals that are currently discarded.
- **Missingness indicators.** A blank `monthly_spend` means contractor, and that fact is itself
  a signal — we know it carries signal, because the leakage detector already looks for exactly
  this.
- **Text length** for free-text columns.

Same constraint as the tuned settings: each derived feature must be a declared recipe so
`train_model.py` can reproduce it. A model the export can't rebuild is a model trapped in the app.

### 3. Threshold selection

The threshold sweep is already *shown* — "the default cut-off is 0.50, but nothing makes it the
right one" — but you can't act on it. Letting you state the trade-off ("a missed churn costs
more than a false alarm") and having the model adopt that operating point is a large, cheap win:
recall can often be doubled without retraining anything.

### 4. Two text-quality defects found live

- **The LLM hallucinated an example from a different dataset.** A card about a `plan` column
  (values: basic/standard/premium) read *"For example, 'USA' and 'usa' are considered different
  values."* That's the few-shot example in the prompt bleeding into the output. Seen twice
  across two runs, so it isn't a one-off.
- **Outlier cards quote the IQR fences as if they were the data.** *"These extreme values are
  between -19.32 and 63.15"* for monthly spend; *"-4.50 to 7.50"* for ticket counts. Negative
  bounds on strictly positive columns read as nonsense. This one is ours, not the model's.

Neither breaks anything, but both undermine trust in cards you're being asked to judge.

### 5. Streaming the improvement loop

It already reports progress round by round internally (`on_progress`), but the endpoint is a
single blocking POST. A search takes tens of seconds to minutes; watching it work would be
better than a paragraph of explanation next to a spinner.

---

## Known limitations

- **The improvement loop is slow by nature.** It fits a model many times over. Sized to run
  behind a button — twelve draws per search, families more than 0.05 behind the leader skipped —
  but it's still tens of seconds on a few thousand rows.
- **`signup_date`-style columns are dropped, not derived from.** See item 1 above.
- **No account system.** Sessions are ephemeral and access tokens are per-session. This is a
  deliberate scope choice, not an oversight — but it means there's no "my past sessions" view.
- **Single-process.** Fine for the intended use; a multi-worker deployment would need the
  retention sweep moved out of the app process.
