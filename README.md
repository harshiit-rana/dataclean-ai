# DataClean AI

An open-source, AI-powered CSV cleaning tool that keeps a human in the loop: upload a CSV,
tell it what you're trying to do, and it profiles your data, ranks issues by relevance to
your goal, and proposes fixes as suggestion cards -- Found / Proposed Action / Reasoning /
Tradeoff. Nothing changes until you approve it, card by card.

If you're cleaning data to train a model, name the column you want to predict and the tool
changes what it looks for -- see "Cleaning towards a model" below.

For the whole picture in one place -- what's built, what was deliberately left out, what's
planned next and what the known limitations are -- see **[STATUS.md](STATUS.md)**.

## Before you start: get an LLM API key

This project doesn't ship with a shared/default key for either provider it supports --
pick one, get your own key, and put it in `.env` (see "Getting started" below and the
"LLM provider" section for the tradeoff between them). Without a key, the Profiling and
Explanation agents cannot run.

## Getting started (Docker, recommended)

```bash
git clone <this-repo-url>
cd dataclean-ai
cp .env.example .env
# edit .env: set LLM_PROVIDER (anthropic or groq) and the matching API key

docker-compose up
```

- Backend: http://localhost:8000 (health check at `/health`)
- Frontend: http://localhost:5173

Try it with one of the messy datasets in `samples/` (`customers.csv`,
`sales_transactions.csv`, `employee_records.csv`) -- each has a mix of missing values,
duplicates, inconsistent formats, mixed types, and outliers baked in.

Two larger ones go further:

- `subscription_churn.csv` -- 1,200 rows with genuine signal in them, for the model path.
  Name `churned` as the column to predict. It also has 144 customers spread across those rows,
  so the readiness panel will point out that a random split scatters the same customer over
  both sides -- pick "Keep related rows together" on the model plan to see the score correct
  itself.
- `retail_orders.csv` -- carries the errors no statistic can find, for the business-rules
  path: negative ages, discounts over 100%, deliveries dated before their order, and line
  totals that don't match price x quantity. None of those are outliers, and the IQR check
  walks straight past all four.

## Getting started (without Docker)

**Backend:**
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate  # or source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
# Redis 8.0+ must be running locally (redis:8-alpine via Docker is the easiest way)
uvicorn app.main:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Architecture

- **Backend**: FastAPI + Server-Sent Events, orchestrated by a LangGraph `StateGraph`
  (`goal_intake -> profiling_agent -> explanation_agent -> human_approval [interrupt] ->
  cleaning_agent -> conversational_loop -> audit_agent`). The graph pauses at
  `human_approval` using LangGraph's native `interrupt()`, checkpointed to Redis so a
  session survives a server restart while it's waiting on you.
- **Agents**: any LangChain `BaseChatModel` -- provider picked at runtime by `LLM_PROVIDER`
  (`app/services/llm_client.py`), so nodes never call anything provider-specific. See "LLM
  provider" below.
- **Profiling**: deterministic pandas/IQR-based detectors
  (which is wrapped defensively and never allowed to crash a session) -- the LLM only adds
  semantic interpretation and goal-aware re-prioritization on top.
- **Frontend**: React + Vite + Tailwind, SSE consumed via `@microsoft/fetch-event-source`
  (native `EventSource` can't do POST, and starting the graph requires one).

## LLM provider

Set `LLM_PROVIDER` in `.env` to `anthropic` (default) or `groq`:

- **`anthropic`** (Claude, default `claude-sonnet-5`): the better-verified option for this
  app specifically. Two of the three LLM calls need reliable structured output -- the
  Profiling Agent needs a strict JSON array back (has a retry-once on parse failure, falls
  back to baseline scores rather than crashing), and the Explanation Agent needs an exact
  `FOUND:`/`PROPOSED_ACTION:`/`REASONING:`/`TRADEOFF:` marker format with **no retry today**,
  so a model that drifts from the format silently produces blank suggestion cards. Cheaper
  alternative: `claude-haiku-4-5` via `CLAUDE_MODEL`.
- **`groq`** (default `llama-3.3-70b-versatile`): far cheaper, generous free tier, fast
  inference. Llama 3.3 70B is a solid instruction-follower and should handle both structured
  calls fine, but it's less battle-tested here than Claude -- if you see blank suggestion
  cards, that's the marker-format gap above, not a bug elsewhere.

Only the API key matching your chosen provider is required (see `.env.example`).

## Project status

The full loop works end to end:

1. **Upload** a CSV, describe what you're using it for, and optionally name the column you
   want to predict
2. **Profiling** runs deterministic pandas/IQR detectors — plus leakage and target checks if
   you named one — then an LLM re-ranks findings against your stated goal
3. **Review** shows your actual data in a table — click any column to see its stats, its
   role, and what was flagged in it — alongside streamed suggestion cards
4. **Approve or reject** each fix individually; only approved ones execute. Reject one and
   you can say why — what you type is copied into the report word for word and is never sent
   to the model
5. **Clean** applies them, showing before/after numbers per fix, with per-fix **undo**
6. **Ask for more** in plain English ("drop rows where salary is under 30000")
7. **Build a model** from the cleaned data, if you named a target — see below
8. **Export** the cleaned CSV, a PDF audit report documenting every change (including the
   ones you declined, in your own words), the model card, a runnable training script and the
   fitted model

Your uploaded file is never modified. Cleaning writes a separate `cleaned.csv`, and there's
a test asserting the original is byte-identical after a run.

## Data retention

An ordinary session's files (the uploaded CSV, cleaned copies, models and audit reports) and its saved state in
Redis are deleted 24 hours after the session was last used (`UPLOAD_RETENTION_SECONDS`). A background sweep runs
hourly inside the backend process, with no cron job or sidecar to set up. Set the retention to `0` to keep files
indefinitely, which is handy while debugging one session and a bad idea anywhere real.

Redis persists to a named volume with `appendonly` enabled, so sessions survive a container
restart or a `docker compose down`.

## Projects

Work you want to come back to can be kept as a **project**: press "Keep this project" at the top of any page and
give it a name. A project is kept until you delete it, and from then on its key is always required, even in
development mode. The key is shown once, when you keep the project: save it, because it is the only way back in
from another browser and there is no recovery (there are no accounts). This browser remembers your projects and
lists them on the upload page; "Open a project with its key" brings one back elsewhere. "New key" replaces a key
that has been shared or exposed, and "Delete project" removes every file and the saved state for good.

### Where your data lives, and what that means

Everything is stored in one folder, `UPLOAD_DIR` (the `backend_uploads` volume under Docker Compose), one
sub-folder per session or project. The files are **not encrypted at rest**. The project key protects the app's
HTTP interface; it does not protect the disk. Anyone with access to the host, the Docker volume, or a backup of
either can read every project. Deleting a project unlinks its files; it does not securely erase them, and copies
in backups or snapshots remain. On a shared or hosted machine, put `UPLOAD_DIR` on an encrypted volume, restrict it
to the app's user, and treat backups as holding your users' data. See
[`docs/security/local-storage.md`](docs/security/local-storage.md) and
[`docs/decisions/0004-platform-architecture.md`](docs/decisions/0004-platform-architecture.md).

## Light and dark

The app starts in whichever mode your operating system is set to, and there's a toggle in the
top-right corner of every page. Once you use it, your choice is remembered and the OS stops
overruling it — so you can keep this one app light on a machine that's otherwise dark.

The charts follow the same signal the rest of the page does (the `dark` class on `<html>`),
never `prefers-color-scheme` directly. Both palettes were validated against the app's own
surfaces rather than assumed.

## Session access

Session ids are server-generated UUIDs, which makes them unguessable but not private —
anything that learns one (a log line, a shared screenshot, a proxy, browser history sync)
could then read that session's data. So `POST /api/upload` also issues a one-time access
token, the server stores only its SHA-256 hash, and every request scoped to that session must
carry the token in an `X-Session-Token` header. The web UI keeps it in `sessionStorage` and
attaches it for you, downloads included.

It is deliberately **not** an account system: there are no users here and sessions are
ephemeral. What it buys is that holding the id is no longer sufficient.

Tokens are **required by default**. They are off only when development is switched on
explicitly — `APP_ENV=development`, which the shipped `.env.example` and `docker-compose.yml`
set so `docker compose up` and a browser is still all it takes to try the app — or when
`REQUIRE_SESSION_TOKENS=false` is set. With nothing set, `APP_ENV` means `production`. The app
logs a prominent warning at every start while tokens are off.

**Deploying:** do not carry `APP_ENV=development` over to a hosted instance. Unset it (or set
`production`) and tokens are enforced, and unexpected errors stop echoing their details to the
browser. *Upgrading:* a self-hosted instance that relied on the old default (development when
unset) now requires tokens; the web UI sends them already, but a script calling the API must
send `X-Session-Token`. Sessions created before tokens were switched on have no stored hash and
keep working, so enabling it doesn't strand work in progress.

**Not implemented:** the `conversational_loop` graph node is an unwired stub — conversational
cleaning is served by `POST /api/chat` instead, since the graph has already completed by the
time you're chatting and re-entering it per message would re-run profiling for nothing.

## Cleaning towards a model

On the upload step you can name the column you're ultimately trying to predict. Every other
field in the goal form only tints how the agents word things; this one changes what the tool
does:

- **Leakage detection.** A column that predicts the target almost perfectly is usually one
  recorded *after* the outcome was known -- `termination_date` when predicting attrition,
  `invoice_paid_at` when predicting default. Left in, it inflates every offline score and the
  model collapses in production. Emptiness is tested as well as value, because that's how the
  classic offender shows up: `termination_date` is blank for exactly the people who stayed.
- **The target is protected.** A fix that would fill in missing values in the target column
  is rewritten to drop those rows instead, and dropping the target column outright is
  blocked. That's enforced in code when the card is built, not asked for in a prompt, so what
  you approve is what runs.
- **Whole-file statistics are flagged.** Filling a column with its mean bakes a number
  computed from every row into the file. Split into train and test afterwards and the test
  rows helped produce the values the model trained on, so your scores come out flattering.
  The card says so; approving anyway is your call.
- **Unusable targets are called out.** Missing answers, a single class, four rows in the
  minority class -- these are reported as blockers rather than quietly cleaned around.

`GET /api/readiness/{session_id}` gives a verdict (`ready` / `warnings` / `blocked` /
`not_applicable`) against the data as it stands right now, and the same summary appears in
the audit report. It is deliberately willing to answer "no": a tool that always hands back a
model is less useful than one that can say the data won't support it, and why.

None of this activates unless you name a target. Cleaning a file for a report behaves exactly
as it did before.

## Building a model

Once the data is clean, `/model/{session}` trains one. The same human-in-the-loop shape as
the cleaning side: the app proposes with reasons, you decide.

**The plan comes first.** The decisions that matter in a model build happen before any
fitting — which columns count as inputs, how the data is split, what gets tried. Leaky
columns and row-unique identifiers arrive pre-unticked with the reason attached; untick the
suggestion if you know better. If the data has dates, a time-based split is offered, because
a random split lets the model learn from the future to predict the past.

**Preprocessing lives inside the pipeline.** Imputation, scaling and encoding are steps in
the sklearn `Pipeline`, so `fit` sees only the training fold and the test rows are transformed
with statistics they had no part in producing. This is why cleaning deliberately leaves
`fill_mean` and friends alone rather than baking them into `cleaned.csv` — doing it there
would leak, and every score afterwards would flatter the model.

**It will tell you when the model is worthless.** A model that learned nothing still reports
70% accuracy on a 70/30 target, which reads like success. So the result is judged against
what luck alone could produce on a test set of that size, and against the score you'd get by
always guessing the same answer. Measured on data with no signal in it, 23 of 25 runs are
correctly flagged and none are called useful.

**You get the standard diagnostics**, as interactive charts with a table view behind each
one: ROC and precision-recall curves, confusion matrix, per-class breakdown, calibration, a
threshold trade-off table, cross-validation spread, learning curve, permutation importance and
a feature correlation matrix. For regression: predicted-vs-actual, residuals, residual
distribution and error by range.

**It builds inputs, not just picks them.** A `signup_date` used to be thrown away for having
too many values, when how long ago someone signed up is exactly what predicts churn. Dates
become their parts, free text becomes its length, blanks become a flag you can learn from, and
a column with hundreds of values keeps its common ones instead of being discarded. Each is a
recipe you tick, with the evidence attached, and each is rebuilt by the exported script — a
derived input the export can't reproduce would be a model trapped in this app. The improvement
loop tries them for you and tells you whether they helped.

**You say what a mistake costs, and it moves the line.** Where a classifier's cut-off sits is
a business call — *a missed case costs five times a false alarm*, *catch at least 80%* — so
you state the trade-off and the app chooses the threshold. It chooses it on out-of-fold
predictions over the training rows, never on the test set, then reports what the line actually
delivered on the held-out rows next to what the folds promised. The threshold reaches the
exported script as a declared constant.

**And you can use it again.** Upload next month's file and it is cleaned with the fixes you
approved — using the median and limits this session learned, not the new file's own — then
every row is predicted at the cut-off you chose, and the app tells you whether the new rows
still look like what the model learned from. Rows a fix removed come back with the reason, so
nothing goes missing when you join the predictions back.

**Every verdict shows its working.** "Ready" is a claim about what *isn't* wrong, and a reader
can't confirm that by looking at the data. So the readiness panel opens onto the checks behind
it — what each one concluded, the numbers, and what would change the answer — including the
ones that **couldn't run and why**. "We didn't look" and "we looked and it was fine" are
different statements, and without this they'd both appear as silence. The same table goes into
the audit report, because the conclusion is what someone has to defend months later.

**And it will keep trying, without cheating.** "Improve it" searches model settings and the
other model types, optionally until a target you name. The obvious way to build that is also
a machine for fooling yourself: try enough configurations against the held-out rows and one
of them clears any bar, not because it is better but because it happens to suit those rows.
So the search never sees the test set. Every round competes on cross-validated folds of the
training data, a round is only accepted if it beats the incumbent by more than the folds
disagree with each other, and the test set is opened once at the end for a number nothing was
selected against.

It stops on its own and says why — target reached, target out of reach, or the rounds stopped
paying for themselves. It shows the rounds it rejected too, because *"we tried tuning it and
tuning did not help"* is a finding about your data: it means the ceiling is the columns you
have, and better inputs would help where more searching would not. Whatever it settles on
goes back through the normal training path, so the charts, the model card and the exported
script all work on it, and the script carries the tuned settings rather than reverting to the
defaults.

**And it counts how often you look.** The search can be kept honest by code; you can't. Retrain,
read the score, change a column, retrain — ten rounds of that and the number on the page is the
best of ten draws on the same fifth of the rows. So every look at the held-out rows is recorded.
The first is an independent estimate; the second and third carry a note that if you changed
anything because of an earlier score this one no longer is; from the fourth, the verdict itself
says the score is optimistic and points you to the way out. Nothing is blocked — a strict limit
would make small files unusable. The ways out are rows that didn't exist yet ("Use this on a
new file" records a check on a file's real answers, never on rows the model was built from) or
a final holdout sealed *before* your first look, which only an explicit "open it" ever scores,
once. Every number on the page is labelled as one of three kinds — fold score, held-out
evaluation, new data — because they aren't interchangeable. The reasoning is in
[`docs/decisions/0001-evaluation-protocol.md`](docs/decisions/0001-evaluation-protocol.md).

**And you leave with something you can run.** The export includes `train_model.py`, which
rebuilds the model from scratch with the same preprocessing, split and settings, plus the
fitted `model.joblib` and a model card. There's a test that runs the generated script in a
subprocess and checks it reproduces the reported metrics — if it ever doesn't, one of them is
wrong, and that should be discoverable.

Models come from a fixed registry (logistic/ridge, random forest, gradient boosting) for the
same reason the cleaning operations do: the app never writes or executes generated code.

## How fixes are executed (and why there's no sandbox)

The PRD this was built from specifies generating pandas code and running it in a sandbox.
This implementation does something different: the LLM **selects** an operation from a fixed
whitelist (`backend/app/services/operations.py`) and supplies its arguments. It never writes
code, and nothing ever calls `exec`.

Why:

- **Safety.** A real sandbox (no network, no filesystem, memory and time limits) is hard to
  get right, and a restricted-globals `exec()` is not one. For a tool people self-host,
  arbitrary code execution is a bad default.
- **Cost.** The operation is chosen during the per-issue call the Explanation Agent already
  makes, so executing approved fixes costs zero extra tokens. There's also no isolated
  compute to pay for in production.
- **Coverage.** The profiler detects a closed set of issue types, so code generation buys
  almost nothing — the whitelist covers filtering, renaming, replacing, rounding, type
  coercion, formatting, outliers and null handling.

The tradeoff is real: an instruction that can't be expressed as a whitelisted operation is
declined with an explanation rather than approximated. That's the intended behaviour.

## Scope notes

`ChromaDB` and `Great Expectations` appear in the original design's tech list but aren't
referenced by any agent or workflow node's actual spec, so they're not wired into this
scaffold. `CHROMA_PERSIST_DIR` stays in `.env.example` as a reserved/unused variable in case
that changes later.

## Testing

```bash
cd backend
python -m pytest --cov=app -n auto --dist loadfile   # in parallel, as CI runs it
```

Fixture datasets in `backend/tests/fixtures/` are built to guarantee specific issue types
(missing values, duplicates, mixed types, inconsistent date formats, outliers) so the
profiler's detectors have known-good test targets. The end-to-end graph test
(`tests/integration/test_graph_end_to_end.py`) uses LangGraph's in-memory checkpointer and
a mocked LLM client, so it doesn't need Redis or a real API key. A second integration test
(`tests/integration/test_http_session_end_to_end.py`) runs a whole session over HTTP — upload to
apply — on the churn sample with the real graph and models and a fake LLM, in about twenty seconds.

```bash
cd frontend
npm test        # vitest
npm run build   # tsc + vite
```

## License

MIT, see `LICENSE`.
