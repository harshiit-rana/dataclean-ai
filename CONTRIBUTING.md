# Contributing to DataClean AI

Thanks for your interest in contributing! The full pipeline works end to end -- see
`README.md`'s "Project status" for what that covers.

## Getting set up

Follow the "Getting started" section in `README.md` first. Once the backend and frontend
are both running locally, you're ready to make changes.

## Where things live

- `backend/app/services/operations.py` -- the whitelist of cleaning operations. **Start
  here** if you want to make the tool able to fix more things: add an operation, and both
  the suggestion-card and conversational paths can use it. See the README on why the app
  selects from a whitelist rather than generating code.
- `backend/app/services/ml_readiness.py` -- everything that only becomes possible once the
  user names a column to predict: leakage detection, target checks, the guard that stops a
  fix from corrupting the target, and the readiness verdict. Entirely deterministic
  pandas/numpy, like `profiling.py`. It imports from `profiling.py`, never the other way
  round -- keep that direction.
- `backend/app/graph/nodes/` -- one file per LangGraph node. `conversational_loop.py` is an
  unwired stub; everything else is live.
- `backend/app/services/modeling.py` -- the model registry and the sklearn pipeline. Models
  are described as `(module, class, params)` rather than built by a callable, so the model
  the app fits and the model the exported script fits come from the same three values. Add a
  model here and both paths pick it up.
- `backend/app/services/diagnostics.py` -- everything behind the charts, as numbers. Two
  tiers: cheap ones run for every candidate, expensive ones (cross-validation, learning
  curve, permutation importance) for the chosen model only.
- `backend/app/services/model_export.py` / `model_card.py` -- the artifacts the user leaves
  with.
- `backend/app/api/` -- one file per route group.
- `frontend/src/pages/` -- one file per page (Upload, Review, Clean, Model, Export).
- `frontend/src/components/charts/` -- one file per chart family, all built on the shared
  `ChartCard`.

## Before opening a PR

- Backend: `cd backend && python -m pytest` should pass.
- Frontend: `cd frontend && npm test && npm run build` should both pass.
- Keep stub signatures intact if you're not implementing them yet -- other code and tests
  are written against those signatures.

## Conventions worth knowing

- **Graph state is JSON-native.** `GraphState` holds `list[dict]`, not nested Pydantic
  models — nested models get serialized into checkpoint envelopes that don't reliably
  reconstruct, which breaks resuming a session. Validate into the real models at node
  boundaries instead. The docstring on `GraphState` explains this.
- **Degrade, don't crash.** A failed LLM call, an operation that can't run, a PDF that won't
  render — none of these should end a session that already touched the user's data. Look at
  `audit_agent.py` and `cleaning_agent.py` for the pattern.
- **The user's words are never paraphrased.** Rejection reasons are copied verbatim into the
  audit report and are deliberately never sent to the model; there's a test asserting that.
  `ActionLogEntry` keeps `rejection_reason` (the user's words) and `failure_reason` (why the
  app couldn't run a fix) in separate fields for the same reason — they both land on a
  `rejected` entry, and putting machine text in the first one makes the report attribute the
  app's own error message to the user.
- **Target safety is enforced in code, not asked for in a prompt.** A fix that would impute
  the column being predicted has to be impossible, not discouraged. The guard runs when the
  suggestion card is *built* so the user approves the corrected operation -- substituting it
  silently at execution time would make the approval meaningless.
- **Warnings have to stay believable.** The leakage thresholds in `ml_readiness.py` are
  deliberately high. A warning that fires on ordinary strong correlation teaches users to
  ignore it, and it's only worth having if it's believed the one time it matters.
- **Preprocessing belongs inside the pipeline, never in the dataframe.** Imputing or scaling
  before the split lets test rows influence the values written into training rows. There's a
  test asserting the fitted imputer's learned median came from the training fold alone --
  keep it passing.
- **A check that can't run has to say so.** `readiness_checks.Check` has three outcomes that
  matter, not two: passed, failed, and `not_run` with a `blocked_by`. `assess_readiness` used
  to return early when the target was missing, so six later checks silently never happened and
  the report was indistinguishable from one where they had all passed. Never add an early
  return there — record a skip instead. `blockers` and `warnings` are derived views over the
  checks, so a new check gets into the summary, the panel and the audit report for free.
- **A business rule is data, never code.** `business_rules.Rule` is a frozen dataclass with a
  closed set of kinds, and evaluating one is a match over four cases — no `eval`, no expression
  strings, no `df.query`. Rules arrive from proposals, saved sessions and API requests, all of
  which are untrusted input, and untrusted input that can express arbitrary computation is a
  hole no amount of validation closes reliably. Same reasoning as the operations whitelist. To
  support a new kind of rule, add a kind and a branch — not an escape hatch.
- **A proposed rule does nothing.** `propose()` guesses from column names and values; nothing
  is evaluated until it appears in the session's approved list. Proposals are also filtered to
  those that would find something *in this file*: a list padded with plausible rules that catch
  nothing is how people learn to approve without reading, which would make the approval step —
  the thing the whole app is built around — worthless.
- **Rule checks run outside the target-dependent block.** `age = -5` is wrong whether or not
  anyone is predicting anything, so `_add_rule_checks` runs before every early return in
  `assess_readiness`, and rule findings reach the summary even on a `not_applicable` verdict.
  The audit report mirrors this with `_rules_section`, which exists only because the
  model-readiness section is skipped entirely when no target was named.
- **Run history survives `model_store.clear`.** `clear` fires when the data changes underneath
  a model, and drops the result, card, script and joblib — but not `model_history.json`. That
  moment is exactly when the history is most useful: the user has just changed something and
  wants to know whether it helped. Each entry carries a fingerprint of the data it trained on,
  so a run from before the change is *labelled* rather than deleted or silently compared.
  History bookkeeping must never fail a training run — losing a result that exists because its
  record couldn't be written is the worst outcome available.
- **An artifact the export promises is built on demand, not as a side effect.** `train_model.py`
  used to reach disk only when someone generated the model card, so "Export everything" after
  skipping the write-up produced an archive without the script — the one artifact the model
  path exists to produce. `routes_export._ensure_training_script` now writes it when the
  bundle is requested, the same way the audit report is refreshed on demand. If you add an
  artifact to the bundle, make sure something other than an unrelated button creates it.
- **A page that needs navigation state has to recover without it.** `CleanPage` takes the
  approved decisions from router state, which a browser reload discards -- and it used to
  answer that with "start from the review step", stranding anyone who pressed F5. It now
  looks for a persisted action log first. This mattered more once the business rules panel
  moved onto that page: a reload left approved rules with no route back to them at all.
  Anything reachable only through router state is one refresh away from being unreachable.
- **An issue description is quoted, so it has to be true of this column.** The LLM writes the
  card from `ColumnIssue.description` and `stats`, and it repeats what it is told. A canned
  example (`'USA' vs 'usa'`) in every casing description turned up verbatim on a card about
  plan tiers, and quoting the IQR fences as "the expected range" put a negative lower bound on
  monthly spend. Describe what was found in the data -- real values from the column, the
  observed range of the ordinary rows -- never the device used to find it.
- **A blocked verdict binds.** `/api/model/{id}/train` calls `assess_readiness` and refuses
  when the verdict is `blocked`. This was advisory until a dataset with 4,349 of 4,407 rows
  unlabelled went through: readiness said `blocked`, the training route never asked, and the
  app fitted on the 58 remaining rows and reported the result as useful. A verdict the app
  ignores is decoration. If you add a route that produces a model, it asks first.
- **Count the rows you can learn from, not the rows in the file.** `enough_rows` measures
  `df[target].notna().sum()`. The same dataset had it reporting "4,407 rows, comfortably above
  the 50 needed" directly above a check reporting 4,349 rows with no answer. Any check whose
  subject is training has to measure the rows that will actually be trained on.
- **Too little test data is its own verdict, not a bad score.** `assess_result` returns
  `not_enough_test_data` below 30 test rows or 10 in the rarer class, before any other verdict
  is considered — because `no_better_than_guessing` is just as much a claim about the test set
  as `useful` is, and neither can be made from twelve rows. Related: `_auc_noise_threshold` is
  deliberately uncapped. The `min(0.75, ...)` that used to be there fired only on the smallest
  test sets — the exact case the formula exists to be strict about — and lowered the bar.
- **A date-based split has to be forward-in-time all the way down.** Same rule as grouping:
  `split_strategy="time"` sets `ordered=True` on every fold the run makes -- model choice,
  tuning, the improvement loop, the cut-off, the cross-validation spread, the learning curve --
  so each chunk is validated on a model trained only on rows before it. Rows arrive in time
  order from `split_data`; keep it that way. The cut-off uses an explicit fold loop rather than
  `cross_val_predict`, which refuses folds that don't cover every row. The train/test cut itself
  is `modeling.time_cut`, which moves to the nearest point where the timestamp changes so a day
  is never split across the two; `model_export._split_block` writes the same lines into
  `train_model.py`, and a test executes that block against the app's split. Change one, change
  both.
- **A grouped split has to be grouped all the way down.** Choosing `split_strategy="group"`
  changes four things, not one: the train/test split uses `GroupShuffleSplit`, cross-validation
  uses `GroupKFold`, the improvement loop's search folds are grouped, and the grouping column
  is withheld from the features. Miss any of them and the leak simply moves — a grouped
  train/test split with ungrouped CV reports an inflated cross-validated score sitting directly
  beside the honest one, and nothing on the page says why they disagree. `readiness_checks`
  owns the shape test (`MIN_DISTINCT_FOR_ENTITY`, `MIN_REPETITION_FOR_ENTITY`) and the model
  plan offers exactly the columns it flags — two answers to one question is the bug to avoid.
- **Nothing selects on the test set -- including the plain train.** `train_candidates` used to
  keep whichever candidate scored best on the held-out rows; it now calls
  `choose_on_training_folds`, and a test flips every test-set answer and asserts the choice
  does not move. The leakage suggestions in the plan and the cross-validation spread and
  learning curve on the results page follow the same rule: they read the training rows only,
  and `compute_deep_diagnostics` takes `groups` aligned with `X_train`. The improvement loop
  searches against cross-validated
  folds of the training data and opens the held-out set exactly once, at the end. If you add
  a round, score it through `tuning.score_config` or `tuning.search` — never by fitting and
  checking `X_test`. A round also has to clear `_is_real_gain`, which refuses anything smaller
  than the fold-to-fold spread; without that the loop "improves" forever on noise.
- **A statistic learned from one file is applied to the next as learned.** Five operations
  read something from the data they run on (`operations.LEARNED_OPERATIONS`: fills and outlier
  limits). When a session is applied to a new file, `replay.freeze` replays the log over
  `original.csv` and reads each of those off the data at the step it ran, and the new file gets
  those values -- never its own. They travel through `apply_operation(..., learned=)`, never
  through `params`: parameters come from the LLM and from chat, and neither may supply one. If
  you add an operation that reads a statistic, add it to `LEARNED_OPERATIONS` and to `learn`.
- **`clean.py` mirrors the operations by hand, so a test holds it to them.** Each operation has
  a template in `cleaning_export._lines` that must do exactly what the operation does. If you
  change an operation, change its template; if you add one, add a template and a step to
  `EVERY_OPERATION` in `test_cleaning_export.py`, which fails until the registry is covered and
  compares the script's output with the app's byte for byte.
- **No row leaves without a reason.** Replay keeps the upload's row numbers as the index through
  every step and records which step removed each row; the predictions file carries every
  uploaded row. Operations must preserve the index (filter with a mask, never `reset_index`) or
  this breaks.
- **A page that starts work on mount has to ask whether the work is already done.** The Review
  page starts profiling when it mounts, and `/api/profile` used to start the graph from the
  top every time -- so a reload re-profiled, rewrote every card at one LLM call each, and threw
  away undecided cards. It now reads the checkpoint first: paused at approval means replay the
  saved cards (no LLM calls), finished means a 409 pointing at the cleaning page, and only a
  session that never ran or died part-way starts again.
- **Dependencies have a floor and a cap.** `requirements.txt` floors are the versions the suite
  last passed against and caps are the next release allowed to break the API. They used to be
  floors only, some years stale, so a fresh build got whatever was newest that day. To move up:
  raise the bound, run the suite, commit both together.
- **CI runs on a pinned image, and its actions are kept off deprecated runtimes.** Both jobs use
  `ubuntu-24.04`, not `ubuntu-latest`: the alias moves to a new Ubuntu release on GitHub's
  schedule, and a new release can change the system libraries WeasyPrint needs and break the
  backend job with no change to this repo. Moving to a newer image is a deliberate commit with
  the run read afterwards, not something that happens to you. When GitHub's annotations on a
  run say an action targets a deprecated Node.js runtime, upgrade the action's major version
  and read its release notes for changed defaults first -- the `setup-node` majors changed
  automatic caching, for instance.
- **A POST stream is opened once.** `streamPost` sets `openWhenHidden: true`. The library's
  default aborts the request when the tab is hidden and re-sends it when the tab returns --
  harmless for a GET, and for every stream here it meant re-running profiling, re-applying
  approved fixes, or starting a second search. If you add a stream, go through `streamPost`.
- **Work that outlives its request saves itself.** `_improve_and_save` persists the result in
  the worker thread, not in the route that awaits it. The stream's generator is cancelled when
  the client goes away; anything after that `await` would never run, and a minute of fitting
  would be lost to a closed tab. Refuse what can be refused *before* the stream opens, so it
  arrives as an ordinary JSON error with a status code rather than an event inside a 200.
- **A test that loops over a result can pass by looping over nothing.** The check that tuned
  settings reach the exported script iterated over `improvement.settings` -- which was empty on
  its data, because the search never kept a setting there. So it passed while the path it
  guarded was broken: the first real kept setting crashed "Improve it". When a test's subject
  is conditional on the data, force the condition (a monkeypatched `improve` here) rather than
  hoping the fixture triggers it.
- **Names and contact details are suggested out of the inputs.** `_looks_like_personal_details`
  matches the column's name against a closed list of words, because the data alone cannot show
  that a surname is meaningless -- on the churn sample it ranked second for importance on pure
  noise. When a name *does* carry signal it is usually worse, because what it carries is what
  it stands in for. Suggested, never enforced.
- **A derived input is a declared recipe, and the export rebuilds it.** `features.Recipe` is
  a frozen dataclass with a closed set of four kinds; `apply` and `as_pandas` are two
  renderings of the same recipe and a test applies both to one frame and compares them. If
  they drift, `train_model.py` silently builds different inputs from the ones that were
  scored and every number in the app becomes unreproducible. To add a kind, add a branch to
  both -- never an escape hatch that computes an expression.
- **A proposal never reads the target.** `features.propose` runs over the whole file, test rows
  included, so what it offers must not depend on the answers. `TestProposalsNeverReadTheTarget`
  replaces the target with shuffles, a constant, another vocabulary and a column of blanks and
  requires the same proposals every time; a new recipe that looks at the target fails it. Anything
  that does need the answers belongs inside the pipeline, fitted on the training fold.
- **Derive before the split only when the recipe is row-wise.** Date parts, text length and a
  missingness flag depend on nothing but the row they are on, so computing them on the whole
  frame leaks nothing. Anything that learns from the column -- which values are common, a
  target encoding, a mean -- goes inside the pipeline where `fit` sees the training fold
  only. `frequent_categories` is in the pipeline for exactly this reason.
- **Recipe ids are re-proposed, never trusted.** `features.select` re-runs `propose` and keeps
  the ids that come back, so a stale id for a column that no longer supports the recipe is
  dropped quietly rather than failing a run, and a recipe naming an excluded column cannot be
  smuggled in -- a missing flag on a column withheld for leaking the answer would leak it
  straight back.
- **Row-unique is not the same as an identifier.** Dates and free text are near-unique too,
  and `_looks_like_an_identifier` used to claim both. That pre-ticked them for removal, which
  hid the recipes that would have made them usable; the RSNA run dropped a column of
  radiology reports on those grounds. Uniqueness excludes floats, dates and free text before
  it means anything. `classify_features` owns the wording for all three.
- **The cut-off is chosen on the training folds, never on the test set.** `operating_point`
  takes a trade-off (costs, a recall floor, a precision floor, F1) and picks the threshold on
  `cross_val_predict` over the training rows -- the test set is opened afterwards to report
  what the line delivered. The API deliberately has no field for a threshold: letting the
  client send one invites reading it off the test-set sweep, which is selecting on the test
  set in miniature. `expected` (folds) and `observed` (test) are both kept so the gap is
  visible. There is a test that flips every test-set label and asserts the chosen line does
  not move.
- **Charts read the app's surface, never the OS.** `useChartTheme` looks only at the `dark`
  class on `<html>` — the same signal Tailwind's `darkMode: ["class"]` uses. Falling back to
  `prefers-color-scheme` is how the correlation heatmap ended up painting near-zero cells
  charcoal on a white page: the machine was dark, the app was not.
- **Charts ship a table view.** Some palette slots sit below 3:1 on the light surface, and a
  tooltip is a way to inspect a value rather than read one. Every chart goes through
  `ChartCard` with its data as a table.
- **Session routes are guarded at the router, not per-route.** `verify_session_access` is
  applied in `main.py` where the routers are included, so a new session-scoped route cannot
  ship without the check by forgetting a decorator. If you add a router, add it to that list.
  On the client side the mirror of this is `sessionFetch` in `api/client.ts` — nothing scoped
  to a session should call `fetch` directly, and downloads go through `DownloadButton` because
  an `<a download>` can't set a header and the token must never go in the URL.
- **The app is allowed to say no.** Data that can't support a model, and a model that learned
  nothing, both get said plainly. A tool that always produces a confident answer is less
  useful than one that can tell you it hasn't got one.

The source design doc isn't published in this repo — ask the maintainer if you need a
section referenced in a docstring.
