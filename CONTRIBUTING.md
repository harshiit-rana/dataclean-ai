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
- `backend/tests/integration/test_http_session_end_to_end.py` runs a whole session over HTTP --
  upload, goal, profile, resume, readiness, plan, train, improve, export, apply -- on the churn
  sample with a fake LLM and the real graph and models (about 20 seconds). It exists because
  wiring defects between steps have been found by running real data by hand. If you change a
  response shape a later step reads, it fails; update it in the same change, deliberately. The
  fake card writer is kept close to the real prompt contract (an OPERATION line per issue
  type), so keep it in step with `explanation_prompts.py`.
- If you added anything that **decides** something -- picks a model, a setting, a feature, a
  threshold, a column to drop, a number shown beside the score -- add a case for that decision
  path to `backend/tests/unit/test_evaluation_integrity.py`. It runs the path twice, once with
  every held-out answer flipped, and requires the same decision both times. A new round in
  the improvement loop, a feature-selection step, an ensemble, a calibration fit, a search
  strategy: each one is a decision path, and each one needs its case before it is merged.

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
- **With a target named, a card never offers a fix that reads a whole-file statistic.**
  `ml_readiness.DATASET_STATISTIC_OPERATIONS` (median/mean/mode fills, outlier caps and drops)
  compute their value over every row, test rows included, before any split. For a model input
  `enforce_target_safety(..., source="card")` returns no operation and the reason, and the card
  carries `warning_kind="left_to_model"`; the pipeline does the same job on the training rows
  alone. `source="chat"` still runs them with the warning, because a typed instruction is the
  user's call. If you add an operation that learns a value from the data, add it to that set
  (and to `LEARNED_OPERATIONS`).
- **A route may read the action log to say more, never to be able to work.** The model routes
  take `get_graph_if_any`, not `get_graph`, and `action_log.read` returns an empty log when
  there is no graph or the checkpoint cannot be read. A run then simply carries no
  `whole_file_statistics`; it is never refused because Redis was slow.
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
- **Run-to-run verdicts need the same rows.** `model_store.compare_runs` pairs two runs only when
  `split_key` (strategy, columns, seed, sizes) and `data_fingerprint` match; the history entry
  stores `split_key` and the run's predictions are kept in `run_predictions/`, aligned by row id
  because the two runs need not have listed the test rows in the same order. Do not colour a
  difference in the UI from a fixed threshold -- take `comparisons[run_id].verdict`, and treat
  `null` as "no evidence", not "no change". The predictions file is deleted with the history
  entry it belongs to; keep the two in step if you touch the history trimming.
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
- **A regression verdict stands on the lower end of the R² interval.** `assess_result` takes the
  held-out metrics' `intervals` and `_regression_verdict_from_interval` uses `low`: at or below 0
  is guessing, below `R2_USEFUL_FLOOR` (0.3) is marginal, otherwise useful. Pass the intervals
  from every place a verdict is made (the train path and the sealed `finalise`); without them it
  falls back to the old fixed cut-offs, which ignore how many test rows there were. R² has no
  lower limit, so a wide negative-to-positive range on a small set is the correct message.
- **A date-based split has to be forward-in-time all the way down, and a timestamp is never split.**
  Same rule as grouping: `split_strategy="time"` hands every fold the run makes -- model choice,
  tuning, the improvement loop, the cut-off, calibration, the cross-validation spread, the learning
  curve -- a `time_folds.TimeFoldRule` where the bare `ordered=True` used to go, and
  `tuning.cv_splitter` turns it into `ForwardInTimeFolds`. That splitter is given the timestamps
  explicitly, keyed by row label (so it also serves a subset of the rows, as inside the cut-off's own
  folds), and moves every chunk boundary to the nearest change of timestamp, so rows sharing a
  timestamp are never on both sides of a fold -- `TimeSeriesSplit` cut at row positions and let them
  straddle. With distinct timestamps it is exactly `TimeSeriesSplit`. A fold producer that is not
  passed the rule silently reverts to row positions, so `test_time_ordered_folds.py` spies on every
  one. Each chunk is validated on a model trained only on rows before it. Rows arrive in time
  order from `split_data`; keep it that way. The cut-off uses an explicit fold loop rather than
  `cross_val_predict`, which refuses folds that don't cover every row. The train/test cut itself
  is `modeling.time_cut`, which moves to the nearest point where the timestamp changes so a day
  is never split across the two; `model_export._split_block` writes the same lines into
  `train_model.py`, and a test executes that block against the app's split. Change one, change
  both.
- **Splitting rules that the script must reproduce are one function, copied by source.**
  `modeling.split_keeping_identical_rows_together` is self-contained and `model_export` embeds
  its text with `inspect.getsource`, as it does `clipping.IQRClipper`. Where the app's rule is a
  few lines of pandas (the date cut), a test executes the generated block and compares the app's
  test indices. Either way, a split the script cannot reproduce makes every exported number
  wrong.
- **Calibration is part of what the export rebuilds.** When the result records a calibration
  method, `train_model.py` names the base pipeline `base_pipeline` and wraps it in
  `CalibratedClassifierCV` with the same method, `ensemble` setting and *explicit list of folds*
  (a splitter handed over would get no groups): `StratifiedKFold` for a plain split,
  `StratifiedGroupKFold` on the group column or the copy key, `TimeSeriesSplit` for a date split,
  with the fold count the app used. These must match `tuning.cv_splitter` exactly -- changing one
  without the other makes the exported script fit a different mapping. `test_calibrated_export.py` runs the script in a subprocess for each of those and compares
  accuracy and Brier score with the app's to 1e-4. A new way of splitting the training rows into
  folds has to be added to `model_export._calibration_block` and `_calibration_imports` together,
  and the script prints the Brier score of any two-class model so the probabilities are checked,
  not just the answers.
- **Copies kept together at the split are kept together in every fold.** When
  `keep_identical_rows_together` took effect, `train_candidates` and `improve` hand the copy ids
  to every fold-producing call as `groups`, through the same path a grouped split uses; an entity
  grouping, when there is one, wins. `test_identical_rows_folds.py` spies on `tuning.cv_splitter`
  and checks the splitter's own indices -- no group on both sides of any fold.
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
  than the fold-to-fold spread; without that the loop "improves" forever on noise. Selecting
  within the folds has a bias of its own: the best of several fold scores is biased upward, so a round that picks a
  winner out of a search (`tuning.search`) must re-score it, and the incumbent it would replace, on folds the search did
  not select on (`improvement.confirm`, which uses `tuning.score_config(..., confirmation=True)`) and apply the gain rule
  there too. Under a date split or ungrouped-regression groups the folds cannot be reshuffled, so the confirmation uses
  one more fold: say so rather than claim more.
- **Humans select on the test set too, so the app counts the looks.** An automatic decision can
  be tested not to read the held-out rows; a person who retrains and compares scores cannot be.
  `docs/decisions/0001-evaluation-protocol.md` fixes the protocol: which routes are a *look*
  (train, improve, improve/stream — not a re-read of a stored result, and not apply), that looks
  are counted per split definition across dataset versions, that the status goes
  independent → reused (looks 2–3) → exhausted (4+) and never blocks, and that a fresh holdout is
  new labelled rows or a holdout sealed *before* the first look — never a re-split. **A new route
  that returns held-out performance adds a row to that ADR's table and a case to the
  property suite (`test_evaluation_integrity.py`).** The record lives in
  `services/evaluation_ledger.py` (`test_set_ledger.json`): never truncated, kept across
  `model_store.clear` (it is deliberately not in `ALL_ARTIFACTS`), keyed by the split rule via
  `split_key` using the *requested* fraction and seed carried on `result["split"]`, and it
  stores the full dataset fingerprint. Writing to it never raises.
- **New labelled rows are evidence about a model, not rows to learn from or to score on twice.**
  `routes_apply._check_new_answers` leaves out rows whose inputs match the original upload,
  `evaluation_ledger.already_evaluated` recognises a file by its content hash, and a
  `new_data_evaluation` is tied to the model's `run_id` so a retrained model starts clean. It is
  never counted as a look at the held-out rows and never replaces the original score.
- **Only `finalise` reads the sealed rows.** `split_with_sealed_holdout` takes them first, by
  `sealed_holdout_rows` -- a function of the data, the seed and the split rule, never of the
  inputs, models or settings -- and the ordinary split divides what is left. `train_candidates` and
  `improvement.improve` never see them, and `test_sealed_holdout.py` and the integrity suite flip
  their answers and require every decision and every development-set report to stay put. If you add
  a route that could touch them, it belongs in that test.
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
- **Whatever rewrites `cleaned.csv` records a dataset version.** The working data is replaced in
  place by the cleaning run, by undo and by chat, so a model, a test-set look or an export needs a
  name for the state it describes. `session_store.record_dataset_version_in(directory, event)` is
  called after each write (`cleaning_run`, `undo:<action>`, `chat:<action>`; `upload` at upload)
  and appends `{version, fingerprint, rows, columns, at, event}` to the meta sidecar only when
  the content hash differs from the latest version. It never raises — it is bookkeeping about
  work already done. If you add a route that rewrites the data, call it; a test that walks the
  routes should fail if you do not.
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
- **Anything that learns from the data goes in the pipeline, and the export carries its source.**
  `clipping.IQRClipper` learns its limits in `fit`, so inside `build_pipeline` it sees the
  training fold only. `model_export` copies the class with `inspect.getsource` instead of keeping
  a second hand-written version, so the app and `train_model.py` cannot drift; a test runs the
  script and compares its accuracy with the app's. A custom estimator pickles by reference, so
  `model.joblib` needs the class importable to load: say so wherever the joblib is offered, and
  prefer scikit-learn's own classes when one will do the job.
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
- **Every score says what kind of number it is.** Fold scores, held-out scores and new-data
  scores are not interchangeable, and presenting them alike is how a maximised fold score ends up
  beside an untouched held-out one and gets compared. `ScoreLabel` has three kinds --
  `exploratory` (training folds), `held_out` (with the reuse status) and `new_data` -- and any
  surface that shows a metric uses one, on the section header when everything beneath it shares
  a kind. The comparison of candidate models leads with the fold scores that chose between
  them; the other candidates' held-out scores are behind a button with the reason they are not
  the basis for anything, and `model_card._comparison_table` does the same in the written card.
- **One metric per page.** Anything on the results page that scores a model asks
  `tuning.scoring_for(task, y)` -- ROC AUC for two classes, accuracy for more, R² for a number --
  and says which it used. The learning curve and permutation importance scored in accuracy
  while the spread beside them was AUC, so on a lopsided target they read as disagreeing about
  one model. The metric is fixed from the training rows, not the test rows, and each chart's
  output carries its `metric` so the page can name it.
- **A new model gets a `complexity_rank`.** The one-standard-error rule keeps the simplest
  candidate the folds cannot separate from the best, and "simplest" is the rank written on its
  `ModelSpec` (a linear model before a forest before boosted trees), not an accident of the
  registry's order. Add the rank when you add a family, and see `choose_on_training_folds` and
  the improvement loop's first round, which both go through `one_standard_error_choice`.
- **Candidates are compared on identical folds.** `selection.distinguishable` pairs each model's
  fold scores, which is only meaningful if fold *i* holds the same rows for every candidate.
  `tuning.cv_splitter` is deterministic for a given target and groups, and a test checks that
  selection hands every candidate the same splits; if you add randomness to a splitter, seed it
  from the data, not from the call. Do not interpret the interval with a plain paired t-test:
  `uncertainty.fold_difference` applies the Nadeau-Bengio correction for overlapping training
  sets, and the page's wording depends on it.
- **Calibration is fitted on `calibration.TrainingFolds`, never on the test rows.** The folds are
  `tuning.cv_splitter`'s -- whole groups, copies together, forward in time -- built from whichever
  rows `fit` is handed, which is why the fold object looks groups up by the label of each row of
  `X` and why `operating_point.out_of_fold_scores` no longer resets `X`'s index. `ensemble=False`
  is deliberate: one model and one mapping keeps a sigmoid's AUC exactly where it was and keeps the
  saved model one estimator; scikit-learn refuses it for forward-in-time folds, which is the only
  place the averaged form is used. After fitting, `cv` is replaced by an int so the saved model
  does not need this application's class to load. Nothing may build a fold splitter without the groups -- not even to ask whether folds are possible -- or `test_identical_rows_folds.py` rightly flags it. `TestCalibratingTheProbabilities` in
  `test_evaluation_integrity.py` is the guard; keep it passing when you touch any of this.
- **The learning curve is read in one place.** `diagnostics.read_learning_curve` decides what the curve
  says from each point's `validation_std`; the deep diagnostics carry its result and both the card and
  the chart print it, so do not add a second reading in the frontend. A curve without spreads (an older
  result) falls back to the fixed sizes.
- **Drift is judged per column on what the column actually has.** `drift.compare` gives every column
  its own filled-in counts and refuses to judge one below `MIN_VALUES_TO_JUDGE`; it compares blank
  shares (a change beyond chance is a finding in itself); and a derived input is passed its source
  column through `blanks_from`, because its own values are never blank. A new derived kind needs its
  source wired the same way in `routes_apply`. A column is called moved only when its PSI clears both
  the convention (0.1 / 0.25) and `chance_threshold`, which is set in `_judge` once the number of
  compared columns is known; do not decide a column's status before that, or the correction for
  testing many columns at once is lost.
- **A model with a random fit is marked `seeded`.** `ModelSpec.seeded` (set through `_SEEDED_MODELS`)
  is what makes `modeling.seed_stability` refit the chosen model with `SEEDS_TO_CHECK`; a model that
  takes a `random_state` but fits deterministically must not be marked, or the check reports a spread of
  zero as if it had been worth running. The check uses training folds only and is covered by
  `TestCheckingTheChoiceAgainstOtherSeeds` in the integrity suite.
- **The best-single-column baseline is bounded and reads training rows only.**
  `modeling.best_single_column_baseline` fits the simplest registry model once per input column, so it is
  skipped past `BASELINE_MAX_COLUMNS` / `BASELINE_MAX_ROWS` and says why in the result; raise those limits only
  after measuring the cost. It must be scored on the same folds as the candidates (same `folds`, same
  splitter) or the paired difference is meaningless. It is covered by
  `TestComparingWithTheBestSingleColumn` in the integrity suite.
- **A new thing that shapes which model is chosen gets a statement in `why_this_model`.** Each statement is
  built from fields of the result and lists them as `evidence` (a dotted path and its value);
  `assert_every_statement_is_backed` in `tests/unit/test_why_this_model.py` checks every path resolves to the
  value shown. Never put text from a language model in
  it, and never claim more than the fields support: say "too close to call" when the folds cannot separate two
  models.
- **A fact that changes how a score should be read belongs in the card's `_protocol_section`.** It is written in
  code from the result (never by the language model), its optional parts appear only when they apply, and each
  needs a case in `tests/unit/test_card_evaluation_protocol.py`.
- **A new caution is added through `caution_items` with an id and a severity.** Decide its severity against the
  definitions beside `SEVERITY_ORDER`, and add a case in `tests/unit/test_cautions_ranking.py`; do not append
  text to the card's cautions list separately, or the results page (which shows the top three) will not have it.
- **A new place that declines to judge uses `not_judged.not_judged` and the `NotJudged` component.** Give it
  the counts (`what`, `have`, `need`) so the sentence reads "Not enough … to judge: N rows, at least M needed.",
  or a `because` when the obstacle is not a count; do not write a one-off sentence or colour it as a failure.
  Add the block beside the existing headline rather than replacing it, and a case in
  `tests/unit/test_not_judged.py`.
- **A refusal a page can offer a way out of gets a `code`.** Raise `TrainingFailedError(detail, code=...)` or
  `ModelingError(message, code=...)`, add the code to the list in `TrainingFailedError`'s docstring and to
  `TrainingProblem` with the action it leads to, and a case in `tests/unit/test_training_error_codes.py`. Codes are
  part of the API: never reuse one for a different meaning. A refusal with no useful action stays code-less and
  shows as a message.
- **Anything that changes the model's evaluation marks the audit report stale.** Use
  `audit_agent.mark_report_stale_if_written(directory)` where the result, the sealed holdout or the new-data
  record changes, and read the model into the report with `evaluation_ledger.attach_later_evidence`, so the report,
  the card and the page see the same record. The report's model facts come from `model_card.protocol_facts`; do not
  restate them.
- **A readiness check's action navigates; it never applies.** Build it with `readiness_checks.action(kind, label,
  column)` and add the kind to `ACTION_KINDS`, to `actionPath` in `CheckTrail.tsx` and to
  `tests/unit/test_readiness_check_actions.py`. If the place it leads to can be pre-set, pre-set it from the
  address and still leave the decision to the user.
- **Anything slow that the improvement loop runs checks the deadline.** `improve` takes a `time_limit_seconds` and
  hands `tuning.search` a `deadline` (a `time.monotonic()` instant); a new round or search strategy checks
  `out_of_time()` before it starts and records itself as skipped (`skip_for_time`) rather than silently not running.
  The deadline must be read from `time.monotonic` through the module's `time` so tests can substitute a clock; never
  compare it with a score.
- **A new search strategy implements `tuning.SearchStrategy` and joins `tuning.STRATEGIES`.** It is handed the folds,
  metric, budget and deadline already decided; it must check the deadline between pieces of work, report
  `stopped_for_time`, and put its own `name` in `SearchResult.strategy`. Keep the interface to what the strategies
  that exist need: add a parameter when a second strategy needs it, not before.
- **A search never reports, or lets the loop compare, a score on fewer folds than the incumbent had.** Halving ranks on
  partial folds inside a rung, but what it returns is the survivor's mean over every fold of the same splitter. A new
  strategy that uses a cheaper resource must end the same way, and must reuse the splitter it is handed (spy tests in
  `test_successive_halving.py` show how to check group-aware and forward folds at every rung).
- **The automatic strategy choice is a pure function with documented thresholds.** Change `HALVING_MIN_*` in
  `tuning.py` and the boundary cases in `tests/unit/test_choose_strategy.py` together, and say in the commit what
  data justified it: a threshold moved without a benchmark is a guess, and the tests exist to make that visible.
- **A new model family gets a `cost_weight`.** Set it on its `ModelSpec` (see `_COST_WEIGHTS` in `modeling.py`) from
  a rough timing against logistic regression, or the processing limit will treat it as cheap. A new step that scales
  with rows times candidates times folds belongs in `estimate_cost`, and anything that reduces a request to fit
  (`plan_within_limit`) must say so in `resource_limits` rather than do it silently, and never touch the test set.
  Tests that take minutes go in `tests/slow`, marked `slow`; the default run leaves them out.
- **Only comparisons repeat their folds.** `tuning.cv_splitter(..., repeated=True)` returns a repeated
  splitter below `REPEATED_CV_BELOW_ROWS` for a random split; `score_config`, `search` and the shown
  fold spread ask for it. Anything that needs a partition -- the cut-off's out-of-fold scores,
  calibration (`cross_val_predict` refuses anything else) -- must not. `SearchResult.repeats` says how
  many there were: divide `fits` by it before using a fold count as a sample size, as the
  one-standard-error rule does, and give `uncertainty.fold_difference` the per-repeat
  `test_train_ratio`.
- **The multi-class verdict's noise floor comes from shuffling, not a formula.** `ovr_noise_floor`
  shuffles the test answers against the model's own probabilities and reads the tail
  (`1 - 0.05 / model_count`), bounded by `_FLOOR_WORK`; it must stay free of refits and must only ever
  *shuffle* the answers. Every path that makes a verdict for more than two answers (the train route and
  the sealed `finalise`) passes it to `assess_result` as `ovr_auc_floor`; without it the verdict falls
  back to the accuracy rule, with a note. Add a new multi-class metric to the baselines too, or its
  column has nothing to be compared with.
- **A group key is built in one place and copied by source.** `modeling.with_group_key` returns the
  frame, the column to split on and the columns to withhold; one column is used as it is (the
  behaviour before composite keys), several become a column named "a + b" made by
  `composite_group_key`. The order of the key's labels decides which groups land in which fold, so the
  export copies that function's source rather than rewriting it, and a sealed holdout is re-split from
  `split["group_columns"]`. Any new place that groups rows must call `with_group_key` and withhold
  everything it returns -- a key column left in the inputs is the memorisation the split exists to
  prevent.
- **A date split's gap is applied in three places that must agree.** `modeling.split_data`
  (`time_gap_days`) drops the training rows less than the gap before the first test row;
  `time_folds.TimeFoldRule.gap_days` does the same inside every fold, because
  `tuning.cv_splitter` hands it to `ForwardInTimeFolds`; and `model_export._gap_lines` writes the cut
  into `train_model.py` (a test executes the generated block against the app's split) while the
  calibration folds carry `gap_days` too. `split["gap_days"]` and `["gap_rows_dropped"]` are the record,
  `evaluation_ledger.split_key` includes the gap only when it is set so earlier keys do not move, and
  the sealed-holdout opening re-splits with the stored gap or its row-count check would refuse it.
- **The split recommendation reads the answers only from the training side of the default split.**
  `validation_advice.recommend_split` is pure; its one use of the target, the drift test, is handed
  `training_rows` by `routes_model._recommended_split` and nothing else, and
  `TestRecommendingTheSplit` in `test_evaluation_integrity.py` flips every held-back answer and shows
  the advice does not move. Entity candidates come from `_group_columns`, which shares its test with
  `readiness_checks.check_repeated_entities`: do not re-tune thresholds in the advice. Whether a run
  "overrode" it is decided in `_note_the_split_choice` from the split the run used, never from a flag
  the client sends.
- **The held-out fraction is a request, recorded and keyed.** `validation_advice.recommend_test_size`
  turns the labelled-row count (and the rarer answer's count for two answers) into a fraction in
  steps of five percent, with the reason; it reads the verdict's own thresholds from `modeling`, so
  the two cannot disagree about what is judgeable. `TrainRequest.test_size` carries the choice to
  `split_data`, `improve`, the exported script and `split["requested_test_size"]`, which the ledger
  keys on; omitting it keeps the key every earlier run had. Never derive it from which rows were
  held out. A sealed holdout ignores it.
- **A baseline row is complete or it is not a baseline.** `modeling.baseline_score` returns the
  main strategy at the top level and any others in `others`, each with a value for every metric
  the models report for that task, and the page, the sealed table and the card loop over both. A
  naive strategy that leaves a column blank invites comparing a model with nothing, and the common-
  answer guesser's F1 of 0 on a lopsided target is exactly that. Expected scores of a random
  strategy are computed, not sampled, so the row is deterministic. The top-level `accuracy` is what
  `assess_result` compares with; do not change it as a side effect of adding a row. A regression's
  second row is the median, and each model's `mae_vs_baseline` is read against it
  (`modeling.relative_error`); the lower-is-better columns are listed in `LOWER_IS_BETTER` in
  `ModelResults.tsx`.
- **A promise-versus-delivery message needs a range behind it.** The operating-point card judges
  "over-promised" by whether the training folds' figure is outside `observed_intervals`, never by
  a fixed gap; a result without intervals gets no message. `uncertainty.threshold_intervals`
  resamples at the line the training folds chose (the test rows had no say in it) and drops
  resamples where a rate is undefined instead of scoring them 0.
- **A cut-off is chosen on the probabilities it will be applied to.** `train_candidates` passes
  `operating_point.apply` the calibrated estimator factory as `build` when calibration is on, and
  `calibrated_with` so the result and the card say so. Never choose the line on one scale and apply
  it on another. The price is nested folds (outer x (inner + 1) fits); `test_calibrated_cutoff.py`
  pins the count, and a change that makes it worse should say why.
- **A new two-class metric goes in four places.** `modeling.evaluate` computes it, `uncertainty`
  recomputes it as a weighted sum (so it gets a range, and a test holds the two equal), the card's
  `_METRIC_LABELS` explains it in plain English, and the frontend's `CLASSIFICATION_KEYS` lists it.
  Brier score and calibration error are the template: they share their bins with the chart through
  `diagnostics.calibration_slots`, so the figure printed beside the chart is the chart's own, and
  they are shown lower-is-better without a pass/fail colour.
- **A headline held-out number carries its interval, and the interval's arithmetic is held to
  `evaluate`.** `uncertainty.py` recomputes every headline metric as a weighted sum over one set
  of predictions (a resample is a vector of draw counts). If you add a metric to `evaluate`, add
  it there and to the equality test in `test_uncertainty.py`; an interval around a number the
  page does not show is worse than none. A grouped split resamples groups, never rows.
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
