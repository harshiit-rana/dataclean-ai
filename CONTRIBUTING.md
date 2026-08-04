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
- **Charts ship a table view.** Some palette slots sit below 3:1 on the light surface, and a
  tooltip is a way to inspect a value rather than read one. Every chart goes through
  `ChartCard` with its data as a table.
- **The app is allowed to say no.** Data that can't support a model, and a model that learned
  nothing, both get said plainly. A tool that always produces a confident answer is less
  useful than one that can tell you it hasn't got one.

The source design doc isn't published in this repo — ask the maintainer if you need a
section referenced in a docstring.
