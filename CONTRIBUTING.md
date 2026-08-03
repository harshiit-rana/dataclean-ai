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
- `backend/app/graph/nodes/` -- one file per LangGraph node. `conversational_loop.py` is an
  unwired stub; everything else is live.
- `backend/app/api/` -- one file per route group.
- `frontend/src/pages/` -- one file per page (Upload, Review, Clean, Export).

## Before opening a PR

- Backend: `cd backend && python -m pytest` should pass.
- Frontend: `cd frontend && npm run build` should pass.
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

The source design doc isn't published in this repo — ask the maintainer if you need a
section referenced in a docstring.
