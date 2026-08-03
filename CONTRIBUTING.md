# Contributing to DataClean AI

Thanks for your interest in contributing! This project is an early-stage scaffold -- large
parts of it (cleaning execution, the conversational follow-up loop, audit/export) are
intentionally unimplemented stubs. See `README.md`'s "Project status" section for what's
real today.

## Getting set up

Follow the "Getting started" section in `README.md` first. Once the backend and frontend
are both running locally, you're ready to make changes.

## Where things live

- `backend/app/graph/nodes/` -- one file per LangGraph node. `goal_intake.py`,
  `profiling_agent.py`, `explanation_agent.py`, and `human_approval.py` are real;
  `cleaning_agent.py`, `conversational_loop.py`, and `audit_agent.py` are stubs with a
  `# TODO: PRD §x.x` pointing at the relevant spec section.
- `backend/app/api/` -- one file per route group. Stub routes return `501` with a message
  pointing at what needs implementing.
- `frontend/src/pages/` -- one file per page. Pages 1 and 2 (Upload, Review) are wired up
  end-to-end; Pages 3 and 4 (Clean, Export) are placeholder components.

## Before opening a PR

- Backend: `cd backend && python -m pytest` should pass.
- Frontend: `cd frontend && npm run build` should pass.
- Keep stub signatures intact if you're not implementing them yet -- other code and tests
  are written against those signatures.

## Picking up a stub

If you're implementing one of the stub nodes/routes, ask the maintainer for the PRD section
referenced in its docstring/TODO (the source design doc isn't published in this repo) --
the surrounding real code (e.g. `explanation_agent.py` for streaming patterns,
`profiling_agent.py` for the retry pattern) is the best reference for the conventions this
codebase already follows.
