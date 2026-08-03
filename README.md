# DataClean AI

An open-source, AI-powered CSV cleaning tool that keeps a human in the loop: upload a CSV,
tell it what you're trying to do, and it profiles your data, ranks issues by relevance to
your goal, and proposes fixes as suggestion cards -- Found / Proposed Action / Reasoning /
Tradeoff. Nothing changes until you approve it, card by card.

## Before you start: get an Anthropic API key

This project calls the Claude API directly and does not ship with a shared/default key.
Get your own key at **[console.anthropic.com](https://console.anthropic.com)**, then put it
in `.env` as `ANTHROPIC_API_KEY` (see "Getting started" below). Without it, the Profiling
and Explanation agents cannot run.

## Getting started (Docker, recommended)

```bash
git clone <this-repo-url>
cd dataclean-ai
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY

docker-compose up
```

- Backend: http://localhost:8000 (health check at `/health`)
- Frontend: http://localhost:5173

Try it with one of the messy datasets in `samples/` (`customers.csv`,
`sales_transactions.csv`, `employee_records.csv`) -- each has a mix of missing values,
duplicates, inconsistent formats, mixed types, and outliers baked in.

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
- **Agents**: `langchain-anthropic` (`ChatAnthropic`), model configurable via
  `CLAUDE_MODEL` in `.env` (defaults to `claude-sonnet-5`; `claude-haiku-4-5` is a cheaper
  swap-in, see `.env.example`).
- **Profiling**: deterministic pandas/IQR-based detectors, independent of `ydata-profiling`
  (which is wrapped defensively and never allowed to crash a session) -- Claude only adds
  semantic interpretation and goal-aware re-prioritization on top.
- **Frontend**: React + Vite + Tailwind, SSE consumed via `@microsoft/fetch-event-source`
  (native `EventSource` can't do POST, and starting the graph requires one).

## Project status

This repo is a scaffold: the full stack is wired up, but only one path is fully
implemented end-to-end so far.

**Working today:** upload a CSV → describe your goal → the Profiling Agent and Explanation
Agent run for real (real pandas detection + real Claude calls) and stream suggestion cards
to the Review page → the graph pauses at the approval step and the session is checkpointed
in Redis.

**Stubbed (correct signatures, not yet implemented):** actually applying approved fixes,
the conversational follow-up loop, undo, and the audit report / CSV+PDF export. Each stub
has a `# TODO: PRD §x.x` comment pointing at the relevant spec section. See
`CONTRIBUTING.md` for where things live if you want to pick one up.

## Scope notes

`ChromaDB` and `Great Expectations` appear in the original design's tech list but aren't
referenced by any agent or workflow node's actual spec, so they're not wired into this
scaffold. `CHROMA_PERSIST_DIR` stays in `.env.example` as a reserved/unused variable in case
that changes later.

## Testing

```bash
cd backend
python -m pytest --cov=app
```

Fixture datasets in `backend/tests/fixtures/` are built to guarantee specific issue types
(missing values, duplicates, mixed types, inconsistent date formats, outliers) so the
profiler's detectors have known-good test targets. The end-to-end graph test
(`tests/integration/test_graph_end_to_end.py`) uses LangGraph's in-memory checkpointer and
mocked Claude responses, so it doesn't need Redis or a real API key.

```bash
cd frontend
npm run build
```

## License

MIT, see `LICENSE`.
