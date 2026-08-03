# DataClean AI

An open-source, AI-powered CSV cleaning tool that keeps a human in the loop: upload a CSV,
tell it what you're trying to do, and it profiles your data, ranks issues by relevance to
your goal, and proposes fixes as suggestion cards -- Found / Proposed Action / Reasoning /
Tradeoff. Nothing changes until you approve it, card by card.

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
- **Profiling**: deterministic pandas/IQR-based detectors, independent of `ydata-profiling`
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

1. **Upload** a CSV and describe what you're using it for
2. **Profiling** runs deterministic pandas/IQR detectors, then an LLM re-ranks findings
   against your stated goal
3. **Review** shows your actual data in a table — click any column to see its stats, its
   role, and what was flagged in it — alongside streamed suggestion cards
4. **Approve or reject** each fix individually; only approved ones execute
5. **Clean** applies them, showing before/after numbers per fix, with per-fix **undo**
6. **Ask for more** in plain English ("drop rows where salary is under 30000")
7. **Export** the cleaned CSV plus a PDF audit report documenting every change — including
   the ones you declined, in your own words

Your uploaded file is never modified. Cleaning writes a separate `cleaned.csv`, and there's
a test asserting the original is byte-identical after a run.

**Not implemented:** the `conversational_loop` graph node is an unwired stub — conversational
cleaning is served by `POST /api/chat` instead, since the graph has already completed by the
time you're chatting and re-entering it per message would re-run profiling for nothing.

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
python -m pytest --cov=app
```

Fixture datasets in `backend/tests/fixtures/` are built to guarantee specific issue types
(missing values, duplicates, mixed types, inconsistent date formats, outliers) so the
profiler's detectors have known-good test targets. The end-to-end graph test
(`tests/integration/test_graph_end_to_end.py`) uses LangGraph's in-memory checkpointer and
a mocked LLM client, so it doesn't need Redis or a real API key.

```bash
cd frontend
npm run build
```

## License

MIT, see `LICENSE`.
