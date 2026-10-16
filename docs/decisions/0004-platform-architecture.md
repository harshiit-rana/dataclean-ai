# 0004. Platform architecture: projects that are kept, background jobs, and where the data lives

- **Status:** Accepted, 16 October 2026. The owner chose option (b) on issue #102 with the conditions quoted in
  "The owner's conditions"; this record turns that choice into concrete designs before anything is built.
- **Unblocks:** #120 (projects that are kept), #103 (background jobs), #104 (resume an interrupted search),
  #105 (the retention sweep out of the API process). **Revisits:** #114 (threat review), which runs after this
  work because it changes the security model.

## The question

Every session is ephemeral today. Files and the LangGraph checkpoint expire 24 hours after the session was last
touched (`upload_retention_seconds`, `redis_ttl_seconds`), there are no accounts, access is a per-session token
(`session_access.py`), and long work (training, an improvement search, applying a model) runs in a thread inside the
API request. That suits one file at a time. It rules out experiment history beyond a day, monitoring over time, a
champion that lasts, and the planned deep-learning training and retraining, and a deploy or crash mid-search loses
the search.

The options in #102:

- **(a)** stay a session tool and explicitly exclude long-lived features;
- **(b)** add persistent *projects* (named, kept until deleted) with durable storage and a job runner, without
  multi-user collaboration;
- **(c)** a fuller platform: accounts, sharing, object storage.

## Decision

**(b).** Rejected:

- **(a)** closes off what the owner has said comes next: long AutoML searches that survive a restart, experiment
  history, deep-learning training and retraining.
- **(c)** is a much larger change for features nobody has asked for yet (sharing, roles), and accounts bring
  password storage, recovery and session management, each a security surface of its own. Nothing in (b) prevents
  moving to (c) later: a project's folder and token become an account's project and a credential.

## The owner's conditions

Quoted from the decision on #102, each answered by a section below:

1. Project tokens securely generated and stored. -> *Identity*
2. Project data isolated. -> *Storage*
3. Deletion removes associated files and metadata. -> *Deletion*
4. Storage location configurable, with the security implications of local disk documented. -> *Storage*
5. Keep projects until explicitly deleted. -> *Retention*
6. No full user accounts yet. -> *Identity*

## Design

### What a project is

A project is a session its owner chose to keep. Upload still creates a session exactly as today; `POST
/api/session/{id}/keep` with a name turns it into a project. Nothing about the session's data, folder, checkpoint or
routes changes, so every existing feature works on a project unchanged, and a session that is never kept behaves
exactly as it does today. One dataset lineage per project (the upload and its cleaned versions, models and runs), as
a session has now; several datasets per project is not part of this decision.

Why not a new top-level object: a separate `projects/` tree would duplicate the session store, the access check,
the run folders and the sweep, and every route would need to learn a second id. Making "kept" a property of a
session reaches the same place with one flag.

### Identity

- **No accounts.** The project token *is* the session token: 32 random bytes from `secrets.token_urlsafe`, issued
  at upload, of which the server stores only the SHA-256 hash in `meta.json` and compares in constant time
  (`session_access.py`). A leaked `meta.json` does not grant access.
- **A kept project always requires its token,** whatever `APP_ENV` says. Development mode turns token enforcement
  off for ephemeral sessions so `docker compose up` is all it takes to try the app; a project is kept indefinitely,
  so that convenience would leave it readable by anyone who learns its id for as long as it exists. The frontend
  already sends the token on every request, so this costs a developer nothing.
- A session created before tokens existed has no hash. Keeping it issues a new token, stores its hash and returns
  the token once.
- **The token is shown once,** when the project is kept, with an instruction to save it: it is the only way back
  into the project from another browser. The browser keeps the tokens of kept projects in `localStorage` (ephemeral
  sessions stay in `sessionStorage`, as now) and lists those projects on the upload page. Losing the token loses
  access; there is no recovery, because recovery is an account feature.
- Rotating a token (`POST /api/session/{id}/token/rotate`, guarded by the current token) is included so a token that
  has been shared or exposed can be replaced without moving the data.

### Storage

- **One configurable root,** `UPLOAD_DIR` as today (default `./uploads`, `/app/uploads` on the compose volume).
  It already is configurable; it is documented as *the* storage location, and holds sessions, projects, job records
  and the sweep lock.
- **Isolation** is the folder per session id (a server-generated UUID, never a user-supplied name), with every route
  that touches it behind the token check attached at the router (`main.py`). A project adds nothing outside its
  folder.
- **The checkpoint stays in Redis.** The LangGraph state lives where it does now; the compose Redis already runs with
  append-only persistence on a volume.
- **Local disk, stated plainly** (goes into the README and `docs/security/`): the files are stored unencrypted.
  Anyone with access to the host, the Docker volume or a backup of either can read every project, tokens
  notwithstanding; the token protects the HTTP interface, not the disk. Deletion unlinks files; it does not
  securely erase them, and copies in backups or snapshots remain. A hosted deployment should put `UPLOAD_DIR` on an
  encrypted volume, restrict it to the app's user, and treat backups as holding the users' data. Object storage
  (S3 and the like) is not part of (b); the folder layout is kept simple enough to move onto one later.

### Retention

- An ephemeral session keeps today's rule: swept `upload_retention_seconds` after it was last touched.
- **A kept project is never swept.** `purge_expired_sessions` skips folders whose `meta.json` says `kept`.
- **The checkpoint follows the same rule, decided in one place.** Today Redis expires a checkpoint by its own TTL
  and the sweep deletes the files separately, which cannot express "kept". The Redis TTL is removed; the sweep, when
  it removes a session's folder, also deletes that session's checkpoint thread (`adelete_thread`). Checkpoint keys
  written before this change still carry their old TTL, so keeping a project also persists its existing keys.

### Deletion

`DELETE /api/session/{id}` (guarded by the token) removes, in this order: any queued or running job of the session
(cancelled, see below), the session's checkpoint thread in Redis, and the session folder with everything in it
(uploads, cleaned versions, run folders and fitted models, job records, reports). It works for ephemeral sessions
too. The response says what was removed. It is irreversible, and the page asks for confirmation by typing the
project's name.

### Background jobs

Training, the improvement search, applying a model and reproducing a run become **jobs**.

- **A job record is a folder on disk:** `<session>/jobs/<job_id>/` holding `job.json` (kind, request, state
  `queued | running | done | failed | interrupted | cancelled`, heartbeat, result reference) and `events.jsonl`, the
  progress events in order. The progress stream (`/improve/stream` and friends) reads `events.jsonl` from an offset,
  so a page that reconnects, or an API process that restarted, picks the stream up where it was.
- **Two runners behind one interface,** chosen by `JOB_RUNNER`:
  - `inline` (default outside compose, and in tests): the job runs in a thread of the API process as today, but
    writes its record and events, so the behaviour and the tests stay synchronous when they need to be.
  - `worker`: the API only writes the job as queued; a separate worker process (`python -m app.worker`, a second
    container in compose) claims it and runs it. Killing the API does not touch the search.
- **Claiming is an atomic rename** of the job's state file on the shared volume, so two workers cannot take the same
  job. A running job writes a heartbeat; a job whose heartbeat is stale (the worker died) is marked interrupted and,
  for an improvement search, resumed from its last completed round (#104); other kinds start again.
- **No new dependency, and no Redis in the job path.** RQ pickles callables and forks per job (and does not run
  on Windows, where this project is developed), Arq is async-only and barely maintained, Celery is far heavier
  than a handful of long jobs per deployment needs. The shared volume is already required for the data the jobs
  read and write, and a folder per job is something a person can read when a job goes wrong. The cost is that a
  deployment across several hosts would need a shared filesystem or a real queue; that is (c) territory.
- The service functions (`train_candidates`, `improve`, the apply path) do not change; only how they are called.

### The retention sweep (#105)

The sweep takes a lock file in `UPLOAD_DIR` (`O_CREAT | O_EXCL`, with an expiry so a crashed sweeper does not block
it for ever) before each pass, so two API processes, or an API and a worker, never sweep at once. With
`JOB_RUNNER=worker` the worker runs the sweep; the in-process sweep remains for the single-container setup.

## What `docker compose up` becomes

Three containers become four: `redis`, `backend` (API), **`worker`** (same image, `python -m app.worker`, same
volume, `JOB_RUNNER=worker`), `frontend`. Opening `http://localhost:5173` works as before: upload, clean, train,
with no setup beyond the `.env` the README already describes. What is new on the page: "Keep this project"
(name it, save the token shown once), the list of kept projects on the upload page, and "Delete project". A search
keeps running through `docker compose restart backend`. Running the backend alone, without compose, uses the inline
runner and behaves as today.

## What this does not include

Accounts, sharing between people, roles, object storage, a deployment across several hosts, scheduled retraining
and the deep-learning engine. The last two will be jobs of their own kinds when they are built, which is part of why
the job runner comes first.

## Backlog effect

| Issue | Effect |
| --- | --- |
| #120 Projects that are kept (new) | Keep, name, list, delete, rotate; tokens enforced on projects; sweep skips them; checkpoint TTL removed and deleted with the folder. |
| #103 Background jobs | Job records, inline and worker runners, streams read from the events file, worker container. |
| #104 Resume an interrupted search | Round state written after each round; a resumed job skips completed rounds, same seeds, the trail says where it resumed. |
| #105 Sweep out of the API process | Lock file; the worker sweeps when there is one. |
| #114 Threat review | Runs after these; covers project tokens, deletion and the local-disk statement as well as prompt injection. |

## What would change this

- Someone needing to share a project with another person: that is accounts, option (c).
- A deployment across several hosts: a real queue and object storage in place of the folder-based ones.
- Disk use by kept projects becoming a problem: a per-project size report and limits, not expiry, since the owner
  chose "kept until deleted".
