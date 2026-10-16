# Local storage: what is on disk, who can read it, and what deletion does

Written for #120 under the platform decision (`docs/decisions/0004-platform-architecture.md`). The threat review
(#114) builds on it.

## What is stored, and where

One folder, `UPLOAD_DIR` (default `./uploads`; the `backend_uploads` volume under Docker Compose), holds a
sub-folder per session, named by a server-generated UUID:

| File or folder | What it holds |
| --- | --- |
| `original.csv` | The uploaded file, byte for byte. |
| `cleaned.csv`, dataset versions | The data after each cleaning step. |
| `runs/<run_id>/` | Each trained model: the fitted pipeline (`model.joblib`), its result, held-out predictions. |
| `meta.json` | The goal, the business rules, the dataset versions, the project name, and the **SHA-256 hash** of the access token (never the token). |
| audit reports, model cards, exports | Generated documents, which quote column names and values. |

The LangGraph state (the cleaning conversation, approvals, the action log) lives in Redis, persisted append-only to
the `redis_data` volume.

## Who can read it

- **Through the app:** only a request carrying the session's token, when tokens are enforced. A kept project always
  requires its token, whatever `APP_ENV` says; an ordinary session requires it unless development mode turns
  enforcement off (the app logs a warning at every start when it does).
- **On the machine: anyone with access to the files.** Nothing is encrypted at rest. The token protects the HTTP
  interface, not the disk: an administrator, another process running as the same user, a container with the volume
  mounted, or anyone holding a backup or snapshot of the volume can read every session and project, including the
  data in Redis's append-only file.
- The token itself is not on disk, only its hash, so reading `meta.json` does not let someone into the app as the
  project's owner. It does not need to: they can read the data directly.

## What deletion does

`DELETE /api/session/{id}` ("Delete project" on the page) removes the session's checkpoint thread in Redis and then
the session folder with every file in it. The sweep does the same for an ordinary session past its retention
period, and never for a kept project.

- Files are **unlinked, not securely erased.** Their blocks stay on the disk until reused, and can be recovered with
  forensic tools from an unencrypted disk.
- **Backups and snapshots keep their copies.** Deleting a project does not reach into them.
- Redis rewrites its append-only file periodically; until it does, deleted checkpoint data can remain in the file.

## What a deployment should do

- Put `UPLOAD_DIR` and the Redis volume on an **encrypted volume** (LUKS, BitLocker, an encrypted cloud disk).
  This is what makes the "not securely erased" point moot.
- Restrict the folder to the user the app runs as (`chmod 700` or the equivalent).
- Treat **backups as holding users' data**: encrypt them, limit who can restore them, and expire them on a schedule
  you can state to users.
- Never run with development mode (`APP_ENV=development`) on a machine other people can reach: ordinary sessions are
  open there to anyone who learns an id.
- The app never loads a model file supplied by a user; `model.joblib` files are written by the app itself. A pickle
  is code, so anyone who can write to `UPLOAD_DIR` can make the app run code when it loads a model. Write access to
  the folder must be as restricted as read access.
