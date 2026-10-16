import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  ApiError,
  deleteSession,
  fetchProject,
  keepProject,
  rotateProjectToken,
  type ProjectInfo,
} from "@/api/client";
import { forgetSession, getSessionToken, rememberKeptProject, replaceSessionToken } from "@/lib/sessionToken";

/** Keep the work in progress as a project, until it is deleted (#120, docs/decisions/0004).
 *
 * An ordinary session is deleted a day after it was last used. Keeping it names it, keeps it
 * until the owner deletes it, and from then on its key is always required. The key is shown
 * once, to save: it is the only way back in from another browser, and there is no recovery,
 * because recovery is an account feature and there are no accounts. */
type Mode = "idle" | "naming" | "deleting";

function KeyNotice({ token, onDone }: { token: string; onDone: () => void }) {
  const [copied, setCopied] = useState(false);
  return (
    <div role="status" className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
      <p className="font-medium">Save this project key now.</p>
      <p className="mt-1 text-muted-foreground">
        It is the only way back into this project from another browser, and it will not be shown
        again. Anyone who has it can open the project.
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <code className="break-all rounded bg-muted px-2 py-1 text-xs" aria-label="Project key">
          {token}
        </code>
        <Button
          size="sm"
          variant="outline"
          onClick={() => {
            navigator.clipboard?.writeText(token).then(() => setCopied(true), () => setCopied(false));
          }}
        >
          {copied ? "Copied" : "Copy key"}
        </Button>
        <Button size="sm" variant="ghost" onClick={onDone}>
          I've saved it
        </Button>
      </div>
    </div>
  );
}

export function ProjectBar() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [project, setProject] = useState<ProjectInfo | null>(null);
  const [mode, setMode] = useState<Mode>("idle");
  const [name, setName] = useState("");
  const [confirmName, setConfirmName] = useState("");
  const [shownKey, setShownKey] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    setProject(null);
    setMode("idle");
    setShownKey(null);
    fetchProject(sessionId).then(setProject).catch(() => setProject(null));
  }, [sessionId]);

  if (!sessionId || !project) return null;

  const fail = (err: unknown) => setError(err instanceof ApiError ? err.detail : "That didn't work. Try again.");

  const keep = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const kept = await keepProject(sessionId, name);
      const token = kept.token ?? getSessionToken(sessionId);
      if (token) rememberKeptProject({ id: sessionId, name: kept.name ?? name, keptAt: kept.kept_at }, token);
      // A newly kept project shows its key once. A rename does not.
      if (!project.kept && token) setShownKey(token);
      setProject(kept);
      setMode("idle");
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const rotate = async () => {
    setBusy(true);
    setError(null);
    try {
      const { token } = await rotateProjectToken(sessionId);
      replaceSessionToken(sessionId, token);
      setShownKey(token);
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await deleteSession(sessionId);
      forgetSession(sessionId);
      navigate("/");
    } catch (err) {
      fail(err);
      setBusy(false);
    }
  };

  const label = project.kept ? project.name ?? "Untitled project" : null;
  const expected = project.kept ? project.name ?? "" : "delete";

  return (
    <section aria-label="Project" className="mx-auto max-w-5xl px-8 pt-4 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        {project.kept ? (
          <>
            <span>
              Project: <strong>{label}</strong>
              <span className="text-muted-foreground"> · kept until you delete it</span>
            </span>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => { setName(project.name ?? ""); setMode("naming"); }}>
              Rename
            </Button>
            <Button size="sm" variant="outline" disabled={busy} onClick={rotate}>
              New key
            </Button>
          </>
        ) : (
          <>
            <span className="text-muted-foreground">
              This work is deleted a day after you last use it.
            </span>
            <Button size="sm" variant="outline" disabled={busy} onClick={() => setMode("naming")}>
              Keep this project
            </Button>
          </>
        )}
        <Button size="sm" variant="ghost" disabled={busy} onClick={() => { setConfirmName(""); setMode("deleting"); }}>
          {project.kept ? "Delete project" : "Delete now"}
        </Button>
      </div>

      {mode === "naming" && (
        <form onSubmit={keep} className="mt-2 flex flex-wrap items-end gap-2">
          <div className="flex flex-col gap-1">
            <Label htmlFor="project-name">Project name</Label>
            <Input
              id="project-name"
              value={name}
              maxLength={100}
              onChange={(e) => setName(e.target.value)}
              className="w-72"
              autoFocus
            />
          </div>
          <Button type="submit" size="sm" disabled={busy || !name.trim()}>
            {project.kept ? "Save name" : "Keep it"}
          </Button>
          <Button type="button" size="sm" variant="ghost" onClick={() => setMode("idle")}>
            Cancel
          </Button>
        </form>
      )}

      {mode === "deleting" && (
        <form onSubmit={remove} className="mt-2 rounded-md border border-destructive/40 bg-destructive/5 p-3">
          <p>
            This deletes the uploaded file, every cleaned version, every model and report, and the
            saved progress, for good. Files are removed from this server's disk, but copies in any
            backups of it remain.
          </p>
          <div className="mt-2 flex flex-wrap items-end gap-2">
            <div className="flex flex-col gap-1">
              <Label htmlFor="confirm-delete">
                Type <strong>{expected}</strong> to confirm
              </Label>
              <Input
                id="confirm-delete"
                value={confirmName}
                onChange={(e) => setConfirmName(e.target.value)}
                className="w-72"
                autoFocus
              />
            </div>
            <Button type="submit" size="sm" variant="destructive" disabled={busy || confirmName.trim() !== expected}>
              Delete for good
            </Button>
            <Button type="button" size="sm" variant="ghost" onClick={() => setMode("idle")}>
              Cancel
            </Button>
          </div>
        </form>
      )}

      {shownKey && <KeyNotice token={shownKey} onDone={() => setShownKey(null)} />}
      {error && <p role="alert" className="mt-2 text-destructive">{error}</p>}
    </section>
  );
}
