import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { buttonClasses, Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, fetchProject } from "@/api/client";
import { listKeptProjects, rememberKeptProject, forgetSession } from "@/lib/sessionToken";

/** The projects this browser holds keys for, and a way in with a saved key (#120).
 *
 * There are no accounts, so the server cannot list "your" projects: the list is what this
 * browser remembers. A project opened elsewhere comes back with its id and key. */
export function KeptProjects() {
  const navigate = useNavigate();
  const [projects, setProjects] = useState(listKeptProjects);
  const [opening, setOpening] = useState(false);
  const [id, setId] = useState("");
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);

  const open = async (event: React.FormEvent) => {
    event.preventDefault();
    const sessionId = id.trim();
    const token = key.trim();
    setError(null);
    // Remembered first so the check below carries the key; forgotten again if it is refused.
    rememberKeptProject({ id: sessionId, name: "Opening…", keptAt: null }, token);
    try {
      const project = await fetchProject(sessionId);
      rememberKeptProject({ id: sessionId, name: project.name ?? project.file_name ?? sessionId, keptAt: project.kept_at }, token);
      navigate(`/model/${sessionId}`);
    } catch (err) {
      forgetSession(sessionId);
      setProjects(listKeptProjects());
      setError(
        err instanceof ApiError && (err.status === 403 || err.status === 404)
          ? "No project matches that id and key."
          : "The project couldn't be opened. Try again."
      );
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Your kept projects</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {projects.length === 0 ? (
          <p className="text-muted-foreground">
            None in this browser. Work you keep as a project stays until you delete it.
          </p>
        ) : (
          <ul className="flex flex-col gap-2">
            {projects.map((p) => (
              <li key={p.id} className="flex flex-wrap items-center justify-between gap-2">
                <span>
                  <strong>{p.name}</strong>
                  {p.keptAt && <span className="text-muted-foreground"> · kept {p.keptAt.slice(0, 10)}</span>}
                </span>
                <Link to={`/model/${p.id}`} className={buttonClasses("outline", "sm")} aria-label={`Open ${p.name}`}>
                  Open
                </Link>
              </li>
            ))}
          </ul>
        )}
        {opening ? (
          <form onSubmit={open} className="flex flex-col gap-2">
            <div className="flex flex-col gap-1">
              <Label htmlFor="open-project-id">Project id</Label>
              <Input id="open-project-id" value={id} onChange={(e) => setId(e.target.value)} autoFocus />
            </div>
            <div className="flex flex-col gap-1">
              <Label htmlFor="open-project-key">Project key</Label>
              <Input id="open-project-key" type="password" value={key} onChange={(e) => setKey(e.target.value)} />
            </div>
            <div className="flex gap-2">
              <Button type="submit" size="sm" disabled={!id.trim() || !key.trim()}>
                Open project
              </Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => setOpening(false)}>
                Cancel
              </Button>
            </div>
          </form>
        ) : (
          <div>
            <Button size="sm" variant="ghost" onClick={() => setOpening(true)}>
              Open a project with its key
            </Button>
          </div>
        )}
        {error && <p role="alert" className="text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
