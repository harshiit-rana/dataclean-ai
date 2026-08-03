import { useEffect, useRef } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { buttonClasses } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ActionLog } from "@/components/clean/ActionLog";
import { useCleaningRun } from "@/hooks/useCleaningRun";
import { API_BASE_URL } from "@/api/client";

/** Page 3, PRD §12.3: runs the approved fixes and hands back the cleaned CSV.
 *
 * The cleaned file is generated server-side into the session folder and downloaded from
 * there -- the user's own file on their machine is never touched. */
export function CleanPage() {
  const { sessionId } = useParams();
  const location = useLocation();
  const run = useCleaningRun(sessionId ?? null);
  const startedRef = useRef(false);

  const decisions =
    (location.state as { decisions?: Record<string, "approved" | "rejected"> } | null)
      ?.decisions ?? null;

  useEffect(() => {
    // Guard against React StrictMode's double-effect firing the run twice.
    if (startedRef.current || !sessionId || !decisions) return;
    startedRef.current = true;
    run.start(decisions, {});
  }, [sessionId, decisions, run]);

  if (!decisions) {
    return (
      <div className="mx-auto flex max-w-2xl flex-col gap-4 p-8">
        <h1 className="text-2xl font-semibold">Cleaning</h1>
        <p className="text-muted-foreground">
          This page runs the fixes you approved on the review step. Start from there to clean
          a dataset.
        </p>
        <Link to="/" className="text-sm text-primary underline">
          Upload a CSV
        </Link>
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Applying your fixes</h1>
        <p className="text-muted-foreground">
          Only the fixes you approved are applied. Your original file is never modified --
          this creates a new, cleaned copy.
        </p>
      </div>

      {run.status === "error" && <p className="text-sm text-destructive">{run.error}</p>}

      <ActionLog actions={run.actions} status={run.status} />

      {run.status === "done" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Your cleaned data is ready</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4 text-sm">
            <p className="text-muted-foreground">
              {run.appliedCount} fix{run.appliedCount === 1 ? "" : "es"} applied
              {run.skippedCount > 0 && `, ${run.skippedCount} skipped`}.
            </p>
            <div className="flex flex-wrap gap-2">
              <Link to={`/export/${sessionId}`} className={buttonClasses()}>
                View report &amp; download
              </Link>
              <a
                href={`${API_BASE_URL}/api/export/${sessionId}/csv`}
                download
                className={buttonClasses("outline")}
              >
                Cleaned CSV only
              </a>
              <Link to="/" className={buttonClasses("ghost")}>
                Clean another file
              </Link>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
