import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { buttonClasses } from "@/components/ui/button";
import { DownloadButton } from "@/components/ui/download-button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ActionLog } from "@/components/clean/ActionLog";
import { AppliedActions } from "@/components/clean/AppliedActions";
import { ChatPanel } from "@/components/clean/ChatPanel";
import { ReadinessPanel } from "@/components/readiness/ReadinessPanel";
import { RulesSection } from "@/components/rules/RulesSection";
import { useCleaningRun } from "@/hooks/useCleaningRun";
import { fetchActionLog, type ActionLogEntry } from "@/api/client";

/** Page 3, PRD §12.3: runs the approved fixes and hands back the cleaned CSV.
 *
 * The cleaned file is generated server-side into the session folder and downloaded from
 * there -- the user's own file on their machine is never touched. */
export function CleanPage() {
  const { sessionId } = useParams();
  const location = useLocation();
  const run = useCleaningRun(sessionId ?? null);
  const startedRef = useRef(false);
  const [actionLog, setActionLog] = useState<ActionLogEntry[] | null>(null);
  // Undo and chat both rewrite cleaned.csv, so the readiness verdict has to be recomputed
  // whenever the action log changes rather than only once when the run finishes.
  const [readinessKey, setReadinessKey] = useState(0);
  // Whether this page is showing a run it started, or one it recovered from the server
  // after a reload. "unknown" until the lookup has been attempted.
  const [recovery, setRecovery] = useState<"unknown" | "loading" | "found" | "nothing">(
    "unknown"
  );

  const handleActionsChanged = useCallback((actions: ActionLogEntry[]) => {
    setActionLog(actions);
    setReadinessKey((key) => key + 1);
  }, []);

  // Counted from the persisted log once there is one, because undo and the chat box both
  // change what has been applied after the streamed run finished -- the run's own tallies
  // stop being true the moment either is used.
  const counts = useMemo(() => {
    if (actionLog === null) {
      return {
        applied: run.appliedCount,
        skipped: run.skippedCount,
        superseded: run.supersededCount,
      };
    }
    return {
      applied: actionLog.filter((a) => a.status === "applied").length,
      skipped: actionLog.filter((a) => a.status === "rejected").length,
      superseded: actionLog.filter((a) => a.status === "superseded").length,
    };
  }, [actionLog, run.appliedCount, run.skippedCount, run.supersededCount]);

  const navState = location.state as {
    decisions?: Record<string, "approved" | "rejected">;
    skipReasons?: Record<string, string>;
  } | null;
  const decisions = navState?.decisions ?? null;
  // Memoized so the `?? {}` fallback doesn't hand the effect a new object every render.
  const skipReasons = useMemo(() => navState?.skipReasons ?? {}, [navState]);

  useEffect(() => {
    // Guard against React StrictMode's double-effect firing the run twice.
    if (startedRef.current || !sessionId || !decisions) return;
    startedRef.current = true;
    run.start(decisions, skipReasons);
  }, [sessionId, decisions, skipReasons, run]);

  // Once the run finishes, swap the live view for the persisted log so fixes can be undone.
  useEffect(() => {
    if (run.status !== "done" || !sessionId || actionLog !== null) return;
    fetchActionLog(sessionId)
      .then((data) => setActionLog(data.actions))
      .catch(() => setActionLog(null)); // keep the live view rather than blanking the page
  }, [run.status, sessionId, actionLog]);

  // Recovering a session that was already cleaned.
  //
  // `decisions` arrives in navigation state, so it is gone the moment the page is reloaded --
  // and this page used to answer that with "start from the review step", stranding a user who
  // had merely pressed refresh. Everything it shows is persisted server-side, so the honest
  // response is to fetch it. This became a functional gap rather than a cosmetic one when the
  // business rules panel moved here: a reload left approved rules with no route back to them.
  useEffect(() => {
    if (decisions || !sessionId || actionLog !== null || recovery !== "unknown") return;
    setRecovery("loading");
    fetchActionLog(sessionId)
      .then((data) => {
        setActionLog(data.actions.length > 0 ? data.actions : null);
        setRecovery(data.actions.length > 0 ? "found" : "nothing");
      })
      .catch(() => setRecovery("nothing"));
  }, [decisions, sessionId, actionLog, recovery]);

  // The run either just finished in this tab, or finished earlier and was recovered.
  const finished = run.status === "done" || recovery === "found";

  if (!decisions && recovery !== "found") {
    if (recovery === "unknown" || recovery === "loading") {
      return (
        <div className="mx-auto flex max-w-2xl flex-col gap-4 p-8">
          <p className="text-muted-foreground">Looking for this session...</p>
        </div>
      );
    }
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
        <h1 className="text-2xl font-semibold">
          {recovery === "found" ? "Your cleaned data" : "Applying your fixes"}
        </h1>
        <p className="text-muted-foreground">
          {recovery === "found"
            ? "Picking up where you left off. Everything below is read from this session, so a reload doesn't lose it."
            : "Only the fixes you approved are applied. Your original file is never modified -- this creates a new, cleaned copy."}
        </p>
      </div>

      {run.status === "error" && <p className="text-sm text-destructive">{run.error}</p>}

      {/* Live progress while the run streams, then the persisted log with undo controls. */}
      {actionLog === null ? (
        <ActionLog actions={run.actions} status={run.status} />
      ) : (
        <AppliedActions
          sessionId={sessionId!}
          actions={actionLog}
          onChange={handleActionsChanged}
        />
      )}

      {finished && sessionId && (
        <RulesSection
          sessionId={sessionId}
          refreshKey={readinessKey}
          onRulesChanged={() => setReadinessKey((key) => key + 1)}
        />
      )}

      {finished && sessionId && (
        <ReadinessPanel sessionId={sessionId} refreshKey={readinessKey} canKeepCleaning />
      )}

      {finished && sessionId && (
        <ChatPanel sessionId={sessionId} onActionsChanged={handleActionsChanged} />
      )}

      {finished && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Your cleaned data is ready</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4 text-sm">
            <p className="text-muted-foreground">
              {counts.applied} fix{counts.applied === 1 ? "" : "es"} applied
              {counts.skipped > 0 && `, ${counts.skipped} skipped`}
              {counts.superseded > 0 &&
                `, ${counts.superseded} no longer needed once the others ran`}
              .
            </p>
            <div className="flex flex-wrap gap-2">
              <Link to={`/export/${sessionId}`} className={buttonClasses()}>
                View report &amp; download
              </Link>
              <Link to={`/model/${sessionId}`} className={buttonClasses("outline")}>
                Build a model
              </Link>
              <DownloadButton
                sessionId={sessionId!}
                path={`/api/export/${sessionId}/csv`}
                filename="cleaned.csv"
                variant="outline"
              >
                Cleaned CSV only
              </DownloadButton>
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
