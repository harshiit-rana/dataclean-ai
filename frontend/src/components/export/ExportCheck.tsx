import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { ApiError, fetchExportVerification, verifyExport } from "@/api/client";
import type { ExportAcceptance } from "@/api/types";
import { cn } from "@/lib/utils";

/** Whether the exported scripts rebuild this session's model -- checked on request, because it
 * retrains the model, and never something the download waits for. */
export function ExportCheck({ sessionId }: { sessionId: string }) {
  const [check, setCheck] = useState<ExportAcceptance | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExportVerification(sessionId)
      .then(setCheck)
      .catch(() => setCheck(null)); // no model yet, or nothing checked: the button still offers it
  }, [sessionId]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setCheck(await verifyExport(sessionId));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setRunning(false);
    }
  };

  const checked = check && check.status !== "not_checked";
  return (
    <div className="flex flex-col gap-2 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" onClick={run} disabled={running}>
          {running ? "Rebuilding the model from the export…" : "Check the export rebuilds the model"}
        </Button>
        <span className="text-xs text-muted-foreground">
          Runs clean.py and train_model.py on your upload and compares every held-out prediction.
        </span>
      </div>
      {checked && (
        <p
          className={cn(
            "text-sm",
            check.status === "passed" ? "text-emerald-700 dark:text-emerald-400" : "text-amber-700 dark:text-amber-500"
          )}
        >
          {check.message}
        </p>
      )}
      {checked && check.first_differences && check.first_differences.length > 0 && (
        <ul className="list-disc pl-5 text-xs text-muted-foreground">
          {check.first_differences.map((d) => (
            <li key={d.row}>
              row {d.row}: {d.difference}
            </li>
          ))}
        </ul>
      )}
      {checked && check.probes && check.probes.disagreements.length > 0 && (
        <p className="text-xs text-muted-foreground">
          Probe rows where they disagree: {check.probes.disagreements.map((d) => d.probe).join("; ")}.
        </p>
      )}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}
