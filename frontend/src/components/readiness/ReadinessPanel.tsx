import { useCallback, useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ApiError, fetchReadiness } from "@/api/client";
import type { ReadinessResponse } from "@/api/types";

interface ReadinessPanelProps {
  sessionId: string;
  /** Bump to re-fetch — an undo or a conversational edit changes the file underneath. */
  refreshKey?: number;
}

const VERDICTS: Record<string, { title: string; blurb: string; className: string }> = {
  ready: {
    title: "Ready for modelling",
    blurb: "Nothing found that would stop you training a model on this data.",
    className: "border-emerald-500/40 bg-emerald-500/5",
  },
  warnings: {
    title: "Usable, with caveats",
    blurb: "Nothing blocking — but read these before you trust a score from this data.",
    className: "border-amber-500/40 bg-amber-500/5",
  },
  blocked: {
    title: "Not ready yet",
    blurb: "These need resolving before this data can support the model you asked for.",
    className: "border-destructive/40 bg-destructive/5",
  },
};

/** Answers "can I actually build the model I said I wanted from this?" against the data as
 * it stands right now.
 *
 * Renders nothing at all when the session has no prediction target — someone cleaning a file
 * for a report shouldn't be shown a modelling verdict they never asked for. */
export function ReadinessPanel({ sessionId, refreshKey = 0 }: ReadinessPanelProps) {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    fetchReadiness(sessionId)
      .then((data) => {
        setReadiness(data);
        setError(null);
      })
      .catch((err) =>
        setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err))
      );
  }, [sessionId]);

  useEffect(load, [load, refreshKey]);

  if (error) {
    return (
      <p className="text-xs text-muted-foreground">
        Model readiness couldn't be checked: {error}
      </p>
    );
  }
  if (!readiness || readiness.verdict === "not_applicable") return null;

  const verdict = VERDICTS[readiness.verdict] ?? VERDICTS.warnings;
  const { remaining_issues: remaining } = readiness;

  return (
    <Card className={cn("border", verdict.className)}>
      <CardHeader>
        <CardTitle className="text-base">{verdict.title}</CardTitle>
        <p className="text-sm text-muted-foreground">{verdict.blurb}</p>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 text-sm">
        <p className="text-muted-foreground">
          Predicting <span className="font-medium text-foreground">{readiness.target_column}</span>
          {readiness.task_type && ` (${readiness.task_type})`} from{" "}
          {readiness.row_count.toLocaleString()} rows and {Math.max(0, readiness.column_count - 1)}{" "}
          other columns.
        </p>

        {readiness.blockers.length > 0 && (
          <div>
            <p className="font-medium">Blocking</p>
            <ul className="mt-1 list-disc pl-5 text-muted-foreground">
              {readiness.blockers.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {readiness.warnings.length > 0 && (
          <div>
            <p className="font-medium">Worth knowing</p>
            <ul className="mt-1 list-disc pl-5 text-muted-foreground">
              {readiness.warnings.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
        )}

        {readiness.notes.map((note) => (
          <p key={note} className="text-muted-foreground">
            {note}
          </p>
        ))}

        {remaining.total > 0 && (
          <p className="text-xs text-muted-foreground">
            {remaining.total} issue{remaining.total === 1 ? "" : "s"} still present in the data
            {remaining.high > 0 && ` (${remaining.high} high severity)`}. You can keep cleaning
            with the box above.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
