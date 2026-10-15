import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { buttonClasses } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CheckTrail } from "@/components/readiness/CheckTrail";
import { cn } from "@/lib/utils";
import { ApiError, fetchReadiness } from "@/api/client";
import type { ReadinessResponse } from "@/api/types";

interface ReadinessPanelProps {
  sessionId: string;
  /** Bump to re-fetch — an undo or a conversational edit changes the file underneath. */
  refreshKey?: number;
  /** Offer the link on to model building. Off on the model page itself. */
  showModelLink?: boolean;
  /** Whether this page has the conversational cleaning box. Only the clean page does, and
   * pointing at a box that isn't on screen is worse than not mentioning one. */
  canKeepCleaning?: boolean;
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
export function ReadinessPanel({
  sessionId,
  refreshKey = 0,
  showModelLink = true,
  canKeepCleaning = false,
}: ReadinessPanelProps) {
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
          {/* The rows that carry an answer, not the rows in the file. Where they differ the
              gap is stated rather than hidden -- it is usually the most important fact on
              this panel. */}
          {(readiness.trainable_row_count ?? readiness.row_count).toLocaleString()} rows
          {readiness.trainable_row_count != null &&
            readiness.trainable_row_count !== readiness.row_count &&
            ` of ${readiness.row_count.toLocaleString()}`}{" "}
          and {Math.max(0, readiness.column_count - 1)} other columns.
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

        {/* The verdict above is a claim about what *isn't* wrong, which nobody can confirm
            by looking at the data. This is the working behind it. */}
        <CheckTrail checks={readiness.checks ?? []} counts={readiness.check_counts} sessionId={sessionId} />


        {remaining.total > 0 && (
          <p className="text-xs text-muted-foreground">
            {remaining.total} issue{remaining.total === 1 ? "" : "s"} still present in the data
            {remaining.high > 0 && ` (${remaining.high} high severity)`}.
            {canKeepCleaning
              ? " Describe another change below to keep cleaning."
              : " Go back to the cleaning step to keep working on them."}
          </p>
        )}

        {/* Offered even when there are blockers: the model page names them and explains what
            to do, which is more use than a disabled button with no reason attached. */}
        {showModelLink && (
          <div className="flex flex-wrap items-center gap-2 pt-1">
            <Link to={`/model/${sessionId}`} className={buttonClasses("default", "sm")}>
              {readiness.verdict === "blocked" ? "See what's blocking a model" : "Build a model"}
            </Link>
            <span className="text-xs text-muted-foreground">
              Trains on your cleaned data. Nothing about the data changes.
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
