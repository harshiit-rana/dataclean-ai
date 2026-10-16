import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import type { ImprovementRound } from "@/api/types";

interface LiveImprovementProps {
  /** Rounds finished so far, in order. */
  rounds: ImprovementRound[];
  /** What it is doing right now, or null before the first word from the server. */
  stage: string | null;
}

/** The improvement loop, while it runs.
 *
 * This used to be a paragraph of explanation next to a spinner for anything from twenty
 * seconds to a few minutes. The loop already reported each round as it finished; now the
 * page shows them, and says what it is on as each one starts -- the settings search alone can
 * outlast anyone's patience with a screen that isn't moving.
 *
 * Rejected rounds are shown as they land, not tidied away. "Tuning didn't help" arriving live
 * is the same finding as it is in the final trail, just sooner. */
export function LiveImprovement({ rounds, stage }: LiveImprovementProps) {
  const latest = rounds[rounds.length - 1];
  return (
    <Card>
      {/* One short announcement per finished round, not the whole card on every change: the stage line updates
          often, and a live region around it all would read the page aloud over and over (#101). */}
      <p role="status" aria-live="polite" className="sr-only">
        {latest ? `Round finished: ${latest.label}, ${latest.accepted ? "kept" : "not kept"}.` : ""}
      </p>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base">Searching for a better model</CardTitle>
          <ScoreLabel kind="exploratory" />
        </div>
        <p className="text-sm text-muted-foreground">
          Every round is judged on cross-validated folds of the training rows. The test set
          stays sealed until the very end.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-sm">
        {rounds.map((round, index) => (
          <div
            key={`${round.name}-${index}`}
            className={cn(
              "rounded-md border p-3",
              round.accepted ? "border-primary/40 bg-primary/5" : "border-border"
            )}
          >
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <span className="font-medium">
                {round.label}
                {/* The highlight says it in colour; this says it in words. */}
                {round.accepted && <span className="ml-2 text-xs font-normal text-primary">kept</span>}
              </span>
              <span className="tabular-nums text-muted-foreground">
                {round.score != null ? round.score.toFixed(3) : "—"}
                {round.gain != null && (
                  <span className={cn("ml-2", round.accepted ? "text-primary" : "")}>
                    {round.gain >= 0 ? "+" : ""}
                    {round.gain.toFixed(3)}
                  </span>
                )}
              </span>
            </div>
            <p className="mt-1">{round.outcome}</p>
          </div>
        ))}

        <div className="flex items-center gap-3 rounded-md border border-dashed border-border p-3 text-muted-foreground">
          <span
            className="inline-block h-3 w-3 shrink-0 animate-pulse rounded-full bg-primary/60"
            aria-hidden="true"
          />
          <span>{stage ?? "Starting…"}</span>
        </div>
      </CardContent>
    </Card>
  );
}
