import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import type { ImprovementTrail as Trail } from "@/api/types";

const METRIC_LABELS: Record<string, string> = {
  roc_auc: "AUC",
  accuracy: "accuracy",
  r2: "R²",
};

const STOPPED: Record<Trail["stopped_because"], { title: string; blurb: string; tone: string }> = {
  goal_reached: {
    title: "Reached your target",
    blurb: "It stopped as soon as it got there rather than carrying on -- more searching " +
      "after the goal is met buys nothing and risks tuning to the folds.",
    tone: "border-emerald-500/40 bg-emerald-500/5",
  },
  goal_out_of_reach: {
    title: "Couldn't reach your target",
    blurb: "It ran out of things worth trying before getting there. That is a statement " +
      "about the data, not the search: no configuration of these columns gets that far. " +
      "Better inputs would help; more searching would not.",
    tone: "border-amber-500/40 bg-amber-500/5",
  },
  no_further_gains: {
    title: "Stopped when the rounds stopped paying",
    blurb: "Each round has to beat the one before it by more than the folds disagree with " +
      "each other. Once nothing could, carrying on would only be fitting to the folds.",
    tone: "border-border bg-muted/40",
  },
};

function formatScore(value: number | null, metric: string): string {
  if (value === null) return "—";
  return metric === "r2" ? value.toFixed(3) : value.toFixed(3);
}

/** What the improvement loop tried, in order, including what didn't work.
 *
 * Showing the rejected rounds is the point of the panel. A loop that only reported its wins
 * would look like it improved things every time, and the useful signal -- "tuning this data
 * doesn't help, the ceiling is the columns you have" -- would be invisible. */
export function ImprovementTrail({ trail }: { trail: Trail }) {
  const metric = METRIC_LABELS[trail.metric] ?? trail.metric;
  const stopped = STOPPED[trail.stopped_because];
  const settings = Object.entries(trail.settings);

  return (
    <Card>
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0">
        <CardTitle className="text-base">What it tried</CardTitle>
        <ScoreLabel kind="exploratory" />
      </CardHeader>
      <CardContent className="flex flex-col gap-4 text-sm">
        <p className="text-muted-foreground">
          Every round was scored by cross-validation on the training rows. The test set stayed
          sealed until the end, so the headline number above is measured on rows that nothing
          here was chosen against.
        </p>

        <ol className="flex flex-col gap-2">
          {trail.rounds.map((round, index) => (
            <li
              key={`${round.name}-${index}`}
              className={cn(
                "rounded-md border p-3",
                round.accepted ? "border-primary/40 bg-primary/5" : "border-border"
              )}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium">{round.label}</span>
                <span className="tabular-nums text-muted-foreground">
                  {formatScore(round.score, trail.metric)}
                  {round.gain !== null && round.gain !== undefined && (
                    <span
                      className={cn(
                        "ml-2",
                        round.accepted ? "text-primary" : "text-muted-foreground"
                      )}
                    >
                      {round.gain >= 0 ? "+" : ""}
                      {round.gain.toFixed(3)}
                    </span>
                  )}
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{round.what_it_tried}</p>
              <p className="mt-1">{round.outcome}</p>
            </li>
          ))}
        </ol>

        <div className={cn("rounded-md border p-3", stopped.tone)}>
          <p className="font-medium">{stopped.title}</p>
          <p className="mt-1 text-muted-foreground">{stopped.blurb}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Best cross-validated {metric}: {trail.cv_score.toFixed(3)} (±
            {trail.cv_std.toFixed(3)} across folds)
            {trail.goal !== null && ` · your target was ${trail.goal.toFixed(3)}`}
          </p>
        </div>

        {settings.length > 0 && (
          <div>
            <p className="font-medium">Settings it settled on</p>
            <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
              {settings.map(([key, value]) => (
                <div key={key} className="contents">
                  <dt className="text-muted-foreground">{key}</dt>
                  <dd className="tabular-nums">{String(value)}</dd>
                </div>
              ))}
            </dl>
            <p className="mt-2 text-xs text-muted-foreground">
              These are written into the downloadable script, so it reproduces this model
              rather than the defaults.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
