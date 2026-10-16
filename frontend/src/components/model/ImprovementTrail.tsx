import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import type { ImprovementTrail as Trail } from "@/api/types";

const STRATEGY_LABELS: Record<string, string> = {
  random: "random search",
  halving: "successive halving",
};

interface KeptCombination {
  id: string;
  output: string;
  gain: number;
  search_score: number;
  confirmation_score: number;
  incumbent_confirmation_score: number;
}

/** The combinations an interactions round kept, with the fresh-fold check that earned each its place. */
function keptCombinationsOf(detail: Record<string, unknown> | undefined): KeptCombination[] {
  const kept = detail?.kept;
  return Array.isArray(kept) ? (kept as KeptCombination[]) : [];
}

interface RemovedInput {
  column: string;
  importance: number;
  spread: number;
}

/** The inputs a feature-selection round took out, with how little each mattered. */
function removedOf(detail: Record<string, unknown> | undefined): RemovedInput[] {
  const removed = detail?.removed;
  return Array.isArray(removed) ? (removed as RemovedInput[]) : [];
}

interface Rung {
  rung: number;
  settings: number;
  folds: number;
  best_score: number | null;
  kept: number;
}

/** The rungs of a halving search, when the round has them: how many settings were scored on how many
 * folds, and how many went on. Read defensively because the detail is free-form. */
function rungsOf(detail: Record<string, unknown> | undefined): Rung[] {
  const rungs = detail?.rungs;
  return Array.isArray(rungs) ? (rungs as Rung[]) : [];
}

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
  time_limit: {
    title: "Stopped at your time limit",
    blurb: "The search ran out of time, so what is shown is the best it found by then, and the " +
      "rounds marked not run were never tried. A longer limit might find more, or might not. " +
      "Scoring the result once on the held-out rows still happened afterwards.",
    tone: "border-sky-500/40 bg-sky-500/5",
  },
};

/** What searched the settings, by name: the strategies the searches used if the run recorded them (which
 * is what "auto" resolved to), else what was asked for. Empty for a run from before it was recorded. */
function strategyUsed(trail: Trail): string {
  const names = trail.search_strategies?.length
    ? trail.search_strategies
    : trail.search_strategy && trail.search_strategy !== "auto"
      ? [trail.search_strategy]
      : [];
  return names.map((name) => STRATEGY_LABELS[name] ?? name).join(" and ");
}

interface Confirmation {
  status: string;
  search_score: number;
  confirmation_score: number;
  incumbent_confirmation_score: number;
}

/** The fresh-fold check of a search's winner, when the round made one and got a score. */
function confirmationOf(detail: Record<string, unknown> | undefined): Confirmation | null {
  const found = detail?.confirmation as Partial<Confirmation> | undefined;
  return found && typeof found.confirmation_score === "number" && typeof found.incumbent_confirmation_score === "number"
    ? (found as Confirmation)
    : null;
}

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
  // Members of an ensemble and cost weights are described by their own rounds; the list is the plain settings.
  const settings = Object.entries(trail.settings).filter(
    ([key, value]) => key !== "members" && key !== "cost_weighting" && (value === null || typeof value !== "object")
  );

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

        {trail.resumed && (
          <p className="rounded-md border border-sky-500/40 bg-sky-500/5 p-3">
            This search was interrupted and carried on where it stopped. The {trail.resumed.reused} cross-validations
            and searches it had finished were restored from its journal rather than run again
            {trail.resumed.diverged_at !== null
              ? ", until one no longer matched what had been recorded; everything from there was run afresh."
              : "; the result is the one an uninterrupted run gives."}
          </p>
        )}

        <ol className="flex flex-col gap-2">
          {trail.rounds.map((round, index) => {
            const notRun = round.detail?.skipped === "time_limit";
            return (
            <li
              key={`${round.name}-${index}`}
              className={cn(
                "rounded-md border p-3",
                notRun
                  ? "border-dashed border-border bg-muted/30 text-muted-foreground"
                  : round.accepted
                    ? "border-primary/40 bg-primary/5"
                    : "border-border"
              )}
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium">
                  {round.label}
                  {notRun && (
                    <span className="ml-2 text-xs uppercase tracking-wide">Not run: time limit</span>
                  )}
                </span>
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
              {rungsOf(round.detail).length > 0 && (
                <ol className="mt-2 flex flex-col gap-0.5 text-xs text-muted-foreground" aria-label="Rungs">
                  {rungsOf(round.detail).map((rung) => (
                    <li key={rung.rung} className="tabular-nums">
                      Rung {rung.rung}: {rung.settings} setting{rung.settings === 1 ? "" : "s"} on{" "}
                      {rung.folds} fold{rung.folds === 1 ? "" : "s"}
                      {rung.best_score !== null && `, best ${rung.best_score.toFixed(3)}`}
                      {rung.kept > 0 && ` → ${rung.kept} went on`}
                    </li>
                  ))}
                </ol>
              )}
              {typeof round.detail?.strategy_reason === "string" && !round.detail?.note && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Searched by{" "}
                  {STRATEGY_LABELS[String(round.detail.strategy)] ?? String(round.detail.strategy)}:{" "}
                  {round.detail.strategy_reason}.
                </p>
              )}
              {keptCombinationsOf(round.detail).length > 0 && (
                <ul className="mt-2 flex flex-col gap-0.5 text-xs text-muted-foreground" aria-label="Kept combinations">
                  {keptCombinationsOf(round.detail).map((kept) => (
                    <li key={kept.id} className="tabular-nums">
                      {kept.output}: +{kept.gain.toFixed(3)} on the training folds; on fresh folds{" "}
                      {kept.confirmation_score.toFixed(3)} against {kept.incumbent_confirmation_score.toFixed(3)}
                    </li>
                  ))}
                </ul>
              )}
              {removedOf(round.detail).length > 0 && (
                <ul className="mt-2 flex flex-col gap-0.5 text-xs text-muted-foreground" aria-label="Removed inputs">
                  {removedOf(round.detail).map((input) => (
                    <li key={input.column} className="tabular-nums">
                      {input.column}: shuffling it changed the score by {(-input.importance).toFixed(3)} (±
                      {input.spread.toFixed(3)} across folds)
                    </li>
                  ))}
                </ul>
              )}
              {confirmationOf(round.detail) && (
                <p className="mt-1 text-xs text-muted-foreground">
                  Checked on fresh folds: {confirmationOf(round.detail)!.confirmation_score.toFixed(3)} against{" "}
                  {confirmationOf(round.detail)!.incumbent_confirmation_score.toFixed(3)} for what it would replace (
                  {confirmationOf(round.detail)!.search_score.toFixed(3)} on the folds it was picked on).
                </p>
              )}
              {typeof round.detail?.note === "string" && (
                <p className="mt-1 text-xs text-muted-foreground">{round.detail.note}</p>
              )}
            </li>
            );
          })}
        </ol>

        <div className={cn("rounded-md border p-3", stopped.tone)}>
          <p className="font-medium">{stopped.title}</p>
          <p className="mt-1 text-muted-foreground">{stopped.blurb}</p>
          <p className="mt-2 text-xs text-muted-foreground">
            Best cross-validated {metric}: {trail.cv_score.toFixed(3)} (±
            {trail.cv_std.toFixed(3)} across folds)
            {trail.goal !== null && ` · your target was ${trail.goal.toFixed(3)}`}
            {strategyUsed(trail) && ` · settings searched by ${strategyUsed(trail)}`}
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
