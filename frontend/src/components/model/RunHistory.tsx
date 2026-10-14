import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import { cn } from "@/lib/utils";
import type { ModelRun } from "@/api/types";

/** What has been tried in this session, and whether it helped.
 *
 * Results used to overwrite a single file, so retraining discarded the previous run and the
 * one question people actually ask became unanswerable: *did the thing I just changed help?*
 * The verdict on a single run says whether a model is worth anything. Only the sequence says
 * whether a decision was.
 *
 * The change column is the point of the table. A list of scores makes you do the subtraction
 * yourself, and the subtraction is the finding. */

function metricOf(run: ModelRun): number | null {
  const key = run.primary_metric ?? "";
  const value = run.metrics?.[key];
  return typeof value === "number" ? value : null;
}

function describeSetup(run: ModelRun): string {
  const parts: string[] = [`${run.feature_count} columns`];
  if (run.split.strategy === "group" && run.split.group_column) {
    parts.push(`grouped by ${run.split.group_column}`);
  } else if (run.split.strategy === "time" && run.split.time_column) {
    parts.push(`split by ${run.split.time_column}`);
  }
  if (run.tuned) parts.push("tuned");
  if (run.feature_recipes?.length) {
    parts.push(`${run.feature_recipes.length} derived`);
  }
  if (run.clipped_columns?.length) {
    parts.push(`${run.clipped_columns.length} clipped`);
  }
  // How many times the held-out rows had been used when this ran. A run from before the
  // ledger has no count, and is shown as such: reading it as the first look would present a
  // score of unknown history as an independent one.
  parts.push(
    run.evaluation
      ? run.evaluation.status === "independent"
        ? "first look at the test rows"
        : `look ${run.evaluation.look_count} at the test rows (${run.evaluation.status})`
      : "test-row use not recorded"
  );
  // Two runs with the same AUC and different recall are the same model at different lines.
  if (run.operating_point && run.operating_point.objective !== "default" && run.operating_point.threshold != null) {
    parts.push(`cut-off ${run.operating_point.threshold.toFixed(2)}`);
  }
  if (run.excluded_columns.length > 0) {
    parts.push(`without ${run.excluded_columns.join(", ")}`);
  }
  return parts.join(" · ");
}

function when(iso: string): string {
  const parsed = new Date(iso);
  return Number.isNaN(parsed.getTime())
    ? iso
    : parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

const VERDICT_TEXT: Record<string, string> = {
  useful: "Works",
  marginal: "Weak",
  no_better_than_guessing: "No better than guessing",
  not_enough_test_data: "Can't tell",
};

export function RunHistory({
  runs,
  currentFingerprint,
}: {
  runs: ModelRun[];
  currentFingerprint: string;
}) {
  // One run is not a comparison, and a table of one row implies there is something to compare.
  if (runs.length < 2) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What you&apos;ve tried</CardTitle>
        <p className="text-sm text-muted-foreground">
          Every run in this session, newest first. The verdict on one run tells you whether a
          model is any good; the column on the right tells you whether a decision was.
        </p>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="py-2 pr-3 font-medium">Run</th>
                <th className="py-2 pr-3 font-medium">Setup</th>
                <th className="py-2 pr-3 font-medium">
                  Score
                  <ScoreLabel kind="held_out" className="ml-1 normal-case tracking-normal" />
                </th>
                <th className="py-2 pr-3 font-medium">Change</th>
                <th className="py-2 font-medium">Verdict</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run, index) => {
                const score = metricOf(run);
                // Compared against the run before it in time, which is the next row down.
                const previous = runs[index + 1];
                const previousScore = previous ? metricOf(previous) : null;
                const comparable =
                  score != null &&
                  previousScore != null &&
                  previous.data_fingerprint === run.data_fingerprint &&
                  previous.primary_metric === run.primary_metric;
                const delta = comparable ? score - previousScore : null;
                const stale =
                  Boolean(currentFingerprint) &&
                  Boolean(run.data_fingerprint) &&
                  run.data_fingerprint !== currentFingerprint;

                return (
                  <tr key={run.run_id} className="border-b border-border/60 align-top">
                    <td className="py-2 pr-3">
                      <span className="font-medium">{run.model_label ?? run.best_model}</span>
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {when(run.at)}
                        {index === 0 && " · current"}
                      </span>
                    </td>
                    <td className="py-2 pr-3 text-xs text-muted-foreground">
                      {describeSetup(run)}
                      {/* Said rather than hidden: comparing across a cleaning change is often
                          the most interesting comparison there is, but it is a different
                          comparison and has to be labelled as one. */}
                      {stale && (
                        <span className="mt-0.5 block text-amber-700 dark:text-amber-500">
                          trained before the data changed
                        </span>
                      )}
                    </td>
                    <td className="py-2 pr-3 tabular-nums">
                      {score == null ? "—" : score.toFixed(4)}
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {run.primary_metric}
                      </span>
                    </td>
                    <td className="py-2 pr-3 tabular-nums">
                      {delta == null ? (
                        <span className="text-xs text-muted-foreground">
                          {previous ? "not comparable" : "first run"}
                        </span>
                      ) : (
                        <span
                          className={cn(
                            "font-medium",
                            delta > 0.0005 && "text-emerald-600 dark:text-emerald-400",
                            delta < -0.0005 && "text-destructive"
                          )}
                        >
                          {delta > 0 ? "+" : ""}
                          {delta.toFixed(4)}
                        </span>
                      )}
                    </td>
                    <td className="py-2 text-xs">
                      {VERDICT_TEXT[run.verdict ?? ""] ?? run.verdict ?? "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-muted-foreground">
          This is a record, not a restore point — an earlier run can&apos;t be brought back.
          Retraining with the same settings will reproduce it.
        </p>
      </CardContent>
    </Card>
  );
}
