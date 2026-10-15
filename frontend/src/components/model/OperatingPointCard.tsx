import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import type { CutoffMetrics, EvaluationStatus, OperatingPointResult } from "@/api/types";

const pct = (v: number | null | undefined) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

/** Where the cut-off sits and what it delivered.
 *
 * Three columns, not one. The line was chosen on the training folds, so what those folds
 * *promised* at that line is one number and what the sealed test set *delivered* is another
 * -- and the gap between them is how much the choice over-promised. The default at 0.50 sits
 * beside both so the trade-off is visible as a change, not a claim. */
export function OperatingPointCard({
  point,
  evaluation,
  trainedWithCosts = false,
}: {
  point: OperatingPointResult;
  evaluation?: EvaluationStatus;
  /** The model was also trained with the stated costs as row weights, not only cut with them. */
  trainedWithCosts?: boolean;
}) {
  if (point.objective === "default") return null;

  const rows: {
    label: string;
    pick: (m: CutoffMetrics) => string;
    hint?: string;
    /** The rate whose range, if there is one, is printed beside the delivered figure. */
    range?: "precision" | "recall";
  }[] = [
    { label: "Share of real cases caught", pick: (m) => pct(m.recall), hint: "recall", range: "recall" },
    {
      label: "Correct when flagged",
      pick: (m) => pct(m.precision),
      hint: "precision",
      range: "precision",
    },
    { label: "Rows flagged", pick: (m) => `${m.flagged.toLocaleString()} of ${m.rows.toLocaleString()}` },
    { label: "Real cases missed", pick: (m) => m.missed.toLocaleString() },
    { label: "False alarms", pick: (m) => m.false_alarms.toLocaleString() },
  ];

  const promised = point.expected?.recall;
  const delivered = point.observed.recall;
  // Judged against the range the delivered figure could land in on other rows, not a fixed gap:
  // ten points on forty positives is well inside it, and a result with no range has no evidence
  // either way, so it gets no message.
  const range = point.observed_intervals?.metrics?.recall;
  const overPromised = promised != null && range != null && promised > range.high;
  const gapWithinNoise =
    promised != null &&
    range != null &&
    !overPromised &&
    Math.abs(promised - delivered) > 0.02 &&
    promised >= range.low;

  return (
    <Card className={cn(!point.reachable && "border-amber-500/40 bg-amber-500/5")}>
      <CardHeader>
        <CardTitle className="text-base">
          The line sits at {point.threshold.toFixed(2)}, so that {point.description}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          A row is flagged as <span className="font-medium text-foreground">{point.positive_label}</span>{" "}
          when the model gives it a probability of {point.threshold.toFixed(2)} or more. The
          line was chosen on {point.chosen_on}.
        </p>
        {point.calibrated_with && (
          <p className="text-sm text-muted-foreground">
            The probabilities were calibrated ({point.calibrated_with}), so this line is on
            calibrated probabilities: {point.threshold.toFixed(2)} means the model says it is{" "}
            {Math.round(point.threshold * 100)}% likely, and means it.
          </p>
        )}
        {point.objective === "cost" && (
          <p className="text-sm text-muted-foreground">
            {trainedWithCosts
              ? "The model was also trained with your costs: each row counted by the cost of getting it wrong, because on the training folds that lowered the expected cost by more than the folds disagree."
              : "The model was trained without your costs; only this line uses them."}
          </p>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {point.note && (
          <p className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3 text-amber-900 dark:text-amber-200">
            {point.note}
          </p>
        )}

        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
                <th className="border-b border-border py-2 pr-3 font-medium">At this line</th>
                <th className="border-b border-border py-2 pr-3 text-right font-medium">
                  Training folds promised
                  <ScoreLabel kind="exploratory" className="ml-1 normal-case tracking-normal" />
                </th>
                <th className="border-b border-border py-2 pr-3 text-right font-medium">
                  Test set delivered
                  <ScoreLabel
                    kind="held_out"
                    evaluation={evaluation}
                    className="ml-1 normal-case tracking-normal"
                  />
                </th>
                <th className="border-b border-border py-2 text-right font-medium">
                  Test set at 0.50
                  <ScoreLabel
                    kind="held_out"
                    evaluation={evaluation}
                    className="ml-1 normal-case tracking-normal"
                  />
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label} className="border-b border-border/60 last:border-0">
                  <td className="py-1.5 pr-3">
                    {row.label}
                    {row.hint && (
                      <span className="ml-1 text-xs text-muted-foreground">({row.hint})</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground">
                    {point.expected ? row.pick(point.expected) : "—"}
                  </td>
                  <td className="py-1.5 pr-3 text-right font-medium tabular-nums">
                    {row.pick(point.observed)}
                    {row.range && point.observed_intervals?.metrics?.[row.range] && (
                      <span className="ml-1 text-xs font-normal text-muted-foreground">
                        ({pct(point.observed_intervals.metrics[row.range]!.low)}–
                        {pct(point.observed_intervals.metrics[row.range]!.high)})
                      </span>
                    )}
                  </td>
                  <td className="py-1.5 text-right tabular-nums text-muted-foreground">
                    {row.pick(point.at_default)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <p className="text-xs text-muted-foreground">
          {overPromised
            ? `The training folds promised ${pct(promised)} caught and the test set delivered ${pct(delivered)}, with a range of ${pct(range!.low)}–${pct(range!.high)} on other rows: the promise is outside it. That gap is what choosing on the training rows costs -- and it is the honest number, because the test set had no say in where the line went.`
            : gapWithinNoise
              ? `The training folds promised ${pct(promised)} caught and the test set delivered ${pct(delivered)}, but on this many test rows the delivered figure could land anywhere from ${pct(range!.low)} to ${pct(range!.high)}, so the gap is not evidence that the line over-promised.`
              : "The comparison table above stays at 0.50 so every model is measured the same way. This line applies to the chosen model only, and the downloaded model.joblib does not know about it -- the model card shows how to apply it."}
        </p>
      </CardContent>
    </Card>
  );
}
