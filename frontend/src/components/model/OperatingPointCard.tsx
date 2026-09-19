import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { CutoffMetrics, OperatingPointResult } from "@/api/types";

const pct = (v: number | null | undefined) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

/** Where the cut-off sits and what it delivered.
 *
 * Three columns, not one. The line was chosen on the training folds, so what those folds
 * *promised* at that line is one number and what the sealed test set *delivered* is another
 * -- and the gap between them is how much the choice over-promised. The default at 0.50 sits
 * beside both so the trade-off is visible as a change, not a claim. */
export function OperatingPointCard({ point }: { point: OperatingPointResult }) {
  if (point.objective === "default") return null;

  const rows: { label: string; pick: (m: CutoffMetrics) => string; hint?: string }[] = [
    { label: "Share of real cases caught", pick: (m) => pct(m.recall), hint: "recall" },
    { label: "Correct when flagged", pick: (m) => pct(m.precision), hint: "precision" },
    { label: "Rows flagged", pick: (m) => `${m.flagged.toLocaleString()} of ${m.rows.toLocaleString()}` },
    { label: "Real cases missed", pick: (m) => m.missed.toLocaleString() },
    { label: "False alarms", pick: (m) => m.false_alarms.toLocaleString() },
  ];

  const promised = point.expected?.recall;
  const delivered = point.observed.recall;
  const overPromised = promised != null && promised - delivered > 0.1;

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
                </th>
                <th className="border-b border-border py-2 pr-3 text-right font-medium">
                  Test set delivered
                </th>
                <th className="border-b border-border py-2 text-right font-medium">
                  Test set at 0.50
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
            ? `The training folds promised ${pct(promised)} caught and the test set delivered ${pct(delivered)}. That gap is what choosing on the training rows costs -- and it is the honest number, because the test set had no say in where the line went.`
            : "The comparison table above stays at 0.50 so every model is measured the same way. This line applies to the chosen model only, and the downloaded model.joblib does not know about it -- the model card shows how to apply it."}
        </p>
      </CardContent>
    </Card>
  );
}
