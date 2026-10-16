import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { fetchPredictionHistory } from "@/api/client";
import type { PredictionHistory as History, PredictionSummary } from "@/api/types";

/** What the model in use predicted on each file it was applied to, against what it predicted on
 * its held-out rows at training time. A jump in the share flagged, or in the typical value, is
 * often the first sign something changed, and it shows before any answers arrive. */

function headline(summary: PredictionSummary): string {
  if (summary.kind === "binary") {
    return summary.positive_rate == null ? "—" : `${Math.round(summary.positive_rate * 100)}% flagged as ${summary.positive}`;
  }
  if (summary.kind === "values") return `mean ${summary.mean.toLocaleString()}`;
  const [top, share] = Object.entries(summary.shares)[0] ?? ["—", 0];
  return `${Math.round(share * 100)}% ${top}`;
}

function middle(summary: PredictionSummary): string {
  const q = summary.kind === "binary" ? summary.probability_quantiles : summary.kind === "values" ? summary.quantiles : null;
  return q && q["0.5"] != null ? String(q["0.5"]) : "—";
}

export function PredictionHistory({ sessionId, refreshKey }: { sessionId: string; refreshKey?: string }) {
  const [history, setHistory] = useState<History | null>(null);

  useEffect(() => {
    fetchPredictionHistory(sessionId).then(setHistory).catch(() => setHistory(null));
  }, [sessionId, refreshKey]);

  if (!history || history.applications.length === 0) return null;
  const rows = [
    ...(history.reference ? [{ key: "reference", label: "Held-out rows, at training", summary: history.reference }] : []),
    ...history.applications.map((a) => ({ key: a.apply_id, label: `${a.file_name ?? a.apply_id} · ${a.at}`, summary: a.prediction_summary })),
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What it has been predicting</CardTitle>
        <p className="text-sm text-muted-foreground">
          Each file the model was used on, against its own predictions on the held-out rows when it
          was trained, at the same cut-off. A big change here is worth looking into before the
          answers arrive.
        </p>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs uppercase tracking-wide text-muted-foreground">
              <th className="py-1 pr-3 font-medium">Rows predicted</th>
              <th className="py-1 pr-3 font-medium">Count</th>
              <th className="py-1 pr-3 font-medium">Predicted</th>
              <th className="py-1 font-medium">Median {history.reference?.kind === "values" ? "value" : "probability"}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.key} className="border-t border-border/60">
                <td className="py-1 pr-3">{r.label}</td>
                <td className="py-1 pr-3 tabular-nums">{r.summary.rows.toLocaleString()}</td>
                <td className="py-1 pr-3">{headline(r.summary)}</td>
                <td className="py-1 tabular-nums">{middle(r.summary)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </CardContent>
    </Card>
  );
}
