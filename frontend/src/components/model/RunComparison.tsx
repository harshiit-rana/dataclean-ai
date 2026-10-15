import type { RunPairComparison } from "@/api/types";

/** Two runs side by side: what differed, and whether the score moved beyond the noise.
 *
 * The sentence comes first because it is the answer; the settings and the numbers are the
 * evidence for it. Runs that cannot be compared say why and show no difference in score -- a
 * difference between scores on different data is not one. */

function show(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (Array.isArray(value)) return value.length ? value.map(String).join(", ") : "none";
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    return entries.length ? entries.map(([k, v]) => `${k} ${show(v)}`).join(", ") : "none";
  }
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

export function RunComparisonPanel({
  comparison,
  onClose,
}: {
  comparison: RunPairComparison;
  onClose?: () => void;
}) {
  const { a, b, primary } = comparison;
  const name = (run: RunPairComparison["a"]) => run.model_label ?? run.best_model ?? run.run_id;
  return (
    <section aria-label="Comparison of two runs" className="mt-4 rounded-md border border-border p-3 text-sm">
      <div className="flex items-start justify-between gap-2">
        <p className="font-medium">{comparison.summary}</p>
        {onClose && (
          <button type="button" className="text-xs text-muted-foreground underline" onClick={onClose}>
            Close
          </button>
        )}
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        Newer: {name(a)} · older: {name(b)}
        {comparison.comparable && !comparison.same_rows && " · scored on different held-out rows"}
      </p>

      {primary && primary.difference !== undefined && primary.low != null && primary.high != null && (
        <p className="mt-2 text-xs">
          {primary.metric}: {signed(primary.difference)} (95% range {signed(primary.low)} to {signed(primary.high)},
          paired on the same held-out rows)
        </p>
      )}
      {primary && primary.method === "intervals" && (
        <p className="mt-2 text-xs text-muted-foreground">
          Judged from the two runs&apos; own ranges, because they were not scored on the same rows.
        </p>
      )}

      <h3 className="mt-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">What differed</h3>
      {comparison.settings.length === 0 ? (
        <p className="text-xs text-muted-foreground">Nothing in the settings.</p>
      ) : (
        <table className="mt-1 w-full text-xs">
          <thead>
            <tr className="text-left text-muted-foreground">
              <th className="py-1 pr-3 font-medium">Setting</th>
              <th className="py-1 pr-3 font-medium">Older</th>
              <th className="py-1 font-medium">Newer</th>
            </tr>
          </thead>
          <tbody>
            {comparison.settings.map((d) => (
              <tr key={d.field} className="border-t border-border/60 align-top">
                <td className="py-1 pr-3">{d.label}</td>
                <td className="py-1 pr-3">{show(d.b)}</td>
                <td className="py-1">{show(d.a)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {comparison.comparable && comparison.metrics.length > 0 && (
        <>
          <h3 className="mt-3 text-xs font-medium uppercase tracking-wide text-muted-foreground">Scores</h3>
          <table className="mt-1 w-full text-xs tabular-nums">
            <thead>
              <tr className="text-left text-muted-foreground">
                <th className="py-1 pr-3 font-medium">Metric</th>
                <th className="py-1 pr-3 font-medium">Older</th>
                <th className="py-1 pr-3 font-medium">Newer</th>
                <th className="py-1 font-medium">Change</th>
              </tr>
            </thead>
            <tbody>
              {comparison.metrics.map((m) => (
                <tr key={m.metric} className="border-t border-border/60">
                  <td className="py-1 pr-3">{m.metric}</td>
                  <td className="py-1 pr-3">{m.b.toFixed(3)}</td>
                  <td className="py-1 pr-3">{m.a.toFixed(3)}</td>
                  <td className="py-1">{signed(m.difference)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-xs text-muted-foreground">
            Only {primary?.metric ?? "the headline metric"} is judged against the noise; the others are shown as they
            came out.
          </p>
        </>
      )}
    </section>
  );
}
