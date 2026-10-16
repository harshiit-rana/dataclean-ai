import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { CalibrationMethod, CalibrationReport } from "@/api/types";

const METHOD_LABEL: Record<Exclude<CalibrationMethod, "none">, string> = {
  sigmoid: "Sigmoid",
  isotonic: "Isotonic",
};

const METHOD_HELP: Record<Exclude<CalibrationMethod, "none">, string> = {
  sigmoid: "A smooth S-shaped correction with two settings. Needs little data and cannot reorder cases.",
  isotonic:
    "A flexible step-shaped correction. Fits odd distortions, but needs a lot of data and can flatten ties.",
};

const points = (value: number | undefined) =>
  typeof value === "number" ? `${(value * 100).toFixed(1)} points` : "—";

const num = (value: number | undefined) => (typeof value === "number" ? value.toFixed(3) : "—");

/** What the probabilities are worth, and what can be done about it.
 *
 * Calibration is an action the user approves, not something applied behind their back: it
 * changes what every probability means, and with it where the default 0.50 line falls. So this
 * card has three states. After calibration it shows the change; when the probabilities are off
 * by more than chance explains it offers the fix; and when the data cannot support a method it
 * says why instead of failing. A model whose probabilities are fine shows nothing. */
export function CalibrationCard({
  report,
  onCalibrate,
  busy,
}: {
  report?: CalibrationReport | null;
  onCalibrate?: (method: CalibrationMethod) => void;
  busy?: boolean;
}) {
  if (!report) return null;

  if (report.method !== "none") return <Applied report={report} onCalibrate={onCalibrate} busy={busy} />;

  if (report.refused_because) {
    return (
      <Card className="border-amber-500/40 bg-amber-500/5">
        <CardHeader>
          <CardTitle className="text-base">The probabilities were not calibrated</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-muted-foreground">
          {METHOD_LABEL[report.requested as Exclude<CalibrationMethod, "none">] ?? report.requested}{" "}
          calibration was asked for, but: {report.refused_because} The model was left as it was.
        </CardContent>
      </Card>
    );
  }

  if (!report.recommended || !onCalibrate) return null;

  return (
    <Card className="border-sky-500/40 bg-sky-500/5">
      <CardHeader>
        <CardTitle className="text-base">Its probabilities may not mean what they say</CardTitle>
        <p className="text-sm text-muted-foreground">
          On the held-out rows its probabilities are off by{" "}
          {points(report.error)}{" "}
          {report.chance_error != null
            ? `(about ${(report.chance_error * 100).toFixed(1)} would be chance on this many rows) `
            : ""}
          on average: when it says 70% it may be right more or less often than that. That matters if
          you read the number as a probability, and the cut-off is chosen on it.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <p className="text-muted-foreground">
          Calibration learns a correction from cross-validated predictions on the training rows
          only. The held-out rows are scored once more to show the change; they do not take part
          in fitting it. Applying it retrains the model, which counts as another look at those
          rows, and it can change the answers at the default 0.50 line.
        </p>
        <div className="flex flex-col gap-2">
          {(Object.keys(METHOD_LABEL) as Exclude<CalibrationMethod, "none">[]).map((method) => {
            const option = report.options[method];
            return (
              <div key={method} className="flex flex-wrap items-start gap-3">
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!option?.available || busy}
                  onClick={() => onCalibrate(method)}
                >
                  Calibrate with {METHOD_LABEL[method].toLowerCase()}
                </Button>
                <p
                  className={cn(
                    "max-w-xl text-xs",
                    option?.available ? "text-muted-foreground" : "text-amber-700 dark:text-amber-400"
                  )}
                >
                  {option?.available ? METHOD_HELP[method] : option?.reason}
                </p>
              </div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function Applied({
  report,
  onCalibrate,
  busy,
}: {
  report: CalibrationReport;
  onCalibrate?: (method: CalibrationMethod) => void;
  busy?: boolean;
}) {
  const method = report.method as Exclude<CalibrationMethod, "none">;
  const before = report.before ?? {};
  const after = report.after ?? {};
  const rows: { label: string; key: string; hint?: string }[] = [
    { label: "Brier score", key: "brier_score", hint: "lower is better" },
    { label: "Calibration error", key: "expected_calibration_error", hint: "lower is better" },
    { label: "ROC AUC", key: "roc_auc" },
    { label: "Recall at the default 0.50 line", key: "recall" },
    { label: "Precision at the default 0.50 line", key: "precision" },
  ];

  return (
    <Card className="border-emerald-500/30">
      <CardHeader>
        <CardTitle className="text-base">
          Probabilities calibrated ({method})
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          Fitted on {report.fitted_on}. The held-out rows were scored once more, with it, to show
          the change.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border-b border-border px-2 py-1.5 text-left font-medium">
                  <span className="sr-only">Measure</span>
                </th>
                <th className="border-b border-border px-2 py-1.5 text-right font-medium">Before</th>
                <th className="border-b border-border px-2 py-1.5 text-right font-medium">After</th>
              </tr>
            </thead>
            <tbody>
              {rows
                .filter((row) => typeof before[row.key] === "number" && typeof after[row.key] === "number")
                .map((row) => (
                  <tr key={row.key} className="border-b border-border last:border-0">
                    <td className="px-2 py-1.5">
                      {row.label}
                      {row.hint && <span className="ml-2 text-xs text-muted-foreground">{row.hint}</span>}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                      {num(before[row.key])}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">{num(after[row.key])}</td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
        <p className="text-muted-foreground">
          Calibration changes what a probability means, not which cases rank higher
          {method === "sigmoid" && !report.ensemble
            ? ", so the AUC is unchanged. "
            : "; the AUC can move slightly. "}
          It can change the answers at the default 0.50 line, because 0.50 now sits on honest
          probabilities: on a rare answer that flags fewer rows. Choose the cut-off for the trade-off
          that matters rather than reading the default line.
        </p>
        {report.ensemble && (
          <p className="text-muted-foreground">
            This was a forward-in-time split, whose folds cover only the later rows, so the
            calibrated model averages one copy per fold rather than a single model with a single
            mapping; the AUC difference includes that averaging.
          </p>
        )}
        {onCalibrate && (
          <div>
            <Button size="sm" variant="ghost" disabled={busy} onClick={() => onCalibrate("none")}>
              Go back to the uncalibrated model
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
