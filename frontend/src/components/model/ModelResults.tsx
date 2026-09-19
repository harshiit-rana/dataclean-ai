import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import {
  CalibrationChart,
  ClassDistributionChart,
  ConfusionMatrix,
  PerClassChart,
  PrecisionRecallChart,
  RocCurveChart,
  ThresholdChart,
} from "@/components/charts/ClassificationCharts";
import {
  ErrorByRangeChart,
  PredictedVsActualChart,
  ResidualChart,
  ResidualHistogram,
} from "@/components/charts/RegressionCharts";
import {
  ColumnImportanceChart,
  CorrelationHeatmap,
  CrossValidationChart,
  LearningCurveChart,
  PermutationImportanceChart,
} from "@/components/charts/ModelCharts";
import { OperatingPointCard } from "@/components/model/OperatingPointCard";
import type { ModelResult } from "@/api/types";

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  roc_auc: "ROC AUC",
  r2: "R²",
  mae: "Mean error",
  rmse: "RMSE",
};

const VERDICT_STYLES: Record<string, { title: string; className: string }> = {
  useful: { title: "This model works", className: "border-emerald-500/40 bg-emerald-500/5" },
  marginal: { title: "This model is weak", className: "border-amber-500/40 bg-amber-500/5" },
  no_better_than_guessing: {
    title: "This model does not work",
    className: "border-destructive/40 bg-destructive/5",
  },
  // Deliberately not styled as a failure. "We cannot tell" is not "it is bad", and colouring
  // it red would push people towards the wrong conclusion just as hard as calling it useful.
  not_enough_test_data: {
    title: "There isn't enough data to judge this model",
    className: "border-sky-500/40 bg-sky-500/5",
  },
};

function formatMetric(value: unknown): string {
  if (typeof value !== "number") return "—";
  return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
}

export function ModelResults({ result }: { result: ModelResult }) {
  const best = result.results.find((r) => r.model === result.best_model);
  const verdict = VERDICT_STYLES[result.assessment.verdict] ?? VERDICT_STYLES.marginal;
  const diagnostics = best?.diagnostics ?? {};
  const deep = result.deep_diagnostics ?? {};

  const metricKeys =
    result.task === "regression"
      ? ["r2", "mae", "rmse"]
      : ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const presentKeys = metricKeys.filter((key) =>
    result.results.some((r) => typeof r.metrics[key] === "number")
  );

  return (
    <div className="flex flex-col gap-6">
      {/* The verdict leads. A model that learned nothing still reports a healthy-looking
          accuracy on a lopsided target, and that is the single easiest thing to misread. */}
      <Card className={cn("border", verdict.className)}>
        <CardHeader>
          <CardTitle className="text-base">{verdict.title}</CardTitle>
        </CardHeader>
        <CardContent className="text-sm">
          <p>{result.assessment.headline}</p>
          <p className="mt-3 text-muted-foreground">
            Trained on {result.split.train_rows.toLocaleString()} rows and scored on{" "}
            {result.split.test_rows.toLocaleString()} it had never seen
            {result.split.strategy === "time"
              ? `, split by ${result.split.time_column}.`
              : result.split.strategy === "group"
                ? `, kept whole by ${result.split.group_column}.`
                : ", split at random."}
          </p>
          {/* The entity counts, not just the rows. 800 training rows drawn from 4 customers
              is not 800 independent observations, and the row count alone hides that. */}
          {result.split.strategy === "group" && result.split.test_groups != null && (
            <p className="mt-1 text-muted-foreground">
              That is {result.split.train_groups?.toLocaleString()} distinct{" "}
              {result.split.group_column} values to learn from and{" "}
              {result.split.test_groups.toLocaleString()} it has never met. No{" "}
              {result.split.group_column} appears on both sides.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">How the models compared</CardTitle>
          <p className="text-sm text-muted-foreground">
            All trained on the same split. The last row is what you get with no model at all.
          </p>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border-b border-border px-2 py-1.5 text-left font-medium">Model</th>
                {presentKeys.map((key) => (
                  <th
                    key={key}
                    className="border-b border-border px-2 py-1.5 text-right font-medium"
                  >
                    {METRIC_LABELS[key] ?? key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.results.map((entry) => (
                <tr
                  key={entry.model}
                  className={cn(
                    "border-b border-border last:border-0",
                    entry.model === result.best_model && "bg-muted/50"
                  )}
                >
                  <td className="px-2 py-1.5">
                    {entry.label}
                    {entry.model === result.best_model && (
                      <span className="ml-2 text-xs text-muted-foreground">chosen</span>
                    )}
                  </td>
                  {presentKeys.map((key) => (
                    <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                      {entry.failed ? "—" : formatMetric(entry.metrics[key])}
                    </td>
                  ))}
                </tr>
              ))}
              <tr className="text-muted-foreground">
                <td className="px-2 py-1.5 italic">
                  No model at all ({String(result.baseline.strategy ?? "guessing")})
                </td>
                {presentKeys.map((key) => (
                  <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                    {formatMetric(result.baseline[key])}
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </CardContent>
      </Card>

      {result.operating_point && <OperatingPointCard point={result.operating_point} />}

      {result.task === "classification" ? (
        <>
          {diagnostics.confusion_cells && <ConfusionMatrix cells={diagnostics.confusion_cells} />}
          {diagnostics.per_class && <PerClassChart rows={diagnostics.per_class} />}
          {diagnostics.roc_curve && (
            <RocCurveChart
              points={diagnostics.roc_curve}
              auc={typeof best?.metrics.roc_auc === "number" ? best.metrics.roc_auc : undefined}
            />
          )}
          {diagnostics.precision_recall_curve && (
            <PrecisionRecallChart points={diagnostics.precision_recall_curve} />
          )}
          {diagnostics.thresholds && (
            <ThresholdChart
              rows={diagnostics.thresholds}
              chosen={
                result.operating_point && result.operating_point.objective !== "default"
                  ? result.operating_point.threshold
                  : undefined
              }
            />
          )}
          {diagnostics.calibration && <CalibrationChart points={diagnostics.calibration} />}
          {diagnostics.class_distribution && (
            <ClassDistributionChart counts={diagnostics.class_distribution} />
          )}
        </>
      ) : (
        <>
          {diagnostics.predicted_vs_actual && (
            <PredictedVsActualChart points={diagnostics.predicted_vs_actual} />
          )}
          {diagnostics.residuals && <ResidualChart points={diagnostics.residuals} />}
          {diagnostics.residual_histogram && (
            <ResidualHistogram bars={diagnostics.residual_histogram} />
          )}
          {diagnostics.error_by_range && <ErrorByRangeChart rows={diagnostics.error_by_range} />}
        </>
      )}

      {best?.column_importance?.length ? (
        <ColumnImportanceChart rows={best.column_importance} />
      ) : null}
      {deep.permutation_importance?.length ? (
        <PermutationImportanceChart rows={deep.permutation_importance} />
      ) : null}
      {deep.cross_validation ? <CrossValidationChart cv={deep.cross_validation} /> : null}
      {deep.learning_curve?.length ? <LearningCurveChart points={deep.learning_curve} /> : null}
      {result.feature_correlations?.columns?.length ? (
        <CorrelationHeatmap data={result.feature_correlations} />
      ) : null}

      {deep.skipped && (
        <p className="rounded-md border border-border bg-muted p-3 text-xs text-muted-foreground">
          {deep.skipped}
        </p>
      )}
    </div>
  );
}
