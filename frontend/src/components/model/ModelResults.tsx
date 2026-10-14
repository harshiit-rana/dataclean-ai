import { useState } from "react";
import { Button } from "@/components/ui/button";
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
import { HeldOutUse, ScoreLabel } from "@/components/model/ScoreLabel";
import type { CandidateResult, MetricIntervals, ModelResult } from "@/api/types";

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

/** A metric with where it could plausibly land: `0.78 (0.71–0.84)`. The bracket is what lets 60
 * rows and 6,000 rows look different on the page, which a bare number cannot do. */
function MetricWithInterval({
  value,
  interval,
}: {
  value: unknown;
  interval?: { low: number; high: number };
}) {
  return (
    <>
      {formatMetric(value)}
      {interval && (
        <span className="ml-1 text-xs text-muted-foreground">
          ({interval.low.toFixed(3)}–{interval.high.toFixed(3)})
        </span>
      )}
    </>
  );
}

function intervalNote(intervals?: MetricIntervals | null): string | null {
  if (!intervals || Object.keys(intervals.metrics).length === 0) return null;
  const unit = intervals.by === "groups" ? "whole groups" : "rows";
  return (
    `Brackets: the middle ${Math.round(intervals.level * 100)}% of ` +
    `${intervals.resamples.toLocaleString()} resamples of the held-out ${unit} — the range the ` +
    `score could plausibly land in on a different set of ${unit}, not a guarantee.`
  );
}

/** The chosen model's held-out scores, with the no-model baseline for contrast.
 *
 * This is the only place a test-set score is shown by default. Candidates other than the one
 * chosen have theirs hidden (see ModelComparison): showing them invites choosing by them,
 * which is selecting on the test set by hand. */
function HeldOutScores({
  result,
  best,
  keys,
}: {
  result: ModelResult;
  best: CandidateResult;
  keys: string[];
}) {
  if (best.failed || keys.length === 0) return null;
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <caption className="pb-1 text-left text-xs text-muted-foreground">
          Scored once on the held-out rows, for the chosen model only.
          {intervalNote(best.metrics_intervals) && ` ${intervalNote(best.metrics_intervals)}`}
        </caption>
        <thead>
          <tr>
            <th className="border-b border-border px-2 py-1.5 text-left font-medium">&nbsp;</th>
            {keys.map((key) => (
              <th key={key} className="border-b border-border px-2 py-1.5 text-right font-medium">
                {METRIC_LABELS[key] ?? key}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          <tr className="bg-muted/50">
            <td className="px-2 py-1.5 font-medium">{best.label}</td>
            {keys.map((key) => (
              <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                <MetricWithInterval
                  value={best.metrics[key]}
                  interval={best.metrics_intervals?.metrics?.[key]}
                />
              </td>
            ))}
          </tr>
          <tr className="text-muted-foreground">
            <td className="px-2 py-1.5 italic">
              No model at all ({String(result.baseline.strategy ?? "guessing")})
            </td>
            {keys.map((key) => (
              <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                {formatMetric(result.baseline[key])}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );
}

/** How the model has done on rows that did not exist when it was built.
 *
 * The cleanest check there is, because nothing was decided with those rows. Shown beside the
 * held-out score and labelled as a different kind of number: the new rows may define the label
 * a little differently, so this is evidence about the model rather than a replacement for the
 * original evaluation. Nothing is shown until a file with real answers has been applied. */
function NewDataCheck({ evaluation }: { evaluation?: ModelResult["evaluation"] }) {
  const found = evaluation?.new_data;
  if (!found) return null;
  const { latest } = found;
  const label = latest.metric ? (METRIC_LABELS[latest.metric] ?? latest.metric) : "score";
  const when = latest.at ? new Date(latest.at) : null;
  return (
    <div className="mt-3 rounded-md border border-violet-500/40 bg-violet-500/5 p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <ScoreLabel kind="new_data" />
        <span className="font-medium">
          Checked on new data {found.count === 1 ? "once" : `${found.count} times`}
        </span>
      </div>
      <p className="mt-1 text-muted-foreground">
        Latest: {latest.rows?.toLocaleString() ?? "?"} rows that did not exist when this model was
        built, {label} {latest.value != null ? latest.value.toFixed(3) : "—"}
        {latest.reported != null && ` (${latest.reported.toFixed(3)} when it was built)`}
        {when && !Number.isNaN(when.getTime()) && `, ${when.toLocaleDateString()}`}.
        {latest.overlap_excluded > 0 &&
          ` ${latest.overlap_excluded.toLocaleString()} rows the model was built from were left out.`}
      </p>
    </div>
  );
}

const SELECTION_NOTE: Record<NonNullable<ModelResult["selection"]>["method"], string> = {
  cross_validation: "chosen on training folds",
  only_candidate: "the only model trained",
  default_order: "kept by default order",
};

/** The models side by side, on the numbers that actually chose between them.
 *
 * This table used to show every candidate's *test* score and mark the best one "chosen", which
 * reads as though the highest test score won -- and lets anyone pick a different candidate by
 * it, which is selecting on the test set by hand. The winner is chosen on cross-validated
 * folds of the training rows, so those are what lead; the test scores of the others are behind
 * a button, with the reason they are not the basis for anything. */
function ModelComparison({ result, keys }: { result: ModelResult; keys: string[] }) {
  const [showHeldOut, setShowHeldOut] = useState(false);
  const selection = result.selection;
  const metric = selection?.metric ?? null;

  // A result saved before models were chosen on folds has no fold scores to lead with. Its
  // choice *was* made on the test set, so its test-score table is the accurate description.
  if (!selection) {
    return (
      <Card>
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0">
          <CardTitle className="text-base">How the models compared</CardTitle>
          <ScoreLabel kind="held_out" evaluation={result.evaluation} />
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <HeldOutTable result={result} keys={keys} />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center justify-between gap-2">
          <CardTitle className="text-base">How the models compared</CardTitle>
          <ScoreLabel kind="exploratory" />
        </div>
        <p className="text-sm text-muted-foreground">
          Compared on cross-validated folds of the training rows — never on the held-out rows, so
          the choice cannot have been made with the test answers. The held-out score above is
          for the chosen model only.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {selection.rule_note && (
          <p className="rounded-md border border-sky-500/40 bg-sky-500/5 p-3 text-sm">
            {selection.rule_note}
          </p>
        )}
        {selection.method === "default_order" && (
          <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-sm">
            {selection.chosen_on}.
          </p>
        )}
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr>
                <th className="border-b border-border px-2 py-1.5 text-left font-medium">Model</th>
                <th className="border-b border-border px-2 py-1.5 text-right font-medium">
                  Fold score{metric ? ` (${METRIC_LABELS[metric] ?? metric})` : ""}
                </th>
                <th className="border-b border-border px-2 py-1.5 text-right font-medium">
                  Fold spread
                </th>
              </tr>
            </thead>
            <tbody>
              {result.results.map((entry) => {
                const isChosen = entry.model === result.best_model;
                return (
                  <tr
                    key={entry.model}
                    className={cn("border-b border-border last:border-0", isChosen && "bg-muted/50")}
                  >
                    <td className="px-2 py-1.5">
                      {entry.label}
                      {isChosen && (
                        <span className="ml-2 text-xs text-muted-foreground">
                          {selection.rule === "one_standard_error"
                            ? "simplest within one standard error of the best"
                            : SELECTION_NOTE[selection.method]}
                        </span>
                      )}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums">
                      {entry.failed
                        ? "did not train"
                        : entry.cv_score != null
                          ? entry.cv_score.toFixed(3)
                          : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-muted-foreground">
                      {!entry.failed && entry.cv_std != null ? `± ${entry.cv_std.toFixed(3)}` : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {result.results.length > 1 && keys.length > 0 && (
          <div className="text-sm">
            <button
              type="button"
              className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
              aria-expanded={showHeldOut}
              onClick={() => setShowHeldOut((open) => !open)}
            >
              {showHeldOut
                ? "Hide the other models' held-out scores"
                : "Show held-out scores for every model"}
            </button>
            {showHeldOut && (
              <div className="mt-2 flex flex-col gap-2">
                <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3 text-xs">
                  These were not what the choice was based on. Choosing a different model
                  because it scores higher here would bring back the problem the folds avoid:
                  the best of several scores on the same rows looks better than it will on new
                  data.
                </p>
                <ScoreLabel kind="held_out" evaluation={result.evaluation} className="self-start" />
                <HeldOutTable result={result} keys={keys} />
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function HeldOutTable({ result, keys }: { result: ModelResult; keys: string[] }) {
  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr>
          <th className="border-b border-border px-2 py-1.5 text-left font-medium">Model</th>
          {keys.map((key) => (
            <th key={key} className="border-b border-border px-2 py-1.5 text-right font-medium">
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
            {keys.map((key) => (
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
          {keys.map((key) => (
            <td key={key} className="px-2 py-1.5 text-right tabular-nums">
              {formatMetric(result.baseline[key])}
            </td>
          ))}
        </tr>
      </tbody>
    </table>
  );
}

interface ModelResultsProps {
  result: ModelResult;
  /** Opens the sealed final holdout. Present only where the page can do it. */
  onFinalise?: () => void;
  finalising?: boolean;
  finaliseError?: string | null;
  /** Said when the rows were opened earlier, for a different model. */
  finaliseNote?: string | null;
}

/** The sealed final holdout: rows set aside before anything was decided and never scored while
 * iterating. Opened once, by an explicit action, because a score on rows nothing was chosen
 * against is the only number here that has not been through a selection. */
function SealedHoldout({
  result,
  onFinalise,
  finalising,
  finaliseError,
  finaliseNote,
}: ModelResultsProps) {
  const split = result.split;
  if (!split.sealed) {
    if (!split.sealed_unavailable_reason) return null;
    return (
      <p className="rounded-md border border-border bg-muted p-3 text-sm text-muted-foreground">
        A sealed final holdout was asked for but could not be made: {split.sealed_unavailable_reason}{" "}
        The ordinary split was used instead.
      </p>
    );
  }

  const final = result.evaluation?.final;
  const keys =
    result.task === "regression"
      ? ["r2", "mae", "rmse"]
      : ["accuracy", "precision", "recall", "f1", "roc_auc"];
  const shown = keys.filter((key) => typeof final?.metrics?.[key] === "number");
  const late = (split.sealed_after_looks ?? 0) > 0;

  return (
    <Card className="border-violet-500/30">
      <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0">
        <CardTitle className="text-base">Sealed final holdout</CardTitle>
        <ScoreLabel kind="held_out" />
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <p>
          {split.final_rows?.toLocaleString()} rows were set aside and have never been scored. Every
          number above was measured on other rows, so nothing you changed while iterating could have
          been tuned to these.
        </p>
        {late && (
          <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
            Sealed after {split.sealed_after_looks} {split.sealed_after_looks === 1 ? "look" : "looks"}.
            These rows are still unopened, but until now they were training or development rows, so
            they shaped decisions made before the seal. A holdout sealed before the first look is
            cleaner.
          </p>
        )}

        {!final ? (
          <div className="flex flex-col gap-2">
            {finaliseNote && (
              <p className="rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
                {finaliseNote}
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              Opening them scores the current model on them, once. After that they are used up: a
              second model scored on the same rows would not be an independent evaluation, because
              this result may already have shaped it.
            </p>
            <div>
              <Button size="sm" onClick={onFinalise} disabled={!onFinalise || finalising}>
                {finalising ? "Opening…" : "Open the sealed rows (once)"}
              </Button>
            </div>
            {finaliseError && <p className="text-xs text-destructive">{finaliseError}</p>}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {final.assessment && <p className="font-medium">{final.assessment.headline}</p>}
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-sm">
                <caption className="pb-1 text-left text-xs text-muted-foreground">
                  Scored once on the {final.rows?.toLocaleString()} sealed rows.
                  {intervalNote(final.metrics_intervals) && ` ${intervalNote(final.metrics_intervals)}`}
                </caption>
                <thead>
                  <tr>
                    <th className="border-b border-border px-2 py-1.5 text-left font-medium">&nbsp;</th>
                    {shown.map((key) => (
                      <th key={key} className="border-b border-border px-2 py-1.5 text-right font-medium">
                        {METRIC_LABELS[key] ?? key}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr className="bg-muted/50">
                    <td className="px-2 py-1.5 font-medium">Final evaluation</td>
                    {shown.map((key) => (
                      <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                        <MetricWithInterval
                          value={final.metrics[key]}
                          interval={final.metrics_intervals?.metrics?.[key]}
                        />
                      </td>
                    ))}
                  </tr>
                  <tr className="text-muted-foreground">
                    <td className="px-2 py-1.5 italic">
                      No model at all ({String(final.baseline?.strategy ?? "guessing")})
                    </td>
                    {shown.map((key) => (
                      <td key={key} className="px-2 py-1.5 text-right tabular-nums">
                        {formatMetric(final.baseline?.[key])}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
            {final.rows_seen_in_training > 0 && (
              <p className="text-xs text-muted-foreground">
                {final.rows_seen_in_training.toLocaleString()} of these rows match a training row on
                the model&apos;s inputs and answer, so this score is slightly flattered.
              </p>
            )}
            <p className="text-xs text-muted-foreground">
              These rows are now used up. Treat this number as the one to report.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function ModelResults({
  result,
  onFinalise,
  finalising,
  finaliseError,
  finaliseNote,
}: ModelResultsProps) {
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
        <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0">
          <CardTitle className="text-base">{verdict.title}</CardTitle>
          <ScoreLabel kind="held_out" evaluation={result.evaluation} />
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
          {(result.split.test_rows_seen_in_training ?? 0) > 0 && (
            <p className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-2">
              {result.split.test_rows_seen_in_training!.toLocaleString()} of those{" "}
              {result.split.test_rows.toLocaleString()} test rows are exact copies of training
              rows, answer included — the model had already seen those answers, so the score
              above is flattered by them. If they&apos;re duplicated records rather than genuine
              repeats, remove duplicates during cleaning and retrain.
            </p>
          )}
          <HeldOutUse evaluation={result.evaluation} split={result.split} />
          <NewDataCheck evaluation={result.evaluation} />
          {(result.evaluation?.whole_file_statistics?.length ?? 0) > 0 && (
            <p className="mt-2 rounded-md border border-amber-500/40 bg-amber-500/5 p-2 text-xs">
              Trained on data where{" "}
              {result.evaluation!.whole_file_statistics!
                .map((w) => `${w.operation.replace(/_/g, " ")} on ${w.column}`)
                .join(", ")}{" "}
              was worked out over every row, the held-out ones included, before the split. A small
              leak — the score is slightly flattered. Undoing those fixes lets the model do it
              from the training rows only.
            </p>
          )}
          {result.split.strategy === "group" && result.split.test_groups != null && (
            <p className="mt-1 text-muted-foreground">
              That is {result.split.train_groups?.toLocaleString()} distinct{" "}
              {result.split.group_column} values to learn from and{" "}
              {result.split.test_groups.toLocaleString()} it has never met. No{" "}
              {result.split.group_column} appears on both sides.
            </p>
          )}
          {best && (
            <HeldOutScores
              result={result}
              best={best}
              keys={presentKeys}
            />
          )}
        </CardContent>
      </Card>

      <ModelComparison result={result} keys={presentKeys} />

      <SealedHoldout
        result={result}
        onFinalise={onFinalise}
        finalising={finalising}
        finaliseError={finaliseError}
        finaliseNote={finaliseNote}
      />

      {(result.features.derived?.length ?? 0) > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Inputs built from your columns</CardTitle>
            <p className="text-sm text-muted-foreground">
              These aren&apos;t in your file — they were computed from columns a model
              couldn&apos;t otherwise use. <code>train_model.py</code> rebuilds every one.
            </p>
          </CardHeader>
          <CardContent className="flex flex-col gap-2 text-sm">
            {result.features.derived!.map((recipe) => (
              <div key={recipe.id} className="rounded-md border border-border p-3">
                <span className="font-medium">{recipe.column}</span>
                <span className="text-muted-foreground"> → </span>
                <span className="text-muted-foreground">
                  {recipe.outputs.length > 0
                    ? recipe.outputs.join(", ")
                    : "its most common values, the rest pooled"}
                </span>
                <p className="mt-1 text-xs text-muted-foreground">{recipe.reason}</p>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {result.operating_point && (
        <OperatingPointCard point={result.operating_point} evaluation={result.evaluation} />
      )}

      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <ScoreLabel kind="held_out" evaluation={result.evaluation} />
        The charts below, down to the importance chart, are measured on the held-out rows.
      </div>

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
      {(deep.cross_validation || deep.learning_curve?.length) && (
        <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
          <ScoreLabel kind="exploratory" />
          The next two are scored on folds of the training rows only.
        </div>
      )}
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
