import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Button, buttonClasses } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DownloadButton } from "@/components/ui/download-button";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import { FileDropzone } from "@/components/upload/FileDropzone";
import { ApiError, applyToNewFile } from "@/api/client";
import { OPERATION_LABELS } from "@/lib/operations";
import { cn } from "@/lib/utils";
import { NotJudged } from "@/components/NotJudged";
import { ValidationCard } from "@/components/apply/ValidationCard";
import { PreviousApplications } from "@/components/apply/PreviousApplications";
import type { AppliedStep, ApplyResult, DriftColumn, DriftReport } from "@/api/types";

const DRIFT_STYLES: Record<DriftReport["verdict"], { title: string; className: string }> = {
  similar: { title: "Looks like the training data", className: "border-emerald-500/40 bg-emerald-500/5" },
  shifted: { title: "Some inputs have moved", className: "border-amber-500/40 bg-amber-500/5" },
  different: {
    title: "This isn't the data the model learned from",
    className: "border-destructive/40 bg-destructive/5",
  },
  // Not styled as a failure, for the same reason the model verdict isn't: "can't tell" is not
  // "it's bad".
  not_enough_rows: { title: "Too few rows to judge", className: "border-sky-500/40 bg-sky-500/5" },
};

const STATUS_TEXT: Record<AppliedStep["status"], string> = {
  applied: "Applied",
  skipped: "Skipped",
  failed: "Couldn't run",
};

const pct = (v: number | undefined) => (typeof v === "number" ? `${Math.round(v * 100)}%` : "—");

function describeLearned(step: AppliedStep): string | null {
  if (!step.learned) return null;
  if ("fill_value" in step.learned) return `with ${String(step.learned.fill_value)}`;
  if ("lower" in step.learned) {
    const lower = Number(step.learned.lower);
    const upper = Number(step.learned.upper);
    return `range ${lower.toFixed(2)} to ${upper.toFixed(2)}`;
  }
  return null;
}

function blankDetail(column: DriftColumn): string {
  if (!column.blank_status || column.blank_status === "stable") return "";
  return `blank in ${pct(column.training_blank_share)} of training rows, ${pct(column.new_blank_share)} here`;
}

function counts(column: DriftColumn): string {
  return column.training_values != null && column.new_values != null
    ? `${column.training_values.toLocaleString()} / ${column.new_values.toLocaleString()}`
    : "—";
}

function driftDetail(column: DriftColumn): string {
  const blank = blankDetail(column);
  if (column.note) return blank ? `${blank}. ${column.note}` : column.note;
  const values = valueDetail(column);
  return [blank, values].filter(Boolean).join("; ");
}

function valueDetail(column: DriftColumn): string {
  if (column.kind === "categorical") {
    const unseen = column.unseen_share ?? 0;
    return unseen > 0
      ? `${pct(unseen)} of rows carry a value the model never saw (${(column.unseen_examples ?? []).join(", ")})`
      : "Every value was seen in training";
  }
  const outside = column.outside_training_range ?? 0;
  const medians =
    column.training_median != null && column.new_median != null
      ? `median ${column.training_median.toFixed(2)} in training, ${column.new_median.toFixed(2)} here`
      : "";
  return outside > 0 ? `${medians}; ${pct(outside)} outside the training range` : medians;
}

/** Page 6: use a finished session on a new file.
 *
 * A session used to be single-use -- next month's export meant approving every fix again,
 * and the model could only be used by writing Python against the downloaded joblib. This
 * cleans the new file with the session's approved fixes (and the statistics it learned, not
 * the new file's own), checks it against the data the model learned from, and predicts every
 * row. A row a fix removed comes back with the reason rather than going missing. */
export function ApplyPage() {
  const { sessionId } = useParams();
  const [file, setFile] = useState<File | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ApplyResult | null>(null);

  const run = async () => {
    if (!sessionId || !file) return;
    setRunning(true);
    setError(null);
    setResult(null);
    try {
      setResult(await applyToNewFile(sessionId, file));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setRunning(false);
    }
  };

  const drift = result?.drift ? DRIFT_STYLES[result.drift.verdict] : null;

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Use this on a new file</h1>
        <p className="text-muted-foreground">
          Upload next month&apos;s export, or any file with the same columns. It&apos;s cleaned
          with the fixes you approved here, using the values this session learned — the same
          median, the same limits — so it matches the data the model was built on. If you
          trained a model, every row gets a prediction.
        </p>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-3 p-6">
          <FileDropzone onFileSelected={setFile} selectedFile={file} />
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={run} disabled={!file || running}>
              {running ? "Working..." : "Clean and predict"}
            </Button>
            <Link to={`/export/${sessionId}`} className={buttonClasses("ghost", "sm")}>
              Back to the export page
            </Link>
          </div>
        </CardContent>
      </Card>

      {error && (
        <div className="rounded-md border border-destructive/50 p-3 text-sm">
          <p className="font-medium text-destructive">That file couldn&apos;t be used</p>
          <p className="mt-1 text-muted-foreground">{error}</p>
        </div>
      )}

      {result && (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                {result.rows_kept.toLocaleString()} of {result.rows_in.toLocaleString()} rows kept
              </CardTitle>
              <p className="text-sm text-muted-foreground">
                {result.dropped_count > 0
                  ? `${result.dropped_count.toLocaleString()} ${result.dropped_count === 1 ? "row was" : "rows were"} removed by a fix. ${
                      result.predictions
                        ? "They're still in the predictions file, with the reason and no prediction."
                        : "They're listed below with the reason."
                    }`
                  : "No rows were removed."}
              </p>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 text-sm">
              <div className="flex flex-wrap gap-2">
                {result.downloads.predictions && (
                  <DownloadButton
                    sessionId={sessionId!}
                    path={result.downloads.predictions}
                    filename={`predictions_${result.file_name}`}
                    onError={setError}
                  >
                    Download predictions
                  </DownloadButton>
                )}
                <DownloadButton
                  sessionId={sessionId!}
                  path={result.downloads.cleaned}
                  filename={`cleaned_${result.file_name}`}
                  variant="outline"
                  onError={setError}
                >
                  Download the cleaned file
                </DownloadButton>
              </div>
              {result.notes.map((note) => (
                <p key={note} className="rounded-md border border-amber-500/40 bg-amber-500/5 p-3">
                  {note}
                </p>
              ))}
            </CardContent>
          </Card>

          {result.validation && <ValidationCard validation={result.validation} />}

          {result.drift && drift && (
            <Card className={cn("border", drift.className)}>
              <CardHeader>
                <CardTitle className="text-base">{drift.title}</CardTitle>
                {result.drift.not_judged ? (
                  <NotJudged block={result.drift.not_judged}>{result.drift.headline}</NotJudged>
                ) : (
                  <p className="text-sm">{result.drift.headline}</p>
                )}
              </CardHeader>
              {result.drift.columns.length > 0 && (
                <CardContent className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                        <th className="py-2 pr-3 font-medium">Input</th>
                        <th className="py-2 pr-3 font-medium">Compared with training</th>
                        <th className="py-2 pr-3 text-right font-medium">Shift (PSI)</th>
                        <th className="py-2 pr-3 text-right font-medium">Chance reaches</th>
                        <th className="py-2 pr-3 text-right font-medium">Values (training / here)</th>
                        <th className="py-2 font-medium">What changed</th>
                      </tr>
                    </thead>
                    <tbody>
                      {result.drift.columns.map((column) => (
                        <tr key={column.column} className="border-b border-border/60 align-top last:border-0">
                          <td className="py-1.5 pr-3 font-medium">{column.column}</td>
                          <td
                            className={cn(
                              "py-1.5 pr-3",
                              column.status === "different" && "text-destructive",
                              column.status === "shifted" && "text-amber-700 dark:text-amber-500"
                            )}
                          >
                            {column.status === "stable"
                              ? "same"
                              : column.status === "not_judged"
                                ? "too few values"
                                : column.status}
                          </td>
                          <td className="py-1.5 pr-3 text-right tabular-nums">
                            {column.psi != null ? column.psi.toFixed(3) : "—"}
                          </td>
                          <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground">
                            {column.chance_threshold != null ? column.chance_threshold.toFixed(3) : "—"}
                          </td>
                          <td className="py-1.5 pr-3 text-right tabular-nums text-muted-foreground">
                            {counts(column)}
                          </td>
                          <td className="py-1.5 text-xs text-muted-foreground">{driftDetail(column)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <p className="mt-2 text-xs text-muted-foreground">
                    PSI under 0.1 is usually read as stable and over 0.25 as a different
                    population. Those are conventions, not statistics, so a column is only called
                    moved when its PSI is also above what chance reaches for samples of these sizes
                    (with every column checked at once, not one at a time). The numbers are here so
                    you can judge.
                  </p>
                </CardContent>
              )}
            </Card>
          )}

          {result.answers && (
            <Card>
              <CardHeader className="flex-row flex-wrap items-center justify-between gap-2 space-y-0">
                <CardTitle className="text-base">Checked against the real answers in this file</CardTitle>
                <ScoreLabel kind="new_data" />
              </CardHeader>
              <CardContent className="flex flex-col gap-2 text-sm">
                {result.answers.not_judged ? (
                  <NotJudged block={result.answers.not_judged}>{result.answers.headline}</NotJudged>
                ) : (
                  <p>{result.answers.headline}</p>
                )}
                {result.answers.counted_as_independent && (
                  <p className="text-xs text-muted-foreground">
                    Recorded as an independent check on this model — these rows did not exist
                    when it was built — and shown beside its original score.
                  </p>
                )}
                {result.answers.already_counted && (
                  <p className="text-xs text-muted-foreground">
                    This exact file has already been counted for this model, so it is not
                    counted a second time.
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {result.predictions && result.predictions.count > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">
                  {result.predictions.count.toLocaleString()} predictions
                </CardTitle>
                {result.predictions.cutoff != null && (
                  <p className="text-sm text-muted-foreground">
                    Answered at the cut-off you chose ({result.predictions.cutoff.toFixed(2)}),
                    the same line the model&apos;s reported numbers were measured at.
                  </p>
                )}
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="py-2 pr-3 font-medium">Row</th>
                      {result.predictions.columns.map((column) => (
                        <th key={column} className="py-2 pr-3 font-medium">
                          {column}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.predictions.preview.map((row) => (
                      <tr key={String(row.row)} className="border-b border-border/60 last:border-0">
                        <td className="py-1.5 pr-3 tabular-nums text-muted-foreground">{row.row}</td>
                        {result.predictions!.columns.map((column) => (
                          <td key={column} className="py-1.5 pr-3 tabular-nums">
                            {String(row[column] ?? "—")}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
                {result.predictions.count > result.predictions.preview.length && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    The first {result.predictions.preview.length} of{" "}
                    {result.predictions.count.toLocaleString()} — the download has them all.
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {result.steps.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">The fixes, as applied to this file</CardTitle>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-left text-xs uppercase tracking-wide text-muted-foreground">
                      <th className="py-2 pr-3 font-medium">Fix</th>
                      <th className="py-2 pr-3 font-medium">Column</th>
                      <th className="py-2 pr-3 font-medium">Result</th>
                      <th className="py-2 text-right font-medium">Rows removed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.steps.map((step) => (
                      <tr key={step.action_id} className="border-b border-border/60 align-top last:border-0">
                        <td className="py-1.5 pr-3">
                          {OPERATION_LABELS[step.operation] ?? step.operation}
                          {describeLearned(step) && (
                            <span className="block text-xs text-muted-foreground">
                              {describeLearned(step)}, as learned from your original file
                            </span>
                          )}
                        </td>
                        <td className="py-1.5 pr-3">{step.column ?? "—"}</td>
                        <td className="py-1.5 pr-3">
                          <span className={cn(step.status === "failed" && "text-destructive")}>
                            {STATUS_TEXT[step.status]}
                          </span>
                          {step.note && (
                            <span className="block text-xs text-muted-foreground">{step.note}</span>
                          )}
                        </td>
                        <td className="py-1.5 text-right tabular-nums">{step.dropped_count || "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </CardContent>
            </Card>
          )}

          {result.dropped.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Rows removed</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="flex flex-col gap-1 text-sm">
                  {result.dropped.map((row) => (
                    <li key={row.row}>
                      <span className="tabular-nums text-muted-foreground">Data row {row.row}:</span>{" "}
                      {row.reason}
                    </li>
                  ))}
                </ul>
                {result.dropped_count > result.dropped.length && (
                  <p className="mt-2 text-xs text-muted-foreground">
                    And {result.dropped_count - result.dropped.length} more.
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </>
      )}

      {sessionId && <PreviousApplications sessionId={sessionId} refreshKey={result?.apply_id} />}
    </div>
  );
}
