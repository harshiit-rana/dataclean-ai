import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ScoreLabel } from "@/components/model/ScoreLabel";
import { cn } from "@/lib/utils";
import type { ModelRun, RunComparison, VersionDiff } from "@/api/types";

/** What has been tried in this session, and whether it helped.
 *
 * Results used to overwrite a single file, so retraining discarded the previous run and the
 * one question people actually ask became unanswerable: *did the thing I just changed help?*
 * The verdict on a single run says whether a model is worth anything. Only the sequence says
 * whether a decision was.
 *
 * The change column is the point of the table. A list of scores makes you do the subtraction
 * yourself, and the subtraction is the finding. */

/** An attempt that produced no model -- refused before fitting or failed during it. */
export function isAttempt(run: ModelRun): boolean {
  return run.status === "failed" || run.status === "refused" || run.status === "interrupted";
}

/** Anything in the history that is not a run with a score: an attempt, or a model brought back. */
export function isEvent(run: ModelRun): boolean {
  return isAttempt(run) || run.status === "restored";
}

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
  // The usual fifth goes unmentioned; anything else is part of the setup, because a different
  // fraction is a different test set.
  const heldBack = run.split.requested_test_size;
  if (heldBack != null && Math.abs(heldBack - 0.2) > 1e-9) {
    parts.push(`${Math.round(heldBack * 100)}% held back`);
  }
  if (run.split.gap_days) parts.push(`${run.split.gap_days}-day gap`);
  if (run.split_overridden) parts.push("overrode the recommended split");
  if (run.tuned) {
    // Which search found the settings, when the run recorded it: two tuned runs searched differently
    // are not quite the same experiment.
    parts.push(run.search_strategies?.length ? `tuned (${run.search_strategies.join(" + ")})` : "tuned");
  }
  if (run.calibration && run.calibration !== "none") parts.push(`calibrated (${run.calibration})`);
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

/** What produced the run, one fact per line, for the run's "how it was made" details. Null for a
 * run from before this was recorded -- shown as "not recorded", never as blank or as unknown values. */
export function describeProvenance(run: ModelRun): string[] | null {
  if (!run.config_hash && !run.environment) return null;
  const lines: string[] = [];
  if (run.config_hash) lines.push(`configuration ${run.config_hash}`);
  if (run.runtime_seconds != null) {
    const seconds = run.runtime_seconds;
    lines.push(seconds < 60 ? `took ${seconds.toFixed(1)} s` : `took ${Math.floor(seconds / 60)} min ${Math.round(seconds % 60)} s`);
  }
  if (run.dataset_version) {
    lines.push(
      `data version ${run.dataset_version.version} (${run.dataset_version.rows.toLocaleString()} rows, ${run.dataset_version.columns} columns)`
    );
  }
  if (run.selection?.method) lines.push(`chosen by ${run.selection.method.replace(/_/g, " ")}`);
  const env = run.environment;
  if (env) {
    lines.push(
      [
        `Python ${env.python}`,
        `scikit-learn ${env.scikit_learn ?? "?"}`,
        `pandas ${env.pandas ?? "?"}`,
        `numpy ${env.numpy ?? "?"}`,
      ].join(", ")
    );
  }
  if (run.seeds) {
    lines.push(`seeds ${Object.entries(run.seeds).map(([k, v]) => `${k} ${v}`).join(", ")}`);
  }
  return lines;
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

function Provenance({ run }: { run: ModelRun }) {
  const lines = describeProvenance(run);
  if (!lines) {
    return <span className="mt-0.5 block text-xs text-muted-foreground">how it was made: not recorded</span>;
  }
  return (
    <details className="mt-0.5 text-xs text-muted-foreground">
      <summary className="cursor-pointer">how it was made</summary>
      <ul className="mt-1 space-y-0.5">
        {lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </details>
  );
}

/** The diff as a few lines a reader can take in at once. */
export function describeDiff(diff: VersionDiff): string[] {
  const lines: string[] = [];
  const rows = diff.rows;
  if (rows.removed || rows.added) {
    const parts = [rows.removed ? `${rows.removed.toLocaleString()} removed` : "", rows.added ? `${rows.added.toLocaleString()} added` : ""].filter(Boolean);
    lines.push(`Rows: ${rows.from.toLocaleString()} → ${rows.to.toLocaleString()} (${parts.join(", ")})`);
  } else if (rows.change) {
    lines.push(`Rows: ${rows.from.toLocaleString()} → ${rows.to.toLocaleString()}`);
  } else {
    lines.push("Rows: no change");
  }
  if (diff.columns?.removed.length) lines.push(`Columns removed: ${diff.columns.removed.join(", ")}`);
  if (diff.columns?.added.length) lines.push(`Columns added: ${diff.columns.added.join(", ")}`);
  if (diff.cells) {
    const changed = Object.entries(diff.cells);
    lines.push(
      changed.length
        ? `Values changed: ${changed.map(([column, count]) => `${column} (${count.toLocaleString()})`).join(", ")}`
        : "Values changed: none on the rows both versions have"
    );
  }
  const label = (a: { operation: string; column_name: string | null }) =>
    `${a.operation.replace(/_/g, " ")}${a.column_name && a.column_name !== "DATASET" ? ` on ${a.column_name}` : ""}`;
  if (diff.actions?.applied.length) lines.push(`Fixes applied since: ${diff.actions.applied.map(label).join("; ")}`);
  if (diff.actions?.undone.length) lines.push(`Fixes undone since: ${diff.actions.undone.map(label).join("; ")}`);
  return [...lines, ...diff.notes];
}

/** Why a run cannot be compared with the one before it, in words: the data changed, the metric
 * changed, or one of them has no score. Empty when they can be compared. */
export function whyNotComparable(run: ModelRun, previous: ModelRun): string[] {
  const reasons: string[] = [];
  if (previous.data_fingerprint !== run.data_fingerprint) reasons.push("data changed");
  if (previous.primary_metric !== run.primary_metric) {
    reasons.push(`different metric (${previous.primary_metric ?? "none"} → ${run.primary_metric ?? "none"})`);
  }
  if (!reasons.length && (metricOf(run) == null || metricOf(previous) == null)) reasons.push("no score to compare");
  return reasons;
}

/** "What changed" for a run trained on an earlier version of the data, fetched when asked for. */
function WhatChanged({ from, to, fetchDiff }: { from: number; to: number; fetchDiff: (from: number, to: number) => Promise<VersionDiff> }) {
  const [lines, setLines] = useState<string[] | null>(null);
  const [failed, setFailed] = useState(false);
  const [loading, setLoading] = useState(false);
  if (lines) {
    return (
      <ul className="mt-1 space-y-0.5 text-muted-foreground">
        {lines.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    );
  }
  return (
    <button
      type="button"
      className="mt-0.5 block text-xs text-primary underline underline-offset-2 disabled:opacity-50"
      disabled={loading}
      onClick={async () => {
        setLoading(true);
        setFailed(false);
        try {
          setLines(describeDiff(await fetchDiff(from, to)));
        } catch {
          setFailed(true);
        } finally {
          setLoading(false);
        }
      }}
    >
      {loading ? "Comparing the data…" : failed ? "Could not compare; try again" : `What changed since (version ${from} → ${to})`}
    </button>
  );
}

/** A refused or failed attempt, or a model brought back, in sequence with the runs. */
function EventRow({ run, latest }: { run: ModelRun; latest: boolean }) {
  const restored = run.status === "restored";
  return (
    <tr className="border-b border-border/60 align-top">
      <td className="py-2 pr-3">
        <span className="font-medium">
          {restored
            ? "Restored"
            : run.status === "refused"
              ? "Refused"
              : run.status === "interrupted"
                ? "Interrupted"
                : "Failed"}
        </span>
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {when(run.at)}
          {latest && !restored && " · latest attempt"}
        </span>
        {!restored && <Provenance run={run} />}
      </td>
      <td className="py-2 pr-3 text-xs text-muted-foreground" colSpan={2}>
        {run.reason ??
          (restored
            ? "An earlier model was made the current one."
            : run.status === "refused"
              ? "Nothing was trained."
              : "Training did not finish.")}
      </td>
      <td className="py-2 pr-3 text-xs text-muted-foreground">{restored ? "no new score" : "no model"}</td>
      <td className="py-2 text-xs">—</td>
    </tr>
  );
}

export function RunHistory({
  runs,
  currentFingerprint,
  comparisons,
  currentRunId,
  storedRunIds,
  onRestore,
  restoring = null,
  restoreError = null,
  olderCount = 0,
  onLoadOlder,
  loadingOlder = false,
  currentDatasetVersion = null,
  fetchDiff,
}: {
  runs: ModelRun[];
  currentFingerprint: string;
  /** Whether each run's change from the one before is real; see `RunComparison`. */
  comparisons?: Record<string, RunComparison>;
  /** The run whose model is in use, from the server; null when none is. */
  currentRunId?: string | null;
  /** Runs whose model is still stored, and so can be brought back. */
  storedRunIds?: string[];
  /** Makes a stored run on the current data the model in use. */
  onRestore?: (runId: string) => void;
  /** The run being brought back, while the request is in flight. */
  restoring?: string | null;
  restoreError?: string | null;
  /** Entries older than the ones shown. Said, never cut silently. */
  olderCount?: number;
  onLoadOlder?: () => void;
  loadingOlder?: boolean;
  /** The data's version now; with `fetchDiff`, a run on an earlier one can show what changed. */
  currentDatasetVersion?: number | null;
  fetchDiff?: (from: number, to: number) => Promise<VersionDiff>;
}) {
  // One run is not a comparison, and a table of one row implies there is something to compare.
  if (runs.length < 2) return null;
  // The model in use, as the server says. Without that, the newest run that produced one: a later
  // failed attempt did not replace it.
  const currentId = currentRunId !== undefined ? currentRunId : runs.find((r) => !isEvent(r))?.run_id;
  const stored = new Set(storedRunIds ?? []);

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
                if (isEvent(run)) return <EventRow key={run.run_id} run={run} latest={index === 0} />;
                const score = metricOf(run);
                // Compared against the last run before it that has a score: an attempt or a restore
                // sits between them in the list but has nothing to subtract.
                const previous = runs.slice(index + 1).find((r) => !isEvent(r));
                const previousScore = previous ? metricOf(previous) : null;
                const comparable =
                  score != null &&
                  previousScore != null &&
                  previous?.data_fingerprint === run.data_fingerprint &&
                  previous?.primary_metric === run.primary_metric;
                const delta = comparable ? score - previousScore : null;
                const comparison = comparisons?.[run.run_id];
                const verdict = comparable ? (comparison?.verdict ?? null) : null;
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
                        {run.run_id === currentId && " · current"}
                      </span>
                      <Provenance run={run} />
                      {/* Only a stored model trained on the data as it is now: an older one's scores
                          describe rows that are no longer the data. */}
                      {onRestore && stored.has(run.run_id) && run.run_id !== currentId && !stale && (
                        <button
                          type="button"
                          className="mt-1 block text-xs text-primary underline underline-offset-2 disabled:opacity-50"
                          disabled={restoring !== null}
                          onClick={() => onRestore(run.run_id)}
                        >
                          {restoring === run.run_id ? "Bringing it back…" : "Make this the current model"}
                        </button>
                      )}
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
                      {stale &&
                        fetchDiff &&
                        currentDatasetVersion != null &&
                        run.dataset_version?.version != null &&
                        run.dataset_version.version !== currentDatasetVersion && (
                          <WhatChanged
                            from={run.dataset_version.version}
                            to={currentDatasetVersion}
                            fetchDiff={fetchDiff}
                          />
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
                          {previous
                            ? `not comparable: ${whyNotComparable(run, previous).join("; ") || "no score to compare"}`
                            : olderCount > 0
                              ? "earlier runs not shown"
                              : "first run"}
                          {/* The data changed between the two: say what, when both versions are known. */}
                          {previous &&
                            fetchDiff &&
                            previous.data_fingerprint !== run.data_fingerprint &&
                            previous.dataset_version?.version != null &&
                            run.dataset_version?.version != null &&
                            previous.dataset_version.version !== run.dataset_version.version && (
                              <WhatChanged
                                from={previous.dataset_version.version}
                                to={run.dataset_version.version}
                                fetchDiff={fetchDiff}
                              />
                            )}
                        </span>
                      ) : (
                        <>
                          {/* Coloured only when the change is outside what the test rows could
                              produce by chance. A difference inside the noise used to read as
                              "this change helped" or "hurt" -- the one question this table is
                              for -- and a run with nothing to judge it by gets no colour. */}
                          <span
                            className={cn(
                              "font-medium",
                              verdict === "better" && "text-emerald-600 dark:text-emerald-400",
                              verdict === "worse" && "text-destructive"
                            )}
                          >
                            {delta > 0 ? "+" : ""}
                            {delta.toFixed(4)}
                          </span>
                          {verdict === "within_the_noise" && (
                            <span className="mt-0.5 block text-xs text-muted-foreground">
                              within the noise
                            </span>
                          )}
                          {comparison?.method === "paired" &&
                            comparison.low != null &&
                            comparison.high != null && (
                              <span className="mt-0.5 block text-xs text-muted-foreground">
                                95% range {comparison.low >= 0 ? "+" : ""}
                                {comparison.low.toFixed(3)} to {comparison.high >= 0 ? "+" : ""}
                                {comparison.high.toFixed(3)}
                              </span>
                            )}
                        </>
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
        {olderCount > 0 && (
          <p className="mt-3 text-xs text-muted-foreground">
            {olderCount} older {olderCount === 1 ? "entry" : "entries"} not shown.{" "}
            {onLoadOlder && (
              <button
                type="button"
                className="text-primary underline underline-offset-2 disabled:opacity-50"
                disabled={loadingOlder}
                onClick={onLoadOlder}
              >
                {loadingOlder ? "Loading…" : "Show older runs"}
              </button>
            )}
          </p>
        )}
        {restoreError && <p className="mt-3 text-xs text-destructive">{restoreError}</p>}
        <p className="mt-3 text-xs text-muted-foreground">
          The models of recent runs are kept. One trained on the data as it is now can be made the
          current model again, and its card, script and download follow it. A run from before a
          cleaning change can only be reproduced by retraining.
        </p>
      </CardContent>
    </Card>
  );
}
