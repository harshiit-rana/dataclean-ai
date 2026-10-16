import { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ApiError,
  decideComparison,
  evaluateComparison,
  fetchDecisions,
  rollBack,
  startComparison,
} from "@/api/client";
import type {
  Comparison,
  ModelRun,
  PromotionConstraint,
  PromotionCriteria,
  PromotionDecision,
  PromotionMetric,
} from "@/api/types";
import { cn } from "@/lib/utils";

/** Replacing the model in use: a challenger, criteria fixed before anything is scored, an
 * evaluation on newly labelled rows both models have never seen, and a recorded decision.
 *
 * The order on the page is the order that makes the comparison honest. The criteria are set
 * and sent before the evaluation file is even asked for, and once it is evaluated they cannot
 * be changed. Nothing here promotes anything by itself. */

const CLASSIFICATION: PromotionMetric[] = ["roc_auc", "accuracy", "precision", "recall", "f1", "brier"];
const REGRESSION: PromotionMetric[] = ["r2", "mae", "rmse"];

const CONSTRAINT_LABEL: Record<PromotionConstraint["kind"], string> = {
  metric_at_least: "at least",
  metric_at_most: "at most",
  no_worse_than_champion: "no worse than the champion, give or take",
  no_new_required_columns: "needs no column the champion did not",
};

function errorText(err: unknown): string {
  return err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err);
}

function ConstraintRow({
  constraint,
  metrics,
  onChange,
  onRemove,
}: {
  constraint: PromotionConstraint;
  metrics: PromotionMetric[];
  onChange: (c: PromotionConstraint) => void;
  onRemove: () => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <select
        aria-label="Kind of constraint"
        className="rounded border border-border bg-background px-1 py-0.5"
        value={constraint.kind}
        onChange={(e) => {
          const kind = e.target.value as PromotionConstraint["kind"];
          if (kind === "no_new_required_columns") onChange({ kind });
          else if (kind === "no_worse_than_champion") onChange({ kind, metric: metrics[0], tolerance: 0 });
          else onChange({ kind, metric: metrics[0], value: 0 });
        }}
      >
        {Object.entries(CONSTRAINT_LABEL).map(([kind, label]) => (
          <option key={kind} value={kind}>
            {label}
          </option>
        ))}
      </select>
      {"metric" in constraint && (
        <select
          aria-label="Metric"
          className="rounded border border-border bg-background px-1 py-0.5"
          value={constraint.metric}
          onChange={(e) => onChange({ ...constraint, metric: e.target.value as PromotionMetric })}
        >
          {metrics.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
      )}
      {"value" in constraint && (
        <input
          aria-label="Value"
          type="number"
          step="0.01"
          className="w-20 rounded border border-border bg-background px-1 py-0.5"
          value={constraint.value}
          onChange={(e) => onChange({ ...constraint, value: Number(e.target.value) })}
        />
      )}
      {"tolerance" in constraint && (
        <input
          aria-label="Tolerance"
          type="number"
          step="0.01"
          min="0"
          className="w-20 rounded border border-border bg-background px-1 py-0.5"
          value={constraint.tolerance}
          onChange={(e) => onChange({ ...constraint, tolerance: Number(e.target.value) })}
        />
      )}
      <button type="button" className="text-muted-foreground underline" onClick={onRemove}>
        remove
      </button>
    </div>
  );
}

export function ReplaceModel({
  sessionId,
  runs,
  storedRunIds,
  currentRunId,
  task,
  onModelChanged,
}: {
  sessionId: string;
  /** The run history: challengers are its stored runs other than the model in use. */
  runs: ModelRun[];
  storedRunIds?: string[];
  currentRunId: string | null | undefined;
  task: string;
  /** The model in use changed (a promotion or a rollback): the page reloads its result. */
  onModelChanged: () => void;
}) {
  const metrics = task === "regression" ? REGRESSION : CLASSIFICATION;
  // Only runs whose model is still stored can be compared, and not the model in use with itself.
  const challengers = runs.filter(
    (r) =>
      (!r.status || r.status === "succeeded") &&
      r.run_id !== currentRunId &&
      (!storedRunIds || storedRunIds.includes(r.run_id))
  );
  const [challenger, setChallenger] = useState<string>("");
  const [criteria, setCriteria] = useState<PromotionCriteria>({
    deciding_metric: metrics[0],
    min_improvement: 0,
    constraints: [],
  });
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [reason, setReason] = useState("");
  const [override, setOverride] = useState(false);
  const [decisions, setDecisions] = useState<PromotionDecision[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDecisions(sessionId).then(setDecisions).catch(() => setDecisions([]));
  }, [sessionId]);

  const run = async (work: () => Promise<void>) => {
    setBusy(true);
    setError(null);
    try {
      await work();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy(false);
    }
  };

  if (!currentRunId) return null;

  const evaluation = comparison?.evaluation ?? null;
  const lastPromotion = [...decisions]
    .reverse()
    .find(
      (d) =>
        d.decision === "promoted" &&
        !decisions.some((r) => r.decision === "rolled_back" && r.from_run_id === d.challenger_run_id && r.at > d.at)
    );

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Replace the model in use</CardTitle>
        <p className="text-sm text-muted-foreground">
          Compare it with another run on newly labelled rows neither has seen, against criteria you
          set first. The criteria cannot change once the comparison is evaluated, and nothing is
          promoted unless you decide it.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 text-sm">
        {!comparison && (
          <>
            {challengers.length === 0 ? (
              <p className="text-muted-foreground">
                Train another model first: there is no other run to compare the model in use with.
              </p>
            ) : (
              <>
                <label className="flex flex-col gap-1">
                  <span className="text-xs font-medium">Challenger</span>
                  <select
                    aria-label="Challenger"
                    className="rounded border border-border bg-background px-2 py-1"
                    value={challenger}
                    onChange={(e) => setChallenger(e.target.value)}
                  >
                    <option value="">Choose a run…</option>
                    {challengers.map((r) => (
                      <option key={r.run_id} value={r.run_id}>
                        {r.model_label ?? r.best_model} · {r.at}
                      </option>
                    ))}
                  </select>
                </label>
                <div className="flex flex-wrap items-center gap-2 text-xs">
                  <span className="font-medium">Decided on</span>
                  <select
                    aria-label="Deciding metric"
                    className="rounded border border-border bg-background px-1 py-0.5"
                    value={criteria.deciding_metric}
                    onChange={(e) => setCriteria({ ...criteria, deciding_metric: e.target.value as PromotionMetric })}
                  >
                    {metrics.map((m) => (
                      <option key={m}>{m}</option>
                    ))}
                  </select>
                  <span>improving by at least</span>
                  <input
                    aria-label="Minimum improvement"
                    type="number"
                    step="0.01"
                    min="0"
                    className="w-20 rounded border border-border bg-background px-1 py-0.5"
                    value={criteria.min_improvement}
                    onChange={(e) => setCriteria({ ...criteria, min_improvement: Number(e.target.value) })}
                  />
                  <span>and beyond the noise</span>
                </div>
                <div className="flex flex-col gap-2">
                  <span className="text-xs font-medium">And it must also be…</span>
                  {criteria.constraints.map((c, i) => (
                    <ConstraintRow
                      key={i}
                      constraint={c}
                      metrics={metrics}
                      onChange={(next) =>
                        setCriteria({ ...criteria, constraints: criteria.constraints.map((x, j) => (j === i ? next : x)) })
                      }
                      onRemove={() =>
                        setCriteria({ ...criteria, constraints: criteria.constraints.filter((_, j) => j !== i) })
                      }
                    />
                  ))}
                  <button
                    type="button"
                    className="self-start text-xs text-primary underline"
                    onClick={() =>
                      setCriteria({
                        ...criteria,
                        constraints: [...criteria.constraints, { kind: "metric_at_least", metric: metrics[0], value: 0 }],
                      })
                    }
                  >
                    Add a constraint
                  </button>
                </div>
                <Button
                  disabled={!challenger || busy}
                  onClick={() =>
                    run(async () => setComparison(await startComparison(sessionId, challenger, criteria)))
                  }
                >
                  Fix these criteria and continue
                </Button>
              </>
            )}
          </>
        )}

        {comparison && !evaluation && (
          <div className="flex flex-col gap-2">
            <p>
              Criteria fixed. Now a labelled file with rows neither model was built from — rows from
              the training file are left out if it has any.
            </p>
            <input
              aria-label="Labelled evaluation file"
              type="file"
              accept=".csv"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <Button
              disabled={!file || busy}
              onClick={() =>
                run(async () => setComparison(await evaluateComparison(sessionId, comparison.comparison_id, file!)))
              }
            >
              {busy ? "Scoring both models…" : "Score both on this file"}
            </Button>
          </div>
        )}

        {comparison && evaluation && (
          <div className="flex flex-col gap-3">
            <p className={cn("font-medium", evaluation.all_passed ? "text-emerald-700 dark:text-emerald-400" : "text-amber-700 dark:text-amber-500")}>
              {evaluation.all_passed
                ? "The challenger meets every criterion."
                : "The challenger does not meet every criterion."}
            </p>
            <p className="text-xs text-muted-foreground">
              Both scored on the same {evaluation.rows_compared.toLocaleString()} rows.
              {Object.entries(evaluation.rows_excluded)
                .filter(([, n]) => n > 0)
                .map(([why, n]) => ` ${n.toLocaleString()} left out: ${why.replace(/_/g, " ")}.`)}
            </p>
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-muted-foreground">
                  <th className="py-1 pr-3 font-medium">Criterion</th>
                  <th className="py-1 pr-3 font-medium">Met</th>
                  <th className="py-1 font-medium">Evidence</th>
                </tr>
              </thead>
              <tbody>
                {[{ ...evaluation.deciding, kind: `deciding: ${evaluation.deciding.metric}` }, ...evaluation.constraints].map(
                  (c, i) => (
                    <tr key={i} className="border-t border-border/60 align-top">
                      <td className="py-1 pr-3">
                        {c.kind.replace(/_/g, " ")}
                        {c.metric && !c.kind.startsWith("deciding") ? ` (${c.metric})` : ""}
                      </td>
                      <td className="py-1 pr-3">{c.passed ? "yes" : "no"}</td>
                      <td className="py-1">{c.evidence}</td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
            {comparison.status !== "decided" && (
              <div className="flex flex-col gap-2">
                <label className="flex flex-col gap-1">
                  <span className="text-xs font-medium">Your reason (kept as you write it)</span>
                  <textarea
                    aria-label="Reason"
                    className="rounded border border-border bg-background p-2 text-sm"
                    value={reason}
                    onChange={(e) => setReason(e.target.value)}
                  />
                </label>
                {!evaluation.all_passed && (
                  <label className="flex items-center gap-2 text-xs">
                    <input type="checkbox" checked={override} onChange={(e) => setOverride(e.target.checked)} />
                    Promote anyway — recorded as an override, and needs a reason
                  </label>
                )}
                <div className="flex gap-2">
                  <Button
                    disabled={busy || (!evaluation.all_passed && !override)}
                    onClick={() =>
                      run(async () => {
                        const entry = await decideComparison(sessionId, comparison.comparison_id, "promote", reason, override);
                        setDecisions([...decisions, entry]);
                        setComparison({ ...comparison, status: "decided" });
                        onModelChanged();
                      })
                    }
                  >
                    Promote the challenger
                  </Button>
                  <Button
                    variant="outline"
                    disabled={busy}
                    onClick={() =>
                      run(async () => {
                        const entry = await decideComparison(sessionId, comparison.comparison_id, "reject", reason, false);
                        setDecisions([...decisions, entry]);
                        setComparison({ ...comparison, status: "decided" });
                      })
                    }
                  >
                    Keep the model in use
                  </Button>
                </div>
              </div>
            )}
            {comparison.status === "decided" && (
              <button
                type="button"
                className="self-start text-xs text-primary underline"
                onClick={() => {
                  setComparison(null);
                  setReason("");
                  setOverride(false);
                  setFile(null);
                }}
              >
                Start another comparison
              </button>
            )}
          </div>
        )}

        {error && <p className="text-xs text-destructive">{error}</p>}

        {decisions.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium">Decisions</span>
            <ul className="text-xs text-muted-foreground">
              {decisions.map((d) => (
                <li key={d.decision_id}>
                  {d.at}: {d.decision.replace("_", " ")}
                  {d.override ? " (override)" : ""}
                  {d.reason ? ` — “${d.reason}”` : ""}
                </li>
              ))}
            </ul>
            {lastPromotion && (
              <button
                type="button"
                className="self-start text-xs text-primary underline disabled:opacity-50"
                disabled={busy}
                onClick={() =>
                  run(async () => {
                    const entry = await rollBack(sessionId, reason);
                    setDecisions([...decisions, entry]);
                    onModelChanged();
                  })
                }
              >
                Roll back to the model the last promotion replaced
              </button>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
