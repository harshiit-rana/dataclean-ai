import { useState } from "react";
import { Link } from "react-router-dom";
import { buttonClasses } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ReadinessAction, ReadinessCheck, ReadinessCheckCounts } from "@/api/types";

/** Where an action leads. Each is a page the user decides on, with the choice set up for them; none
 * applies anything. The model page reads `exclude`, `split` and `group` from the address to pre-set its
 * form, and the user still has to press train. */
export function actionPath(action: ReadinessAction, sessionId: string): string {
  const column = action.column ? encodeURIComponent(action.column) : "";
  switch (action.kind) {
    case "exclude_column":
      return `/model/${sessionId}?exclude=${column}`;
    case "use_group_split":
      return `/model/${sessionId}?split=group&group=${column}`;
    case "choose_target":
      return "/";
  }
}

const STATUS_STYLES: Record<ReadinessCheck["status"], { mark: string; className: string }> = {
  failed: { mark: "Failed", className: "border-destructive/40 bg-destructive/5" },
  warning: { mark: "Worth knowing", className: "border-amber-500/40 bg-amber-500/5" },
  not_run: { mark: "Not checked", className: "border-border bg-muted/40" },
  passed: { mark: "Passed", className: "border-border" },
};

function CheckRow({ check, sessionId }: { check: ReadinessCheck; sessionId: string }) {
  const style = STATUS_STYLES[check.status];
  return (
    <li className={cn("rounded-md border p-3", style.className)}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-medium">{check.label}</span>
        <span className="text-xs uppercase tracking-wide text-muted-foreground">
          {style.mark}
        </span>
      </div>
      <p className="mt-1">{check.finding}</p>
      {check.evidence && <p className="mt-1 text-xs text-muted-foreground">{check.evidence}</p>}
      {check.remedy && <p className="mt-1 text-xs">{check.remedy}</p>}
      {check.status !== "passed" && (check.actions?.length ?? 0) > 0 && (
        <div className="mt-2 flex flex-wrap gap-2">
          {check.actions!.map((action) => (
            <Link
              key={`${action.kind}:${action.column ?? ""}`}
              to={actionPath(action, sessionId)}
              className={buttonClasses("outline", "sm")}
            >
              {action.label}
            </Link>
          ))}
        </div>
      )}
    </li>
  );
}

/** Why the app believes what it believes about this data.
 *
 * The verdict above this is a claim of absence — "nothing found that would stop you" — and a
 * reader cannot check that by looking at the data. This is the working.
 *
 * Leads with what failed or could not run, and collapses the passes behind a count. Twelve
 * green ticks on open is compliance theatre: people stop reading, and then the one amber row
 * disappears into the noise. */
export function CheckTrail({
  checks,
  counts,
  sessionId,
}: {
  checks: ReadinessCheck[];
  counts: ReadinessCheckCounts;
  sessionId: string;
}) {
  const [showPassed, setShowPassed] = useState(false);
  if (!checks.length) return null;

  const notable = checks.filter((c) => c.status !== "passed");
  const passed = checks.filter((c) => c.status === "passed");

  return (
    <details className="rounded-md border border-border">
      <summary className="cursor-pointer list-none px-3 py-2 text-xs text-muted-foreground">
        Why this verdict — {counts.total} check{counts.total === 1 ? "" : "s"}
        {counts.failed > 0 && `, ${counts.failed} failed`}
        {counts.warning > 0 && `, ${counts.warning} worth knowing`}
        {counts.not_run > 0 && `, ${counts.not_run} not checked`}
      </summary>

      <div className="flex flex-col gap-2 px-3 pb-3 text-sm">
        {notable.length > 0 && (
          <ul className="flex flex-col gap-2">
            {notable.map((check) => (
              <CheckRow key={check.name} check={check} sessionId={sessionId} />
            ))}
          </ul>
        )}

        {passed.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => setShowPassed((open) => !open)}
              className="self-start text-xs text-muted-foreground underline"
            >
              {showPassed
                ? "Hide the checks that passed"
                : `${passed.length} other check${passed.length === 1 ? "" : "s"} passed`}
            </button>
            {showPassed && (
              <ul className="flex flex-col gap-2">
                {passed.map((check) => (
                  <CheckRow key={check.name} check={check} sessionId={sessionId} />
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </details>
  );
}
