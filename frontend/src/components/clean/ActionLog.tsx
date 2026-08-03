import { Card, CardContent } from "@/components/ui/card";
import type { AppliedAction } from "@/hooks/useCleaningRun";
import type { ColumnStats } from "@/api/types";
import { OPERATION_LABELS } from "@/lib/operations";

interface ActionLogProps {
  actions: AppliedAction[];
  status: "idle" | "running" | "done" | "error";
}

/** Renders one human-readable "what changed" line per stat that actually moved, so the
 * user sees concrete impact rather than a raw stats dump. */
function changeSummary(before?: ColumnStats, after?: ColumnStats): string[] {
  if (!before || !after) return [];
  const changes: string[] = [];

  if (before.row_count !== after.row_count) {
    const removed = before.row_count - after.row_count;
    changes.push(
      removed > 0
        ? `${removed.toLocaleString()} rows removed (${before.row_count.toLocaleString()} → ${after.row_count.toLocaleString()})`
        : `${(-removed).toLocaleString()} rows added`
    );
  }
  if (before.null_count !== undefined && after.null_count !== undefined && before.null_count !== after.null_count) {
    changes.push(`missing values: ${before.null_count.toLocaleString()} → ${after.null_count.toLocaleString()}`);
  }
  if (before.max !== undefined && after.max !== undefined && before.max !== after.max) {
    changes.push(`max value: ${before.max.toLocaleString()} → ${after.max.toLocaleString()}`);
  }
  if (
    before.unique_value_count !== undefined &&
    after.unique_value_count !== undefined &&
    before.unique_value_count !== after.unique_value_count
  ) {
    changes.push(
      `distinct values: ${before.unique_value_count.toLocaleString()} → ${after.unique_value_count.toLocaleString()}`
    );
  }
  return changes;
}

export function ActionLog({ actions, status }: ActionLogProps) {
  if (actions.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">
        {status === "running" ? "Starting..." : "No fixes were applied."}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {actions.map((action) => {
        const done = Boolean(action.after_stats);
        const changes = changeSummary(action.before_stats, action.after_stats);
        return (
          <Card key={action.action_id}>
            <CardContent className="flex flex-col gap-2 p-4 text-sm">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <span className="font-medium">{action.column_name}</span>
                  <span className="text-muted-foreground">
                    {" "}
                    — {OPERATION_LABELS[action.operation] ?? action.operation}
                  </span>
                </div>
                {done ? (
                  <span className="shrink-0 text-xs text-muted-foreground">done</span>
                ) : (
                  <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-primary" />
                )}
              </div>

              {changes.length > 0 ? (
                <ul className="flex flex-col gap-0.5 text-xs text-muted-foreground">
                  {changes.map((c) => (
                    <li key={c}>{c}</li>
                  ))}
                </ul>
              ) : (
                done && (
                  // Reformatting operations (date/case/whitespace) rewrite values without
                  // moving row counts, null counts, or distinct counts -- without this the
                  // row just reads "done" with no evidence anything happened.
                  <p className="text-xs text-muted-foreground">
                    values rewritten in place — row and value counts unchanged
                  </p>
                )
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
