import { Card, CardContent } from "@/components/ui/card";
import type { AppliedAction } from "@/hooks/useCleaningRun";
import type { ColumnStats } from "@/api/types";

interface ActionLogProps {
  actions: AppliedAction[];
  status: "idle" | "running" | "done" | "error";
}

const OPERATION_LABELS: Record<string, string> = {
  fill_median: "Filled missing values with the median",
  fill_mean: "Filled missing values with the mean",
  fill_mode: "Filled missing values with the most common value",
  fill_constant: "Filled missing values with a fixed value",
  drop_rows_with_nulls: "Dropped rows with missing values",
  drop_duplicate_rows: "Removed duplicate rows",
  cast_to_numeric: "Converted to numbers",
  cast_to_datetime: "Converted to dates",
  standardize_case: "Made capitalization consistent",
  trim_whitespace: "Removed extra spaces",
  standardize_dates: "Standardized date formats",
  cap_outliers: "Clamped extreme values",
  drop_outlier_rows: "Dropped rows with extreme values",
  drop_column: "Removed the column",
};

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

              {changes.length > 0 && (
                <ul className="flex flex-col gap-0.5 text-xs text-muted-foreground">
                  {changes.map((c) => (
                    <li key={c}>{c}</li>
                  ))}
                </ul>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
