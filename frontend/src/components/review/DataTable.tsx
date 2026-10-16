import { memo } from "react";
import { cn } from "@/lib/utils";
import type { ColumnIssue, PreviewResponse, Severity } from "@/api/types";

interface DataTableProps {
  preview: PreviewResponse;
  issuesByColumn: Record<string, ColumnIssue[]>;
  selectedColumn: string | null;
  onSelectColumn: (name: string) => void;
}

const severityRank: Record<Severity, number> = { high: 3, medium: 2, low: 1 };

function worstSeverity(issues: ColumnIssue[]): Severity | null {
  if (issues.length === 0) return null;
  return issues.reduce((worst, i) =>
    severityRank[i.severity] > severityRank[worst.severity] ? i : worst
  ).severity;
}

const severityDot: Record<Severity, string> = {
  high: "bg-red-500",
  medium: "bg-amber-500",
  low: "bg-blue-500",
};

function formatCell(value: string | number | boolean | null) {
  if (value === null) return { text: "null", isNull: true };
  return { text: String(value), isNull: false };
}

/** Memoized: the review page re-renders as suggestion text streams in, but the table's
 * inputs only change when profiling finishes or a column is selected. Re-rendering every
 * cell on each token flush is pure waste. */
export const DataTable = memo(function DataTable({
  preview,
  issuesByColumn,
  selectedColumn,
  onSelectColumn,
}: DataTableProps) {
  return (
    <div className="rounded-lg border border-border">
      <div className="flex items-center justify-between border-b border-border px-4 py-2 text-xs text-muted-foreground">
        <span>
          Showing {preview.preview_row_count} of {preview.total_rows} rows &middot;{" "}
          {preview.total_columns} columns
        </span>
        <span>Choose a column header for details</span>
      </div>

      {/* Height-capped rather than rendering all 100 rows inline: the suggestion cards are the
          point of this page, and a full-length table pushed them a long way below the fold --
          far enough that a wheel gesture over the table never reached them. */}
      {/* Focusable so the rows can be scrolled from the keyboard as well as the wheel (#101). */}
      <div
        className="max-h-[26rem] overflow-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
        tabIndex={0}
        role="region"
        aria-label="Data preview"
      >
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr>
              {preview.columns.map((col) => {
                const issues = issuesByColumn[col.name] ?? [];
                const severity = worstSeverity(issues);
                const isSelected = selectedColumn === col.name;
                return (
                  <th
                    key={col.name}
                    scope="col"
                    className={cn(
                      // Sticky so the column names stay readable while the rows scroll under
                      // them. The background has to be opaque or the rows show through.
                      "sticky top-0 z-10 whitespace-nowrap border-b border-border bg-background p-0 text-left font-medium",
                      isSelected && "bg-muted"
                    )}
                  >
                    {/* A button, not a clickable cell, so the details open from the keyboard too (#101). */}
                    <button
                      type="button"
                      onClick={() => onSelectColumn(col.name)}
                      aria-pressed={isSelected}
                      className="flex w-full items-center gap-1.5 px-3 py-2 text-left font-medium transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-primary"
                    >
                      {severity && (
                        <span
                          className={cn("h-2 w-2 shrink-0 rounded-full", severityDot[severity])}
                          role="img"
                          aria-label={`${severity} severity issue`}
                        />
                      )}
                      {col.name}
                      {col.is_critical && (
                        <span className="text-xs font-normal text-primary" title="Marked critical">
                          <span aria-hidden="true">&#9733;</span>
                          <span className="sr-only">(critical)</span>
                        </span>
                      )}
                      {col.is_target && (
                        <span
                          className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-primary"
                          title="The column you're predicting"
                        >
                          target
                        </span>
                      )}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, idx) => (
              <tr key={idx} className="border-b border-border last:border-0">
                {preview.columns.map((col) => {
                  const { text, isNull } = formatCell(row[col.name] ?? null);
                  return (
                    <td
                      key={col.name}
                      className={cn(
                        "max-w-[220px] truncate whitespace-nowrap px-3 py-1.5",
                        selectedColumn === col.name && "bg-muted",
                        isNull && "italic text-muted-foreground"
                      )}
                      title={text}
                    >
                      {text}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
});
