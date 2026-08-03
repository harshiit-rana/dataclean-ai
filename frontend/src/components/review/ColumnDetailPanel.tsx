import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ColumnIssue, PreviewColumn, PreviewResponse } from "@/api/types";

interface ColumnDetailPanelProps {
  column: PreviewColumn;
  issues: ColumnIssue[];
  preview: PreviewResponse;
  onClose: () => void;
}

const ISSUE_LABELS: Record<string, string> = {
  missing_values: "Missing values",
  duplicates: "Duplicate rows",
  type_mismatch: "Wrong data type",
  outliers: "Outliers",
  inconsistent_format: "Inconsistent formatting",
  high_cardinality: "Mostly unique values",
};

export function ColumnDetailPanel({ column, issues, preview, onClose }: ColumnDetailPanelProps) {
  const filledCount = preview.total_rows - column.null_count;
  const fillRate = preview.total_rows > 0 ? (filledCount / preview.total_rows) * 100 : 0;
  const looksLikeIdentifier =
    column.unique_count === preview.total_rows && preview.total_rows > 0;

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="flex items-center gap-2 text-base">
          {column.name}
          {column.is_critical && <Badge variant="low">critical</Badge>}
        </CardTitle>
        <button
          onClick={onClose}
          className="text-sm text-muted-foreground hover:text-foreground"
          aria-label="Close column details"
        >
          Close
        </button>
      </CardHeader>

      <CardContent className="flex flex-col gap-4 text-sm">
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div>
            <p className="text-xs text-muted-foreground">Type</p>
            <p className="font-medium">{column.dtype}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Filled</p>
            <p className="font-medium">{fillRate.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Missing</p>
            <p className="font-medium">{column.null_count.toLocaleString()}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Distinct values</p>
            <p className="font-medium">{column.unique_count.toLocaleString()}</p>
          </div>
        </div>

        <div>
          <p className="mb-1 font-medium">Role in your data</p>
          <p className="text-muted-foreground">
            {column.is_critical
              ? "You marked this column as critical for your goal, so issues here are prioritized above others."
              : looksLikeIdentifier
                ? "Every value in this column is distinct, so it most likely identifies each row rather than describing it — useful for joining or tracing rows, but rarely meaningful to analyze on its own."
                : `This column holds ${column.unique_count.toLocaleString()} distinct values across ${preview.total_rows.toLocaleString()} rows.`}
          </p>
        </div>

        <div>
          <p className="mb-2 font-medium">
            {issues.length > 0
              ? `Issues found (${issues.length})`
              : "No issues found in this column"}
          </p>
          {issues.length === 0 ? (
            <p className="text-muted-foreground">
              Nothing flagged here — this column looks clean.
            </p>
          ) : (
            <ul className="flex flex-col gap-2">
              {issues.map((issue) => (
                <li key={issue.issue_id} className="rounded-md border border-border p-3">
                  <div className="mb-1 flex items-center gap-2">
                    <Badge variant={issue.severity}>{issue.severity}</Badge>
                    <span className="font-medium">
                      {ISSUE_LABELS[issue.issue_type] ?? issue.issue_type}
                    </span>
                  </div>
                  <p className="text-muted-foreground">{issue.description}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
