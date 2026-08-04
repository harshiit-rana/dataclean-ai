import { useState, type ReactNode } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";

export interface TableColumn {
  key: string;
  label: string;
  /** Right-align and tabular-align numbers; left-align text. */
  numeric?: boolean;
  format?: (value: unknown) => string;
}

interface ChartCardProps {
  title: string;
  /** One line saying how to read the chart. Charts that need a paragraph are the wrong chart. */
  description?: string;
  children: ReactNode;
  /** The same values as a table. Not optional in spirit: it's how a value stays reachable
   * when hue is the only thing separating two marks, and how a screen reader gets at it. */
  tableColumns?: TableColumn[];
  tableRows?: Record<string, unknown>[];
  footnote?: ReactNode;
  className?: string;
}

function formatCell(value: unknown, column: TableColumn): string {
  if (column.format) return column.format(value);
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4).replace(/0+$/, "");
  }
  return String(value);
}

/** Wrapper for every chart in the app: title, one-line reading instruction, and a table view.
 *
 * The table isn't a nicety. Three of the categorical slots sit below 3:1 against the light
 * surface, and a tooltip is not a way to *read* data -- it's a way to inspect it. The table
 * is where every value stays reachable without hover, colour vision, or a mouse. */
export function ChartCard({
  title,
  description,
  children,
  tableColumns,
  tableRows,
  footnote,
  className,
}: ChartCardProps) {
  const [showTable, setShowTable] = useState(false);
  const hasTable = Boolean(tableColumns?.length && tableRows?.length);

  return (
    <Card className={className}>
      <CardHeader className="flex-row items-start justify-between gap-4 space-y-0">
        <div>
          <CardTitle className="text-base">{title}</CardTitle>
          {description && (
            <p className="mt-1 text-sm text-muted-foreground">{description}</p>
          )}
        </div>
        {hasTable && (
          <button
            type="button"
            onClick={() => setShowTable((open) => !open)}
            className="shrink-0 rounded-md border border-border px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            aria-expanded={showTable}
          >
            {showTable ? "Show chart" : "Show table"}
          </button>
        )}
      </CardHeader>
      <CardContent>
        {showTable && hasTable ? (
          <div className="max-h-[420px] overflow-auto">
            <table className="w-full border-collapse text-xs">
              <thead className="sticky top-0 bg-background">
                <tr>
                  {tableColumns!.map((column) => (
                    <th
                      key={column.key}
                      scope="col"
                      className={cn(
                        "border-b border-border px-2 py-1.5 font-medium",
                        column.numeric ? "text-right" : "text-left"
                      )}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tableRows!.map((row, index) => (
                  <tr key={index} className="border-b border-border last:border-0">
                    {tableColumns!.map((column) => (
                      <td
                        key={column.key}
                        className={cn(
                          "px-2 py-1",
                          column.numeric ? "text-right tabular-nums" : "text-left"
                        )}
                      >
                        {formatCell(row[column.key], column)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          children
        )}
        {footnote && <div className="mt-3 text-xs text-muted-foreground">{footnote}</div>}
      </CardContent>
    </Card>
  );
}
