import type { ReactNode } from "react";
import { useChartTheme } from "@/lib/chartTheme";

/** One tooltip shape for every chart, so hovering anything reads the same way.
 *
 * Identity comes from a colour swatch beside the label, never from colouring the text --
 * the light aqua and yellow slots are illegible as type on the surface. */
export function ChartTooltip({
  active,
  payload,
  label,
  labelFormatter,
  valueFormatter,
}: {
  active?: boolean;
  payload?: { name?: string; value?: number | string; color?: string; dataKey?: string }[];
  label?: string | number;
  labelFormatter?: (label: string | number) => string;
  valueFormatter?: (value: number | string, name?: string) => string;
}) {
  const theme = useChartTheme();
  if (!active || !payload?.length) return null;

  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 text-xs shadow-sm">
      {label !== undefined && (
        <p className="mb-1 font-medium" style={{ color: theme.secondaryText }}>
          {labelFormatter ? labelFormatter(label) : label}
        </p>
      )}
      {payload.map((entry, index) => (
        <div key={index} className="flex items-center gap-2">
          <span
            aria-hidden
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ background: entry.color }}
          />
          <span className="text-muted-foreground">{entry.name ?? entry.dataKey}</span>
          <span className="ml-auto pl-3 font-medium tabular-nums">
            {valueFormatter && entry.value !== undefined
              ? valueFormatter(entry.value, entry.name)
              : entry.value}
          </span>
        </div>
      ))}
    </div>
  );
}

/** Legend for two or more series. A single-series chart gets none -- its title already says
 * what is plotted, and a one-swatch box just restates it. */
export function ChartLegend({ items }: { items: { label: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground">
      {items.map((item) => (
        <span key={item.label} className="flex items-center gap-1.5">
          <span
            aria-hidden
            className="inline-block h-0.5 w-4 shrink-0 rounded-full"
            style={
              item.dashed
                ? {
                    backgroundImage: `repeating-linear-gradient(90deg, ${item.color} 0 4px, transparent 4px 7px)`,
                  }
                : { background: item.color }
            }
          />
          {item.label}
        </span>
      ))}
    </div>
  );
}

/** Shown in place of a chart when a diagnostic couldn't be computed, rather than an empty
 * frame that looks like a rendering bug. */
export function ChartUnavailable({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-[220px] items-center justify-center rounded-md border border-dashed border-border px-6 text-center text-sm text-muted-foreground">
      {children}
    </div>
  );
}

export const CHART_HEIGHT = 260;
export const CHART_MARGIN = { top: 8, right: 16, bottom: 8, left: 0 };
