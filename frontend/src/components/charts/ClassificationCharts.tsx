import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartCard } from "@/components/charts/ChartCard";
import {
  CHART_HEIGHT,
  CHART_MARGIN,
  ChartLegend,
  ChartTooltip,
} from "@/components/charts/primitives";
import { AXIS_TICK_STYLE, inkOn, rampColor, useChartTheme } from "@/lib/chartTheme";
import type {
  CalibrationPoint,
  ClassCount,
  ConfusionCell,
  PerClassRow,
  PrecisionRecallPoint,
  RocPoint,
  ThresholdRow,
} from "@/api/types";

const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
const ratio = (value: unknown) => (typeof value === "number" ? value.toFixed(3) : "—");

export function RocCurveChart({ points, auc }: { points: RocPoint[]; auc?: number }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  return (
    <ChartCard
      title="How well it separates the two outcomes"
      description="The further the curve bows above the diagonal, the better. The diagonal is what random guessing looks like."
      tableColumns={[
        { key: "threshold", label: "Cut-off", numeric: true, format: ratio },
        { key: "fpr", label: "False alarm rate", numeric: true, format: ratio },
        { key: "tpr", label: "Caught rate", numeric: true, format: ratio },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
      footnote={
        auc !== undefined ? `Area under the curve: ${auc.toFixed(3)} (0.50 is a coin flip).` : undefined
      }
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={points} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} strokeDasharray="" vertical={false} />
          <XAxis
            dataKey="fpr"
            type="number"
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "False alarms",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          {/* Chance line, drawn first so the model's curve sits above it. */}
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke={theme.mutedText}
            strokeDasharray="4 4"
            ifOverflow="extendDomain"
          />
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(value) => `False alarms ${pct(Number(value))}`}
                valueFormatter={(value) => pct(Number(value))}
              />
            }
          />
          <Line
            type="monotone"
            dataKey="tpr"
            name="Caught"
            stroke={theme.series[0]}
            strokeWidth={2}
            strokeLinecap="round"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "This model", color: theme.series[0] },
          { label: "Random guessing", color: theme.mutedText, dashed: true },
        ]}
      />
    </ChartCard>
  );
}

export function PrecisionRecallChart({ points }: { points: PrecisionRecallPoint[] }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  return (
    <ChartCard
      title="Catching more cases versus being right about them"
      description="Moving right catches more of the real cases; the curve falling shows what that costs in false alarms."
      tableColumns={[
        { key: "threshold", label: "Cut-off", numeric: true, format: ratio },
        { key: "recall", label: "Caught", numeric: true, format: ratio },
        { key: "precision", label: "Correct when flagged", numeric: true, format: ratio },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={points} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="recall"
            type="number"
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "Share of real cases caught",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(value) => `Caught ${pct(Number(value))}`}
                valueFormatter={(value) => pct(Number(value))}
              />
            }
          />
          <Line
            type="monotone"
            dataKey="precision"
            name="Correct when flagged"
            stroke={theme.series[0]}
            strokeWidth={2}
            strokeLinecap="round"
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function CalibrationChart({ points }: { points: CalibrationPoint[] }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  return (
    <ChartCard
      title="Can you trust the probabilities?"
      description="When it says 70% likely, does that happen 70% of the time? Points on the diagonal mean yes."
      tableColumns={[
        { key: "predicted", label: "Model said", numeric: true, format: (v) => pct(Number(v)) },
        { key: "actual", label: "Actually happened", numeric: true, format: (v) => pct(Number(v)) },
        { key: "count", label: "Rows", numeric: true },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
      footnote="A model can rank cases perfectly and still have badly wrong probabilities. That only matters if you treat the number as a probability rather than a ranking."
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={points} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="predicted"
            type="number"
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "What the model predicted",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <ReferenceLine
            segment={[
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ]}
            stroke={theme.mutedText}
            strokeDasharray="4 4"
            ifOverflow="extendDomain"
          />
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(value) => `Predicted ${pct(Number(value))}`}
                valueFormatter={(value) => pct(Number(value))}
              />
            }
          />
          <Line
            type="monotone"
            dataKey="actual"
            name="Actually happened"
            stroke={theme.series[0]}
            strokeWidth={2}
            dot={{ r: 4, fill: theme.series[0], stroke: theme.surface, strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "This model", color: theme.series[0] },
          { label: "Perfectly calibrated", color: theme.mutedText, dashed: true },
        ]}
      />
    </ChartCard>
  );
}

export function ThresholdChart({ rows, chosen }: { rows: ThresholdRow[]; chosen?: number }) {
  const theme = useChartTheme();
  if (!rows?.length) return null;

  return (
    <ChartCard
      title={chosen != null ? "Where the cut-off sits" : "Where should the cut-off sit?"}
      description={
        chosen != null
          ? `The line you asked for sits at ${chosen.toFixed(2)}; the faint one is the default 0.50. This sweep is over the test rows -- the line itself was not chosen from it.`
          : "Flag more cases and you catch more but are wrong more often. This is a business call, not a modelling one."
      }
      tableColumns={[
        { key: "threshold", label: "Cut-off", numeric: true, format: (v) => Number(v).toFixed(2) },
        { key: "flagged", label: "Flagged", numeric: true },
        { key: "caught", label: "Caught", numeric: true },
        { key: "missed", label: "Missed", numeric: true },
        { key: "precision", label: "Correct when flagged", numeric: true, format: ratio },
        { key: "recall", label: "Share caught", numeric: true, format: ratio },
      ]}
      tableRows={rows as unknown as Record<string, unknown>[]}
      footnote={
        chosen != null
          ? "To move it, change the trade-off under \"Change columns or retrain\" and train again."
          : "The default cut-off is 0.50, but nothing makes it the right one. If a missed case costs more than a false alarm, say so under \"Change columns or retrain\" and the app will move it."
      }
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={rows} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="threshold"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "Cut-off",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <ReferenceLine x={0.5} stroke={theme.axis} strokeDasharray={chosen != null ? "3 3" : undefined} />
          {chosen != null && (
            <ReferenceLine
              x={chosen}
              stroke={theme.series[2] ?? theme.axis}
              strokeWidth={2}
              label={{ value: "your line", position: "top", fill: theme.mutedText, fontSize: 11 }}
            />
          )}
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(value) => `Cut-off ${Number(value).toFixed(2)}`}
                valueFormatter={(value) => pct(Number(value))}
              />
            }
          />
          <Line
            type="monotone"
            dataKey="precision"
            name="Correct when flagged"
            stroke={theme.series[0]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="recall"
            name="Share of real cases caught"
            stroke={theme.series[1]}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "Correct when flagged", color: theme.series[0] },
          { label: "Share of real cases caught", color: theme.series[1] },
        ]}
      />
    </ChartCard>
  );
}

/** Confusion matrix as a heatmap.
 *
 * Built as a CSS grid rather than a charting primitive: a matrix is a table with colour, the
 * counts have to be readable in every cell, and the row-normalized share is what makes a
 * lopsided matrix interpretable at all. */
export function ConfusionMatrix({ cells }: { cells: ConfusionCell[] }) {
  const theme = useChartTheme();
  if (!cells?.length) return null;

  const labels = Array.from(new Set(cells.map((c) => c.actual)));
  const lookup = new Map(cells.map((c) => [`${c.actual}|${c.predicted}`, c]));

  return (
    <ChartCard
      title="Where it gets things wrong"
      description="Rows are what actually happened; columns are what the model said. The diagonal is correct."
      tableColumns={[
        { key: "actual", label: "Actual" },
        { key: "predicted", label: "Predicted" },
        { key: "count", label: "Rows", numeric: true },
        {
          key: "share_of_actual",
          label: "Share of actual",
          numeric: true,
          format: (v) => pct(Number(v)),
        },
      ]}
      tableRows={cells as unknown as Record<string, unknown>[]}
    >
      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-0.5 text-sm"
          style={{ gridTemplateColumns: `auto repeat(${labels.length}, minmax(96px, 1fr))` }}
        >
          <div />
          {labels.map((label) => (
            <div key={label} className="px-2 pb-1 text-center text-xs text-muted-foreground">
              said &ldquo;{label}&rdquo;
            </div>
          ))}
          {labels.map((actual) => (
            <div key={actual} className="contents">
              <div className="flex items-center justify-end pr-2 text-xs text-muted-foreground">
                was &ldquo;{actual}&rdquo;
              </div>
              {labels.map((predicted) => {
                const cell = lookup.get(`${actual}|${predicted}`);
                const share = cell?.share_of_actual ?? 0;
                const background = rampColor(theme.sequential, share);
                return (
                  <div
                    key={predicted}
                    className="rounded-md px-2 py-4 text-center"
                    style={{ background, color: inkOn(background) }}
                    title={`${cell?.count ?? 0} rows — ${pct(share)} of everything that was "${actual}"`}
                  >
                    <div className="text-base font-semibold tabular-nums">
                      {(cell?.count ?? 0).toLocaleString()}
                    </div>
                    <div className="text-xs opacity-80 tabular-nums">{pct(share)}</div>
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>
    </ChartCard>
  );
}

export function PerClassChart({ rows }: { rows: PerClassRow[] }) {
  const theme = useChartTheme();
  if (!rows?.length) return null;

  return (
    <ChartCard
      title="Performance on each answer"
      description="An overall score can look healthy while the rare answer is missed almost entirely."
      tableColumns={[
        { key: "label", label: "Answer" },
        { key: "precision", label: "Correct when predicted", numeric: true, format: ratio },
        { key: "recall", label: "Share found", numeric: true, format: ratio },
        { key: "f1", label: "F1", numeric: true, format: ratio },
        { key: "support", label: "Test rows", numeric: true },
      ]}
      tableRows={rows as unknown as Record<string, unknown>[]}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <BarChart data={rows} margin={CHART_MARGIN} barGap={2}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="label"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => pct(Number(value))} />}
          />
          <Bar
            dataKey="precision"
            name="Correct when predicted"
            fill={theme.series[0]}
            radius={[4, 4, 0, 0]}
            maxBarSize={24}
            isAnimationActive={false}
          />
          <Bar
            dataKey="recall"
            name="Share found"
            fill={theme.series[1]}
            radius={[4, 4, 0, 0]}
            maxBarSize={24}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "Correct when predicted", color: theme.series[0] },
          { label: "Share found", color: theme.series[1] },
        ]}
      />
    </ChartCard>
  );
}

export function ClassDistributionChart({ counts }: { counts: ClassCount[] }) {
  const theme = useChartTheme();
  if (!counts?.length) return null;

  const total = counts.reduce((sum, c) => sum + c.count, 0);

  return (
    <ChartCard
      title="How the answers are split"
      description="A lopsided split is why accuracy alone can flatter a model that has learned nothing."
      tableColumns={[
        { key: "label", label: "Answer" },
        { key: "count", label: "Test rows", numeric: true },
      ]}
      tableRows={counts as unknown as Record<string, unknown>[]}
      footnote={`${total.toLocaleString()} rows in the test set.`}
    >
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={counts} layout="vertical" margin={{ ...CHART_MARGIN, left: 8 }}>
          <CartesianGrid stroke={theme.grid} horizontal={false} />
          <XAxis
            type="number"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            type="category"
            dataKey="label"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={80}
          />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toLocaleString()} />}
          />
          {/* One series, one colour: shading bars by their own length would double-encode. */}
          <Bar
            dataKey="count"
            name="Test rows"
            fill={theme.series[0]}
            radius={[0, 4, 4, 0]}
            maxBarSize={24}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}
