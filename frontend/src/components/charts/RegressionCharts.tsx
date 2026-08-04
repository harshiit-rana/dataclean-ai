import {
  Bar,
  BarChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { ChartCard } from "@/components/charts/ChartCard";
import {
  CHART_HEIGHT,
  CHART_MARGIN,
  ChartLegend,
  ChartTooltip,
} from "@/components/charts/primitives";
import { AXIS_TICK_STYLE, useChartTheme } from "@/lib/chartTheme";
import type {
  ErrorByRangeRow,
  PredictedVsActualPoint,
  ResidualBar,
  ResidualPoint,
} from "@/api/types";

const compact = (value: number) =>
  Math.abs(value) >= 1000 ? `${(value / 1000).toFixed(1)}k` : value.toLocaleString();

export function PredictedVsActualChart({ points }: { points: PredictedVsActualPoint[] }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  const values = points.flatMap((p) => [p.actual, p.predicted]);
  const min = Math.min(...values);
  const max = Math.max(...values);

  return (
    <ChartCard
      title="Predicted against actual"
      description="Every dot is one row. Dots on the diagonal were predicted exactly; distance from it is the error."
      tableColumns={[
        { key: "actual", label: "Actual", numeric: true },
        { key: "predicted", label: "Predicted", numeric: true },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <ScatterChart margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} />
          <XAxis
            type="number"
            dataKey="actual"
            domain={[min, max]}
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "Actual",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="predicted"
            domain={[min, max]}
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={56}
          />
          <ZAxis range={[36, 36]} />
          <ReferenceLine
            segment={[
              { x: min, y: min },
              { x: max, y: max },
            ]}
            stroke={theme.mutedText}
            strokeDasharray="4 4"
          />
          <Tooltip
            cursor={{ stroke: theme.axis }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toLocaleString()} />}
          />
          <Scatter
            data={points}
            name="Rows"
            fill={theme.series[0]}
            fillOpacity={0.55}
            stroke={theme.surface}
            strokeWidth={1}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "Rows", color: theme.series[0] },
          { label: "Perfect prediction", color: theme.mutedText, dashed: true },
        ]}
      />
    </ChartCard>
  );
}

export function ResidualChart({ points }: { points: ResidualPoint[] }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  return (
    <ChartCard
      title="What is left over"
      description="Errors should scatter evenly around zero. A fan, a curve or a step means the model is systematically wrong somewhere."
      tableColumns={[
        { key: "predicted", label: "Predicted", numeric: true },
        { key: "residual", label: "Actual minus predicted", numeric: true },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
      footnote="A single overall score hides this completely — a model can have a good R² and still be badly wrong for one whole group of rows."
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <ScatterChart margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            type="number"
            dataKey="predicted"
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "Predicted",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            type="number"
            dataKey="residual"
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={56}
          />
          <ZAxis range={[36, 36]} />
          <ReferenceLine y={0} stroke={theme.axis} />
          <Tooltip
            cursor={{ stroke: theme.axis }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toLocaleString()} />}
          />
          <Scatter
            data={points}
            name="Rows"
            fill={theme.series[0]}
            fillOpacity={0.55}
            stroke={theme.surface}
            strokeWidth={1}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function ResidualHistogram({ bars }: { bars: ResidualBar[] }) {
  const theme = useChartTheme();
  if (!bars?.length) return null;

  return (
    <ChartCard
      title="How the errors are distributed"
      description="A hump centred on zero is healthy. A hump shifted off zero means the model is biased high or low across the board."
      tableColumns={[
        { key: "centre", label: "Error", numeric: true },
        { key: "count", label: "Rows", numeric: true },
      ]}
      tableRows={bars as unknown as Record<string, unknown>[]}
    >
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={bars} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="centre"
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <ReferenceLine x={0} stroke={theme.axis} />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={
              <ChartTooltip
                labelFormatter={(value) => `Error around ${compact(Number(value))}`}
                valueFormatter={(value) => `${Number(value).toLocaleString()} rows`}
              />
            }
          />
          <Bar
            dataKey="count"
            name="Rows"
            fill={theme.series[0]}
            radius={[4, 4, 0, 0]}
            maxBarSize={24}
            isAnimationActive={false}
          />
        </BarChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

export function ErrorByRangeChart({ rows }: { rows: ErrorByRangeRow[] }) {
  const theme = useChartTheme();
  if (!rows?.length) return null;

  return (
    <ChartCard
      title="Where it is most wrong"
      description="Typical error within each slice of the target's range. Models are often accurate in the middle and poor at the extremes."
      tableColumns={[
        { key: "range", label: "Actual value" },
        { key: "mean_absolute_error", label: "Typical error", numeric: true },
        { key: "mean_error", label: "Bias (+ over, − under)", numeric: true },
        { key: "count", label: "Rows", numeric: true },
      ]}
      tableRows={rows as unknown as Record<string, unknown>[]}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <BarChart data={rows} layout="vertical" margin={{ ...CHART_MARGIN, left: 8 }}>
          <CartesianGrid stroke={theme.grid} horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={compact}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            type="category"
            dataKey="range"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={140}
          />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toLocaleString()} />}
          />
          <Bar
            dataKey="mean_absolute_error"
            name="Typical error"
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
