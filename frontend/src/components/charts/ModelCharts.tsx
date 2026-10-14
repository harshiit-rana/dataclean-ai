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
import { AXIS_TICK_STYLE, inkOn, useChartTheme } from "@/lib/chartTheme";
import type {
  ColumnImportance,
  CorrelationResult,
  CrossValidation,
  LearningCurvePoint,
  PermutationImportance,
} from "@/api/types";

const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
const ratio = (value: unknown) => (typeof value === "number" ? value.toFixed(3) : "—");

export function ColumnImportanceChart({ rows }: { rows: ColumnImportance[] }) {
  const theme = useChartTheme();
  if (!rows?.length) return null;

  return (
    <ChartCard
      title="What the model is using"
      description="Share of the model's decision each column accounts for."
      tableColumns={[
        { key: "column", label: "Column" },
        { key: "importance", label: "Share", numeric: true, format: (v) => pct(Number(v)) },
      ]}
      tableRows={rows as unknown as Record<string, unknown>[]}
      footnote="A column that dominates completely is worth a second look — it can mean the column gives away the answer."
    >
      <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 34)}>
        <BarChart data={rows} layout="vertical" margin={{ ...CHART_MARGIN, left: 8 }}>
          <CartesianGrid stroke={theme.grid} horizontal={false} />
          <XAxis
            type="number"
            tickFormatter={pct}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            type="category"
            dataKey="column"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={130}
          />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => pct(Number(value))} />}
          />
          <Bar
            dataKey="importance"
            name="Share of the decision"
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

/** The metric's name as a reader would say it. Every chart that scores something says which
 * score, because they used to disagree: the spread was AUC, the curve was accuracy. */
const METRIC_NAMES: Record<string, string> = {
  roc_auc: "ROC AUC",
  accuracy: "accuracy",
  r2: "R²",
};
const metricName = (metric?: string) => (metric ? (METRIC_NAMES[metric] ?? metric) : "score");

export function PermutationImportanceChart({ rows }: { rows: PermutationImportance[] }) {
  const theme = useChartTheme();
  if (!rows?.length) return null;
  const measure = metricName(rows[0].metric);

  return (
    <ChartCard
      title="What the model would lose without each column"
      description={`Measured by scrambling each column in turn and seeing how much the ${measure} drops.`}
      tableColumns={[
        { key: "column", label: "Column" },
        { key: "importance", label: `${measure} lost`, numeric: true, format: ratio },
        { key: "std", label: "Variation", numeric: true, format: ratio },
      ]}
      tableRows={rows as unknown as Record<string, unknown>[]}
      footnote="More trustworthy than the chart above: the built-in kind favours columns with many distinct values whether or not they help. A column at or below zero is carrying nothing."
    >
      <ResponsiveContainer width="100%" height={Math.max(160, rows.length * 34)}>
        <BarChart data={rows} layout="vertical" margin={{ ...CHART_MARGIN, left: 8 }}>
          <CartesianGrid stroke={theme.grid} horizontal={false} />
          <XAxis
            type="number"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            type="category"
            dataKey="column"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={130}
          />
          <ReferenceLine x={0} stroke={theme.axis} />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toFixed(4)} />}
          />
          <Bar
            dataKey="importance"
            name="Score lost without it"
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

export function LearningCurveChart({ points }: { points: LearningCurvePoint[] }) {
  const theme = useChartTheme();
  if (!points?.length) return null;

  const measure = metricName(points[0].metric);
  const last = points[points.length - 1];
  const gap = last.training_score - last.validation_score;
  const climbing = last.validation_score - points[0].validation_score > 0.05;

  const reading =
    gap > 0.15
      ? "The model scores much better on rows it trained on than on unseen ones — it is partly memorizing. More data would help more than a different model."
      : climbing
        ? "The score on unseen rows was still climbing as data was added, so more rows would probably still be worth having."
        : "The score flattened out, so more rows of the same kind are unlikely to help. Better or different columns would matter more.";

  return (
    <ChartCard
      title="Would more data help?"
      description={`How the ${measure} changes as the model is trained on more and more rows.`}
      tableColumns={[
        { key: "training_rows", label: "Rows trained on", numeric: true },
        { key: "training_score", label: `${measure} on rows it saw`, numeric: true, format: ratio },
        { key: "validation_score", label: `${measure} on unseen rows`, numeric: true, format: ratio },
      ]}
      tableRows={points as unknown as Record<string, unknown>[]}
      footnote={reading}
    >
      <ResponsiveContainer width="100%" height={CHART_HEIGHT}>
        <LineChart data={points} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="training_rows"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            label={{
              value: "Rows trained on",
              position: "insideBottom",
              offset: -4,
              fill: theme.mutedText,
              fontSize: 11,
            }}
          />
          <YAxis
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={48}
            domain={["auto", "auto"]}
          />
          <Tooltip
            content={
              <ChartTooltip
                labelFormatter={(value) => `${Number(value).toLocaleString()} rows`}
                valueFormatter={(value) => Number(value).toFixed(3)}
              />
            }
          />
          <Line
            type="monotone"
            dataKey="training_score"
            name="On rows it saw"
            stroke={theme.series[1]}
            strokeWidth={2}
            dot={{ r: 4, fill: theme.series[1], stroke: theme.surface, strokeWidth: 2 }}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="validation_score"
            name="On unseen rows"
            stroke={theme.series[0]}
            strokeWidth={2}
            dot={{ r: 4, fill: theme.series[0], stroke: theme.surface, strokeWidth: 2 }}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
      <ChartLegend
        items={[
          { label: "On unseen rows", color: theme.series[0] },
          { label: "On rows it saw", color: theme.series[1] },
        ]}
      />
    </ChartCard>
  );
}

export function CrossValidationChart({ cv }: { cv: CrossValidation }) {
  const theme = useChartTheme();
  if (!cv?.folds?.length) return null;

  const data = cv.folds.map((score, index) => ({ fold: `Split ${index + 1}`, score }));
  const wide = cv.spread > 0.15;

  return (
    <ChartCard
      title="Would a different split have said the same thing?"
      description={`The same model scored on ${cv.folds.length} different splits of the training rows.`}
      tableColumns={[
        { key: "fold", label: "Split" },
        { key: "score", label: cv.metric, numeric: true, format: ratio },
      ]}
      tableRows={data as unknown as Record<string, unknown>[]}
      footnote={
        wide
          ? `Average ${cv.mean.toFixed(3)}, ranging ${Math.min(...cv.folds).toFixed(3)} to ${Math.max(...cv.folds).toFixed(3)}. That is a wide range — the headline score could have landed anywhere in it depending on which rows were held back, so treat it as approximate.`
          : `Average ${cv.mean.toFixed(3)}, ranging ${Math.min(...cv.folds).toFixed(3)} to ${Math.max(...cv.folds).toFixed(3)}. A narrow range, so the headline score is fairly stable.`
      }
    >
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={CHART_MARGIN}>
          <CartesianGrid stroke={theme.grid} vertical={false} />
          <XAxis
            dataKey="fold"
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
          />
          <YAxis
            domain={[0, 1]}
            stroke={theme.axis}
            tick={{ ...AXIS_TICK_STYLE, fill: theme.mutedText }}
            width={44}
          />
          <ReferenceLine
            y={cv.mean}
            stroke={theme.mutedText}
            strokeDasharray="4 4"
            // Inside the plot area: "right" puts the label past the axis where the chart
            // margin clips it, and a reference line labelled "av" explains nothing.
            label={{
              value: "average",
              position: "insideTopRight",
              fill: theme.mutedText,
              fontSize: 10,
            }}
          />
          <Tooltip
            cursor={{ fill: theme.grid, fillOpacity: 0.4 }}
            content={<ChartTooltip valueFormatter={(value) => Number(value).toFixed(3)} />}
          />
          <Bar
            dataKey="score"
            name={cv.metric}
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

/** Correlation heatmap. Diverging blue↔red with a neutral midpoint, because the sign matters
 * as much as the size -- and a neutral middle is what makes "no relationship" read as nothing. */
export function CorrelationHeatmap({ data }: { data: CorrelationResult }) {
  const theme = useChartTheme();
  if (!data?.columns?.length || data.columns.length < 2) return null;

  const lookup = new Map(data.cells.map((c) => [`${c.row}|${c.column}`, c.correlation]));

  const cellColor = (value: number) => {
    const strength = Math.min(1, Math.abs(value));
    if (strength < 0.05) return theme.divergingMid;
    const pole = value > 0 ? theme.divergingHigh : theme.divergingLow;
    // Mix toward the neutral midpoint so weak relationships stay pale.
    return `color-mix(in oklab, ${pole} ${Math.round(strength * 100)}%, ${theme.divergingMid})`;
  };

  return (
    <ChartCard
      title="Which columns move together"
      // "pale is none" was only true on the light surface -- in dark the neutral cell is the
      // dark one. Saying "faint" describes the same thing in both.
      description="Blue is a negative relationship, red a positive one; the fainter the cell, the weaker the relationship."
      tableColumns={[
        { key: "row", label: "Column" },
        { key: "column", label: "Compared with" },
        { key: "correlation", label: "Correlation", numeric: true, format: ratio },
      ]}
      tableRows={data.cells as unknown as Record<string, unknown>[]}
      footnote={
        data.strong_pairs.length
          ? `${data.strong_pairs
              .map((p) => `${p.a} and ${p.b}`)
              .join("; ")} move almost identically. Two columns that say the same thing split the credit between them in the importance chart, so both look less useful than they are.`
          : "No pair of columns is near-identical, so the importance ranking above is not being split between duplicates."
      }
    >
      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-0.5 text-xs"
          style={{ gridTemplateColumns: `auto repeat(${data.columns.length}, minmax(60px, 1fr))` }}
        >
          <div />
          {data.columns.map((column) => (
            <div
              key={column}
              className="px-1 pb-1 text-center text-[10px] text-muted-foreground"
              title={column}
            >
              <span className="block truncate">{column}</span>
            </div>
          ))}
          {data.columns.map((row) => (
            <div key={row} className="contents">
              <div
                className="flex items-center justify-end pr-2 text-[10px] text-muted-foreground"
                title={row}
              >
                <span className="max-w-[110px] truncate">{row}</span>
              </div>
              {data.columns.map((column) => {
                const value = lookup.get(`${row}|${column}`) ?? 0;
                const background = cellColor(value);
                return (
                  <div
                    key={column}
                    className="rounded px-1 py-2.5 text-center tabular-nums"
                    style={{
                      background,
                      color: Math.abs(value) > 0.6 ? inkOn(theme.divergingHigh) : undefined,
                    }}
                    title={`${row} vs ${column}: ${value.toFixed(3)}`}
                  >
                    {value.toFixed(2)}
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
