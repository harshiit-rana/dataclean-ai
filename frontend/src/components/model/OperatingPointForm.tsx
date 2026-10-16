import { useId } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { OperatingObjective, OperatingPointRequest } from "@/api/types";

interface OperatingPointFormProps {
  /** The two answers, sorted. The form is only rendered when there are exactly two. */
  classLabels: string[];
  value: OperatingPointRequest;
  onChange: (next: OperatingPointRequest) => void;
}

const OBJECTIVES: { value: OperatingObjective; label: string; description: string }[] = [
  {
    value: "default",
    label: "Use the usual cut-off (0.50)",
    description: "Flag a row when the model thinks it's more likely than not. Fine when neither kind of mistake is worse than the other.",
  },
  {
    value: "cost",
    label: "One kind of mistake costs more than the other",
    description: "Say how much worse a missed case is than a false alarm (or the reverse) and the line moves to keep the total cost down.",
  },
  {
    value: "min_recall",
    label: "Catch at least a set share of real cases",
    description: "The line moves down until enough are caught, and no further -- every step lower is more false alarms for nothing.",
  },
  {
    value: "min_precision",
    label: "Be right at least a set share of the time when flagging",
    description: "The line moves up until flagged rows are reliable enough, and no further -- every step higher is a real case missed.",
  },
  {
    value: "f1",
    label: "Balance the two",
    description: "The line that best trades catching cases against false alarms, with no preference between them.",
  },
];

/** Where the cut-off sits, stated as a trade-off rather than a number.
 *
 * The threshold sweep on the results page has shown the trade for a while without letting
 * anyone act on it -- and the obvious way to act on it, picking the number that looks best
 * on that chart, is choosing on the test set. So the user says what matters and the server
 * chooses the line on the training folds. That is why there is no slider here. */
export function OperatingPointForm({ classLabels, value, onChange }: OperatingPointFormProps) {
  const titleId = useId();
  const positive = value.positive_label ?? classLabels[classLabels.length - 1];
  const set = (patch: Partial<OperatingPointRequest>) => onChange({ ...value, ...patch });

  // Kept as strings while editing so a half-typed "0." doesn't get coerced to 0.
  const rateText = value.target_rate == null ? "" : String(Math.round(value.target_rate * 100));
  const parseRate = (text: string): number | null => {
    const n = Number.parseFloat(text);
    return Number.isFinite(n) && n > 0 && n <= 100 ? n / 100 : null;
  };

  return (
    <Card role="group" aria-labelledby={titleId}>
      <CardHeader>
        <CardTitle id={titleId} className="text-base">What matters more: catching cases, or not crying wolf?</CardTitle>
        <p className="text-sm text-muted-foreground">
          A model gives each row a probability, and somewhere a line is drawn. Where it sits
          is a business call, not a modelling one. Say what matters and the app will choose
          the line on the training rows -- the held-out test set is never consulted, so the
          numbers it reports at that line are honest ones.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <label htmlFor="positive-label" className="text-muted-foreground">
            A &ldquo;case&rdquo; is a row where the answer is
          </label>
          <Select
            id="positive-label"
            className="h-9 w-auto"
            value={positive}
            onChange={(e) => set({ positive_label: e.target.value })}
          >
            {classLabels.map((label) => (
              <option key={label} value={label}>
                {label}
              </option>
            ))}
          </Select>
        </div>

        {OBJECTIVES.map((option) => {
          const selected = value.objective === option.value;
          return (
            <label
              key={option.value}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-md border border-border p-3",
                selected && "bg-muted/40"
              )}
            >
              <input
                type="radio"
                name="operating-objective"
                className="mt-0.5"
                checked={selected}
                onChange={() => set({ objective: option.value })}
              />
              <span className="flex-1 text-sm">
                <span className="font-medium">{option.label}</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">{option.description}</span>

                {selected && option.value === "cost" && (
                  <span className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    Missing a real case costs
                    <input
                      aria-label="Cost of missing a case"
                      type="number"
                      min={0}
                      step="any"
                      inputMode="decimal"
                      value={value.cost_of_missing ?? 1}
                      onChange={(e) => set({ cost_of_missing: Number(e.target.value) })}
                      className="h-8 w-20 rounded-md border border-border bg-transparent px-2 tabular-nums"
                    />
                    and a false alarm costs
                    <input
                      aria-label="Cost of a false alarm"
                      type="number"
                      min={0}
                      step="any"
                      inputMode="decimal"
                      value={value.cost_of_false_alarm ?? 1}
                      onChange={(e) => set({ cost_of_false_alarm: Number(e.target.value) })}
                      className="h-8 w-20 rounded-md border border-border bg-transparent px-2 tabular-nums"
                    />
                    <span className="text-muted-foreground">(only the ratio matters)</span>
                  </span>
                )}

                {selected && (option.value === "min_recall" || option.value === "min_precision") && (
                  <span className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    {option.value === "min_recall" ? "Catch at least" : "Be right at least"}
                    <input
                      aria-label="Target rate"
                      type="number"
                      min={1}
                      max={100}
                      step={1}
                      inputMode="numeric"
                      value={rateText}
                      placeholder="80"
                      onChange={(e) => set({ target_rate: parseRate(e.target.value) })}
                      className="h-8 w-20 rounded-md border border-border bg-transparent px-2 tabular-nums"
                    />
                    {option.value === "min_recall" ? "% of real cases" : "% of the time when flagging"}
                  </span>
                )}
              </span>
            </label>
          );
        })}
      </CardContent>
    </Card>
  );
}

/** Whether the trade-off is complete enough to send. A floor without a rate is refused by
 * the server with a 422; better to say so before the button is pressed. */
export function operatingPointIsComplete(value: OperatingPointRequest): boolean {
  if (value.objective === "min_recall" || value.objective === "min_precision") {
    return value.target_rate != null;
  }
  if (value.objective === "cost") {
    return (value.cost_of_missing ?? 1) > 0 || (value.cost_of_false_alarm ?? 1) > 0;
  }
  return true;
}
