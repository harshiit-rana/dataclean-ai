import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { CleaningAggression, GoalContext, UseCase } from "@/api/types";

interface GoalFormProps {
  onSubmit: (goal: GoalContext) => void;
  submitting: boolean;
}

const USE_CASES: { value: UseCase; label: string }[] = [
  { value: "ml_training", label: "Training a machine learning model" },
  { value: "reporting", label: "Building a report or dashboard" },
  { value: "analysis", label: "General analysis / exploration" },
  { value: "pipeline_validation", label: "Validating a data pipeline" },
];

const AGGRESSION_LEVELS: { value: CleaningAggression; label: string }[] = [
  { value: "conservative", label: "Conservative -- flag more, change less" },
  { value: "moderate", label: "Moderate -- balanced default" },
  { value: "thorough", label: "Thorough -- clean aggressively" },
];

/** PRD US-01: collects goal context (use case, expected output, critical columns,
 * cleaning aggression) before profiling starts, so the Profiling/Explanation agents can
 * rank issues by relevance to what the user is actually trying to do. */
export function GoalForm({ onSubmit, submitting }: GoalFormProps) {
  const [useCase, setUseCase] = useState<UseCase>("analysis");
  const [expectedOutput, setExpectedOutput] = useState("");
  const [criticalColumns, setCriticalColumns] = useState("");
  const [aggression, setAggression] = useState<CleaningAggression>("moderate");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    onSubmit({
      use_case: useCase,
      expected_output: expectedOutput,
      critical_columns: criticalColumns
        .split(",")
        .map((c) => c.trim())
        .filter(Boolean),
      cleaning_aggression: aggression,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <div className="flex flex-col gap-2">
        <Label htmlFor="use_case">What are you using this data for?</Label>
        <Select id="use_case" value={useCase} onChange={(e) => setUseCase(e.target.value as UseCase)}>
          {USE_CASES.map((uc) => (
            <option key={uc.value} value={uc.value}>
              {uc.label}
            </option>
          ))}
        </Select>
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="expected_output">What should the cleaned data look like?</Label>
        <Textarea
          id="expected_output"
          required
          placeholder="e.g. one row per customer, no nulls in the revenue column"
          value={expectedOutput}
          onChange={(e) => setExpectedOutput(e.target.value)}
        />
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="critical_columns">Any columns that matter most? (comma-separated, optional)</Label>
        <Input
          id="critical_columns"
          placeholder="e.g. customer_id, revenue"
          value={criticalColumns}
          onChange={(e) => setCriticalColumns(e.target.value)}
        />
      </div>

      <div className="flex flex-col gap-2">
        <Label htmlFor="aggression">How aggressive should cleaning be?</Label>
        <Select
          id="aggression"
          value={aggression}
          onChange={(e) => setAggression(e.target.value as CleaningAggression)}
        >
          {AGGRESSION_LEVELS.map((a) => (
            <option key={a.value} value={a.value}>
              {a.label}
            </option>
          ))}
        </Select>
      </div>

      <Button type="submit" disabled={submitting || !expectedOutput}>
        {submitting ? "Starting..." : "Start profiling"}
      </Button>
    </form>
  );
}
