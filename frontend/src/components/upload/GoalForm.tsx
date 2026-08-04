import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import type { CleaningAggression, GoalContext, TaskType, UseCase } from "@/api/types";

interface GoalFormProps {
  onSubmit: (goal: GoalContext) => void;
  submitting: boolean;
  /** Column names from the uploaded file. Empty until the upload finishes, which is why the
   * target picker is disabled rather than absent -- it should be visible from the start so
   * the question gets considered. */
  columns: string[];
}

const TASK_TYPES: { value: TaskType; label: string }[] = [
  { value: "auto", label: "Work it out from the column" },
  { value: "classification", label: "Predict a category (yes/no, a label)" },
  { value: "regression", label: "Predict a number (an amount, a score)" },
];

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
export function GoalForm({ onSubmit, submitting, columns }: GoalFormProps) {
  const [useCase, setUseCase] = useState<UseCase>("analysis");
  const [expectedOutput, setExpectedOutput] = useState("");
  const [criticalColumns, setCriticalColumns] = useState("");
  const [aggression, setAggression] = useState<CleaningAggression>("moderate");
  const [targetColumn, setTargetColumn] = useState("");
  const [taskType, setTaskType] = useState<TaskType>("auto");

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
      target_column: targetColumn || null,
      task_type: taskType,
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

      {/* The one answer that changes what the tool actually does, rather than just how it
          words things: it turns on leakage detection and hard-blocks fixes that would
          corrupt the answers the model is supposed to learn. */}
      <div className="flex flex-col gap-2">
        <Label htmlFor="target_column">
          Which column are you trying to predict? (optional)
        </Label>
        <Select
          id="target_column"
          value={targetColumn}
          disabled={columns.length === 0}
          onChange={(e) => setTargetColumn(e.target.value)}
        >
          <option value="">
            {columns.length === 0
              ? "Choose a CSV first"
              : "Not building a model — just cleaning"}
          </option>
          {columns.map((column) => (
            <option key={column} value={column}>
              {column}
            </option>
          ))}
        </Select>
        <p className="text-xs text-muted-foreground">
          Naming it lets us catch columns that give away the answer, and stop any fix that
          would invent the values you want predicted.
        </p>
      </div>

      {targetColumn && (
        <div className="flex flex-col gap-2">
          <Label htmlFor="task_type">What kind of answer is {targetColumn}?</Label>
          <Select
            id="task_type"
            value={taskType}
            onChange={(e) => setTaskType(e.target.value as TaskType)}
          >
            {TASK_TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </Select>
        </div>
      )}

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
