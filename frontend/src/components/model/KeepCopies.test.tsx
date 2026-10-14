import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelPlanForm } from "./ModelPlanForm";
import { requestFromResult } from "@/pages/ModelPage";
import type { ModelPlan, ModelResult } from "@/api/types";

const PLAN: ModelPlan = {
  session_id: "s1",
  target_column: "churned",
  task_type: "classification",
  row_count: 300,
  trainable_row_count: 300,
  features: { numeric: ["spend"], categorical: [], dropped: [] },
  suggested_exclusions: [],
  excluded_columns: [],
  candidate_models: [
    { name: "logistic_regression", label: "Logistic regression", description: "", explains_features: true },
  ],
  split_options: [
    { value: "random", label: "Random split", description: "" },
    { value: "time", label: "Split by date", description: "" },
  ],
  date_columns: ["signup_date"],
  group_columns: [],
  blockers: [],
  warnings: [],
  trained: false,
  identical_rows: { copies: 60, share: 0.2, rows: 300 },
  keep_identical_rows_together: true,
};

function train(plan: Partial<ModelPlan> = {}) {
  const onTrain = vi.fn();
  render(<ModelPlanForm plan={{ ...PLAN, ...plan }} onTrain={onTrain} training={false} />);
  return onTrain;
}

describe("ModelPlanForm — identical rows", () => {
  it("offers it with the count as evidence when copies exist", () => {
    train();
    expect(screen.getByText(/60 rows/)).toBeInTheDocument();
    expect(screen.getByText(/20%\) are exact copies/)).toBeInTheDocument();
  });

  it("is ticked when the plan says copies exist", () => {
    train();
    expect(screen.getByRole("checkbox", { name: /Keep identical rows together/ })).toBeChecked();
  });

  it("sends the choice with the request", () => {
    const onTrain = train();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].keep_identical_rows_together).toBe(true);
  });

  it("can be unticked, for genuine repeats", () => {
    const onTrain = train();
    fireEvent.click(screen.getByRole("checkbox", { name: /Keep identical rows together/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].keep_identical_rows_together).toBe(false);
  });

  it("is not offered when there are no copies", () => {
    const onTrain = train({ identical_rows: null, keep_identical_rows_together: false });
    expect(screen.queryByText(/Keep identical rows together/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].keep_identical_rows_together).toBe(false);
  });

  it("is not sent for a split that already keeps its units whole", () => {
    const onTrain = train();
    fireEvent.click(screen.getByRole("radio", { name: /Split by date/ }));
    expect(screen.queryByText(/Keep identical rows together/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].keep_identical_rows_together).toBe(false);
  });

  it("comes back showing what the last run used", () => {
    train({ trained: true, keep_identical_rows_together: false });
    expect(screen.getByRole("checkbox", { name: /Keep identical rows together/ })).not.toBeChecked();
  });
});

describe("requestFromResult — identical rows", () => {
  const result = (kept: boolean | undefined) =>
    ({
      task: "classification",
      target_column: "churned",
      features: { numeric: ["spend"], categorical: [], dropped: [], excluded: [] },
      split: { strategy: "random", time_column: null, group_column: null, keep_duplicates_together: kept },
      operating_point: null,
    }) as unknown as ModelResult;

  it("keeps the choice when 'Improve it' is pressed after a refresh", () => {
    expect(requestFromResult(result(true)).keep_identical_rows_together).toBe(true);
  });

  it("is off for a result from before the option existed", () => {
    expect(requestFromResult(result(undefined)).keep_identical_rows_together).toBe(false);
  });
});
