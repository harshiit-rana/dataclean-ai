import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelPlanForm } from "./ModelPlanForm";
import type { ModelPlan } from "@/api/types";

/** The plan the RSNA knee dataset produced: a large file, almost none of it labelled, and a
 * verdict the server will refuse to train on. The form used to offer the button anyway and
 * print the file's row count beside it. */
const PLAN: ModelPlan = {
  session_id: "s1",
  target_column: "Effusion",
  task_type: "classification",
  row_count: 4407,
  trainable_row_count: 58,
  features: { numeric: ["signal"], categorical: ["region"], dropped: [] },
  suggested_exclusions: [],
  excluded_columns: [],
  candidate_models: [
    { name: "logistic_regression", label: "Logistic regression", description: "", explains_features: true },
  ],
  split_options: [{ value: "random", label: "Random split", description: "" }],
  date_columns: [],
  blockers: [],
  warnings: [],
  trained: false,
};

function renderForm(overrides: Partial<ModelPlan> = {}) {
  const onTrain = vi.fn();
  render(<ModelPlanForm plan={{ ...PLAN, ...overrides }} onTrain={onTrain} training={false} />);
  return onTrain;
}

describe("ModelPlanForm", () => {
  it("counts the rows it can train on, not the rows in the file", () => {
    renderForm();
    expect(screen.getByText(/58 rows of 4,407/)).toBeInTheDocument();
  });

  it("does not mention a second number when every row is usable", () => {
    renderForm({ row_count: 300, trainable_row_count: 300 });
    expect(screen.getByText(/300 rows\./)).toBeInTheDocument();
    expect(screen.queryByText(/ of 300/)).not.toBeInTheDocument();
  });

  it("will not offer to train on data the server has already refused", () => {
    renderForm({ blockers: ["4,349 rows still have no value for 'Effusion'."] });
    expect(screen.getByRole("button", { name: /Train/ })).toBeDisabled();
  });

  it("says why the button is dead", () => {
    renderForm({ blockers: ["4,349 rows still have no value for 'Effusion'."] });
    expect(screen.getByText(/Fix what's blocking this above/)).toBeInTheDocument();
  });

  it("still trains when nothing is blocking", () => {
    renderForm();
    expect(screen.getByRole("button", { name: /Train/ })).toBeEnabled();
  });
});
