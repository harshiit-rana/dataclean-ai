import { fireEvent, render, screen } from "@testing-library/react";
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
  group_columns: [],
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

describe("ModelPlanForm — grouped split", () => {
  const GROUPED: Partial<ModelPlan> = {
    group_columns: ["customer_id", "site"],
    split_options: [
      { value: "random", label: "Random split", description: "" },
      { value: "group", label: "Keep related rows together", description: "" },
    ],
  };

  it("disables grouping when nothing in the data repeats", () => {
    renderForm({
      split_options: [
        { value: "random", label: "Random split", description: "" },
        { value: "group", label: "Keep related rows together", description: "" },
      ],
      group_columns: [],
    });
    expect(screen.getByRole("radio", { name: /Keep related rows together/ })).toBeDisabled();
    expect(screen.getByText(/No column in this data looks like one thing/)).toBeInTheDocument();
  });

  it("offers the repeating columns once grouping is chosen", () => {
    renderForm(GROUPED);
    fireEvent.click(screen.getByRole("radio", { name: /Keep related rows together/ }));
    expect(screen.getByRole("option", { name: /Group by customer_id/ })).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Group by site/ })).toBeInTheDocument();
  });

  it("warns that the grouping column stops being an input", () => {
    renderForm(GROUPED);
    fireEvent.click(screen.getByRole("radio", { name: /Keep related rows together/ }));
    expect(screen.getByText(/won't be used as an input/)).toBeInTheDocument();
  });

  it("sends the chosen column with the training request", () => {
    const onTrain = renderForm(GROUPED);
    fireEvent.click(screen.getByRole("radio", { name: /Keep related rows together/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain).toHaveBeenCalledWith(
      expect.objectContaining({ split_strategy: "group", group_column: "customer_id" })
    );
  });

  it("does not send a group column on a random split", () => {
    const onTrain = renderForm(GROUPED);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain).toHaveBeenCalledWith(
      expect.objectContaining({ split_strategy: "random", group_column: null })
    );
  });
});

describe("ModelPlanForm — where the cut-off sits", () => {
  const TWO_CLASS: Partial<ModelPlan> = { class_labels: ["no", "yes"] };

  it("only asks about the cut-off for a two-class target", () => {
    renderForm({ class_labels: ["a", "b", "c"] });
    expect(screen.queryByText(/What matters more/)).not.toBeInTheDocument();
    renderForm({ task_type: "regression", class_labels: [] });
    expect(screen.queryByText(/What matters more/)).not.toBeInTheDocument();
  });

  it("sends the default when nothing was chosen", () => {
    const onTrain = renderForm(TWO_CLASS);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain).toHaveBeenCalledWith(
      expect.objectContaining({
        operating_point: expect.objectContaining({ objective: "default", positive_label: "yes" }),
      })
    );
  });

  it("never sends a threshold -- only the trade-off", () => {
    const onTrain = renderForm(TWO_CLASS);
    fireEvent.click(screen.getByRole("radio", { name: /One kind of mistake costs more/ }));
    fireEvent.change(screen.getByLabelText("Cost of missing a case"), { target: { value: "5" } });
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    const sent = onTrain.mock.calls[0][0].operating_point;
    expect(sent).toMatchObject({ objective: "cost", cost_of_missing: 5, cost_of_false_alarm: 1 });
    expect(sent).not.toHaveProperty("threshold");
  });

  it("will not train on a floor with no number in it", () => {
    renderForm(TWO_CLASS);
    fireEvent.click(screen.getByRole("radio", { name: /Catch at least a set share/ }));
    expect(screen.getByRole("button", { name: /Train/ })).toBeDisabled();
    expect(screen.getByText(/Give the cut-off trade-off a number first/)).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Target rate"), { target: { value: "80" } });
    expect(screen.getByRole("button", { name: /Train/ })).toBeEnabled();
  });

  it("sends the floor as a rate, not a percentage", () => {
    const onTrain = renderForm(TWO_CLASS);
    fireEvent.click(screen.getByRole("radio", { name: /Be right at least/ }));
    fireEvent.change(screen.getByLabelText("Target rate"), { target: { value: "70" } });
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].operating_point).toMatchObject({
      objective: "min_precision",
      target_rate: 0.7,
    });
  });

  it("lets the user say which answer is a case", () => {
    const onTrain = renderForm({ class_labels: ["churned", "retained"] });
    // Sorted last is "retained", which is the wrong thing to be catching.
    fireEvent.change(screen.getByLabelText(/A .case. is a row where the answer is/), {
      target: { value: "churned" },
    });
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].operating_point.positive_label).toBe("churned");
  });
});

describe("ModelPlanForm — derived inputs", () => {
  const DERIVABLE: Partial<ModelPlan> = {
    suggested_features: [
      {
        id: "date_parts:signup_date",
        kind: "date_parts",
        column: "signup_date",
        outputs: ["signup_date__weekday"],
        reason: "Dates spanning 897 days.",
        evidence: {},
      },
      {
        id: "missing_flag:spend",
        kind: "missing_flag",
        column: "spend",
        outputs: ["spend__missing"],
        reason: "Blank in 18% of rows.",
        evidence: {},
      },
    ],
    feature_recipes: ["date_parts:signup_date", "missing_flag:spend"],
  };

  it("sends the ticked recipes with the train request", () => {
    const onTrain = renderForm(DERIVABLE);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].feature_recipes).toEqual([
      "date_parts:signup_date",
      "missing_flag:spend",
    ]);
  });

  it("lets one be turned off without touching the others", () => {
    const onTrain = renderForm(DERIVABLE);
    fireEvent.click(screen.getByRole("checkbox", { name: /signup_date/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].feature_recipes).toEqual(["missing_flag:spend"]);
  });

  it("drops a recipe when its source column is excluded", () => {
    /* Otherwise a column withheld for giving away the answer walks straight back in as a
       missing flag. The server refuses it too; this keeps the form from asking. */
    const onTrain = renderForm({
      ...DERIVABLE,
      features: { numeric: ["spend"], categorical: [], dropped: [] },
    });
    fireEvent.click(screen.getByRole("checkbox", { name: /^spend$/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].feature_recipes).toEqual(["date_parts:signup_date"]);
  });

  it("counts the derived inputs alongside the columns", () => {
    renderForm(DERIVABLE);
    expect(screen.getByText(/plus 2 derived/)).toBeInTheDocument();
  });

  it("says nothing at all when there is nothing to derive", () => {
    renderForm();
    expect(screen.queryByText(/Inputs we can build/)).not.toBeInTheDocument();
  });
});
