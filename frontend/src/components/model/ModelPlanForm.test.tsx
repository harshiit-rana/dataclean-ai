import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelPlanForm } from "./ModelPlanForm";
import { requestFromResult } from "@/pages/ModelPage";
import type { ModelPlan, ModelResult } from "@/api/types";

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

describe("ModelPlanForm — personal details", () => {
  it("says why a name is suggested for leaving out", () => {
    renderForm({
      features: { numeric: [], categorical: ["last_name", "region"], dropped: [] },
      suggested_exclusions: [
        { column: "last_name", reason: "This is a person's name.", kind: "personal", severity: "high" },
      ],
      excluded_columns: ["last_name"],
    });
    expect(screen.getByText(/A person's name or contact details/)).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /last_name/ })).not.toBeChecked();
  });
});

describe("ModelPlanForm — what the suggestions were measured on", () => {
  const leaky: Partial<ModelPlan> = {
    features: { numeric: ["signal"], categorical: [], dropped: [] },
    suggested_exclusions: [
      { column: "outcome_code", reason: "Gives away the answer.", kind: "leakage", severity: "high" },
    ],
    excluded_columns: ["outcome_code"],
  };

  it("says which rows the leakage check read", () => {
    renderForm({
      ...leaky,
      exclusions_basis: "Leakage checked on the 960 training rows of the default random split.",
    });
    expect(screen.getByText(/checked on the 960 training rows/)).toBeInTheDocument();
  });

  it("says nothing when there is nothing suggested", () => {
    renderForm({ exclusions_basis: "Leakage checked on the 960 training rows." });
    expect(screen.queryByText(/training rows of/)).not.toBeInTheDocument();
    expect(screen.queryByText(/checked on the 960/)).not.toBeInTheDocument();
  });
});

describe("ModelPlanForm — how much to hold back (#44)", () => {
  const REC = {
    fraction: 0.3,
    reason: "Raised to 30%: 20% would leave about 8 of the rarer answer to test on, and at least 10 are needed before a score means anything. 30% leaves 90 test rows (12 of the rarer answer) -- enough to judge a model.",
    feasible: true,
    labelled_rows: 300,
    test_rows: 90,
    train_rows: 210,
    test_rare_rows: 12,
  };
  const select = () => screen.getByRole("combobox", { name: /Fraction held back for testing/ }) as HTMLSelectElement;

  it("pre-selects the recommended fraction, labelled as such, with the reason", () => {
    renderForm({ recommended_test_size: REC, trainable_row_count: 300 });
    expect(select().value).toBe("0.3");
    expect(screen.getByRole("option", { name: /30% \(recommended\)/ })).toBeInTheDocument();
    expect(screen.getByText(/Raised to 30%: 20% would leave about 8/)).toBeInTheDocument();
  });

  it("sends the recommendation when the user changes nothing", () => {
    const onTrain = renderForm({ recommended_test_size: REC, trainable_row_count: 300 });
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].test_size).toBe(0.3);
  });

  it("sends what the user chose instead, and says they overrode it", () => {
    const onTrain = renderForm({ recommended_test_size: REC, trainable_row_count: 300 });
    fireEvent.change(select(), { target: { value: "0.15" } });
    expect(screen.getByText(/You chose 15% instead of the recommended 30%/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].test_size).toBe(0.15);
  });

  it("offers a recommendation that is not one of the usual steps", () => {
    renderForm({ recommended_test_size: { ...REC, fraction: 0.12 } });
    expect(screen.getByRole("option", { name: /12% \(recommended\)/ })).toBeInTheDocument();
    expect(select().value).toBe("0.12");
  });

  it("says plainly when no fraction can work, in an amber box", () => {
    renderForm({
      recommended_test_size: { ...REC, fraction: 0.3, feasible: false, reason: "No fraction works for this file: with 150 labelled rows ..." },
    });
    expect(screen.getByText(/No fraction works for this file/)).toHaveClass("bg-amber-500/10");
  });

  it("falls back to the usual fifth for a plan that carries no recommendation", () => {
    const onTrain = renderForm();
    expect(select().value).toBe("0.2");
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].test_size).toBe(0.2);
  });

  it("sets the choice aside when a sealed holdout is taken, whose parts are fixed", () => {
    const onTrain = renderForm({
      recommended_test_size: REC,
      sealed_holdout_offer: { available: true, final_rows: 45, train_rows: 210, development_rows: 45 },
      sealed_holdout: true,
    });
    expect(select()).toBeDisabled();
    expect(screen.getByText(/fixed at 70 \/ 15 \/ 15/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].test_size).toBeNull();
  });
});

describe("ModelPlanForm — a recommended way to split (#43)", () => {
  const GROUP_PLAN: Partial<ModelPlan> = {
    group_columns: ["customer_id", "store"],
    date_columns: ["created_on"],
    split_options: [
      { value: "random", label: "Random split", description: "" },
      { value: "time", label: "Split by date", description: "" },
      { value: "group", label: "Keep related rows together", description: "" },
    ],
    recommended_split: {
      strategy: "group",
      column: "customer_id",
      reasons: [
        "'customer_id' repeats in 1,180 of 1,218 rows (240 distinct values, about 5.1 rows each): a random split would put the same customer_id on both sides.",
      ],
      confidence: "high",
    },
  };
  const radio = (name: RegExp) => screen.getByRole("radio", { name }) as HTMLInputElement;

  it("pre-selects the recommended strategy and column", () => {
    renderForm(GROUP_PLAN);
    expect(radio(/Keep related rows together/).checked).toBe(true);
    expect(radio(/Random split/).checked).toBe(false);
    expect((screen.getByDisplayValue(/customer_id/) as HTMLSelectElement).value).toBe("customer_id");
  });

  it("shows the reason with its evidence and the confidence", () => {
    renderForm(GROUP_PLAN);
    const box = screen.getByTestId("split-recommendation");
    expect(box).toHaveTextContent("Recommended: Keeping related rows together on customer_id");
    expect(box).toHaveTextContent("high confidence");
    expect(box).toHaveTextContent("'customer_id' repeats in 1,180 of 1,218 rows");
  });

  it("marks the option it recommends", () => {
    renderForm(GROUP_PLAN);
    expect(radio(/Keep related rows together/).closest("label")).toHaveTextContent("recommended");
    expect(radio(/Random split/).closest("label")).not.toHaveTextContent("recommended");
  });

  it("sends the recommendation when the user changes nothing", () => {
    const onTrain = renderForm(GROUP_PLAN);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    const request = onTrain.mock.calls[0][0];
    expect(request.split_strategy).toBe("group");
    expect(request.group_column).toBe("customer_id");
  });

  it("lets the user choose differently, says so, and sends the choice", () => {
    const onTrain = renderForm(GROUP_PLAN);
    fireEvent.click(radio(/Random split/));
    expect(screen.getByTestId("split-recommendation")).toHaveTextContent(
      "You chose a random split instead of the recommendation"
    );
    expect(screen.getByTestId("split-recommendation")).toHaveTextContent("the run will record that you did");
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].split_strategy).toBe("random");
  });

  it("says nothing about overriding while the recommendation is being followed", () => {
    renderForm(GROUP_PLAN);
    expect(screen.queryByText(/instead of the recommendation/)).not.toBeInTheDocument();
  });

  it("pre-selects a date split and its column", () => {
    const onTrain = renderForm({
      ...GROUP_PLAN,
      recommended_split: {
        strategy: "time",
        column: "created_on",
        reasons: ["You described the goal in terms of the future ('forecast')."],
        confidence: "medium",
      },
    });
    expect(radio(/Split by date/).checked).toBe(true);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0]).toMatchObject({ split_strategy: "time", time_column: "created_on" });
  });

  it("pre-selects a random split with the reason when there is nothing to protect against", () => {
    renderForm({
      recommended_split: {
        strategy: "random",
        column: null,
        reasons: ["No repeated entities or time order found.", "A random split, stratified so the rarer answer reaches both sides."],
        confidence: "medium",
      },
    });
    expect(radio(/Random split/).checked).toBe(true);
    expect(screen.getByTestId("split-recommendation")).toHaveTextContent("No repeated entities or time order found.");
  });

  it("falls back to a random split rather than selecting an option the form cannot offer", () => {
    renderForm({
      date_columns: [],
      recommended_split: { strategy: "time", column: "created_on", reasons: ["x"], confidence: "medium" },
    });
    expect(radio(/Random split/).checked).toBe(true);
    expect(screen.queryByTestId("split-recommendation")).not.toBeInTheDocument();
  });

  it("is unchanged for a plan with no recommendation", () => {
    renderForm();
    expect(radio(/Random split/).checked).toBe(true);
    expect(screen.queryByTestId("split-recommendation")).not.toBeInTheDocument();
  });
});

describe("ModelPlanForm — a gap between training and test (#20)", () => {
  const DATED: Partial<ModelPlan> = {
    date_columns: ["created_on"],
    split_options: [
      { value: "random", label: "Random split", description: "" },
      { value: "time", label: "Split by date", description: "" },
    ],
  };
  const gapInput = () => screen.getByLabelText("Days to leave between training and test") as HTMLInputElement;
  const chooseDate = () => fireEvent.click(screen.getByRole("radio", { name: /Split by date/ }));

  it("is offered only for a date split", () => {
    renderForm(DATED);
    expect(screen.queryByLabelText("Days to leave between training and test")).not.toBeInTheDocument();
    chooseDate();
    expect(gapInput()).toBeInTheDocument();
  });

  it("explains what it is for", () => {
    renderForm(DATED);
    chooseDate();
    expect(screen.getByText(/cancelled within 30/)).toBeInTheDocument();
    expect(screen.getByText(/Rows inside the gap are dropped, and the result says how many/)).toBeInTheDocument();
  });

  it("sends the number of days chosen", () => {
    const onTrain = renderForm(DATED);
    chooseDate();
    fireEvent.change(gapInput(), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0]).toMatchObject({ split_strategy: "time", time_gap_days: 30 });
  });

  it("sends no gap when none is entered", () => {
    const onTrain = renderForm(DATED);
    chooseDate();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].time_gap_days).toBeNull();
  });

  it("sends none for a random split even if a number was typed before switching", () => {
    const onTrain = renderForm(DATED);
    chooseDate();
    fireEvent.change(gapInput(), { target: { value: "30" } });
    fireEvent.click(screen.getByRole("radio", { name: /Random split/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].time_gap_days).toBeNull();
  });

  it("will not train on a gap that is not a sensible number of days", () => {
    renderForm(DATED);
    chooseDate();
    fireEvent.change(gapInput(), { target: { value: "-5" } });
    expect(screen.getByText(/whole number of days between 0 and 3,650/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Train/ })).toBeDisabled();
    fireEvent.change(gapInput(), { target: { value: "14" } });
    expect(screen.getByRole("button", { name: /Train/ })).toBeEnabled();
  });
});


describe("requestFromResult — the gap (#20)", () => {
  const stored = (split: Record<string, unknown>) =>
    ({
      task: "classification",
      features: { numeric: [], categorical: [], dropped: [], excluded: [] },
      split: { strategy: "time", time_column: "day", group_column: null, train_rows: 1, test_rows: 1, ...split },
    }) as unknown as ModelResult;

  it("carries the gap into a rebuilt request, so improving or calibrating keeps it", () => {
    expect(requestFromResult(stored({ gap_days: 30 })).time_gap_days).toBe(30);
  });

  it("carries none when there was none, or for a split that is not by date", () => {
    expect(requestFromResult(stored({ gap_days: 0 })).time_gap_days).toBeNull();
    expect(requestFromResult(stored({ strategy: "random", gap_days: 30 })).time_gap_days).toBeNull();
  });
});
