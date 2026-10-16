import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LiveImprovement } from "./LiveImprovement";
import { requestFromResult } from "@/pages/ModelPage";
import type { ImprovementRound, ModelResult } from "@/api/types";

const round = (overrides: Partial<ImprovementRound>): ImprovementRound =>
  ({
    name: "baseline",
    label: "Starting point",
    what_it_tried: "",
    score: 0.799,
    std: 0.01,
    gain: null,
    accepted: true,
    outcome: "Logistic regression leads at 0.799 AUC.",
    detail: {},
    ...overrides,
  }) as ImprovementRound;

describe("LiveImprovement — what a screen reader hears (#101)", () => {
  it("announces only the latest finished round, briefly", () => {
    render(
      <LiveImprovement
        rounds={[round({}), round({ name: "hyperparameters", label: "Settings", accepted: false })]}
        stage="Trying other models."
      />
    );
    expect(screen.getByRole("status")).toHaveTextContent("Round finished: Settings, not kept.");
  });

  it("says a kept round is kept in words, not only colour", () => {
    render(<LiveImprovement rounds={[round({ accepted: true })]} stage={null} />);
    expect(screen.getByText("kept")).toBeInTheDocument();
  });
});

describe("LiveImprovement", () => {
  it("says what it is doing before any round has finished", () => {
    render(<LiveImprovement rounds={[]} stage={null} />);
    expect(screen.getByText("Starting…")).toBeInTheDocument();
  });

  it("labels its scores as exploratory — they are fold scores, not the held-out result", () => {
    render(<LiveImprovement rounds={[round({})]} stage={null} />);
    expect(screen.getByText(/Exploratory · training folds/)).toBeInTheDocument();
  });

  it("names the round in progress", () => {
    render(
      <LiveImprovement rounds={[]} stage="Searching the settings of logistic regression." />
    );
    expect(screen.getByText(/Searching the settings of logistic regression/)).toBeInTheDocument();
  });

  it("shows finished rounds as they land, rejected ones included", () => {
    render(
      <LiveImprovement
        rounds={[
          round({}),
          round({
            name: "hyperparameters",
            label: "Model settings",
            score: 0.805,
            gain: 0.006,
            accepted: false,
            outcome: "No setting beat the defaults by more than the folds disagree.",
          }),
        ]}
        stage="Tuning the runners-up."
      />
    );
    expect(screen.getByText(/leads at 0.799 AUC/)).toBeInTheDocument();
    expect(screen.getByText(/No setting beat the defaults/)).toBeInTheDocument();
    expect(screen.getByText("+0.006")).toBeInTheDocument();
  });
});

describe("requestFromResult", () => {
  /* After a refresh the request that trained the model is gone, and "Improve it" fell back to
     the plan's defaults -- quietly turning a grouped run into a random one. */
  const RESULT = {
    task: "classification",
    target_column: "churned",
    features: {
      numeric: ["signal", "signup_date__days"],
      categorical: [],
      dropped: [],
      excluded: ["first_name", "customer_id"],
      derived: [
        {
          id: "date_parts:signup_date",
          kind: "date_parts",
          column: "signup_date",
          outputs: ["signup_date__days"],
          reason: "",
          evidence: {},
        },
      ],
    },
    split: {
      strategy: "group",
      time_column: null,
      group_column: "customer_id",
      train_rows: 800,
      test_rows: 200,
      train_groups: 80,
      test_groups: 20,
    },
    operating_point: {
      objective: "cost",
      description: "",
      cost_of_missing: 5,
      cost_of_false_alarm: 1,
      target_rate: null,
      positive_label: "yes",
      threshold: 0.31,
    },
  } as unknown as ModelResult;

  it("keeps the split, the exclusions, the trade-off and the derived inputs", () => {
    expect(requestFromResult(RESULT)).toMatchObject({
      split_strategy: "group",
      group_column: "customer_id",
      excluded_columns: ["first_name", "customer_id"],
      feature_recipes: ["date_parts:signup_date"],
      operating_point: { objective: "cost", cost_of_missing: 5, positive_label: "yes" },
    });
  });

  it("sends no trade-off when the default cut-off was used", () => {
    const atDefault = {
      ...RESULT,
      operating_point: { ...RESULT.operating_point, objective: "default" },
    } as ModelResult;
    expect(requestFromResult(atDefault).operating_point).toBeNull();
  });
});
