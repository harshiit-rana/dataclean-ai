import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ModelPlanForm } from "./ModelPlanForm";
import { ModelResults } from "./ModelResults";
import { requestFromResult } from "@/pages/ModelPage";
import type { ModelPlan, ModelResult } from "@/api/types";

const PLAN: ModelPlan = {
  session_id: "s1",
  target_column: "churned",
  task_type: "classification",
  row_count: 800,
  trainable_row_count: 800,
  features: { numeric: ["spend"], categorical: [], dropped: [] },
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
  sealed_holdout_offer: {
    available: true,
    reason: null,
    train_rows: 560,
    development_rows: 120,
    final_rows: 120,
  },
  sealed_holdout: false,
};

function train(plan: Partial<ModelPlan> = {}) {
  const onTrain = vi.fn();
  render(<ModelPlanForm plan={{ ...PLAN, ...plan }} onTrain={onTrain} training={false} />);
  return onTrain;
}

describe("ModelPlanForm — sealed holdout", () => {
  it("offers it with what each part would hold", () => {
    train();
    expect(screen.getByRole("checkbox", { name: /Keep a sealed final holdout/ })).toBeInTheDocument();
    expect(screen.getByText(/Sets aside 120 rows before anything is decided/)).toBeInTheDocument();
    expect(screen.getByText(/Training uses about 560 rows/)).toBeInTheDocument();
  });

  it("is opt-in", () => {
    const onTrain = train();
    expect(screen.getByRole("checkbox", { name: /Keep a sealed final holdout/ })).not.toBeChecked();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].sealed_holdout).toBe(false);
  });

  it("sends the choice", () => {
    const onTrain = train();
    fireEvent.click(screen.getByRole("checkbox", { name: /Keep a sealed final holdout/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].sealed_holdout).toBe(true);
  });

  it("says it only counts before the first look", () => {
    train();
    expect(screen.getByText(/only counts if you take it before your first look/)).toBeInTheDocument();
  });

  it("is withheld, with the reason, when the file cannot afford it", () => {
    const onTrain = train({
      sealed_holdout_offer: {
        available: false,
        reason: "The final part would be too small to judge.",
      },
    });
    const box = screen.getByRole("checkbox", { name: /Keep a sealed final holdout/ });
    expect(box).toBeDisabled();
    expect(
      screen.getByText(/Not available for this file. The final part would be too small to judge/)
    ).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].sealed_holdout).toBe(false);
  });

  it("comes back showing what the last run used", () => {
    train({ trained: true, sealed_holdout: true });
    expect(screen.getByRole("checkbox", { name: /Keep a sealed final holdout/ })).toBeChecked();
  });
});

function result(split: Partial<ModelResult["split"]> = {}, evaluation: unknown = undefined): ModelResult {
  return {
    task: "classification",
    target_column: "churned",
    features: { numeric: ["spend"], categorical: [], dropped: [] },
    split: {
      strategy: "random",
      time_column: null,
      group_column: null,
      train_rows: 560,
      test_rows: 120,
      train_groups: null,
      test_groups: null,
      sealed: true,
      final_rows: 120,
      final_fraction: 0.15,
      sealed_after_looks: 0,
      ...split,
    },
    baseline: { strategy: "always answer no", accuracy: 0.7, roc_auc: 0.5 },
    primary_metric: "roc_auc",
    results: [
      {
        model: "logistic_regression",
        label: "Logistic regression",
        description: "",
        metrics: { accuracy: 0.8, roc_auc: 0.82 },
        feature_importance: [],
        column_importance: [],
        diagnostics: {},
        failed: null,
        cv_score: 0.8,
        cv_std: 0.02,
      },
    ],
    best_model: "logistic_regression",
    selection: {
      chosen: "logistic_regression",
      method: "only_candidate",
      metric: null,
      folds: 0,
      scores: {},
      chosen_on: "the only model trained",
    },
    assessment: { verdict: "useful", headline: "This model separates the two outcomes." },
    deep_diagnostics: {},
    evaluation,
  } as unknown as ModelResult;
}

const FINAL = {
  at: "2026-10-14T10:00:00+00:00",
  rows: 120,
  metrics: { accuracy: 0.77, roc_auc: 0.79 },
  baseline: { strategy: "always answer no", accuracy: 0.7, roc_auc: 0.5 },
  assessment: { verdict: "useful", headline: "On the sealed rows the model still separates the outcomes." },
  primary_metric: "roc_auc",
  rows_seen_in_training: 0,
};
const EVAL = { look_count: 1, limit: 3, status: "independent", message: "", looks_on_other_splits: 0 };

describe("ModelResults — the sealed holdout", () => {
  it("says what was set aside and that it was never scored", () => {
    render(<ModelResults result={result()} onFinalise={vi.fn()} />);
    expect(screen.getByText(/120 rows were set aside and have never been scored/)).toBeInTheDocument();
  });

  it("offers one button to open it, with the cost of doing so", () => {
    render(<ModelResults result={result()} onFinalise={vi.fn()} />);
    expect(screen.getByRole("button", { name: /Open the sealed rows \(once\)/ })).toBeInTheDocument();
    expect(screen.getByText(/After that they are used up/)).toBeInTheDocument();
  });

  it("opens it only when asked", () => {
    const onFinalise = vi.fn();
    render(<ModelResults result={result()} onFinalise={onFinalise} />);
    expect(onFinalise).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: /Open the sealed rows/ }));
    expect(onFinalise).toHaveBeenCalledTimes(1);
  });

  it("is disabled while it is being opened", () => {
    render(<ModelResults result={result()} onFinalise={vi.fn()} finalising />);
    expect(screen.getByRole("button", { name: /Opening/ })).toBeDisabled();
  });

  it("shows the final evaluation beside the no-model baseline once opened, and no button", () => {
    render(<ModelResults result={result({}, { ...EVAL, final: FINAL })} onFinalise={vi.fn()} />);
    expect(screen.getByText(/still separates the outcomes/)).toBeInTheDocument();
    expect(screen.getByText("Final evaluation")).toBeInTheDocument();
    expect(screen.getByText("0.790")).toBeInTheDocument();
    expect(screen.getByText(/Scored once on the 120 sealed rows/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Open the sealed rows/ })).not.toBeInTheDocument();
    expect(screen.getByText(/These rows are now used up/)).toBeInTheDocument();
  });

  it("prints the interval beside the final evaluation", () => {
    const withInterval = {
      ...FINAL,
      metrics_intervals: {
        level: 0.95,
        resamples: 1000,
        by: "rows",
        metrics: { roc_auc: { low: 0.7, high: 0.86 } },
      },
    };
    render(<ModelResults result={result({}, { ...EVAL, final: withInterval })} onFinalise={vi.fn()} />);
    expect(screen.getByText("(0.700–0.860)")).toBeInTheDocument();
    expect(screen.getByText(/the middle 95% of 1,000 resamples of the held-out rows/)).toBeInTheDocument();
  });

  it("says when a seal was taken after looks, without calling it pristine", () => {
    render(<ModelResults result={result({ sealed_after_looks: 3 })} onFinalise={vi.fn()} />);
    expect(screen.getByText(/Sealed after 3 looks/)).toBeInTheDocument();
    expect(screen.getByText(/shaped decisions made before the seal/)).toBeInTheDocument();
  });

  it("says nothing about lateness when it was sealed before the first look", () => {
    render(<ModelResults result={result()} onFinalise={vi.fn()} />);
    expect(screen.queryByText(/Sealed after/)).not.toBeInTheDocument();
  });

  it("repeats the error if opening fails", () => {
    render(
      <ModelResults
        result={result()}
        onFinalise={vi.fn()}
        finaliseError="The sealed rows are not what was sealed."
      />
    );
    expect(screen.getByText(/not what was sealed/)).toBeInTheDocument();
  });

  it("shows the note when the rows were opened for an earlier model", () => {
    render(
      <ModelResults
        result={result()}
        onFinalise={vi.fn()}
        finaliseNote="The sealed rows were already opened, for an earlier model."
      />
    );
    expect(screen.getByText(/already opened, for an earlier model/)).toBeInTheDocument();
  });

  it("says a seal that could not be made, and that the ordinary split was used", () => {
    render(
      <ModelResults
        result={result({
          sealed: false,
          sealed_unavailable_reason: "The final part would be too small to judge.",
        })}
        onFinalise={vi.fn()}
      />
    );
    expect(screen.getByText(/asked for but could not be made/)).toBeInTheDocument();
    expect(screen.getByText(/The ordinary split was used instead/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Open the sealed rows/ })).not.toBeInTheDocument();
  });

  it("shows nothing for a result with no seal at all", () => {
    render(<ModelResults result={result({ sealed: false })} onFinalise={vi.fn()} />);
    expect(screen.queryByText(/Sealed final holdout/)).not.toBeInTheDocument();
  });
});

describe("requestFromResult — sealed holdout", () => {
  const rebuilt = (sealed: boolean | undefined) =>
    ({
      task: "classification",
      target_column: "churned",
      features: { numeric: ["spend"], categorical: [], dropped: [], excluded: [] },
      split: { strategy: "random", time_column: null, group_column: null, sealed },
      operating_point: null,
    }) as unknown as ModelResult;

  it("keeps the seal when 'Improve it' is pressed after a refresh", () => {
    expect(requestFromResult(rebuilt(true)).sealed_holdout).toBe(true);
  });

  it("is off for a result with no seal", () => {
    expect(requestFromResult(rebuilt(undefined)).sealed_holdout).toBe(false);
  });
});
