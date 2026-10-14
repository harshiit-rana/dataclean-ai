import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { ClipOutliers } from "./ClipOutliers";
import { ModelPlanForm } from "./ModelPlanForm";
import { RunHistory } from "./RunHistory";
import { requestFromResult } from "@/pages/ModelPage";
import type { ClippingOffer, ModelPlan, ModelResult, ModelRun } from "@/api/types";

const OFFER: ClippingOffer = {
  column: "monthly_spend",
  reason:
    "3 unusually high (largest 4200) and 1 unusually low (smallest -90), while the rest run from 10 to 130.",
  evidence: {
    far_out: 4,
    count: 8,
    share: 0.03,
    low: 5,
    high: 3,
    lowest: -90,
    highest: 4200,
    typical_min: 10,
    typical_max: 130,
  },
};

describe("ClipOutliers", () => {
  it("renders nothing when no column has anything worth clipping", () => {
    const { container } = render(<ClipOutliers offers={[]} chosen={[]} onChange={vi.fn()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows what was found in the column, not just its name", () => {
    render(<ClipOutliers offers={[OFFER]} chosen={[]} onChange={vi.fn()} />);
    expect(screen.getByText(/largest 4200/)).toBeInTheDocument();
  });

  it("is off until ticked", () => {
    render(<ClipOutliers offers={[OFFER]} chosen={[]} onChange={vi.fn()} />);
    expect(screen.getByRole("checkbox")).not.toBeChecked();
  });

  it("reports a tick and an untick", () => {
    const onChange = vi.fn();
    const { rerender } = render(<ClipOutliers offers={[OFFER]} chosen={[]} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenLastCalledWith(["monthly_spend"]);
    rerender(<ClipOutliers offers={[OFFER]} chosen={["monthly_spend"]} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenLastCalledWith([]);
  });

  it("promises training-only limits and no lost rows", () => {
    render(<ClipOutliers offers={[OFFER]} chosen={[]} onChange={vi.fn()} />);
    expect(screen.getByText(/training rows only/)).toBeInTheDocument();
    expect(screen.getByText(/no row is removed/)).toBeInTheDocument();
  });
});

const PLAN: ModelPlan = {
  session_id: "s1",
  target_column: "churned",
  task_type: "classification",
  row_count: 300,
  trainable_row_count: 300,
  features: { numeric: ["monthly_spend"], categorical: [], dropped: [] },
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
  suggested_clipping: [OFFER],
};

describe("ModelPlanForm — clipping", () => {
  it("sends nothing to clip unless the user ticked something", () => {
    const onTrain = vi.fn();
    render(<ModelPlanForm plan={PLAN} onTrain={onTrain} training={false} />);
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].clip_outliers).toEqual([]);
  });

  it("sends the ticked columns", () => {
    const onTrain = vi.fn();
    render(<ModelPlanForm plan={PLAN} onTrain={onTrain} training={false} />);
    const label = screen.getByText(/largest 4200/).closest("label") as HTMLElement;
    fireEvent.click(within(label).getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].clip_outliers).toEqual(["monthly_spend"]);
  });

  it("comes back showing what the last run clipped", () => {
    const onTrain = vi.fn();
    render(
      <ModelPlanForm
        plan={{ ...PLAN, trained: true, features: { ...PLAN.features, clipped: ["monthly_spend"] } }}
        onTrain={onTrain}
        training={false}
      />
    );
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].clip_outliers).toEqual(["monthly_spend"]);
  });

  it("drops a clip when its column is unticked as an input", () => {
    /* A column left out of the model has nothing to clip; sending it would be a lie. */
    const onTrain = vi.fn();
    render(
      <ModelPlanForm
        plan={{ ...PLAN, features: { ...PLAN.features, clipped: ["monthly_spend"] } }}
        onTrain={onTrain}
        training={false}
      />
    );
    fireEvent.click(screen.getByRole("checkbox", { name: /^monthly_spend$/ }));
    fireEvent.click(screen.getByRole("button", { name: /Train/ }));
    expect(onTrain.mock.calls[0][0].clip_outliers).toEqual([]);
  });

  it("shows no clipping section when nothing is worth clipping", () => {
    render(
      <ModelPlanForm plan={{ ...PLAN, suggested_clipping: [] }} onTrain={vi.fn()} training={false} />
    );
    expect(screen.queryByText(/Tame extreme values/)).not.toBeInTheDocument();
  });
});

describe("requestFromResult — clipping", () => {
  const RESULT = {
    task: "classification",
    target_column: "churned",
    features: { numeric: ["monthly_spend"], categorical: [], dropped: [], excluded: [], clipped: ["monthly_spend"] },
    split: { strategy: "random", time_column: null, group_column: null },
    operating_point: null,
  } as unknown as ModelResult;

  it("keeps the clipping when 'Improve it' is pressed after a refresh", () => {
    expect(requestFromResult(RESULT).clip_outliers).toEqual(["monthly_spend"]);
  });

  it("is empty for a result from before clipping existed", () => {
    const old = { ...RESULT, features: { ...RESULT.features, clipped: undefined } } as ModelResult;
    expect(requestFromResult(old).clip_outliers).toEqual([]);
  });
});

describe("RunHistory — clipping", () => {
  const run = (id: string, clipped: string[]) =>
    ({
      run_id: id,
      at: "2026-10-14T10:00:00+00:00",
      data_fingerprint: "abc",
      target_column: "churned",
      task: "classification",
      best_model: "logistic_regression",
      model_label: "Logistic regression",
      models_tried: ["logistic_regression"],
      primary_metric: "roc_auc",
      metrics: { roc_auc: 0.8 },
      baseline: {},
      verdict: "useful",
      headline: "It works.",
      split: { strategy: "random", time_column: null, group_column: null, train_rows: 240, test_rows: 60 },
      feature_count: 4,
      excluded_columns: [],
      tuned: false,
      tuned_settings: {},
      stopped_because: null,
      clipped_columns: clipped,
    }) as ModelRun;

  it("says a run clipped its inputs, so two runs can be told apart", () => {
    render(
      <RunHistory runs={[run("r2", ["monthly_spend"]), run("r1", [])]} currentFingerprint="abc" />
    );
    expect(screen.getByText(/1 clipped/)).toBeInTheDocument();
  });

  it("says nothing about clipping for a run that did not", () => {
    render(<RunHistory runs={[run("r2", []), run("r1", [])]} currentFingerprint="abc" />);
    expect(screen.queryByText(/clipped/)).not.toBeInTheDocument();
  });
});
