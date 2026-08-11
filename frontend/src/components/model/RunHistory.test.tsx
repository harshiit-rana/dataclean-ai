import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RunHistory } from "./RunHistory";
import type { ModelRun } from "@/api/types";

function run(overrides: Partial<ModelRun> = {}): ModelRun {
  return {
    run_id: Math.random().toString(36).slice(2),
    at: "2026-08-07T10:00:00+00:00",
    data_fingerprint: "aaa",
    target_column: "outcome",
    task: "classification",
    best_model: "logistic_regression",
    model_label: "Logistic regression",
    models_tried: ["logistic_regression"],
    primary_metric: "roc_auc",
    metrics: { roc_auc: 0.8 },
    baseline: { accuracy: 0.6 },
    verdict: "useful",
    headline: "It works.",
    split: {
      strategy: "random",
      time_column: null,
      group_column: null,
      train_rows: 240,
      test_rows: 60,
    },
    feature_count: 3,
    excluded_columns: [],
    tuned: false,
    tuned_settings: {},
    stopped_because: null,
    ...overrides,
  };
}

const renderHistory = (runs: ModelRun[], fingerprint = "aaa") =>
  render(<RunHistory runs={runs} currentFingerprint={fingerprint} />);

describe("RunHistory", () => {
  it("says nothing when there is only one run", () => {
    const { container } = renderHistory([run()]);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the change between consecutive runs", () => {
    renderHistory([run({ metrics: { roc_auc: 0.86 } }), run({ metrics: { roc_auc: 0.8 } })]);
    expect(screen.getByText("+0.0600")).toBeInTheDocument();
  });

  it("shows a drop as a drop", () => {
    renderHistory([run({ metrics: { roc_auc: 0.75 } }), run({ metrics: { roc_auc: 0.8 } })]);
    expect(screen.getByText("-0.0500")).toBeInTheDocument();
  });

  it("records what made each run different", () => {
    renderHistory([
      run({ excluded_columns: ["noise"], feature_count: 2 }),
      run(),
    ]);
    expect(screen.getByText(/without noise/)).toBeInTheDocument();
  });

  it("refuses to compare runs trained on different data", () => {
    renderHistory([
      run({ metrics: { roc_auc: 0.9 }, data_fingerprint: "bbb" }),
      run({ metrics: { roc_auc: 0.8 }, data_fingerprint: "aaa" }),
    ], "bbb");
    expect(screen.getByText("not comparable")).toBeInTheDocument();
    expect(screen.queryByText("+0.1000")).not.toBeInTheDocument();
  });

  it("flags a run that predates a cleaning change", () => {
    renderHistory([run({ data_fingerprint: "bbb" }), run({ data_fingerprint: "aaa" })], "bbb");
    expect(screen.getByText(/trained before the data changed/)).toBeInTheDocument();
  });

  it("names the grouped split, because it is the thing that moved the score", () => {
    renderHistory([
      run({
        split: {
          strategy: "group",
          time_column: null,
          group_column: "customer_id",
          train_rows: 240,
          test_rows: 60,
        },
      }),
      run(),
    ]);
    expect(screen.getByText(/grouped by customer_id/)).toBeInTheDocument();
  });

  it("marks a tuned run", () => {
    renderHistory([run({ tuned: true }), run()]);
    expect(screen.getByText(/tuned/)).toBeInTheDocument();
  });

  it("is honest that an old run cannot be restored", () => {
    renderHistory([run(), run()]);
    expect(screen.getByText(/a record, not a restore point/)).toBeInTheDocument();
  });
});
