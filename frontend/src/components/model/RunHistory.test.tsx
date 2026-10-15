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

describe("RunHistory — the cut-off is part of the setup", () => {
  it("names the line when one was chosen", () => {
    render(
      <RunHistory
        runs={[
          run({
            operating_point: { objective: "cost", description: "", threshold: 0.31, positive_label: "yes", observed: null },
          }),
          run({ operating_point: { objective: "default", description: "", threshold: 0.5, positive_label: "yes", observed: null } }),
        ]}
        currentFingerprint="aaa"
      />
    );
    expect(screen.getByText(/cut-off 0.31/)).toBeInTheDocument();
    expect(screen.queryByText(/cut-off 0.50/)).not.toBeInTheDocument();
  });
});

describe("RunHistory — how the held-out rows had been used", () => {
  it("never calls a run from before the ledger independent", () => {
    renderHistory([run(), run()]);
    expect(screen.getAllByText(/test-row use not recorded/).length).toBe(2);
    expect(screen.queryByText(/first look/)).not.toBeInTheDocument();
  });

  it("says a first run was the first look", () => {
    renderHistory([
      run({ evaluation: { look_count: 2, status: "reused" } }),
      run({ evaluation: { look_count: 1, status: "independent" } }),
    ]);
    expect(screen.getByText(/first look at the test rows/)).toBeInTheDocument();
  });

  it("says which look a reused run was, and what that made it", () => {
    renderHistory([
      run({ evaluation: { look_count: 4, status: "exhausted" } }),
      run({ evaluation: { look_count: 1, status: "independent" } }),
    ]);
    expect(screen.getByText(/look 4 at the test rows \(exhausted\)/)).toBeInTheDocument();
  });
});

describe("RunHistory — what the Score column is", () => {
  it("labels the column as the held-out score", () => {
    renderHistory([run(), run()]);
    const header = screen.getByRole("columnheader", { name: /Score/ });
    expect(header).toHaveTextContent("Held-out evaluation");
  });
});

describe("RunHistory — a change is only coloured when it is real", () => {
  const withVerdicts = (comparisons: Record<string, import("@/api/types").RunComparison>, runs: ModelRun[]) =>
    render(<RunHistory runs={runs} currentFingerprint="aaa" comparisons={comparisons} />);

  const pair = (newScore: number, oldScore = 0.8) => {
    const newer = run({ run_id: "new", metrics: { roc_auc: newScore } });
    const older = run({ run_id: "old", metrics: { roc_auc: oldScore } });
    return [newer, older];
  };

  it("colours a change green when it is clearly outside the noise", () => {
    withVerdicts(
      { new: { method: "paired", verdict: "better", difference: 0.02, low: 0.006, high: 0.034 } },
      pair(0.82)
    );
    expect(screen.getByText("+0.0200")).toHaveClass("text-emerald-600");
  });

  it("colours a clear drop red", () => {
    withVerdicts(
      { new: { method: "paired", verdict: "worse", difference: -0.05, low: -0.08, high: -0.02 } },
      pair(0.75)
    );
    expect(screen.getByText("-0.0500")).toHaveClass("text-destructive");
  });

  it("leaves a change inside the noise uncoloured and says so", () => {
    // +0.01 used to read as an improvement because it was above the old 0.0005 line.
    withVerdicts(
      { new: { method: "paired", verdict: "within_the_noise", difference: 0.01, low: -0.02, high: 0.04 } },
      pair(0.81)
    );
    const delta = screen.getByText("+0.0100");
    expect(delta).not.toHaveClass("text-emerald-600");
    expect(delta).not.toHaveClass("text-destructive");
    expect(screen.getByText("within the noise")).toBeInTheDocument();
  });

  it("shows the paired range so the size of the doubt is visible", () => {
    withVerdicts(
      { new: { method: "paired", verdict: "within_the_noise", difference: 0.01, low: -0.02, high: 0.04 } },
      pair(0.81)
    );
    expect(screen.getByText(/95% range -0\.020 to \+0\.040/)).toBeInTheDocument();
  });

  it("shows no range when the verdict came from comparing two intervals", () => {
    withVerdicts({ new: { method: "intervals", verdict: "within_the_noise" } }, pair(0.81));
    expect(screen.getByText("within the noise")).toBeInTheDocument();
    expect(screen.queryByText(/95% range/)).not.toBeInTheDocument();
  });

  it("does not colour a run it has no evidence about, however large the difference", () => {
    withVerdicts({ new: { method: null, verdict: null } }, pair(0.9));
    const delta = screen.getByText("+0.1000");
    expect(delta).not.toHaveClass("text-emerald-600");
    expect(screen.queryByText("within the noise")).not.toBeInTheDocument();
  });

  it("does not colour runs from a server that sent no comparisons at all", () => {
    renderHistory(pair(0.9));
    expect(screen.getByText("+0.1000")).not.toHaveClass("text-emerald-600");
  });

  it("never judges two runs it already refuses to compare", () => {
    withVerdicts(
      { new: { method: "paired", verdict: "better", difference: 0.1, low: 0.05, high: 0.15 } },
      [
        run({ run_id: "new", metrics: { roc_auc: 0.9 }, data_fingerprint: "bbb" }),
        run({ run_id: "old", metrics: { roc_auc: 0.8 }, data_fingerprint: "aaa" }),
      ]
    );
    expect(screen.getByText("not comparable")).toBeInTheDocument();
    expect(screen.queryByText(/95% range/)).not.toBeInTheDocument();
  });
});

describe("RunHistory — how much was held back is part of the setup (#44)", () => {
  const withFraction = (requested_test_size: number | null | undefined) =>
    run({ split: { strategy: "random", time_column: null, group_column: null, train_rows: 240, test_rows: 60, requested_test_size } });

  it("names an unusual fraction, because a different fraction is a different test set", () => {
    renderHistory([withFraction(0.3), withFraction(0.2)]);
    expect(screen.getByText(/30% held back/)).toBeInTheDocument();
  });

  it("says nothing for the usual fifth, or for a run from before it was recorded", () => {
    renderHistory([withFraction(0.2), withFraction(undefined)]);
    expect(screen.queryByText(/held back/)).not.toBeInTheDocument();
  });
});

describe("RunHistory — a split nobody recommended is part of the setup (#43)", () => {
  it("says when the run overrode the recommendation", () => {
    renderHistory([run({ split_overridden: true }), run({ split_overridden: false })]);
    expect(screen.getAllByText(/overrode the recommended split/).length).toBe(1);
  });

  it("says nothing for a run that followed it, or one from before it was recorded", () => {
    renderHistory([run({ split_overridden: false }), run({ split_overridden: null })]);
    expect(screen.queryByText(/overrode the recommended split/)).not.toBeInTheDocument();
  });
});

describe("RunHistory — the gap is part of the setup (#20)", () => {
  const withGap = (gap_days: number | null | undefined) =>
    run({ split: { strategy: "time", time_column: "day", group_column: null, train_rows: 240, test_rows: 60, gap_days } });

  it("names the embargo, because a different gap holds out different rows", () => {
    renderHistory([withGap(30), withGap(0)]);
    expect(screen.getByText(/30-day gap/)).toBeInTheDocument();
  });

  it("says nothing without one", () => {
    renderHistory([withGap(0), withGap(undefined)]);
    expect(screen.queryByText(/-day gap/)).not.toBeInTheDocument();
  });
});
