import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunComparisonPanel } from "./RunComparison";
import { RunHistory } from "./RunHistory";
import type { ModelRun, RunPairComparison } from "@/api/types";

const side = (run_id: string, label: string) => ({
  run_id,
  at: "2026-10-16T10:00:00+00:00",
  model_label: label,
  best_model: "logistic_regression",
  primary_metric: "roc_auc",
});

const COMPARABLE: RunPairComparison = {
  a: side("new", "Logistic regression"),
  b: side("old", "Logistic regression"),
  comparable: true,
  reasons: [],
  same_rows: true,
  settings: [
    { field: "excluded_columns", label: "Columns left out", a: ["region"], b: [] },
    { field: "feature_count", label: "Inputs used", a: 2, b: 3 },
  ],
  metrics: [
    { metric: "roc_auc", a: 0.81, b: 0.8, difference: 0.01 },
    { metric: "accuracy", a: 0.74, b: 0.75, difference: -0.01 },
  ],
  primary: { metric: "roc_auc", method: "paired", verdict: "within_the_noise", difference: 0.01, low: -0.02, high: 0.04 },
  summary: "Leaving out 'region' did not change roc_auc beyond the noise.",
};

describe("RunComparisonPanel (#69)", () => {
  it("leads with the sentence, then what differed and the paired range", () => {
    render(<RunComparisonPanel comparison={COMPARABLE} />);
    expect(screen.getByText("Leaving out 'region' did not change roc_auc beyond the noise.")).toBeInTheDocument();
    expect(screen.getByText("Columns left out")).toBeInTheDocument();
    expect(screen.getByText("region")).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
    expect(screen.getByText(/roc_auc: \+0\.010 \(95% range -0\.020 to \+0\.040/)).toBeInTheDocument();
    expect(screen.getByText("accuracy")).toBeInTheDocument();
  });

  it("explains an incomparable pair and shows no score difference", () => {
    const incomparable: RunPairComparison = {
      ...COMPARABLE,
      comparable: false,
      same_rows: false,
      reasons: ["they were trained on different data"],
      primary: null,
      summary: "These two runs cannot be compared: they were trained on different data.",
    };
    render(<RunComparisonPanel comparison={incomparable} />);
    expect(screen.getByText(/cannot be compared: they were trained on different data/)).toBeInTheDocument();
    expect(screen.queryByText(/95% range/)).not.toBeInTheDocument();
    expect(screen.queryByText("Scores")).not.toBeInTheDocument();
  });

  it("says nothing differed for identical settings", () => {
    render(
      <RunComparisonPanel
        comparison={{ ...COMPARABLE, settings: [], summary: "Nothing in the settings differs, and the two scores agree within the noise." }}
      />
    );
    expect(screen.getByText("Nothing in the settings.")).toBeInTheDocument();
  });
});

function run(run_id: string, overrides: Partial<ModelRun> = {}): ModelRun {
  return {
    run_id,
    at: "2026-08-07T10:00:00+00:00",
    data_fingerprint: "aaa",
    target_column: "outcome",
    task: "classification",
    best_model: "logistic_regression",
    model_label: "Logistic regression",
    models_tried: ["logistic_regression"],
    primary_metric: "roc_auc",
    metrics: { roc_auc: 0.8 },
    baseline: {},
    verdict: "useful",
    headline: "",
    split: { strategy: "random", time_column: null, group_column: null, train_rows: 240, test_rows: 60 },
    feature_count: 3,
    excluded_columns: [],
    tuned: false,
    tuned_settings: {},
    stopped_because: null,
    ...overrides,
  };
}

describe("RunHistory — picking two runs to compare (#69)", () => {
  it("compares the two picked runs, newer first, and shows the panel", async () => {
    const fetchComparison = vi.fn().mockResolvedValue(COMPARABLE);
    render(
      <RunHistory runs={[run("new"), run("middle"), run("old")]} currentFingerprint="aaa" fetchComparison={fetchComparison} />
    );
    const boxes = screen.getAllByRole("checkbox", { name: /Compare the run from/ });
    expect(boxes).toHaveLength(3);
    fireEvent.click(boxes[2]);
    expect(screen.getByText("Pick one more run to compare it with.")).toBeInTheDocument();
    fireEvent.click(boxes[0]);
    await waitFor(() => expect(screen.getByLabelText("Comparison of two runs")).toBeInTheDocument());
    expect(fetchComparison).toHaveBeenCalledWith("new", "old");
  });

  it("closes the panel and clears the picks", async () => {
    render(
      <RunHistory runs={[run("new"), run("old")]} currentFingerprint="aaa" fetchComparison={vi.fn().mockResolvedValue(COMPARABLE)} />
    );
    const boxes = screen.getAllByRole("checkbox");
    fireEvent.click(boxes[0]);
    fireEvent.click(boxes[1]);
    await waitFor(() => screen.getByLabelText("Comparison of two runs"));
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByLabelText("Comparison of two runs")).not.toBeInTheDocument();
    expect(screen.getAllByRole("checkbox").every((box) => !(box as HTMLInputElement).checked)).toBe(true);
  });

  it("offers no picking without a way to compare", () => {
    render(<RunHistory runs={[run("new"), run("old")]} currentFingerprint="aaa" />);
    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
  });
});
