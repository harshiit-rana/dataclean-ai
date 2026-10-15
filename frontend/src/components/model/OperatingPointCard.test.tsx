import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { OperatingPointCard } from "./OperatingPointCard";
import type { CutoffMetrics, OperatingPointResult } from "@/api/types";

const at = (over: Partial<CutoffMetrics>): CutoffMetrics => ({
  threshold: 0.5,
  precision: 0.7,
  recall: 0.6,
  f1: 0.65,
  accuracy: 0.8,
  flagged: 30,
  caught: 21,
  missed: 14,
  false_alarms: 9,
  rows: 160,
  ...over,
});

const POINT: OperatingPointResult = {
  objective: "cost",
  description: "a missed case costs 5x a false alarm",
  cost_of_missing: 5,
  cost_of_false_alarm: 1,
  target_rate: null,
  positive_label: "yes",
  threshold: 0.31,
  chosen_on: "out-of-fold predictions over the 640 training rows (5 folds); the test set was not consulted",
  folds: 5,
  reachable: true,
  note: null,
  expected: at({ threshold: 0.31, recall: 0.91, precision: 0.55, flagged: 200, rows: 640 }),
  observed: at({ threshold: 0.31, recall: 0.88, precision: 0.52, flagged: 52 }),
  at_default: at({}),
};

const RANGES = (
  metrics: Record<string, { low: number; high: number }>
): OperatingPointResult["observed_intervals"] => ({ level: 0.95, resamples: 1000, by: "rows", metrics });

describe("OperatingPointCard", () => {
  it("says nothing at the default -- there is nothing to explain", () => {
    const { container } = render(<OperatingPointCard point={{ ...POINT, objective: "default" }} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("leads with the line and the reason it sits there", () => {
    render(<OperatingPointCard point={POINT} />);
    expect(screen.getByText(/The line sits at 0.31, so that a missed case costs 5x a false alarm/)).toBeInTheDocument();
    expect(screen.getByText(/the test set was not consulted/)).toBeInTheDocument();
  });

  it("shows what was promised, what was delivered, and the default side by side", () => {
    render(<OperatingPointCard point={POINT} />);
    const recall = screen.getByText(/Share of real cases caught/).closest("tr")!;
    expect(recall).toHaveTextContent("91%");
    expect(recall).toHaveTextContent("88%");
    expect(recall).toHaveTextContent("60%");
  });

  it("says so when the floor could not be reached", () => {
    render(
      <OperatingPointCard
        point={{ ...POINT, reachable: false, note: "No cut-off catches 99% of cases on the training folds." }}
      />
    );
    expect(screen.getByText(/No cut-off catches 99%/)).toBeInTheDocument();
  });

  it("names the gap when the promise is outside the range the delivered figure could land in", () => {
    render(
      <OperatingPointCard
        point={{
          ...POINT,
          expected: at({ recall: 0.95 }),
          observed: at({ recall: 0.7 }),
          observed_intervals: RANGES({ recall: { low: 0.62, high: 0.78 } }),
        }}
      />
    );
    expect(screen.getByText(/promised 95% caught and the test set delivered 70%/)).toBeInTheDocument();
    expect(screen.getByText(/the promise is outside it/)).toBeInTheDocument();
  });

  it("does not call a gap an over-promise when the promise is inside that range", () => {
    // A ten-point gap on a handful of cases: the old fixed rule would have flagged this.
    render(
      <OperatingPointCard
        point={{
          ...POINT,
          expected: at({ recall: 0.8 }),
          observed: at({ recall: 0.7 }),
          observed_intervals: RANGES({ recall: { low: 0.45, high: 0.9 } }),
        }}
      />
    );
    expect(screen.queryByText(/the promise is outside it/)).not.toBeInTheDocument();
    expect(screen.getByText(/could land anywhere from 45% to 90%, so the gap is not evidence/)).toBeInTheDocument();
  });

  it("says nothing about a gap when there is no range to judge it by", () => {
    render(
      <OperatingPointCard
        point={{ ...POINT, expected: at({ recall: 0.95 }), observed: at({ recall: 0.5 }), observed_intervals: null }}
      />
    );
    expect(screen.queryByText(/promised 95% caught/)).not.toBeInTheDocument();
    expect(screen.queryByText(/not evidence/)).not.toBeInTheDocument();
  });

  it("prints the range beside the delivered recall and precision", () => {
    render(
      <OperatingPointCard
        point={{
          ...POINT,
          observed_intervals: RANGES({ recall: { low: 0.8, high: 0.94 }, precision: { low: 0.4, high: 0.63 } }),
        }}
      />
    );
    expect(screen.getByText(/\(80%–\s*94%\)/)).toBeInTheDocument();
    expect(screen.getByText(/\(40%–\s*63%\)/)).toBeInTheDocument();
  });

  it("prints no range for a result that has none", () => {
    render(<OperatingPointCard point={POINT} />);
    expect(screen.queryByText(/\d+%–/)).not.toBeInTheDocument();
  });

  it("says which column is a training-fold promise and which is a held-out result", () => {
    render(<OperatingPointCard point={POINT} />);
    const promised = screen.getByText(/Training folds promised/).closest("th")!;
    const delivered = screen.getByText(/Test set delivered/).closest("th")!;
    expect(promised).toHaveTextContent("Exploratory · training folds");
    expect(delivered).toHaveTextContent("Held-out evaluation");
  });

  it("carries the reuse status onto the held-out columns", () => {
    render(
      <OperatingPointCard
        point={POINT}
        evaluation={{ look_count: 3, limit: 3, status: "reused", message: "", looks_on_other_splits: 0 }}
      />
    );
    const delivered = screen.getByText(/Test set delivered/).closest("th")!;
    expect(delivered).toHaveTextContent("used 3 times");
  });

  it("says the line is on calibrated probabilities when they were calibrated", () => {
    render(<OperatingPointCard point={{ ...POINT, calibrated_with: "sigmoid" }} />);
    expect(
      screen.getByText(/The probabilities were calibrated \(sigmoid\), so this line is on calibrated probabilities/)
    ).toBeInTheDocument();
    expect(screen.getByText(/0\.31 means the model says it is\s+31% likely, and means it/)).toBeInTheDocument();
  });

  it("says nothing of the kind for a line on the model's own probabilities", () => {
    render(<OperatingPointCard point={{ ...POINT, calibrated_with: null }} />);
    expect(screen.queryByText(/calibrated probabilities/)).not.toBeInTheDocument();
    render(<OperatingPointCard point={POINT} />);
    expect(screen.queryByText(/calibrated probabilities/)).not.toBeInTheDocument();
  });

  it("says when the model was also trained with the stated costs", () => {
    render(<OperatingPointCard point={POINT} trainedWithCosts />);
    expect(screen.getByText(/The model was also trained with your costs/)).toBeInTheDocument();
  });

  it("says when only the line uses the costs", () => {
    render(<OperatingPointCard point={POINT} />);
    expect(screen.getByText(/trained without your costs; only this line uses them/)).toBeInTheDocument();
  });

  it("says nothing about training with costs when the line was not set by costs", () => {
    render(<OperatingPointCard point={{ ...POINT, objective: "min_recall" }} />);
    expect(screen.queryByText(/your costs/)).not.toBeInTheDocument();
  });
});
