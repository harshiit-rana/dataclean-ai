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

  it("names the gap when the training folds over-promised", () => {
    render(
      <OperatingPointCard
        point={{ ...POINT, expected: at({ recall: 0.95 }), observed: at({ recall: 0.7 }) }}
      />
    );
    expect(screen.getByText(/promised 95% caught and the test set delivered 70%/)).toBeInTheDocument();
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
});
