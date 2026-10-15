import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { LearningCurveChart, PermutationImportanceChart } from "./ModelCharts";
import type { LearningCurvePoint, PermutationImportance } from "@/api/types";

const curve = (metric?: string): LearningCurvePoint[] => [
  { training_rows: 100, training_score: 0.9, validation_score: 0.7, metric },
  { training_rows: 400, training_score: 0.85, validation_score: 0.78, metric },
];

const importance = (metric?: string): PermutationImportance[] => [
  { column: "tenure", importance: 0.08, std: 0.01, metric },
  { column: "noise", importance: 0.0, std: 0.01, metric },
];

describe("LearningCurveChart — one metric across the page", () => {
  it("names ROC AUC for a two-class target, matching the spread beside it", () => {
    render(<LearningCurveChart points={curve("roc_auc")} />);
    expect(screen.getByText(/How the ROC AUC changes as the model is trained/)).toBeInTheDocument();
  });

  it("names accuracy for more than two classes", () => {
    render(<LearningCurveChart points={curve("accuracy")} />);
    expect(screen.getByText(/How the accuracy changes/)).toBeInTheDocument();
  });

  it("names R² for a numeric target", () => {
    render(<LearningCurveChart points={curve("r2")} />);
    expect(screen.getByText(/How the R² changes/)).toBeInTheDocument();
  });

  it("labels the table columns with the metric", () => {
    render(<LearningCurveChart points={curve("roc_auc")} />);
    fireEvent.click(screen.getByRole("button", { name: /Show table/ }));
    expect(screen.getByText("ROC AUC on rows it saw")).toBeInTheDocument();
    expect(screen.getByText("ROC AUC on unseen rows")).toBeInTheDocument();
  });

  it("falls back to 'score' for a result saved before the metric was recorded", () => {
    render(<LearningCurveChart points={curve(undefined)} />);
    expect(screen.getByText(/How the score changes/)).toBeInTheDocument();
  });
});

describe("PermutationImportanceChart — one metric across the page", () => {
  it("says which score drops when a column is scrambled", () => {
    render(<PermutationImportanceChart rows={importance("roc_auc")} />);
    expect(screen.getByText(/how much the ROC AUC drops/)).toBeInTheDocument();
  });

  it("labels the lost-score column with the metric", () => {
    render(<PermutationImportanceChart rows={importance("roc_auc")} />);
    fireEvent.click(screen.getByRole("button", { name: /Show table/ }));
    expect(screen.getByText("ROC AUC lost")).toBeInTheDocument();
  });

  it("falls back to 'score' for an older result", () => {
    render(<PermutationImportanceChart rows={importance(undefined)} />);
    expect(screen.getByText(/how much the score drops/)).toBeInTheDocument();
  });
});

describe("LearningCurveChart — read against the folds' disagreement (#42)", () => {
  const withSpread = (std: number): LearningCurvePoint[] => [
    { training_rows: 100, training_score: 0.84, validation_score: 0.8, metric: "roc_auc", training_std: std, validation_std: std },
    { training_rows: 400, training_score: 0.84, validation_score: 0.81, metric: "roc_auc", training_std: std, validation_std: std },
  ];

  it("shows the server's reading, which weighs each difference against the folds' spread", () => {
    render(
      <LearningCurveChart
        points={withSpread(0.04)}
        reading={{ kind: "inconclusive", text: "The score barely moved as data was added, but the folds disagree by about 0.057 -- enough to hide a real rise -- so whether more rows would help is inconclusive." }}
      />
    );
    expect(screen.getByText(/enough to hide a real rise/)).toBeInTheDocument();
    expect(screen.queryByText(/flattened out/)).not.toBeInTheDocument();
  });

  it("draws the spread across folds as a band when the points carry it", () => {
    render(<LearningCurveChart points={withSpread(0.02)} reading={{ kind: "flat", text: "The score flattened out." }} />);
    expect(screen.getByText("Spread across folds")).toBeInTheDocument();
  });

  it("falls back to the fixed reading, and no band, for a result from before the spread was kept", () => {
    render(<LearningCurveChart points={curve("roc_auc")} />);
    expect(screen.queryByText("Spread across folds")).not.toBeInTheDocument();
    expect(screen.getByText(/flattened out|still climbing|memorizing/)).toBeInTheDocument();
  });
});
