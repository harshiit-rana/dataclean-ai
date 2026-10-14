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
