import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ReplaceModel } from "./ReplaceModel";
import * as client from "@/api/client";
import type { Comparison, ModelRun } from "@/api/types";

function run(run_id: string, label: string): ModelRun {
  return {
    run_id,
    at: "2026-10-16T10:00:00+00:00",
    data_fingerprint: "aaa",
    target_column: "churned",
    task: "classification",
    best_model: "logistic_regression",
    model_label: label,
    models_tried: [],
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
  };
}

const OPEN: Comparison = {
  comparison_id: "c1",
  created_at: "t",
  champion_run_id: "champ",
  challenger_run_id: "chall",
  criteria: { deciding_metric: "roc_auc", min_improvement: 0.01, constraints: [] },
  status: "open",
  evaluation: null,
};

const EVALUATED = (all_passed: boolean): Comparison => ({
  ...OPEN,
  status: "evaluated",
  evaluation: {
    rows_compared: 180,
    rows_excluded: { in_the_training_file: 20, no_answer: 0 },
    metrics: { champion: { roc_auc: 0.7 }, challenger: { roc_auc: 0.8 } },
    deciding: { kind: "deciding", metric: "roc_auc", passed: all_passed, evidence: "roc_auc 0.8000 against 0.7000" },
    constraints: [{ kind: "metric_at_least", metric: "recall", passed: true, evidence: "recall 0.7100" }],
    all_passed,
  },
});

function renderPanel(onModelChanged = vi.fn()) {
  render(
    <ReplaceModel
      sessionId="s"
      runs={[run("champ", "Logistic regression"), run("chall", "Random forest"), run("gone", "Old one")]}
      storedRunIds={["champ", "chall"]}
      currentRunId="champ"
      task="classification"
      onModelChanged={onModelChanged}
    />
  );
  return onModelChanged;
}

describe("ReplaceModel (#80)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(client, "fetchDecisions").mockResolvedValue([]);
  });

  it("offers only stored runs other than the model in use as challengers", () => {
    renderPanel();
    const options = Array.from((screen.getByLabelText("Challenger") as HTMLSelectElement).options).map((o) => o.value);
    expect(options).toEqual(["", "chall"]);
  });

  it("fixes the criteria before asking for the evaluation file", async () => {
    const start = vi.spyOn(client, "startComparison").mockResolvedValue(OPEN);
    renderPanel();
    expect(screen.queryByLabelText("Labelled evaluation file")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Challenger"), { target: { value: "chall" } });
    fireEvent.change(screen.getByLabelText("Minimum improvement"), { target: { value: "0.01" } });
    fireEvent.click(screen.getByRole("button", { name: "Add a constraint" }));
    fireEvent.change(screen.getByLabelText("Value"), { target: { value: "0.7" } });
    fireEvent.click(screen.getByRole("button", { name: "Fix these criteria and continue" }));
    await waitFor(() => expect(screen.getByLabelText("Labelled evaluation file")).toBeInTheDocument());
    expect(start).toHaveBeenCalledWith("s", "chall", {
      deciding_metric: "roc_auc",
      min_improvement: 0.01,
      constraints: [{ kind: "metric_at_least", metric: "roc_auc", value: 0.7 }],
    });
  });

  it("shows each criterion with its evidence and promotes a challenger that meets them", async () => {
    vi.spyOn(client, "startComparison").mockResolvedValue(OPEN);
    vi.spyOn(client, "evaluateComparison").mockResolvedValue(EVALUATED(true));
    const decide = vi.spyOn(client, "decideComparison").mockResolvedValue({
      decision_id: "d1", at: "t", decision: "promoted", reason: "Better.",
    });
    const changed = renderPanel();
    fireEvent.change(screen.getByLabelText("Challenger"), { target: { value: "chall" } });
    fireEvent.click(screen.getByRole("button", { name: "Fix these criteria and continue" }));
    const input = await screen.findByLabelText("Labelled evaluation file");
    fireEvent.change(input, { target: { files: [new File(["a,b"], "may.csv", { type: "text/csv" })] } });
    fireEvent.click(screen.getByRole("button", { name: "Score both on this file" }));
    await screen.findByText("The challenger meets every criterion.");
    expect(screen.getByText("roc_auc 0.8000 against 0.7000")).toBeInTheDocument();
    expect(screen.getByText(/20 left out: in the training file/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Reason"), { target: { value: "Better." } });
    fireEvent.click(screen.getByRole("button", { name: "Promote the challenger" }));
    await waitFor(() => expect(changed).toHaveBeenCalled());
    expect(decide).toHaveBeenCalledWith("s", "c1", "promote", "Better.", false);
  });

  it("will not promote past a failed criterion without the override", async () => {
    vi.spyOn(client, "startComparison").mockResolvedValue(OPEN);
    vi.spyOn(client, "evaluateComparison").mockResolvedValue(EVALUATED(false));
    renderPanel();
    fireEvent.change(screen.getByLabelText("Challenger"), { target: { value: "chall" } });
    fireEvent.click(screen.getByRole("button", { name: "Fix these criteria and continue" }));
    const input = await screen.findByLabelText("Labelled evaluation file");
    fireEvent.change(input, { target: { files: [new File(["a"], "may.csv")] } });
    fireEvent.click(screen.getByRole("button", { name: "Score both on this file" }));
    await screen.findByText("The challenger does not meet every criterion.");
    const promote = screen.getByRole("button", { name: "Promote the challenger" });
    expect(promote).toBeDisabled();
    fireEvent.click(screen.getByLabelText(/Promote anyway/));
    expect(promote).not.toBeDisabled();
  });

  it("lists the decisions and offers a rollback after a promotion", async () => {
    vi.spyOn(client, "fetchDecisions").mockResolvedValue([
      { decision_id: "d1", at: "2026-10-16T10:00", decision: "promoted", challenger_run_id: "champ",
        champion_run_id: "old", override: true, reason: "Recall first." },
    ]);
    const rollback = vi.spyOn(client, "rollBack").mockResolvedValue({
      decision_id: "d2", at: "2026-10-16T11:00", decision: "rolled_back", from_run_id: "champ", to_run_id: "old",
    });
    const changed = renderPanel();
    expect(await screen.findByText(/promoted \(override\) — “Recall first.”/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Roll back to the model/ }));
    await waitFor(() => expect(changed).toHaveBeenCalled());
    expect(rollback).toHaveBeenCalled();
  });
});
