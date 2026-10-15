import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RunHistory, describeDiff, describeProvenance, whyNotComparable } from "./RunHistory";
import type { ModelRun, VersionDiff } from "@/api/types";

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
    expect(screen.getByText("not comparable: data changed")).toBeInTheDocument();
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

  it("says which search found a tuned run's settings, when it was recorded (#50)", () => {
    renderHistory([run({ tuned: true, search_strategies: ["halving", "random"] }), run()]);
    expect(screen.getByText(/tuned \(halving \+ random\)/)).toBeInTheDocument();
  });

  it("marks a tuned run", () => {
    renderHistory([run({ tuned: true }), run()]);
    expect(screen.getByText(/tuned/)).toBeInTheDocument();
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
    expect(screen.getByText("not comparable: data changed")).toBeInTheDocument();
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

describe("RunHistory — how each run was made (#63)", () => {
  const recorded = run({
    config_hash: "0123456789abcdef",
    runtime_seconds: 74.2,
    dataset_version: { version: 3, fingerprint: "f", rows: 1200, columns: 9, event: "clean" },
    selection: { method: "cross_validation", metric: "roc_auc", rule: "one_standard_error", scores: { a: 0.8 } },
    environment: { python: "3.12.4", scikit_learn: "1.7.0", pandas: "2.2.2", numpy: "2.0.1", joblib: "1.4.2" },
    seeds: { split: 42, model: 42, search: 42, confirmation: 1729 },
  });

  it("lists the configuration, runtime, data version, software and seeds", () => {
    const lines = describeProvenance(recorded)!;
    expect(lines).toContain("configuration 0123456789abcdef");
    expect(lines).toContain("took 1 min 14 s");
    expect(lines).toContain("data version 3 (1,200 rows, 9 columns)");
    expect(lines).toContain("chosen by cross validation");
    expect(lines).toContain("Python 3.12.4, scikit-learn 1.7.0, pandas 2.2.2, numpy 2.0.1");
    expect(lines).toContain("seeds split 42, model 42, search 42, confirmation 1729");
  });

  it("shows the details on the row", () => {
    renderHistory([recorded, run()]);
    expect(screen.getByText("how it was made")).toBeInTheDocument();
    expect(screen.getByText("configuration 0123456789abcdef")).toBeInTheDocument();
  });

  it("says not recorded for a run from before it was kept, rather than blanks", () => {
    expect(describeProvenance(run())).toBeNull();
    renderHistory([run(), run()]);
    expect(screen.getAllByText("how it was made: not recorded")).toHaveLength(2);
  });
});

describe("RunHistory — refused and failed attempts are part of the record (#64)", () => {
  const refused = run({
    status: "refused",
    code: "data_blocked",
    reason: "This data cannot support a model that would mean anything, so nothing was trained.",
    metrics: {},
    best_model: null as unknown as string,
    model_label: null as unknown as string,
    verdict: null as unknown as string,
  });

  it("shows the attempt in sequence with its reason", () => {
    renderHistory([refused, run()]);
    expect(screen.getByText("Refused")).toBeInTheDocument();
    expect(screen.getByText(/nothing was trained/)).toBeInTheDocument();
    expect(screen.getByText("no model")).toBeInTheDocument();
  });

  it("names a failure as a failure", () => {
    renderHistory([run({ status: "failed", reason: "No usable columns.", metrics: {} }), run()]);
    expect(screen.getByText("Failed")).toBeInTheDocument();
    expect(screen.getByText("No usable columns.")).toBeInTheDocument();
  });

  it("compares a run with the last run that had a score, skipping the attempt between them", () => {
    renderHistory([run({ metrics: { roc_auc: 0.85 } }), refused, run({ metrics: { roc_auc: 0.8 } })]);
    expect(screen.getByText("+0.0500")).toBeInTheDocument();
  });

  it("keeps the current label on the model in use, not on a later failed attempt", () => {
    renderHistory([refused, run({ model_label: "Random forest" })]);
    expect(screen.getByText(/· current/)).toBeInTheDocument();
    expect(screen.getByText(/latest attempt/)).toBeInTheDocument();
  });
});

describe("RunHistory — bringing back an earlier model (#67)", () => {
  const older = run({ run_id: "older", model_label: "Random forest" });
  const newer = run({ run_id: "newer", model_label: "Logistic regression" });

  it("offers a stored run on the current data, and calls back with its id", () => {
    const onRestore = vi.fn();
    render(
      <RunHistory runs={[newer, older]} currentFingerprint="aaa" currentRunId="newer"
        storedRunIds={["newer", "older"]} onRestore={onRestore} />
    );
    const buttons = screen.getAllByRole("button", { name: "Make this the current model" });
    expect(buttons).toHaveLength(1);
    fireEvent.click(buttons[0]);
    expect(onRestore).toHaveBeenCalledWith("older");
  });

  it("offers nothing for a run trained before the data changed, or one no longer stored", () => {
    const stale = run({ run_id: "stale", data_fingerprint: "old" });
    render(
      <RunHistory runs={[newer, stale, older]} currentFingerprint="aaa" currentRunId="newer"
        storedRunIds={["newer", "stale"]} onRestore={() => {}} />
    );
    expect(screen.queryByRole("button", { name: "Make this the current model" })).not.toBeInTheDocument();
  });

  it("puts the current label where the server says the model in use is", () => {
    render(
      <RunHistory runs={[newer, older]} currentFingerprint="aaa" currentRunId="older"
        storedRunIds={["newer", "older"]} onRestore={() => {}} />
    );
    const current = screen.getByText(/· current/).closest("tr")!;
    expect(current).toHaveTextContent("Random forest");
  });

  it("shows a restore as an event with no score, skipped by the changes", () => {
    const restoredEvent = run({
      run_id: "event", status: "restored", restored_run_id: "older",
      reason: "Brought back the Random forest trained at 10:00.", metrics: {},
    });
    render(
      <RunHistory runs={[restoredEvent, run({ metrics: { roc_auc: 0.9 } }), run({ metrics: { roc_auc: 0.8 } })]}
        currentFingerprint="aaa" />
    );
    expect(screen.getByText("Restored")).toBeInTheDocument();
    expect(screen.getByText("no new score")).toBeInTheDocument();
    expect(screen.getByText("+0.1000")).toBeInTheDocument();
  });

  it("no longer says an earlier run cannot be brought back", () => {
    renderHistory([newer, older]);
    expect(screen.queryByText(/can't be brought back/)).not.toBeInTheDocument();
    expect(screen.getByText(/can be made the\s+current model again/)).toBeInTheDocument();
  });
});

describe("RunHistory — a search the server never finished (#65)", () => {
  it("shows an interrupted search with what had finished, and no score", () => {
    const interrupted = run({
      status: "interrupted",
      code: "interrupted",
      reason: "The search stopped before it finished: the server restarted. 2 rounds had finished: Baseline, Settings.",
      metrics: {},
    });
    renderHistory([interrupted, run({ metrics: { roc_auc: 0.85 } }), run({ metrics: { roc_auc: 0.8 } })]);
    expect(screen.getByText("Interrupted")).toBeInTheDocument();
    expect(screen.getByText(/2 rounds had finished/)).toBeInTheDocument();
    expect(screen.getByText("+0.0500")).toBeInTheDocument();
  });
});

describe("RunHistory — older runs are said, never cut silently (#70)", () => {
  it("states how many older entries exist and offers them", () => {
    const onLoadOlder = vi.fn();
    render(<RunHistory runs={[run(), run()]} currentFingerprint="aaa" olderCount={7} onLoadOlder={onLoadOlder} />);
    expect(screen.getByText(/7 older entries not shown/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show older runs" }));
    expect(onLoadOlder).toHaveBeenCalled();
  });

  it("does not call the last run on the page the first run when older ones exist", () => {
    render(<RunHistory runs={[run(), run()]} currentFingerprint="aaa" olderCount={1} />);
    expect(screen.getByText("earlier runs not shown")).toBeInTheDocument();
    expect(screen.queryByText("first run")).not.toBeInTheDocument();
  });

  it("says nothing when the whole history is shown", () => {
    renderHistory([run(), run()]);
    expect(screen.queryByText(/older entr/)).not.toBeInTheDocument();
  });
});

describe("RunHistory — what changed in the data since a run (#71)", () => {
  const DIFF: VersionDiff = {
    from: { version: 2, rows: 300, columns: 4, event: "cleaning_run", at: "t" },
    to: { version: 4, rows: 290, columns: 3, event: "undo:a2", at: "t" },
    rebuilt: { from: "replayed", to: "replayed" },
    rows: { from: 300, to: 290, change: -10, removed: 10, added: 0 },
    columns: { added: [], removed: ["note"] },
    cells: { spend: 12 },
    actions: {
      applied: [{ action_id: "a3", operation: "drop_column", column_name: "note" }],
      undone: [{ action_id: "a2", operation: "fill_median", column_name: "spend" }],
    },
    notes: [],
  };

  it("describes the rows, columns, values and fixes", () => {
    expect(describeDiff(DIFF)).toEqual([
      "Rows: 300 → 290 (10 removed)",
      "Columns removed: note",
      "Values changed: spend (12)",
      "Fixes applied since: drop column on note",
      "Fixes undone since: fill median on spend",
    ]);
  });

  it("offers what changed on a run from an earlier version and shows it when asked", async () => {
    const fetchDiff = vi.fn().mockResolvedValue(DIFF);
    const old = run({ data_fingerprint: "old", dataset_version: { version: 2, fingerprint: "f", rows: 300, columns: 4, event: "clean" } });
    render(
      <RunHistory runs={[run(), old]} currentFingerprint="aaa" currentDatasetVersion={4} fetchDiff={fetchDiff} />
    );
    fireEvent.click(screen.getByRole("button", { name: /What changed since \(version 2 → 4\)/ }));
    await waitFor(() => expect(screen.getByText("Columns removed: note")).toBeInTheDocument());
    expect(fetchDiff).toHaveBeenCalledWith(2, 4);
  });

  it("offers nothing for a run on the current data, or one without a recorded version", () => {
    const noVersion = run({ data_fingerprint: "old" });
    render(<RunHistory runs={[run(), noVersion]} currentFingerprint="aaa" currentDatasetVersion={4} fetchDiff={vi.fn()} />);
    expect(screen.queryByRole("button", { name: /What changed/ })).not.toBeInTheDocument();
  });
});

describe("RunHistory — why two runs are not comparable (#97)", () => {
  it("names a data change", () => {
    renderHistory([run({ data_fingerprint: "bbb" }), run({ data_fingerprint: "aaa" })], "bbb");
    expect(screen.getByText(/not comparable: data changed/)).toBeInTheDocument();
  });

  it("names a metric change with both metrics", () => {
    renderHistory([run({ primary_metric: "accuracy", metrics: { accuracy: 0.8 } }), run()]);
    expect(screen.getByText(/not comparable: different metric \(roc_auc → accuracy\)/)).toBeInTheDocument();
  });

  it("names both when both changed", () => {
    expect(
      whyNotComparable(run({ data_fingerprint: "b", primary_metric: "accuracy" }), run())
    ).toEqual(["data changed", "different metric (roc_auc → accuracy)"]);
  });

  it("links to what changed when both runs' data versions are known", () => {
    const fetchDiff = vi.fn();
    const v = (version: number) => ({ version, fingerprint: "f", rows: 300, columns: 4, event: "clean" });
    render(
      <RunHistory
        runs={[run({ data_fingerprint: "bbb", dataset_version: v(3) }), run({ data_fingerprint: "aaa", dataset_version: v(2) })]}
        currentFingerprint="bbb"
        fetchDiff={fetchDiff}
      />
    );
    expect(screen.getByRole("button", { name: /What changed since \(version 2 → 3\)/ })).toBeInTheDocument();
  });

  it("offers no link without the versions", () => {
    render(
      <RunHistory runs={[run({ data_fingerprint: "bbb" }), run({ data_fingerprint: "aaa" })]} currentFingerprint="bbb" fetchDiff={vi.fn()} />
    );
    expect(screen.queryByRole("button", { name: /What changed/ })).not.toBeInTheDocument();
  });
});
