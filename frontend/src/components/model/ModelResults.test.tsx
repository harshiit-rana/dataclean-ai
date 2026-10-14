import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ModelResults } from "./ModelResults";
import { HeldOutUse, ScoreLabel } from "./ScoreLabel";
import type { EvaluationStatus, ModelResult } from "@/api/types";

/** Two candidates where the *fold* winner and the *test-set* winner differ, which is exactly the
 * case the old table got wrong: it printed the test scores and marked the fold winner "chosen",
 * so a reader concluded the highest test score had won. */
function result(overrides: Partial<ModelResult> = {}): ModelResult {
  return {
    task: "classification",
    target_column: "churned",
    features: { numeric: ["tenure"], categorical: [], dropped: [] },
    split: {
      strategy: "random",
      time_column: null,
      group_column: null,
      train_rows: 800,
      test_rows: 200,
      train_groups: null,
      test_groups: null,
    },
    baseline: { strategy: "always answer no", accuracy: 0.7, roc_auc: 0.5 },
    primary_metric: "roc_auc",
    results: [
      {
        model: "logistic_regression",
        label: "Logistic regression",
        description: "",
        metrics: { accuracy: 0.8, roc_auc: 0.81 },
        feature_importance: [],
        column_importance: [],
        diagnostics: {},
        failed: null,
        cv_score: 0.84,
        cv_std: 0.012,
      },
      {
        model: "random_forest_classifier",
        label: "Random forest",
        description: "",
        metrics: { accuracy: 0.86, roc_auc: 0.9 },
        feature_importance: [],
        column_importance: [],
        diagnostics: {},
        failed: null,
        cv_score: 0.8,
        cv_std: 0.03,
      },
    ],
    best_model: "logistic_regression",
    selection: {
      chosen: "logistic_regression",
      method: "cross_validation",
      metric: "roc_auc",
      folds: 5,
      scores: {},
      chosen_on:
        "cross-validated roc_auc over 5 folds of the training rows; the test set was not consulted",
    },
    assessment: { verdict: "useful", headline: "This model separates the two outcomes." },
    deep_diagnostics: {},
    ...overrides,
  } as unknown as ModelResult;
}

const EVAL = (look_count: number, status: EvaluationStatus["status"], message = ""): EvaluationStatus => ({
  look_count,
  limit: 3,
  status,
  message,
  looks_on_other_splits: 0,
});

describe("ModelResults — the table that chose the model (#90)", () => {
  it("leads with the fold score and the spread that chose the model", () => {
    render(<ModelResults result={result()} />);
    const table = screen.getByRole("table", { name: "" , hidden: false });
    expect(screen.getByText(/Fold score \(ROC AUC\)/)).toBeInTheDocument();
    expect(screen.getByText("Fold spread")).toBeInTheDocument();
    expect(screen.getByText("0.840")).toBeInTheDocument();
    expect(screen.getByText("± 0.012")).toBeInTheDocument();
    expect(table).toBeDefined();
  });

  it("marks the chosen row with how it was chosen", () => {
    render(<ModelResults result={result()} />);
    expect(screen.getByText("chosen on training folds")).toBeInTheDocument();
  });

  it("says nothing about the other candidates' held-out scores until asked", () => {
    render(<ModelResults result={result()} />);
    // 0.9 is the forest's held-out AUC, and the point of the change is that it is not on show.
    expect(screen.queryByText("0.900")).not.toBeInTheDocument();
    expect(screen.queryByText(/not what the choice was based on/i)).not.toBeInTheDocument();
  });

  it("can reveal them, with the reason they are not the basis for anything", () => {
    render(<ModelResults result={result()} />);
    fireEvent.click(screen.getByRole("button", { name: /Show held-out scores for every model/ }));
    expect(screen.getByText(/These were not what the choice was based on/)).toBeInTheDocument();
    expect(screen.getByText(/best of several scores on the same rows/)).toBeInTheDocument();
    expect(screen.getByText("0.900")).toBeInTheDocument();
  });

  it("explains in the caption why the choice cannot have used the test answers", () => {
    render(<ModelResults result={result()} />);
    expect(screen.getByText(/never on the held-out rows/)).toBeInTheDocument();
  });

  it("shows only the chosen model's held-out scores in the verdict card, beside the baseline", () => {
    render(<ModelResults result={result()} />);
    expect(screen.getByText(/Scored once on the held-out rows, for the chosen model only/)).toBeInTheDocument();
    expect(screen.getByText(/No model at all \(always answer no\)/)).toBeInTheDocument();
    expect(screen.getAllByText("0.810").length).toBeGreaterThan(0);
  });

  it("says when only one model was trained that nothing was chosen", () => {
    const one = result();
    one.results = [one.results[0]];
    one.selection = { ...one.selection!, method: "only_candidate", metric: null, chosen_on: "the only model trained" };
    render(<ModelResults result={one} />);
    expect(screen.getByText("the only model trained")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Show held-out scores/ })).not.toBeInTheDocument();
  });

  it("states the reason when no candidate could be cross-validated", () => {
    const fallback = result();
    fallback.results = fallback.results.map((r) => ({ ...r, cv_score: null, cv_std: null }));
    fallback.selection = {
      ...fallback.selection!,
      method: "default_order",
      metric: null,
      chosen_on:
        "no candidate could be cross-validated on the training rows, so the simplest one was kept",
    };
    render(<ModelResults result={fallback} />);
    expect(screen.getByText(/so the simplest one was kept/)).toBeInTheDocument();
    expect(screen.getByText("kept by default order")).toBeInTheDocument();
  });

  it("marks a candidate that did not train", () => {
    const broken = result();
    broken.results[1] = { ...broken.results[1], failed: "boom", metrics: {} };
    render(<ModelResults result={broken} />);
    expect(screen.getByText("did not train")).toBeInTheDocument();
  });

  it("describes a result saved before models were chosen on folds as what it was", () => {
    /* Its choice WAS made on the test set, so its test-score table is the honest account. */
    const old = result();
    delete old.selection;
    old.results = old.results.map((r) => ({ ...r, cv_score: undefined, cv_std: undefined }));
    render(<ModelResults result={old} />);
    expect(screen.queryByText("Fold spread")).not.toBeInTheDocument();
    expect(screen.getByText("0.900")).toBeInTheDocument();
  });
});

describe("ModelResults — what each number is (#91)", () => {
  it("labels the fold table exploratory", () => {
    render(<ModelResults result={result()} />);
    expect(screen.getAllByText(/Exploratory · training folds/).length).toBeGreaterThan(0);
  });

  it("labels the verdict and its scores as the held-out evaluation", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.getAllByText(/Held-out evaluation/).length).toBeGreaterThan(0);
  });

  it("shows the reuse status on the label when the rows have been reused", () => {
    render(<ModelResults result={result({ evaluation: EVAL(3, "reused", "Used 3 times.") })} />);
    expect(screen.getAllByText(/Held-out evaluation · used 3 times/).length).toBeGreaterThan(0);
  });

  it("labels the cross-validation charts as exploratory, not as the held-out score", () => {
    const withCv = result({
      deep_diagnostics: {
        cross_validation: { metric: "roc_auc", folds: [0.8, 0.82], mean: 0.81, std: 0.01, spread: 0.02 },
      },
    });
    render(<ModelResults result={withCv} />);
    expect(screen.getByText(/scored on folds of the training rows only/)).toBeInTheDocument();
  });
});

describe("ModelResults — how often the held-out rows have been used (#92)", () => {
  it("states one look quietly, not as a warning", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.getByText(/Evaluated on these held-out rows once \(random split\)/)).toBeInTheDocument();
    expect(screen.queryByText(/Why does using the same rows again/)).not.toBeInTheDocument();
  });

  it("warns from the second look, with the message and the reason", () => {
    render(
      <ModelResults
        result={result({
          evaluation: EVAL(2, "reused", "No longer an independent estimate."),
        })}
      />
    );
    expect(screen.getByText(/Evaluated on these held-out rows 2 times \(random split\)/)).toBeInTheDocument();
    expect(screen.getByText("No longer an independent estimate.")).toBeInTheDocument();
    expect(screen.getByText(/Why does using the same rows again make the score optimistic/)).toBeInTheDocument();
  });

  it("is louder once exhausted", () => {
    const { container } = render(
      <ModelResults
        result={result({ evaluation: EVAL(5, "exhausted", "This score is optimistic.") })}
      />
    );
    expect(screen.getByText(/Evaluated on these held-out rows 5 times/)).toBeInTheDocument();
    expect(container.querySelector(".border-destructive\\/40")).not.toBeNull();
  });

  it("names the split rule", () => {
    const timed = result({ evaluation: EVAL(1, "independent") });
    timed.split = { ...timed.split, strategy: "time", time_column: "signup_date" };
    render(<ModelResults result={timed} />);
    expect(
      screen.getByText(/Evaluated on these held-out rows once \(split by signup_date\)/)
    ).toBeInTheDocument();
  });

  it("shows nothing for a result from before the ledger, rather than a count of one", () => {
    render(<ModelResults result={result()} />);
    expect(screen.queryByText(/Evaluated on these held-out rows/)).not.toBeInTheDocument();
  });

  it("says the earlier looks under another rule overlap, even on a first look of this one", () => {
    const evaluation = { ...EVAL(1, "independent", "Also evaluated 2 models under a different split rule."), looks_on_other_splits: 2 };
    render(<ModelResults result={result({ evaluation })} />);
    expect(screen.getByText(/different split rule/)).toBeInTheDocument();
  });
});

describe("ScoreLabel", () => {
  it("has a one-line explanation for each kind, reachable without a mouse", () => {
    const { container } = render(
      <>
        <ScoreLabel kind="exploratory" />
        <ScoreLabel kind="held_out" />
        <ScoreLabel kind="new_data" />
      </>
    );
    const labels = container.querySelectorAll("span[title]");
    expect(labels.length).toBe(3);
    for (const label of labels) {
      expect(within(label as HTMLElement).getByText(/\./, { selector: ".sr-only" })).toBeInTheDocument();
    }
  });

  it("does not claim a count it never recorded", () => {
    render(<ScoreLabel kind="held_out" evaluation={undefined} />);
    expect(screen.getByText("Held-out evaluation")).toBeInTheDocument();
  });

  it("shows no count on the first look", () => {
    render(<ScoreLabel kind="held_out" evaluation={{ look_count: 1, status: "independent" }} />);
    expect(screen.queryByText(/used \d+ times/)).not.toBeInTheDocument();
  });
});

describe("HeldOutUse", () => {
  const split = { strategy: "group" as const, time_column: null, group_column: "customer_id" };

  it("renders nothing without an evaluation", () => {
    const { container } = render(<HeldOutUse evaluation={undefined} split={split} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("names the grouping in the rule", () => {
    render(<HeldOutUse evaluation={EVAL(1, "independent")} split={split} />);
    expect(screen.getByText(/kept whole by customer_id/)).toBeInTheDocument();
  });
});

describe("ModelResults — whole-file statistics (#12)", () => {
  it("says so when a fix worked a value out over every row", () => {
    const evaluation = {
      ...EVAL(1, "independent"),
      whole_file_statistics: [{ operation: "fill_median", column: "spend" }],
    };
    render(<ModelResults result={result({ evaluation })} />);
    expect(screen.getByText(/fill median on spend/)).toBeInTheDocument();
    expect(screen.getByText(/worked out over every row/)).toBeInTheDocument();
  });

  it("says nothing when there were none", () => {
    render(<ModelResults result={result({ evaluation: { ...EVAL(1, "independent"), whole_file_statistics: [] } })} />);
    expect(screen.queryByText(/worked out over every row/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — checked on new data (#9)", () => {
  const withNewData = (overrides = {}) =>
    result({
      evaluation: {
        ...EVAL(1, "independent"),
        new_data: {
          count: 2,
          latest: {
            at: "2026-10-14T10:00:00+00:00",
            apply_id: "abc",
            rows: 200,
            metric: "roc_auc",
            value: 0.783,
            reported: 0.81,
            overlap_excluded: 0,
            ...overrides,
          },
        },
      },
    });

  it("shows the latest check beside the held-out score, labelled as new data", () => {
    render(<ModelResults result={withNewData()} />);
    expect(screen.getByText(/Checked on new data 2 times/)).toBeInTheDocument();
    expect(screen.getByText(/Latest: 200 rows that did not exist when this model was built/)).toBeInTheDocument();
    expect(screen.getByText(/ROC AUC 0.783 \(0.810 when it was built\)/)).toBeInTheDocument();
    expect(screen.getAllByText("New data").length).toBeGreaterThan(0);
  });

  it("says once when there has been one check", () => {
    render(
      <ModelResults
        result={result({
          evaluation: {
            ...EVAL(1, "independent"),
            new_data: {
              count: 1,
              latest: { at: null, apply_id: null, rows: 50, metric: "roc_auc", value: 0.7, reported: null, overlap_excluded: 0 },
            },
          },
        })}
      />
    );
    expect(screen.getByText(/Checked on new data once/)).toBeInTheDocument();
  });

  it("says how many rows the model was built from were left out", () => {
    render(<ModelResults result={withNewData({ overlap_excluded: 100 })} />);
    expect(screen.getByText(/100 rows the model was built from were left out/)).toBeInTheDocument();
  });

  it("shows nothing before any file with answers has been applied", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByText(/Checked on new data/)).not.toBeInTheDocument();
  });

  it("does not replace the held-out evaluation", () => {
    render(<ModelResults result={withNewData()} />);
    expect(screen.getByText(/Evaluated on these held-out rows once/)).toBeInTheDocument();
  });
});
