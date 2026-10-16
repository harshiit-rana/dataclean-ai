import { fireEvent, render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { axeViolations } from "@/test/axe";
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

  it("gives a failed candidate's reason in the held-out table instead of dashes (#64)", () => {
    const broken = result();
    broken.results[1] = {
      ...broken.results[1],
      failed: "The classes are not in class_weight",
      metrics: {},
    };
    render(<ModelResults result={broken} />);
    fireEvent.click(screen.getByRole("button", { name: /Show held-out scores for every model/ }));
    expect(screen.getByText("Did not train: The classes are not in class_weight")).toBeInTheDocument();
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

describe("ModelResults — how far the score could move (#28)", () => {
  const withIntervals = (by: "rows" | "groups" = "rows") => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.results[0] = {
      ...r.results[0],
      metrics_intervals: {
        level: 0.95,
        resamples: 1000,
        by,
        metrics: { roc_auc: { low: 0.71, high: 0.84 }, accuracy: { low: 0.74, high: 0.86 } },
      },
    };
    return r;
  };

  it("prints the interval beside the score", () => {
    render(<ModelResults result={withIntervals()} />);
    expect(screen.getByText("(0.710–0.840)")).toBeInTheDocument();
    expect(screen.getByText("(0.740–0.860)")).toBeInTheDocument();
  });

  it("says what the brackets are, and that they are not a guarantee", () => {
    render(<ModelResults result={withIntervals()} />);
    expect(screen.getByText(/the middle 95% of 1,000 resamples of the held-out rows/)).toBeInTheDocument();
    expect(screen.getByText(/not a guarantee/)).toBeInTheDocument();
  });

  it("says whole groups were resampled for a grouped split", () => {
    render(<ModelResults result={withIntervals("groups")} />);
    expect(screen.getByText(/resamples of the held-out whole groups/)).toBeInTheDocument();
  });

  it("shows only the bare score when no interval was computed", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByText(/resamples of the held-out/)).not.toBeInTheDocument();
    expect(screen.queryByText(/\(0\.\d{3}–/)).not.toBeInTheDocument();
  });

  it("leaves a metric without an interval alone while printing the others", () => {
    render(<ModelResults result={withIntervals()} />);
    // precision has a value but no interval in the fixture
    expect(screen.queryAllByText(/\(0\.\d{3}–0\.\d{3}\)/).length).toBe(2);
  });
});

describe("ModelResults — the one-standard-error rule (#33)", () => {
  const kept = () => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.selection = {
      ...r.selection!,
      rule: "one_standard_error",
      highest_scorer: "random_forest_classifier",
      standard_error: 0.013,
      rule_note:
        "Random forest scored highest (0.842), but Logistic regression (0.840) is within one standard error (0.013) of it, so the simpler model was kept.",
    };
    return r;
  };

  it("says why the simpler model was kept", () => {
    render(<ModelResults result={kept()} />);
    expect(screen.getByText(/is within one standard error \(0.013\) of it, so the simpler model was kept/)).toBeInTheDocument();
  });

  it("marks the chosen row as the simplest within the noise, not just 'chosen'", () => {
    render(<ModelResults result={kept()} />);
    expect(screen.getByText("simplest within one standard error of the best")).toBeInTheDocument();
    expect(screen.queryByText("chosen on training folds")).not.toBeInTheDocument();
  });

  it("says nothing extra when the rule changed nothing", () => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.selection = { ...r.selection!, rule: "highest_score", rule_note: null };
    render(<ModelResults result={r} />);
    expect(screen.queryByText(/within one standard error/)).not.toBeInTheDocument();
    expect(screen.getByText("chosen on training folds")).toBeInTheDocument();
  });
});

describe("ModelResults — when the folds cannot tell the models apart (#34)", () => {
  const NOTE =
    "Random forest and logistic regression are within the noise of each other on these folds (the gap averaged +0.016 per fold, 95% range -0.015 to +0.047); Logistic regression was kept because it is simpler.";

  const withNote = (note: string | null | undefined) => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.selection = { ...r.selection!, distinguishable: note ? false : true, distinguishable_note: note };
    return r;
  };

  it("says so under the comparison table", () => {
    render(<ModelResults result={withNote(NOTE)} />);
    const sentence = screen.getByText(/are within the noise of each other on these folds/);
    const table = screen.getByRole("columnheader", { name: /Fold score/ }).closest("table")!;
    expect(sentence).toBeInTheDocument();
    // under the table, not above it
    expect(table.compareDocumentPosition(sentence) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("gives the range, so the size of the doubt is visible", () => {
    render(<ModelResults result={withNote(NOTE)} />);
    expect(screen.getByText(/95% range -0\.015 to \+0\.047/)).toBeInTheDocument();
  });

  it("says nothing when the models can be told apart", () => {
    render(<ModelResults result={withNote(null)} />);
    expect(screen.queryByText(/within the noise of each other/)).not.toBeInTheDocument();
  });

  it("says nothing for a result saved before the comparison existed", () => {
    render(<ModelResults result={withNote(undefined)} />);
    expect(screen.queryByText(/within the noise of each other/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — how honest the probabilities are (#24)", () => {
  const calibrated = (): ModelResult => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.baseline = { ...r.baseline, brier_score: 0.2468 };
    r.results[0].metrics = {
      ...r.results[0].metrics,
      brier_score: 0.1799,
      expected_calibration_error: 0.0614,
      brier_baseline: 0.2468,
    };
    r.results[0].metrics_intervals = {
      level: 0.95,
      resamples: 1000,
      by: "rows",
      metrics: {
        brier_score: { low: 0.142, high: 0.2224 },
        expected_calibration_error: { low: 0.0301, high: 0.1102 },
      },
    };
    r.results[0].diagnostics = {
      calibration: [
        { predicted: 0.2, actual: 0.15, count: 80, bin_start: 0.1, bin_end: 0.2 },
        { predicted: 0.8, actual: 0.85, count: 90, bin_start: 0.7, bin_end: 0.8 },
      ],
    };
    return r;
  };

  it("adds the two columns to the held-out scores for a two-class result", () => {
    render(<ModelResults result={calibrated()} />);
    expect(screen.getByRole("columnheader", { name: "Brier score" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Calibration error" })).toBeInTheDocument();
  });

  it("shows each with its range, so a noisy number is not read as exact", () => {
    render(<ModelResults result={calibrated()} />);
    expect(screen.getByText("(0.142–0.222)")).toBeInTheDocument();
    expect(screen.getByText("(0.030–0.110)")).toBeInTheDocument();
  });

  it("shows the Brier score of predicting the overall rate in the no-model row", () => {
    render(<ModelResults result={calibrated()} />);
    expect(screen.getByText("0.247")).toBeInTheDocument();
  });

  it("says that lower is better for these two, which is the opposite of the others", () => {
    render(<ModelResults result={calibrated()} />);
    expect(screen.getByText(/For Brier score and calibration error, lower is better/)).toBeInTheDocument();
  });

  it("states the numbers in the calibration chart's description", () => {
    render(<ModelResults result={calibrated()} />);
    expect(screen.getByText(/off by 6\.1 percentage points \(calibration error\)/)).toBeInTheDocument();
    expect(screen.getByText(/Brier score 0\.180 against 0\.247 for always predicting the overall rate/)).toBeInTheDocument();
  });

  it("does not colour either number as passing or failing", () => {
    const { container } = render(<ModelResults result={calibrated()} />);
    const cell = screen.getByRole("columnheader", { name: "Calibration error" });
    const column = Array.from(cell.parentElement!.children).indexOf(cell);
    const value = cell.closest("table")!.querySelectorAll("tbody tr")[0].children[column];
    expect(value.className).not.toMatch(/emerald|destructive|amber|red-/);
    expect(container.querySelector("[data-calibration-verdict]")).toBeNull();
  });

  it("is unchanged for a result with no probabilities to judge", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByRole("columnheader", { name: "Brier score" })).not.toBeInTheDocument();
    expect(screen.queryByText(/lower is better/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — the no-model rows have a value under every column (#38)", () => {
  const full = (): ModelResult => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.results[0].metrics = {
      accuracy: 0.8,
      precision: 0.62,
      recall: 0.7,
      f1: 0.66,
      roc_auc: 0.81,
      brier_score: 0.18,
      expected_calibration_error: 0.06,
    };
    r.baseline = {
      strategy: "always answer 'no'",
      accuracy: 0.7,
      precision: 0,
      recall: 0,
      f1: 0,
      roc_auc: 0.5,
      brier_score: 0.2,
      expected_calibration_error: 0.01,
      others: [
        {
          strategy: "guess at random, in proportion to how common each answer is",
          accuracy: 0.58,
          precision: 0.3,
          recall: 0.3,
          f1: 0.3,
          roc_auc: 0.5,
          brier_score: 0.42,
          expected_calibration_error: 0.42,
        },
      ],
    };
    return r;
  };

  const baselineRows = () =>
    screen.getAllByText(/No model at all/).map((cell) => cell.closest("tr") as HTMLElement);

  it("shows both naive strategies, each named", () => {
    render(<ModelResults result={full()} />);
    expect(screen.getByText(/No model at all \(always answer 'no'\)/)).toBeInTheDocument();
    expect(screen.getByText(/No model at all \(guess at random, in proportion/)).toBeInTheDocument();
  });

  it("leaves no blank in either row", () => {
    render(<ModelResults result={full()} />);
    const rows = baselineRows();
    expect(rows.length).toBe(2);
    for (const row of rows) {
      expect(within(row).queryByText("—")).not.toBeInTheDocument();
    }
  });

  it("gives the lopsided-target F1 floor that the common-answer guesser cannot", () => {
    render(<ModelResults result={full()} />);
    const [common, random] = baselineRows();
    const f1Column = Array.from(
      screen.getByRole("columnheader", { name: "F1" }).parentElement!.children
    ).findIndex((th) => th.textContent === "F1");
    expect(common.children[f1Column].textContent).toBe("0.000");
    expect(random.children[f1Column].textContent).toBe("0.300");
  });

  it("still shows one row for a result from before there were alternatives", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(baselineRows().length).toBe(1);
  });
});

describe("ModelResults — a median baseline and a relative error for regression (#39)", () => {
  const regression = (): ModelResult => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.task = "regression";
    r.primary_metric = "r2";
    r.selection = { ...r.selection!, metric: "r2" };
    r.results[0].metrics = { r2: 0.39, mae: 1995.29, rmse: 3976.29, mae_vs_baseline: 0.794 };
    r.results[1].metrics = { r2: 0.31, mae: 2100.1, rmse: 4100.2, mae_vs_baseline: 0.836 };
    r.baseline = {
      strategy: "always predict the average (4,077.38)",
      r2: -0.0009,
      mae: 2811.51,
      rmse: 5102.47,
      mae_vs_baseline: 1.119,
      others: [
        {
          strategy: "always predict the middle value, the median (2,788.46)",
          r2: -0.0496,
          mae: 2513.03,
          rmse: 5225.25,
          mae_vs_baseline: 1,
        },
      ],
    };
    return r;
  };

  it("adds the relative error as a column, in plain words", () => {
    render(<ModelResults result={regression()} />);
    expect(screen.getByRole("columnheader", { name: "Error vs. median guess" })).toBeInTheDocument();
    expect(screen.getByText("0.794")).toBeInTheDocument();
  });

  it("shows the average and the median as separate no-model rows", () => {
    render(<ModelResults result={regression()} />);
    expect(screen.getByText(/No model at all \(always predict the average/)).toBeInTheDocument();
    expect(screen.getByText(/No model at all \(always predict the middle value, the median/)).toBeInTheDocument();
  });

  it("leaves no blank in either, and prints the median row's relative error as exactly 1", () => {
    render(<ModelResults result={regression()} />);
    const rows = screen.getAllByText(/No model at all/).map((cell) => cell.closest("tr") as HTMLElement);
    expect(rows.length).toBe(2);
    for (const row of rows) expect(within(row).queryByText("—")).not.toBeInTheDocument();
    expect(within(rows[1]).getByText("1.000")).toBeInTheDocument();
  });

  it("says the error columns are lower-is-better", () => {
    render(<ModelResults result={regression()} />);
    expect(screen.getByText(/For the error columns, lower is better/)).toBeInTheDocument();
  });

  it("is unchanged for a result saved before the median baseline existed", () => {
    const old = regression();
    old.baseline = { strategy: "always predict the average (4,077.38)", r2: -0.0009, mae: 2811.51 };
    old.results.forEach((r) => delete r.metrics.mae_vs_baseline);
    render(<ModelResults result={old} />);
    expect(screen.queryByRole("columnheader", { name: "Error vs. median guess" })).not.toBeInTheDocument();
    expect(screen.getAllByText(/No model at all/).length).toBe(1);
  });
});

describe("ModelResults — the gap is said where the split is described (#20)", () => {
  const dated = (gap?: number, dropped?: number): ModelResult => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.split = { ...r.split, strategy: "time", time_column: "day", gap_days: gap, gap_rows_dropped: dropped };
    return r;
  };

  it("says how long the gap was and how many rows it cost", () => {
    render(<ModelResults result={dated(30, 29)} />);
    expect(screen.getByText(/split by day, with a gap of 30 days between training and test \(29 rows inside it dropped\)/)).toBeInTheDocument();
  });

  it("says nothing extra for a date split without one", () => {
    render(<ModelResults result={dated(0, 0)} />);
    expect(screen.getByText(/split by day\./)).toBeInTheDocument();
    expect(screen.queryByText(/with a gap/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — the macro AUC for more than two answers (#37)", () => {
  const multi = (): ModelResult => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.results[0].metrics = { accuracy: 0.55, precision: 0.5, recall: 0.5, f1: 0.5, roc_auc_ovr_macro: 0.834 };
    r.results[1].metrics = { accuracy: 0.5, precision: 0.45, recall: 0.45, f1: 0.45, roc_auc_ovr_macro: 0.79 };
    r.baseline = {
      strategy: "always answer 'c'",
      accuracy: 0.28, precision: 0.07, recall: 0.25, f1: 0.11, roc_auc_ovr_macro: 0.5,
      others: [{ strategy: "guess at random, in proportion to how common each answer is", accuracy: 0.26, precision: 0.25, recall: 0.25, f1: 0.25, roc_auc_ovr_macro: 0.5 }],
    };
    return r;
  };

  it("shows the column, named for what it is", () => {
    render(<ModelResults result={multi()} />);
    expect(screen.getByRole("columnheader", { name: "ROC AUC (macro)" })).toBeInTheDocument();
    expect(screen.getByText("0.834")).toBeInTheDocument();
  });

  it("gives both no-model rows a coin flip under it, so the column is never compared with a blank", () => {
    render(<ModelResults result={multi()} />);
    const rows = screen.getAllByText(/No model at all/).map((cell) => cell.closest("tr") as HTMLElement);
    expect(rows.length).toBe(2);
    for (const row of rows) {
      expect(within(row).getByText("0.500")).toBeInTheDocument();
      expect(within(row).queryByText("—")).not.toBeInTheDocument();
    }
  });

  it("does not show it for a result that has none, as every two-answer result", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByRole("columnheader", { name: "ROC AUC (macro)" })).not.toBeInTheDocument();
  });
});

describe("ModelResults — whether the choice survives other seeds (#32)", () => {
  const withStability = (seed_stability: NonNullable<ModelResult["selection"]>["seed_stability"]) => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.selection = { ...r.selection!, seed_stability };
    return r;
  };

  it("warns when the seed decides which model comes out ahead", () => {
    render(
      <ModelResults
        result={withStability({
          applicable: true, seeds: [42, 1, 2, 3, 4], spread: 0.035, gap: 0.02, seed_dependent: true,
          note: "Refitting Random forest with 5 different random seeds moves its fold score by 0.035, at least the 0.020 that separates it from Gradient boosting: which of the two comes out ahead depends on the seed, not on the data.",
        })}
      />
    );
    expect(screen.getByText(/depends on the seed, not on the data/)).toHaveClass("bg-amber-500/5");
  });

  it("says quietly that it was checked when the lead is bigger than the seeds move it", () => {
    render(<ModelResults result={withStability({ applicable: true, seeds: [42, 1, 2, 3, 4], spread: 0.005, gap: 0.017, seed_dependent: false, note: null })} />);
    expect(screen.getByText(/Refitted with 5 random seeds, the chosen model's fold score moved by 0.005, less than its 0.017 lead/)).toBeInTheDocument();
  });

  it("says nothing for a model whose fit has no randomness, or for an older result", () => {
    render(<ModelResults result={withStability({ applicable: false, reason: "Logistic regression's fit has no randomness, so the seed cannot change it." })} />);
    expect(screen.queryByText(/random seeds/)).not.toBeInTheDocument();
    expect(screen.queryByText(/depends on the seed/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — what the model adds over its best single input (#40)", () => {
  const withSingle = (best_single_column: NonNullable<ModelResult["baseline"]["best_single_column"]>) => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.baseline = { ...r.baseline, best_single_column };
    return r;
  };

  it("shows the column as a row in the table, and a plain line when the model clearly adds to it", () => {
    render(
      <ModelResults
        result={withSingle({
          applicable: true, column: "tenure", score: 0.71, std: 0.02, adds_over_it: true,
          note: "The best input on its own, 'tenure', scores 0.710 on the training folds; the lead is more than noise.",
        })}
      />
    );
    expect(screen.getByText("Best single input alone: tenure")).toBeInTheDocument();
    expect(screen.getByText(/the lead is more than noise/)).not.toHaveClass("bg-amber-500/5");
  });

  it("warns when the model is not clearly better than that one column", () => {
    render(
      <ModelResults
        result={withSingle({
          applicable: true, column: "key", score: 0.99, std: 0.01, adds_over_it: false,
          note: "The best input on its own, 'key', scores 0.990, and the chosen model is not clearly better.",
        })}
      />
    );
    expect(screen.getByText(/is not clearly better/)).toHaveClass("bg-amber-500/5");
  });

  it("says why it was skipped, and says nothing for an older result", () => {
    const { unmount } = render(
      <ModelResults result={withSingle({ applicable: false, reason: "Skipped: 60 input columns is more than the 40 it is worth fitting." })} />
    );
    expect(screen.getByText(/Best single input not compared: Skipped: 60 input columns/)).toBeInTheDocument();
    unmount();
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByText(/Best single input/)).not.toBeInTheDocument();
  });
});

describe("ModelResults — the cautions under the verdict (#98)", () => {
  it("shows the ranked cautions in the box and drops the duplicate inline notes", () => {
    const r = result({
      evaluation: { ...EVAL(1, "independent"), whole_file_statistics: [{ operation: "fill_median", column: "spend" }] },
    });
    r.cautions = [{ id: "whole_file_statistics", severity: "medium", text: "Fixes were worked out over every row, the held-out rows included." }];
    render(<ModelResults result={r} />);
    expect(screen.getByRole("region", { name: "Before you rely on this" })).toBeInTheDocument();
    // Once, in the box, not again in the verdict card.
    expect(screen.getAllByText(/worked out over every row/)).toHaveLength(1);
  });

  it("falls back to the inline notes for a result served without cautions", () => {
    const r = result({
      evaluation: { ...EVAL(1, "independent"), whole_file_statistics: [{ operation: "fill_median", column: "spend" }] },
    });
    render(<ModelResults result={r} />);
    expect(screen.queryByRole("region", { name: "Before you rely on this" })).not.toBeInTheDocument();
    expect(screen.getByText(/worked out over every row/)).toBeInTheDocument();
  });
});

describe("ModelResults — the cannot-judge states (#96)", () => {
  it("leads a verdict that cannot be made with the shared sentence, then the explanation", () => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.assessment = {
      verdict: "not_enough_test_data",
      headline: "There isn't enough held-back data to judge this model.",
      not_judged: {
        status: "not_judged", what: "test rows", have: 12, need: 30, unit: "rows", because: null,
        sentence: "Not enough test rows to judge: 12 rows, at least 30 needed.",
      },
    };
    render(<ModelResults result={r} />);
    const box = screen.getByRole("status", { name: "Cannot judge" });
    expect(box).toHaveTextContent("Not enough test rows to judge: 12 rows, at least 30 needed.");
    expect(box).toHaveTextContent("There isn't enough held-back data to judge this model.");
  });

  it("calls a strong score on few test rows promising, not working (#119)", () => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.assessment = {
      verdict: "insufficient_evidence",
      held_back_from: "useful",
      min_test_rows_for_useful: 60,
      headline: "This model's score (AUC 0.84) clears the bar this app sets on 40 test rows, but that is too few rows to call the model reliable.",
    };
    render(<ModelResults result={r} />);
    expect(screen.getByText("Promising, but too few test rows to call it reliable")).toBeInTheDocument();
    expect(screen.queryByText("This model works")).not.toBeInTheDocument();
    expect(screen.getByText(/too few rows to call the model reliable/)).toBeInTheDocument();
  });

  it("says why a requested cut-off was not chosen instead of showing nothing", () => {
    const r = result({ evaluation: EVAL(1, "independent") });
    r.operating_point = null;
    r.operating_point_not_judged = {
      status: "not_judged", what: "training rows with the rarer answer", have: 1, need: 2, unit: "rows", because: null,
      sentence: "Not enough training rows with the rarer answer to judge: 1 rows, at least 2 needed.",
      objective: "min_recall", description: "catch at least 90% of real cases",
    };
    render(<ModelResults result={r} />);
    expect(screen.getByText("Where the cut-off sits")).toBeInTheDocument();
    expect(screen.getByText("Not enough training rows with the rarer answer to judge: 1 rows, at least 2 needed.")).toBeInTheDocument();
    expect(screen.getByText(/catch at least 90% of real cases\. None was chosen/)).toBeInTheDocument();
  });

  it("shows nothing about the cut-off when none was asked for", () => {
    render(<ModelResults result={result({ evaluation: EVAL(1, "independent") })} />);
    expect(screen.queryByText("Where the cut-off sits")).not.toBeInTheDocument();
  });
});

describe("ModelResults — accessibility (#101)", () => {
  it("has no accessibility violations, the held-out table open", async () => {
    const { container } = render(<main><ModelResults result={result()} /></main>);
    fireEvent.click(screen.getByRole("button", { name: /Show held-out scores for every model/ }));
    expect(await axeViolations(container)).toEqual([]);
  });
});
