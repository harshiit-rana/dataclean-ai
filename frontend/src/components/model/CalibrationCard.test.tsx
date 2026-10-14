import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { CalibrationCard } from "./CalibrationCard";
import { RunHistory } from "./RunHistory";
import { ModelResults } from "./ModelResults";
import { requestFromResult } from "@/pages/ModelPage";
import type { CalibrationReport, ModelResult, ModelRun } from "@/api/types";

const OPTIONS = {
  sigmoid: { available: true, reason: null },
  isotonic: {
    available: false,
    reason: "Isotonic calibration needs at least 1,000 training rows and this has 640.",
  },
};

function report(overrides: Partial<CalibrationReport> = {}): CalibrationReport {
  return {
    method: "none",
    requested: "none",
    refused_because: null,
    ece_threshold: 0.05,
    error: 0.093,
    chance_error: 0.047,
    recommended: true,
    options: OPTIONS,
    folds: 0,
    ensemble: false,
    fitted_on: null,
    before: null,
    after: null,
    ...overrides,
  };
}

const applied = (overrides: Partial<CalibrationReport> = {}) =>
  report({
    method: "sigmoid",
    requested: "sigmoid",
    recommended: false,
    folds: 5,
    fitted_on: "cross-validation over 5 folds of the 1,600 training rows; the test set was not consulted",
    before: { brier_score: 0.1961, expected_calibration_error: 0.0932, roc_auc: 0.7652, recall: 0.6513, precision: 0.6226 },
    after: { brier_score: 0.1868, expected_calibration_error: 0.0404, roc_auc: 0.7652, recall: 0.5526, precision: 0.6667 },
    ...overrides,
  });

describe("CalibrationCard — the offer", () => {
  it("says how far off the probabilities are and what chance alone would show", () => {
    render(<CalibrationCard report={report()} onCalibrate={() => undefined} />);
    expect(screen.getByText(/off by 9\.3 points/)).toBeInTheDocument();
    expect(screen.getByText(/about 4\.7 would be chance on this many rows/)).toBeInTheDocument();
  });

  it("offers each method and runs the one that is clicked", () => {
    const onCalibrate = vi.fn();
    render(<CalibrationCard report={report()} onCalibrate={onCalibrate} />);
    fireEvent.click(screen.getByRole("button", { name: /Calibrate with sigmoid/ }));
    expect(onCalibrate).toHaveBeenCalledWith("sigmoid");
  });

  it("disables a method the data cannot support and says why", () => {
    const onCalibrate = vi.fn();
    render(<CalibrationCard report={report()} onCalibrate={onCalibrate} />);
    const isotonic = screen.getByRole("button", { name: /Calibrate with isotonic/ });
    expect(isotonic).toBeDisabled();
    expect(screen.getByText(/needs at least 1,000 training rows and this has 640/)).toBeInTheDocument();
    fireEvent.click(isotonic);
    expect(onCalibrate).not.toHaveBeenCalled();
  });

  it("says it uses only the training rows, and that applying it is another look at the test rows", () => {
    render(<CalibrationCard report={report()} onCalibrate={() => undefined} />);
    expect(screen.getByText(/training rows\s+only/)).toBeInTheDocument();
    expect(screen.getByText(/counts as another look at those\s+rows/)).toBeInTheDocument();
    expect(screen.getByText(/can change the answers at the default 0\.50 line/)).toBeInTheDocument();
  });

  it("disables the buttons while a run is in progress", () => {
    render(<CalibrationCard report={report()} onCalibrate={() => undefined} busy />);
    expect(screen.getByRole("button", { name: /Calibrate with sigmoid/ })).toBeDisabled();
  });

  it("says nothing when the probabilities are fine", () => {
    const { container } = render(
      <CalibrationCard report={report({ recommended: false })} onCalibrate={() => undefined} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("says nothing for a result with nothing to calibrate", () => {
    const { container } = render(<CalibrationCard report={null} onCalibrate={() => undefined} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("offers nothing where the page cannot act on it", () => {
    const { container } = render(<CalibrationCard report={report()} />);
    expect(container).toBeEmptyDOMElement();
  });
});

describe("CalibrationCard — after it was applied", () => {
  it("shows before and after side by side, lower-is-better marked", () => {
    render(<CalibrationCard report={applied()} />);
    expect(screen.getByText("Probabilities calibrated (sigmoid)")).toBeInTheDocument();
    expect(screen.getByText("0.196")).toBeInTheDocument();
    expect(screen.getByText("0.187")).toBeInTheDocument();
    expect(screen.getByText("0.093")).toBeInTheDocument();
    expect(screen.getByText("0.040")).toBeInTheDocument();
    expect(screen.getAllByText("lower is better").length).toBe(2);
  });

  it("says the AUC is unchanged for a sigmoid on a random split", () => {
    render(<CalibrationCard report={applied()} />);
    expect(screen.getByText(/so the AUC is unchanged/)).toBeInTheDocument();
  });

  it("does not claim that for isotonic, which can flatten ties", () => {
    render(<CalibrationCard report={applied({ method: "isotonic", requested: "isotonic" })} />);
    expect(screen.queryByText(/so the AUC is unchanged/)).not.toBeInTheDocument();
    expect(screen.getByText(/the AUC can move slightly/)).toBeInTheDocument();
  });

  it("warns that the default line now sits on different probabilities, with recall to show it", () => {
    render(<CalibrationCard report={applied()} />);
    expect(screen.getByText(/It can change the answers at the default 0\.50 line/)).toBeInTheDocument();
    expect(screen.getByText("Recall at the default 0.50 line")).toBeInTheDocument();
    expect(screen.getByText("0.651")).toBeInTheDocument();
    expect(screen.getByText("0.553")).toBeInTheDocument();
  });

  it("explains the averaging after a forward-in-time split", () => {
    render(<CalibrationCard report={applied({ ensemble: true })} />);
    expect(screen.getByText(/averages one copy per fold/)).toBeInTheDocument();
    expect(screen.queryByText(/so the AUC is unchanged/)).not.toBeInTheDocument();
  });

  it("lets the user go back to the model's own probabilities", () => {
    const onCalibrate = vi.fn();
    render(<CalibrationCard report={applied()} onCalibrate={onCalibrate} />);
    fireEvent.click(screen.getByRole("button", { name: /Go back to the uncalibrated model/ }));
    expect(onCalibrate).toHaveBeenCalledWith("none");
  });
});

describe("CalibrationCard — when a method was declined", () => {
  it("says what was asked for, why it was not done, and that the model was left alone", () => {
    render(
      <CalibrationCard
        report={report({
          requested: "isotonic",
          recommended: false,
          refused_because: "Isotonic calibration needs at least 1,000 training rows and this has 640.",
        })}
        onCalibrate={() => undefined}
      />
    );
    expect(screen.getByText("The probabilities were not calibrated")).toBeInTheDocument();
    expect(screen.getByText(/calibration was asked for, but: Isotonic calibration needs at least 1,000/)).toBeInTheDocument();
    expect(screen.getByText(/The model was left as it was/)).toBeInTheDocument();
  });
});

const RESULT = {
  task: "classification",
  target_column: "churned",
  features: { numeric: ["tenure"], categorical: [], dropped: [] },
  split: { strategy: "random", time_column: null, group_column: null, train_rows: 800, test_rows: 200 },
  baseline: { strategy: "always answer no", accuracy: 0.7 },
  primary_metric: "roc_auc",
  results: [
    {
      model: "random_forest_classifier",
      label: "Random forest",
      description: "",
      metrics: { accuracy: 0.8, roc_auc: 0.81 },
      feature_importance: [],
      column_importance: [],
      diagnostics: {},
      failed: null,
    },
  ],
  best_model: "random_forest_classifier",
  assessment: { verdict: "useful", headline: "It works." },
  deep_diagnostics: {},
} as unknown as ModelResult;

describe("ModelResults and the page around the card", () => {
  it("puts the card on the results page", () => {
    render(<ModelResults result={{ ...RESULT, calibration: report() }} onCalibrate={() => undefined} />);
    expect(screen.getByText("Its probabilities may not mean what they say")).toBeInTheDocument();
  });

  it("is absent from a result with no calibration report, as every older result is", () => {
    render(<ModelResults result={RESULT} onCalibrate={() => undefined} />);
    expect(screen.queryByText("Its probabilities may not mean what they say")).not.toBeInTheDocument();
  });

  it("is carried by the request that rebuilds a stored result, so improving it keeps it", () => {
    expect(requestFromResult({ ...RESULT, calibration: applied() }).calibration).toBe("sigmoid");
    expect(requestFromResult(RESULT).calibration).toBe("none");
  });
});

describe("RunHistory — calibration is part of the setup", () => {
  const run = (calibration?: "none" | "sigmoid" | "isotonic"): ModelRun =>
    ({
      run_id: Math.random().toString(36).slice(2),
      at: "2026-08-07T10:00:00+00:00",
      data_fingerprint: "aaa",
      target_column: "outcome",
      task: "classification",
      best_model: "random_forest_classifier",
      model_label: "Random forest",
      models_tried: ["random_forest_classifier"],
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
      calibration,
    }) as unknown as ModelRun;

  it("names the method on a calibrated run, so two runs differing only in it can be told apart", () => {
    render(<RunHistory runs={[run("sigmoid"), run("none")]} currentFingerprint="aaa" />);
    expect(screen.getByText(/calibrated \(sigmoid\)/)).toBeInTheDocument();
    expect(screen.getAllByText(/calibrated/).length).toBe(1);
  });

  it("says nothing for a run from before calibration existed", () => {
    render(<RunHistory runs={[run(undefined), run(undefined)]} currentFingerprint="aaa" />);
    expect(screen.queryByText(/calibrated/)).not.toBeInTheDocument();
  });
});
