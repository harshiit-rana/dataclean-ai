import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ApplyResult } from "@/api/types";

const applyToNewFile = vi.fn();
vi.mock("@/api/client", async (original) => ({
  ...(await original<typeof import("@/api/client")>()),
  applyToNewFile: (...args: unknown[]) => applyToNewFile(...args),
}));
vi.mock("@/components/upload/FileDropzone", () => ({
  FileDropzone: ({ onFileSelected }: { onFileSelected: (f: File) => void }) => (
    <button onClick={() => onFileSelected(new File(["a,b\n1,2"], "march.csv"))}>pick file</button>
  ),
}));

const { ApplyPage } = await import("./ApplyPage");

const RESULT: ApplyResult = {
  session_id: "s1",
  apply_id: "abcdef123456",
  file_name: "march.csv",
  rows_in: 60,
  rows_kept: 59,
  steps: [
    {
      action_id: "a1", operation: "fill_median", column: "spend", status: "applied", note: null,
      rows_before: 60, rows_after: 60, dropped_count: 0, learned: { fill_value: 23.83 },
    },
    {
      action_id: "a2", operation: "drop_rows_with_nulls", column: "churned", status: "skipped",
      note: "This file has no 'churned' column -- expected when these are the rows you want predictions for.",
      rows_before: 60, rows_after: 60, dropped_count: 0, learned: null,
    },
    {
      action_id: "a3", operation: "drop_duplicate_rows", column: null, status: "applied", note: null,
      rows_before: 60, rows_after: 59, dropped_count: 1, learned: null,
    },
  ],
  dropped: [{ row: 60, reason: "it duplicated an earlier row" }],
  dropped_count: 1,
  notes: [],
  drift: {
    verdict: "different",
    headline: "'spend' no longer looks like the data the model learned from.",
    columns: [
      { column: "spend", kind: "numeric", status: "different", psi: 1.9, outside_training_range: 0.8,
        training_median: 23.8, new_median: 223.8 },
      { column: "plan", kind: "categorical", status: "stable", psi: 0.01, unseen_share: 0, unseen_examples: [] },
    ],
  },
  predictions: {
    count: 59,
    columns: ["predicted_churned", "probability_yes"],
    preview: [{ row: 1, predicted_churned: "yes", probability_yes: 0.67 }],
    cutoff: 0.42,
  },
  answers: null,
  downloads: {
    cleaned: "/api/apply/s1/abcdef123456/cleaned",
    predictions: "/api/apply/s1/abcdef123456/predictions",
  },
};

function renderPage() {
  render(
    <MemoryRouter initialEntries={["/apply/s1"]}>
      <Routes>
        <Route path="/apply/:sessionId" element={<ApplyPage />} />
      </Routes>
    </MemoryRouter>
  );
}

async function runWith(result: ApplyResult) {
  applyToNewFile.mockResolvedValueOnce(result);
  renderPage();
  fireEvent.click(screen.getByText("pick file"));
  fireEvent.click(screen.getByRole("button", { name: /Clean and predict/ }));
  await waitFor(() => expect(applyToNewFile).toHaveBeenCalled());
}

describe("ApplyPage", () => {
  beforeEach(() => applyToNewFile.mockReset());

  it("won't run without a file", () => {
    renderPage();
    expect(screen.getByRole("button", { name: /Clean and predict/ })).toBeDisabled();
  });

  it("says how many rows survived and that the dropped ones are still accounted for", async () => {
    await runWith(RESULT);
    expect(await screen.findByText("59 of 60 rows kept")).toBeInTheDocument();
    expect(screen.getByText(/still in the predictions file, with the reason/)).toBeInTheDocument();
    expect(screen.getByText(/Data row 60:/)).toBeInTheDocument();
  });

  it("leads the drift section with the verdict, and names the column", async () => {
    await runWith(RESULT);
    expect(await screen.findByText("This isn't the data the model learned from")).toBeInTheDocument();
    expect(screen.getByText(/median 23.80 in training, 223.80 here; 80% outside the training range/)).toBeInTheDocument();
  });

  it("shows a drift check on too few rows with the shared cannot-judge sentence (#96)", async () => {
    await runWith({
      ...RESULT,
      drift: {
        verdict: "not_enough_rows",
        headline: "10 rows is too few to tell whether this data has shifted.",
        columns: [],
        not_judged: {
          status: "not_judged", what: "new rows", have: 10, need: 30, unit: "rows", because: null,
          sentence: "Not enough new rows to judge: 10 rows, at least 30 needed.",
        },
      },
    });
    const box = await screen.findByRole("status", { name: "Cannot judge" });
    expect(box).toHaveTextContent("Not enough new rows to judge: 10 rows, at least 30 needed.");
    expect(box).toHaveTextContent("10 rows is too few to tell whether this data has shifted.");
  });

  it("shows a check on too few real answers with the same sentence pattern (#96)", async () => {
    await runWith({
      ...RESULT,
      answers: {
        rows: 12, metric: "roc_auc", reported: 0.8, now: null,
        headline: "There isn't enough held-back data to judge this model.",
        not_judged: {
          status: "not_judged", what: "labelled rows in this file", have: 12, need: 30, unit: "rows", because: null,
          sentence: "Not enough labelled rows in this file to judge: 12 rows, at least 30 needed.",
        },
      },
    });
    expect(await screen.findByText("Not enough labelled rows in this file to judge: 12 rows, at least 30 needed.")).toBeInTheDocument();
  });

  it("shows that the session's learned value was reused", async () => {
    await runWith(RESULT);
    expect(await screen.findByText(/with 23.83, as learned from your original file/)).toBeInTheDocument();
  });

  it("explains a skipped fix rather than hiding it", async () => {
    await runWith(RESULT);
    expect(await screen.findByText("Skipped")).toBeInTheDocument();
    expect(screen.getByText(/expected when these are the rows you want predictions for/)).toBeInTheDocument();
  });

  it("names the cut-off the answers were given at", async () => {
    await runWith(RESULT);
    expect(await screen.findByText(/at the cut-off you chose \(0.42\)/)).toBeInTheDocument();
  });

  it("labels a score on the new file's real answers as new data, not as the held-out score", async () => {
    await runWith({
      ...RESULT,
      answers: { rows: 200, metric: "roc_auc", reported: 0.8, now: 0.78, headline: "Holding up." },
    });
    expect(await screen.findByText("Holding up.")).toBeInTheDocument();
    expect(screen.getByText("New data")).toBeInTheDocument();
    expect(screen.queryByText(/Held-out evaluation/)).not.toBeInTheDocument();
  });

  it("says a file was recorded as an independent check", async () => {
    await runWith({
      ...RESULT,
      answers: {
        rows: 200, metric: "roc_auc", reported: 0.8, now: 0.78, headline: "Holding up.",
        counted_as_independent: true, overlap_excluded: 0,
      },
    });
    expect(await screen.findByText(/Recorded as an independent check on this model/)).toBeInTheDocument();
  });

  it("says an identical file is not counted a second time", async () => {
    await runWith({
      ...RESULT,
      answers: {
        rows: 200, metric: "roc_auc", reported: 0.8, now: 0.78, headline: "Holding up.",
        counted_as_independent: false, already_counted: true,
      },
    });
    expect(await screen.findByText(/already been counted for this model/)).toBeInTheDocument();
    expect(screen.queryByText(/Recorded as an independent check/)).not.toBeInTheDocument();
  });

  it("shows no score label when the file carries no answers", async () => {
    await runWith(RESULT);
    await screen.findByText("59 of 60 rows kept");
    expect(screen.queryByText("New data")).not.toBeInTheDocument();
  });

  it("offers only the cleaned file when there is no model", async () => {
    await runWith({ ...RESULT, drift: null, predictions: null, downloads: { cleaned: RESULT.downloads.cleaned } });
    expect(await screen.findByText(/Download the cleaned file/)).toBeInTheDocument();
    expect(screen.queryByText(/Download predictions/)).not.toBeInTheDocument();
    expect(screen.getByText(/They're listed below with the reason/)).toBeInTheDocument();
  });

  it("shows the server's reason when the file can't be used", async () => {
    const { ApiError } = await import("@/api/client");
    applyToNewFile.mockRejectedValueOnce(
      new ApiError("These rows can't be scored", "This file is missing 1 column the model needs: signal.", 422)
    );
    renderPage();
    fireEvent.click(screen.getByText("pick file"));
    fireEvent.click(screen.getByRole("button", { name: /Clean and predict/ }));
    expect(await screen.findByText(/missing 1 column the model needs: signal/)).toBeInTheDocument();
  });
});

describe("ApplyPage — each column judged on its own values (#46)", () => {
  beforeEach(() => applyToNewFile.mockReset());

  const withColumns = (columns: ApplyResult["drift"] extends infer D ? D extends { columns: infer C } ? C : never : never) => ({
    ...RESULT,
    drift: { verdict: "different" as const, headline: "'notes' no longer looks like the data the model learned from.", columns },
  });

  it("shows how many filled-in values each file had", async () => {
    await runWith(withColumns([
      { column: "spend", kind: "numeric", status: "stable", psi: 0.02, training_values: 1000, new_values: 300,
        training_blank_share: 0, new_blank_share: 0, blank_status: "stable" },
    ]));
    expect(await screen.findByText("Values (training / here)")).toBeInTheDocument();
    expect(screen.getByText("1,000 / 300")).toBeInTheDocument();
  });

  it("says when a column had too few values to compare, and why", async () => {
    await runWith(withColumns([
      { column: "remarks", kind: "categorical", status: "not_judged", training_values: 60, new_values: 15,
        training_blank_share: 0.97, new_blank_share: 0.97, blank_status: "stable",
        note: "Only 15 filled-in values in the new file (60 in training, 15 here); at least 30 are needed to compare them." },
    ]));
    expect(await screen.findByText("too few values")).toBeInTheDocument();
    expect(screen.getByText(/Only 15 filled-in values in the new file/)).toBeInTheDocument();
  });

  it("reports a column going blank as what changed", async () => {
    await runWith(withColumns([
      { column: "notes", kind: "numeric", status: "different", psi: 0.03, training_values: 1960, new_values: 180,
        training_blank_share: 0.02, new_blank_share: 0.4, blank_status: "different" },
    ]));
    expect(await screen.findByText(/blank in 2% of training rows, 40% here/)).toBeInTheDocument();
  });
});

describe("ApplyPage — what chance reaches (#45)", () => {
  beforeEach(() => applyToNewFile.mockReset());

  it("shows each column's PSI beside the PSI chance reaches for files of these sizes", async () => {
    await runWith({
      ...RESULT,
      drift: {
        verdict: "different",
        headline: "'n3' no longer looks like the data the model learned from. 1 of the 30 inputs compared moved more than chance explains on files of these sizes.",
        columns: [
          { column: "n3", kind: "numeric", status: "different", psi: 0.607, chance_threshold: 0.153, beyond_chance: true },
          { column: "n13", kind: "numeric", status: "stable", psi: 0.104, chance_threshold: 0.163, beyond_chance: false },
        ],
      },
    });
    expect(await screen.findByText("Chance reaches")).toBeInTheDocument();
    expect(screen.getByText("0.153")).toBeInTheDocument();
    expect(screen.getByText(/1 of the 30 inputs compared moved more than chance explains/)).toBeInTheDocument();
    expect(screen.getByText(/only called moved when its PSI is also above what chance reaches/)).toBeInTheDocument();
  });
});
