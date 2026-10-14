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
