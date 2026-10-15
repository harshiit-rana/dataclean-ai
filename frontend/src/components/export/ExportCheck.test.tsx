import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ExportCheck } from "./ExportCheck";
import * as client from "@/api/client";

describe("ExportCheck (#78)", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.spyOn(client, "fetchExportVerification").mockResolvedValue({ status: "not_checked" });
  });

  it("runs the check and shows a pass", async () => {
    vi.spyOn(client, "verifyExport").mockResolvedValue({
      status: "passed",
      message: "The exported scripts rebuilt the model and it agrees with the app's on all 60 held-out rows.",
      rows_compared: 60,
    });
    render(<ExportCheck sessionId="s" />);
    fireEvent.click(screen.getByRole("button", { name: "Check the export rebuilds the model" }));
    await waitFor(() => expect(screen.getByText(/agrees with the app's on all 60/)).toBeInTheDocument());
  });

  it("lists the first differing rows of a failed check", async () => {
    vi.spyOn(client, "verifyExport").mockResolvedValue({
      status: "failed",
      message: "The rebuilt model disagrees with the app's on 3 of 60 held-out rows.",
      first_differences: [{ row: 12, difference: "answered 'yes' where the app answered 'no'" }],
    });
    render(<ExportCheck sessionId="s" />);
    fireEvent.click(screen.getByRole("button", { name: /Check the export/ }));
    await waitFor(() => expect(screen.getByText(/disagrees with the app's on 3 of 60/)).toBeInTheDocument());
    expect(screen.getByText(/row 12: answered 'yes'/)).toBeInTheDocument();
  });

  it("names the probe rows the app and predict.py disagree on", async () => {
    vi.spyOn(client, "verifyExport").mockResolvedValue({
      status: "failed",
      message: "On 1 of 9 probe rows the exported predict.py and the app disagree.",
      probes: { rows: 9, agreed: 8, disagreements: [{ probe: "plan a value never seen in training" }] },
    });
    render(<ExportCheck sessionId="s" />);
    fireEvent.click(screen.getByRole("button", { name: /Check the export/ }));
    await waitFor(() => expect(screen.getByText(/Probe rows where they disagree: plan a value never seen/)).toBeInTheDocument());
  });

  it("shows the last check when the page opens", async () => {
    vi.spyOn(client, "fetchExportVerification").mockResolvedValue({ status: "passed", message: "Checked earlier." });
    render(<ExportCheck sessionId="s" />);
    await waitFor(() => expect(screen.getByText("Checked earlier.")).toBeInTheDocument());
  });
});
