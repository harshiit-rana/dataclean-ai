import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PreviousApplications } from "./PreviousApplications";
import * as client from "@/api/client";

describe("PreviousApplications (#89)", () => {
  it("lists earlier applications with their downloads", async () => {
    vi.spyOn(client, "fetchApplications").mockResolvedValue([
      {
        apply_id: "a1", file_name: "march.csv", at: "2026-10-16T10:00:00+00:00", rows_in: 60, rows_kept: 58,
        drift_verdict: "no_meaningful_change", rows_with_findings: 3, predictions: true, answers_headline: null,
        model_run_id: "r1", downloads: { cleaned: "/api/apply/s/a1/cleaned", predictions: "/api/apply/s/a1/predictions" },
      },
    ]);
    render(<PreviousApplications sessionId="s" />);
    expect(await screen.findByText("march.csv")).toBeInTheDocument();
    expect(screen.getByText(/58 of 60 rows kept · 3 with something to check/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Predictions" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Cleaned file" })).toBeInTheDocument();
  });

  it("shows nothing before the model was used on a file", async () => {
    vi.spyOn(client, "fetchApplications").mockResolvedValue([]);
    const { container } = render(<PreviousApplications sessionId="s" />);
    await Promise.resolve();
    expect(container).toBeEmptyDOMElement();
  });
});
