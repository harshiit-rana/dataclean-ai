import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PredictionHistory } from "./PredictionHistory";
import * as client from "@/api/client";

describe("PredictionHistory (#87)", () => {
  it("shows the training-time reference and each file's share flagged", async () => {
    vi.spyOn(client, "fetchPredictionHistory").mockResolvedValue({
      run_id: "r1",
      reference: { kind: "binary", positive: "yes", rows: 60, positive_rate: 0.3, probability_quantiles: { "0.5": 0.28 } },
      applications: [
        { apply_id: "a1", file_name: "may.csv", at: "2026-10-16T10:00",
          prediction_summary: { kind: "binary", positive: "yes", rows: 80, positive_rate: 0.72, probability_quantiles: { "0.5": 0.61 } } },
      ],
    });
    render(<PredictionHistory sessionId="s" />);
    expect(await screen.findByText("Held-out rows, at training")).toBeInTheDocument();
    expect(screen.getByText("30% flagged as yes")).toBeInTheDocument();
    expect(screen.getByText("72% flagged as yes")).toBeInTheDocument();
    expect(screen.getByText("0.61")).toBeInTheDocument();
  });

  it("shows nothing before the model was used on a file", async () => {
    vi.spyOn(client, "fetchPredictionHistory").mockResolvedValue({ run_id: "r1", reference: null, applications: [] });
    const { container } = render(<PredictionHistory sessionId="s" />);
    await Promise.resolve();
    expect(container).toBeEmptyDOMElement();
  });
});
