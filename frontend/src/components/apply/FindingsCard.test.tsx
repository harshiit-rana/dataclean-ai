import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { FindingsCard } from "./FindingsCard";

describe("FindingsCard (#85)", () => {
  it("shows the three findings apart, with the summary", () => {
    render(
      <FindingsCard
        findings={{
          inputs: { verdict: "similar", headline: "The inputs look like the training data." },
          predictions: { verdict: "changed", headline: "62% flagged, against 30% at training." },
          performance: { verdict: "unknown", headline: "Unknown -- this file has no answers yet." },
          summary: "The predictions moved although the inputs look similar, and there are no answers yet.",
        }}
      />
    );
    expect(screen.getByText("Inputs")).toBeInTheDocument();
    expect(screen.getByText("Predictions")).toBeInTheDocument();
    expect(screen.getByText("Performance")).toBeInTheDocument();
    expect(screen.getByText("Unknown -- this file has no answers yet.")).toBeInTheDocument();
    expect(screen.getByText(/predictions moved although the inputs look similar/)).toBeInTheDocument();
  });
});
