import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ImprovementTrail } from "./ImprovementTrail";
import type { ImprovementTrail as Trail } from "@/api/types";

function trail(overrides: Partial<Trail> = {}): Trail {
  return {
    rounds: [
      {
        name: "baseline", label: "Starting point", what_it_tried: "Cross-validated 2 model families.",
        score: 0.78, std: 0.02, gain: null, accepted: true, outcome: "Logistic regression leads at 0.780 AUC.", detail: {},
      },
    ],
    metric: "roc_auc",
    goal: null,
    stopped_because: "no_further_gains",
    cv_score: 0.78,
    cv_std: 0.02,
    settings: {},
    ...overrides,
  };
}

describe("ImprovementTrail — the time limit (#47)", () => {
  it("says the search was cut off by the clock and what that means for the result", () => {
    render(<ImprovementTrail trail={trail({ stopped_because: "time_limit", time_limit_seconds: 25 })} />);
    expect(screen.getByText("Stopped at your time limit")).toBeInTheDocument();
    expect(screen.getByText(/rounds marked not run were never tried/)).toBeInTheDocument();
  });

  it("marks a round that did not run, apart from the ones that did", () => {
    render(
      <ImprovementTrail
        trail={trail({
          stopped_because: "time_limit",
          time_limit_seconds: 25,
          skipped_for_time: ["other_models"],
          rounds: [
            ...trail().rounds,
            {
              name: "other_models", label: "The other models, tuned", what_it_tried: "Looked at runners-up.",
              score: null, std: null, gain: null, accepted: false,
              outcome: "Not run: the 25-second time limit was reached first.", detail: { skipped: "time_limit" },
            },
          ],
        })}
      />
    );
    expect(screen.getByText("Not run: time limit")).toBeInTheDocument();
    expect(screen.getByText("Not run: the 25-second time limit was reached first.")).toBeInTheDocument();
    // Only the round that was skipped carries the mark.
    expect(screen.getAllByText("Not run: time limit")).toHaveLength(1);
  });

  it("shows nothing about time for a search that was not limited", () => {
    render(<ImprovementTrail trail={trail()} />);
    expect(screen.queryByText(/time limit/i)).not.toBeInTheDocument();
    expect(screen.getByText("Stopped when the rounds stopped paying")).toBeInTheDocument();
  });

  it("copes with a trail saved before the limit existed", () => {
    const old = trail();
    delete (old as Partial<Trail>).time_limit_seconds;
    delete (old as Partial<Trail>).skipped_for_time;
    render(<ImprovementTrail trail={old} />);
    expect(screen.getByText("Starting point")).toBeInTheDocument();
  });
});

describe("ImprovementTrail — the search strategy (#48)", () => {
  it("names how the settings were searched", () => {
    render(<ImprovementTrail trail={trail({ search_strategy: "random" })} />);
    expect(screen.getByText(/settings searched by random search/)).toBeInTheDocument();
  });

  it("shows an unfamiliar strategy by its own name rather than hiding it", () => {
    render(<ImprovementTrail trail={trail({ search_strategy: "halving" })} />);
    expect(screen.getByText(/settings searched by halving/)).toBeInTheDocument();
  });

  it("says nothing for a run from before it was recorded", () => {
    render(<ImprovementTrail trail={trail()} />);
    expect(screen.queryByText(/settings searched by/)).not.toBeInTheDocument();
  });
});
