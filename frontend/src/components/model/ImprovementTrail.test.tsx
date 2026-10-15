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
    render(<ImprovementTrail trail={trail({ search_strategy: "bayesian" })} />);
    expect(screen.getByText(/settings searched by bayesian/)).toBeInTheDocument();
  });

  it("says nothing for a run from before it was recorded", () => {
    render(<ImprovementTrail trail={trail()} />);
    expect(screen.queryByText(/settings searched by/)).not.toBeInTheDocument();
  });
});

describe("ImprovementTrail — the rungs of a halving search (#49)", () => {
  const halvingRound = {
    name: "hyperparameters", label: "Model settings", what_it_tried: "Tried 63 fits.",
    score: 0.81, std: 0.02, gain: 0.02, accepted: true, outcome: "Better settings found: AUC up +0.020.",
    detail: {
      strategy: "halving",
      rungs: [
        { rung: 1, settings: 14, folds: 2, best_score: 0.79, kept: 5 },
        { rung: 2, settings: 5, folds: 5, best_score: 0.8, kept: 2 },
        { rung: 3, settings: 2, folds: 15, best_score: 0.81, kept: 1 },
      ],
    },
  };

  it("lists each rung: how many settings, on how many folds, and how many went on", () => {
    render(<ImprovementTrail trail={trail({ rounds: [halvingRound], search_strategy: "halving" })} />);
    const rungs = screen.getByRole("list", { name: "Rungs" });
    expect(rungs).toHaveTextContent("Rung 1: 14 settings on 2 folds, best 0.790 → 5 went on");
    expect(rungs).toHaveTextContent("Rung 3: 2 settings on 15 folds, best 0.810 → 1 went on");
    expect(screen.getByText(/settings searched by successive halving/)).toBeInTheDocument();
  });

  it("says why a halving search became a random one", () => {
    const fell = { ...halvingRound, detail: { strategy: "random", note: "Searched at random: only 5 settings to try is too few to halve over." } };
    render(<ImprovementTrail trail={trail({ rounds: [fell] })} />);
    expect(screen.getByText(/only 5 settings to try is too few to halve over/)).toBeInTheDocument();
    expect(screen.queryByRole("list", { name: "Rungs" })).not.toBeInTheDocument();
  });

  it("shows no rungs for a round that has none", () => {
    render(<ImprovementTrail trail={trail()} />);
    expect(screen.queryByRole("list", { name: "Rungs" })).not.toBeInTheDocument();
  });
});

describe("ImprovementTrail — the strategy chosen for each search (#50)", () => {
  const round = (strategy: string, reason: string) => ({
    name: "hyperparameters", label: "Model settings", what_it_tried: "Tried 60 fits.",
    score: 0.8, std: 0.02, gain: 0.01, accepted: false, outcome: "No setting beat the defaults.",
    detail: { strategy, strategy_reason: reason },
  });

  it("says which strategy a search used and why", () => {
    render(
      <ImprovementTrail
        trail={trail({
          rounds: [round("random", "240 training rows is below the 5,000 where scoring fewer folds saves real time")],
          search_strategy: "auto",
          search_strategies: ["random"],
        })}
      />
    );
    expect(
      screen.getByText(/Searched by random search: 240 training rows is below the 5,000/)
    ).toBeInTheDocument();
    expect(screen.getByText(/settings searched by random search/)).toBeInTheDocument();
  });

  it("names what auto resolved to, not the word auto", () => {
    render(<ImprovementTrail trail={trail({ search_strategy: "auto", search_strategies: ["halving", "random"] })} />);
    expect(screen.getByText(/settings searched by successive halving and random search/)).toBeInTheDocument();
    expect(screen.queryByText(/searched by auto/)).not.toBeInTheDocument();
  });

  it("shows the user's own choice as theirs", () => {
    render(
      <ImprovementTrail
        trail={trail({ rounds: [round("halving", "you chose halving search")], search_strategy: "halving", search_strategies: ["halving"] })}
      />
    );
    expect(screen.getByText(/Searched by successive halving: you chose halving search/)).toBeInTheDocument();
  });

  it("says nothing about a run that asked for auto and recorded nothing else", () => {
    render(<ImprovementTrail trail={trail({ search_strategy: "auto" })} />);
    expect(screen.queryByText(/settings searched by/)).not.toBeInTheDocument();
  });
});
