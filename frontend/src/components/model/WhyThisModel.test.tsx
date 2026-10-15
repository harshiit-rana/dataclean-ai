import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { WhyThisModel } from "./WhyThisModel";
import type { ExplanationStatement, ModelExplanation } from "@/api/types";

const SECTIONS = {
  how_chosen: "How the model was chosen",
  alternatives: "Why the others were not chosen",
  constraints: "What constrained the choice",
  improvement: "What the improvement search tried",
} as const;

function statement(overrides: Partial<ExplanationStatement>): ExplanationStatement {
  return {
    id: "x",
    section: "how_chosen",
    tone: "plain",
    text: "A sentence.",
    evidence: [],
    ...overrides,
  };
}

function explanation(statements: ExplanationStatement[], lead_id: string | null = null): ModelExplanation {
  return { sections: { ...SECTIONS }, statements, lead_id };
}

describe("WhyThisModel", () => {
  it("shows each sentence under its section with the numbers it rests on", () => {
    render(
      <WhyThisModel
        explanation={explanation([
          statement({
            id: "how_chosen_method",
            text: "Every model was cross-validated.",
            evidence: [{ field: "selection.folds", label: "folds", value: 5 }],
          }),
          statement({
            id: "runner_up",
            section: "alternatives",
            text: "Forest beat logistic regression.",
            evidence: [{ field: "selection.paired_difference.difference", label: "average fold gap", value: 0.06 }],
          }),
        ])}
      />
    );
    const chosen = screen.getByRole("region", { name: "How the model was chosen" });
    expect(within(chosen).getByText("Every model was cross-validated.")).toBeInTheDocument();
    expect(within(chosen).getByText("folds 5")).toBeInTheDocument();
    const others = screen.getByRole("region", { name: "Why the others were not chosen" });
    expect(within(others).getByText("average fold gap 0.060")).toBeInTheDocument();
  });

  it("leads with the warning, once, when the choice is not clearly better than its rival", () => {
    render(
      <WhyThisModel
        explanation={explanation(
          [
            statement({ id: "how_chosen_method", text: "Cross-validated." }),
            statement({
              id: "runner_up",
              section: "alternatives",
              tone: "warning",
              text: "Too close to call. Forest and logistic regression are within the noise.",
            }),
          ],
          "runner_up"
        )}
      />
    );
    const warning = screen.getByText(/Too close to call/);
    expect(screen.getAllByText(/Too close to call/)).toHaveLength(1);
    expect(warning.closest("li")).toHaveClass("bg-amber-500/5");
    // It comes before the first section heading.
    const heading = screen.getByRole("heading", { name: "How the model was chosen" });
    expect(warning.compareDocumentPosition(heading) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    // And the section it came from no longer has a heading of its own.
    expect(screen.queryByRole("heading", { name: "Why the others were not chosen" })).not.toBeInTheDocument();
  });

  it("leaves out sections with nothing to say", () => {
    render(<WhyThisModel explanation={explanation([statement({ id: "a" })])} />);
    expect(screen.queryByRole("heading", { name: "What the improvement search tried" })).not.toBeInTheDocument();
  });

  it("renders nothing for an older result without an explanation", () => {
    const { container } = render(<WhyThisModel explanation={undefined} />);
    expect(container).toBeEmptyDOMElement();
    const empty = render(<WhyThisModel explanation={explanation([])} />);
    expect(empty.container).toBeEmptyDOMElement();
  });
});
