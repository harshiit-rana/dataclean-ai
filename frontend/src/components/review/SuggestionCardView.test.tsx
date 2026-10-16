import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { SuggestionCardView } from "./SuggestionCardView";
import type { SuggestionCard } from "@/api/types";

const CARD: SuggestionCard = {
  card_id: "card_1",
  issue_id: "issue_1",
  column_name: "salary",
  found: "340 rows have no salary.",
  proposed_action: "Drop those rows.",
  reasoning: "You're training a model, so rows without a salary can't contribute.",
  tradeoff: "You lose 340 of 5,000 rows.",
  operation: "drop_rows_with_nulls",
  operation_params: { column: "salary" },
  warning: null,
};

function renderCard(overrides: Partial<Parameters<typeof SuggestionCardView>[0]> = {}) {
  const props = {
    card: CARD,
    decision: null,
    onDecide: vi.fn(),
    reason: "",
    onReasonChange: vi.fn(),
    ...overrides,
  };
  render(<SuggestionCardView {...props} />);
  return props;
}

describe("SuggestionCardView", () => {
  it("shows all four sections of the card", () => {
    renderCard();
    expect(screen.getByText("340 rows have no salary.")).toBeInTheDocument();
    expect(screen.getByText("Drop those rows.")).toBeInTheDocument();
    expect(screen.getByText(/rows without a salary can't contribute/)).toBeInTheDocument();
    expect(screen.getByText("You lose 340 of 5,000 rows.")).toBeInTheDocument();
  });

  it("does not ask for a reason until the fix is rejected", () => {
    renderCard();
    expect(screen.queryByLabelText(/why are you declining/i)).not.toBeInTheDocument();
  });

  it("asks for a reason once rejected", () => {
    // FR-30's entry point. This box was built on the backend long before anything on the
    // page collected the text, so the feature existed and was unreachable.
    renderCard({ decision: "rejected" });
    expect(screen.getByLabelText(/why are you declining/i)).toBeInTheDocument();
  });

  it("says the reason is copied word for word", () => {
    renderCard({ decision: "rejected" });
    expect(screen.getByText(/exactly as written/i)).toBeInTheDocument();
  });

  it("reports what the user types", () => {
    const props = renderCard({ decision: "rejected" });
    fireEvent.change(screen.getByLabelText(/why are you declining/i), {
      target: { value: "Those are our founders' salaries." },
    });
    expect(props.onReasonChange).toHaveBeenCalledWith("card_1", "Those are our founders' salaries.");
  });

  it("reports approve and reject decisions", () => {
    const props = renderCard();
    fireEvent.click(screen.getByRole("button", { name: /^Approve/ }));
    expect(props.onDecide).toHaveBeenCalledWith("card_1", "approved");
    fireEvent.click(screen.getByRole("button", { name: /^Reject/ }));
    expect(props.onDecide).toHaveBeenCalledWith("card_1", "rejected");
  });

  it("names the column each button decides, and says which one was chosen (#101)", () => {
    renderCard({ decision: "rejected" });
    const column = CARD.column_name;
    expect(screen.getByRole("group", { name: `Decision for ${column}` })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: `Approve the fix for ${column}` })).toHaveAttribute("aria-pressed", "false");
    expect(screen.getByRole("button", { name: `Reject the fix for ${column}` })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows a target-safety warning as an adjustment, not advice", () => {
    renderCard({
      card: { ...CARD, warning: "'salary' is the column you're trying to predict." },
    });
    expect(screen.getByText(/adjusted for your prediction target/i)).toBeInTheDocument();
  });

  it("shows a blocked fix as blocked", () => {
    renderCard({
      card: { ...CARD, operation: null, warning: "Removing it would leave nothing to predict." },
    });
    expect(screen.getByText("Blocked")).toBeInTheDocument();
  });

  it("says a fix is left for the model, not that it is blocked", () => {
    renderCard({
      card: {
        ...CARD,
        operation: null,
        warning_kind: "left_to_model",
        warning: "This is left for the model. It fills each column's gaps itself.",
      },
    });
    expect(screen.getByText("Left for the model")).toBeInTheDocument();
    expect(screen.queryByText("Blocked")).not.toBeInTheDocument();
  });

  it("still reads a card saved before warning_kind existed", () => {
    renderCard({ card: { ...CARD, operation: null, warning: "Nothing to predict." } });
    expect(screen.getByText("Blocked")).toBeInTheDocument();
  });
});
