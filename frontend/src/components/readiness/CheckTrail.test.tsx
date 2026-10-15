import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { actionPath, CheckTrail } from "./CheckTrail";
import type { ReadinessAction, ReadinessCheck } from "@/api/types";

const COUNTS = { passed: 0, warning: 0, failed: 1, not_run: 0, total: 1 };

function check(overrides: Partial<ReadinessCheck>): ReadinessCheck {
  return {
    name: "leakage",
    label: "Columns that give away the answer",
    status: "failed",
    finding: "'leaky' still predicts 'outcome' almost perfectly.",
    evidence: "",
    remedy: "Drop it.",
    blocked_by: null,
    severity: "warning",
    actions: [],
    ...overrides,
  };
}

function show(checks: ReadinessCheck[]) {
  render(
    <MemoryRouter>
      <CheckTrail checks={checks} counts={COUNTS} sessionId="s1" />
    </MemoryRouter>
  );
}

describe("where an action leads", () => {
  const go = (action: Partial<ReadinessAction>) =>
    actionPath({ kind: "exclude_column", label: "x", column: null, ...action }, "s1");

  it("sends an exclusion to the model page with the column to leave out", () => {
    expect(go({ kind: "exclude_column", column: "leaky" })).toBe("/model/s1?exclude=leaky");
  });

  it("sends a grouped split to the model page with the column to group on", () => {
    expect(go({ kind: "use_group_split", column: "customer" })).toBe("/model/s1?split=group&group=customer");
  });

  it("sends choosing a target back to the start", () => {
    expect(go({ kind: "choose_target" })).toBe("/");
  });

  it("escapes a column name that would break the address", () => {
    expect(go({ kind: "exclude_column", column: "a&b c" })).toBe("/model/s1?exclude=a%26b%20c");
  });
});

describe("CheckTrail — the buttons", () => {
  it("renders a button per action that navigates there", () => {
    show([
      check({
        actions: [
          { kind: "exclude_column", label: "Leave 'leaky' out of the model", column: "leaky" },
          { kind: "exclude_column", label: "Leave 'other' out of the model", column: "other" },
        ],
      }),
    ]);
    fireDetails();
    const first = screen.getByRole("link", { name: "Leave 'leaky' out of the model" });
    expect(first).toHaveAttribute("href", "/model/s1?exclude=leaky");
    expect(screen.getByRole("link", { name: "Leave 'other' out of the model" })).toHaveAttribute(
      "href",
      "/model/s1?exclude=other"
    );
  });

  it("offers a grouped split on a repeated entity", () => {
    show([
      check({
        name: "repeated_entities",
        status: "warning",
        actions: [{ kind: "use_group_split", label: "Split by 'customer'", column: "customer" }],
      }),
    ]);
    expect(screen.getByRole("link", { name: "Split by 'customer'" })).toHaveAttribute(
      "href",
      "/model/s1?split=group&group=customer"
    );
  });

  it("offers choosing a target when it is missing", () => {
    show([
      check({
        name: "target_present",
        label: "The target is still in the data",
        actions: [{ kind: "choose_target", label: "Choose a column to predict", column: null }],
      }),
    ]);
    expect(screen.getByRole("link", { name: "Choose a column to predict" })).toHaveAttribute("href", "/");
  });

  it("shows no buttons for a check without actions, or for an older payload", () => {
    show([check({ actions: [] }), check({ name: "old", label: "Old", actions: undefined })]);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("shows no buttons on a check that passed", () => {
    show([
      check({
        status: "passed",
        actions: [{ kind: "exclude_column", label: "Leave it out", column: "leaky" }],
      }),
    ]);
    fireEvent_showPassed();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("keeps the buttons inside their own check", () => {
    show([
      check({ name: "a", label: "First", actions: [{ kind: "exclude_column", label: "Go A", column: "a" }] }),
      check({ name: "b", label: "Second" }),
    ]);
    const rows = screen.getAllByRole("listitem");
    expect(within(rows[0]).getByRole("link", { name: "Go A" })).toBeInTheDocument();
    expect(within(rows[1]).queryByRole("link")).not.toBeInTheDocument();
  });
});

// jsdom renders the content of a closed <details>, so the checks are in the document already; these
// exist to say so where a reader would otherwise wonder.
function fireDetails() {
  expect(screen.getByText(/Why this verdict/)).toBeInTheDocument();
}
function fireEvent_showPassed() {
  const button = screen.queryByRole("button", { name: /other check.* passed/ });
  if (button) button.click();
}
