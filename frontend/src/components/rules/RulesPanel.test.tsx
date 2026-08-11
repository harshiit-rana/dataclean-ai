import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { RulesPanel } from "./RulesPanel";
import type { BusinessRule, ReadinessCheck } from "@/api/types";

function rule(overrides: Partial<BusinessRule> = {}): BusinessRule {
  return {
    kind: "range",
    column: "age",
    label: "",
    minimum: 0,
    maximum: null,
    other_column: null,
    comparison: null,
    multiplier_column: null,
    result_column: null,
    confidence: 0.9,
    rationale: "'age' holds 3 negative values, and nothing this column could measure goes below zero.",
    name: "rule_range_age",
    sentence: "'age' is never below 0",
    expression: "age >= 0",
    ...overrides,
  };
}

const BROKEN: ReadinessCheck = {
  name: "rule_range_age",
  label: "Rule: 'age' is never below 0",
  status: "failed",
  finding: "3 of 300 rows (1.0%) break age >= 0.",
  evidence: "These are not unusual values -- they are impossible ones.",
  remedy: "Decide whether these rows are wrong or the rule is.",
  blocked_by: null,
  severity: "warning",
};

function renderPanel(props: Partial<Parameters<typeof RulesPanel>[0]> = {}) {
  const onChange = vi.fn();
  render(
    <RulesPanel rules={[]} proposed={[]} checks={[]} onChange={onChange} {...props} />
  );
  return onChange;
}

describe("RulesPanel", () => {
  it("shows a proposal with the evidence behind it, not just the guess", () => {
    renderPanel({ proposed: [rule()] });
    expect(screen.getByText(/'age' is never below 0/)).toBeInTheDocument();
    expect(screen.getByText(/holds 3 negative values/)).toBeInTheDocument();
  });

  it("says plainly that proposals are not in force", () => {
    renderPanel({ proposed: [rule()] });
    expect(screen.getByText(/None of these are in force yet/)).toBeInTheDocument();
  });

  it("does not apply a proposal until it is approved", () => {
    const onChange = renderPanel({ proposed: [rule()] });
    expect(onChange).not.toHaveBeenCalled();
  });

  it("approving hands back the complete new set", () => {
    const existing = rule({ name: "rule_range_quantity", column: "quantity" });
    const onChange = renderPanel({ rules: [existing], proposed: [rule()] });
    fireEvent.click(screen.getByRole("button", { name: /Use this rule/ }));
    expect(onChange).toHaveBeenCalledWith([existing, rule()]);
  });

  it("dismissing a proposal removes it without saving anything", () => {
    const onChange = renderPanel({ proposed: [rule()] });
    fireEvent.click(screen.getByRole("button", { name: /Not a rule/ }));
    expect(screen.queryByText(/'age' is never below 0/)).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("shows what an approved rule currently finds", () => {
    renderPanel({ rules: [rule()], checks: [BROKEN] });
    expect(screen.getByText(/3 of 300 rows/)).toBeInTheDocument();
    expect(screen.getByText(/1 broken/)).toBeInTheDocument();
  });

  it("removing a rule hands back the set without it", () => {
    const onChange = renderPanel({ rules: [rule()], checks: [BROKEN] });
    fireEvent.click(screen.getByRole("button", { name: /Remove/ }));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("renders nothing at all when there is nothing to say", () => {
    const { container } = render(
      <RulesPanel rules={[]} proposed={[]} checks={[]} onChange={vi.fn()} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("disables the decisions while a save is in flight", () => {
    renderPanel({ proposed: [rule()], busy: true });
    expect(screen.getByRole("button", { name: /Use this rule/ })).toBeDisabled();
  });
});
