import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { BeforeYouRely, LEADING_CAUTIONS } from "./BeforeYouRely";
import type { ModelCaution } from "@/api/types";

const caution = (id: string, severity: ModelCaution["severity"] = "medium"): ModelCaution => ({
  id,
  severity,
  text: `Text of ${id}.`,
});

describe("BeforeYouRely", () => {
  it("shows the top three and counts the rest", () => {
    const cautions = [caution("a", "high"), caution("b"), caution("c"), caution("d", "low"), caution("e", "low")];
    render(<BeforeYouRely cautions={cautions} />);
    expect(LEADING_CAUTIONS).toBe(3);
    expect(screen.getByText(/Text of a\./)).toBeInTheDocument();
    expect(screen.getByText(/Text of c\./)).toBeInTheDocument();
    expect(screen.queryByText(/Text of d\./)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Show all 5 (2 more)" })).toBeInTheDocument();
  });

  it("shows all of them on request, and can fold them away again", () => {
    render(<BeforeYouRely cautions={[caution("a"), caution("b"), caution("c"), caution("d", "low")]} />);
    fireEvent.click(screen.getByRole("button", { name: /Show all 4/ }));
    expect(screen.getByText(/Text of d\./)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show fewer" }));
    expect(screen.queryByText(/Text of d\./)).not.toBeInTheDocument();
  });

  it("keeps the server's order and labels how much each weighs", () => {
    render(<BeforeYouRely cautions={[caution("a", "high"), caution("b", "low")]} />);
    const items = screen.getAllByRole("listitem");
    expect(items[0]).toHaveTextContent("Matters most");
    expect(items[0]).toHaveTextContent("Text of a.");
    expect(items[1]).toHaveTextContent("Context");
  });

  it("has no button when there is nothing hidden", () => {
    render(<BeforeYouRely cautions={[caution("a"), caution("b")]} />);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("renders nothing without cautions", () => {
    const { container } = render(<BeforeYouRely cautions={[]} />);
    expect(container).toBeEmptyDOMElement();
    expect(render(<BeforeYouRely cautions={undefined} />).container).toBeEmptyDOMElement();
  });
});
