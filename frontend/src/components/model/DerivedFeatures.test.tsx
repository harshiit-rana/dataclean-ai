import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DerivedFeatures } from "./DerivedFeatures";
import type { FeatureRecipe } from "@/api/types";

const DATE: FeatureRecipe = {
  id: "date_parts:signup_date",
  kind: "date_parts",
  column: "signup_date",
  outputs: ["signup_date__year", "signup_date__weekday", "signup_date__days"],
  reason: "'signup_date' holds dates spanning 897 days, which a model can't use as a category.",
  evidence: { span_days: 897 },
};

const POOLED: FeatureRecipe = {
  id: "frequent_categories:city",
  kind: "frequent_categories",
  column: "city",
  outputs: [],
  reason: "'city' has 77 different values, but its 20 most common cover 65% of the rows.",
  evidence: { distinct: 77 },
};

describe("DerivedFeatures", () => {
  it("renders nothing when there is nothing to derive", () => {
    const { container } = render(
      <DerivedFeatures recipes={[]} chosen={[]} onChange={vi.fn()} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the reason, not just the column name", () => {
    render(<DerivedFeatures recipes={[DATE]} chosen={[DATE.id]} onChange={vi.fn()} />);
    expect(screen.getByText(/spanning 897 days/)).toBeInTheDocument();
  });

  it("names the columns it would produce", () => {
    render(<DerivedFeatures recipes={[DATE]} chosen={[DATE.id]} onChange={vi.fn()} />);
    expect(screen.getByText("signup_date__weekday")).toBeInTheDocument();
  });

  it("can be unticked, like every other proposal in this app", () => {
    const onChange = vi.fn();
    render(<DerivedFeatures recipes={[DATE]} chosen={[DATE.id]} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenCalledWith([]);
  });

  it("can be ticked back on", () => {
    const onChange = vi.fn();
    render(<DerivedFeatures recipes={[DATE]} chosen={[]} onChange={onChange} />);
    fireEvent.click(screen.getByRole("checkbox"));
    expect(onChange).toHaveBeenCalledWith([DATE.id]);
  });

  it("does not promise new columns for a recipe that produces none", () => {
    /* frequent_categories changes how a column is encoded rather than adding one, and
       listing a phantom column name would be a small lie about what happens. */
    render(<DerivedFeatures recipes={[POOLED]} chosen={[POOLED.id]} onChange={vi.fn()} />);
    expect(screen.getByText(/20 most common cover 65%/)).toBeInTheDocument();
    expect(screen.queryByText(/city__/)).not.toBeInTheDocument();
  });
});
