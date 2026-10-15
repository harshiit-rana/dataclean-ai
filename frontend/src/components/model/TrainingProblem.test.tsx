import { fireEvent, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { TrainingProblem } from "./TrainingProblem";
import type { ModelPlan } from "@/api/types";

function plan(dropped: { column: string; reason: string }[]): ModelPlan {
  return { features: { numeric: [], categorical: [], dropped } } as unknown as ModelPlan;
}

function show(code: string | null, detail = "The server's explanation.", dropped: { column: string; reason: string }[] = []) {
  const onOpenSetup = vi.fn();
  render(
    <MemoryRouter>
      <TrainingProblem code={code} detail={detail} plan={plan(dropped)} sessionId="s1" onOpenSetup={onOpenSetup} />
    </MemoryRouter>
  );
  return onOpenSetup;
}

describe("TrainingProblem", () => {
  it("links a blocked run to the failing readiness checks", () => {
    show("data_blocked");
    expect(screen.getByText("This data can't support a model yet")).toBeInTheDocument();
    expect(screen.getByText("The server's explanation.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the failing readiness checks" })).toHaveAttribute("href", "/clean/s1");
  });

  it("explains too few labelled rows and links to the readiness checks", () => {
    show("too_few_labelled_rows");
    expect(screen.getByText("Too few rows have an answer")).toBeInTheDocument();
    expect(screen.getByText(/Add more labelled rows/)).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "See the readiness checks" })).toHaveAttribute("href", "/clean/s1");
  });

  it("opens the column chooser, with the reason each column was left out, when none are usable", () => {
    const onOpenSetup = show("no_usable_columns", "There are no usable columns left to predict from.", [
      { column: "customer_id", reason: "Every value is different, so it is an identifier." },
      { column: "notes", reason: "Free text." },
    ]);
    expect(screen.getByText("No columns are left to learn from")).toBeInTheDocument();
    expect(screen.getByText("customer_id")).toBeInTheDocument();
    expect(screen.getByText(/Every value is different, so it is an identifier\./)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Open the column chooser" }));
    expect(onOpenSetup).toHaveBeenCalledTimes(1);
  });

  it("says how many dropped columns it is not listing", () => {
    const many = Array.from({ length: 11 }, (_, i) => ({ column: `c${i}`, reason: "Constant." }));
    show("no_usable_columns", "None.", many);
    expect(screen.getByText("and 3 more in the column chooser.")).toBeInTheDocument();
    expect(screen.queryByText("c8")).not.toBeInTheDocument();
  });

  it.each(["target_missing", "target_unusable"])("sends %s back to choose another column to predict", (code) => {
    show(code);
    expect(screen.getByRole("link", { name: /choose a different column to predict/ })).toHaveAttribute("href", "/");
  });

  it("offers to change the split when the split cannot be made", () => {
    const onOpenSetup = show("split_unusable");
    expect(screen.getByText("The rows can't be divided as asked")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Change the split" }));
    expect(onOpenSetup).toHaveBeenCalledTimes(1);
  });

  it("explains a file that is too big for the processing limit and offers no button (#62)", () => {
    show("too_large", "200,000 training rows is more than can be trained on within the processing limit.");
    expect(screen.getByText("This file is too big to train on here")).toBeInTheDocument();
    expect(screen.getByText(/Train on a smaller sample of the file, or ask whoever runs this server/)).toBeInTheDocument();
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("offers to change the columns or models when none trained", () => {
    const onOpenSetup = show("no_model_trained");
    fireEvent.click(screen.getByRole("button", { name: "Change the columns or models" }));
    expect(onOpenSetup).toHaveBeenCalledTimes(1);
  });

  it("falls back to the plain message for a code it does not know, and for none", () => {
    for (const code of ["something_new", "training_error", null]) {
      const { unmount } = render(
        <MemoryRouter>
          <TrainingProblem code={code} detail="Raw server message." plan={null} sessionId="s1" onOpenSetup={vi.fn()} />
        </MemoryRouter>
      );
      expect(screen.getByText("Training didn't run")).toBeInTheDocument();
      expect(screen.getByText("Raw server message.")).toBeInTheDocument();
      expect(screen.queryByRole("button")).not.toBeInTheDocument();
      expect(screen.queryByRole("link")).not.toBeInTheDocument();
      unmount();
    }
  });

  it("copes with no plan loaded", () => {
    render(
      <MemoryRouter>
        <TrainingProblem code="no_usable_columns" detail="None." plan={null} sessionId="s1" onOpenSetup={vi.fn()} />
      </MemoryRouter>
    );
    expect(screen.getByRole("button", { name: "Open the column chooser" })).toBeInTheDocument();
    expect(screen.queryByText("Why each column was left out:")).not.toBeInTheDocument();
  });
});
