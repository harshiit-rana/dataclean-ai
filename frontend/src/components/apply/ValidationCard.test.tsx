import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ValidationCard } from "./ValidationCard";
import type { FileValidation } from "@/api/types";

const BASE: FileValidation = {
  schema_version: 1,
  blocking: false,
  missing_columns: [],
  unexpected_columns: [],
  findings: [],
  rows_with_findings: 0,
  rows_checked: 40,
};

describe("ValidationCard (#77)", () => {
  it("says a clean file matches", () => {
    render(<ValidationCard validation={BASE} />);
    expect(screen.getByText("The file matches what the model was trained on")).toBeInTheDocument();
  });

  it("lists each finding with its rows and example values", () => {
    render(
      <ValidationCard
        validation={{
          ...BASE,
          unexpected_columns: ["notes"],
          rows_with_findings: 7,
          findings: [
            { column: "plan", finding: "unseen_category", severity: "warning", count: 7, rows: [2, 4, 9, 11, 12],
              message: "a value not seen in training", values: ["gold"] },
          ],
        }}
      />
    );
    expect(screen.getByText("7 of 40 rows have something to check")).toBeInTheDocument();
    expect(screen.getByText("Values not seen in training")).toBeInTheDocument();
    expect(screen.getByText(/rows 2, 4, 9, 11, 12/)).toBeInTheDocument();
    expect(screen.getByText("“gold”")).toBeInTheDocument();
    expect(screen.getByText(/Not used by the model, so ignored: notes/)).toBeInTheDocument();
    expect(screen.getByText(/dataclean_validation column/)).toBeInTheDocument();
  });
});
