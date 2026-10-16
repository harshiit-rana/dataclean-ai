import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { axeViolations } from "@/test/axe";
import { DataTable } from "./DataTable";
import type { PreviewResponse } from "@/api/types";

/** Found in the keyboard walkthrough (#101): the column headers opened their details on a click only. */

const PREVIEW = {
  session_id: "s1",
  columns: [
    { name: "spend", dtype: "float64", null_count: 2, unique_count: 40, is_critical: true, is_target: false },
    { name: "churned", dtype: "object", null_count: 0, unique_count: 2, is_critical: false, is_target: true },
  ],
  rows: [{ spend: 12.5, churned: "yes" }, { spend: null, churned: "no" }],
  total_rows: 2,
  total_columns: 2,
  preview_row_count: 2,
} as unknown as PreviewResponse;

function renderTable(selected: string | null = null, onSelect = vi.fn()) {
  return {
    onSelect,
    ...render(
      <main>
        <DataTable
          preview={PREVIEW}
          issuesByColumn={{ spend: [{ severity: "high" } as never] }}
          selectedColumn={selected}
          onSelectColumn={onSelect}
        />
      </main>
    ),
  };
}

describe("DataTable", () => {
  it("makes every column header a button that opens its details", () => {
    const { onSelect } = renderTable();
    const header = screen.getByRole("button", { name: /spend/ });
    header.focus();
    expect(header).toHaveFocus();
    fireEvent.click(header);
    expect(onSelect).toHaveBeenCalledWith("spend");
  });

  it("says which column is open", () => {
    renderTable("churned");
    expect(screen.getByRole("button", { name: /churned/ })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /spend/ })).toHaveAttribute("aria-pressed", "false");
  });

  it("names a critical column in words, not only with a star", () => {
    renderTable();
    expect(screen.getByRole("button", { name: /spend.*critical/ })).toBeInTheDocument();
  });

  it("lets the rows be scrolled from the keyboard", () => {
    renderTable();
    expect(screen.getByRole("region", { name: "Data preview" })).toHaveAttribute("tabindex", "0");
  });

  it("has no accessibility violations", async () => {
    const { container } = renderTable("spend");
    expect(await axeViolations(container)).toEqual([]);
  });
});
