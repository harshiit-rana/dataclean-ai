import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ChartCard } from "./ChartCard";

const COLUMNS = [
  { key: "label", label: "Answer" },
  { key: "recall", label: "Share found", numeric: true },
];
const ROWS = [
  { label: "No", recall: 0.83 },
  { label: "Yes", recall: 0.71 },
];

describe("ChartCard", () => {
  it("shows the chart by default", () => {
    render(
      <ChartCard title="Per class" tableColumns={COLUMNS} tableRows={ROWS}>
        <div>the chart</div>
      </ChartCard>
    );
    expect(screen.getByText("the chart")).toBeInTheDocument();
  });

  it("reveals every value as a table", () => {
    // Not a nicety. Three of the categorical slots sit below 3:1 against the light surface,
    // and a tooltip is a way to inspect a value, not to read one. The table is where the
    // numbers stay reachable without hover, colour vision, or a mouse.
    render(
      <ChartCard title="Per class" tableColumns={COLUMNS} tableRows={ROWS}>
        <div>the chart</div>
      </ChartCard>
    );

    fireEvent.click(screen.getByRole("button", { name: /show table/i }));

    expect(screen.queryByText("the chart")).not.toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Answer" })).toBeInTheDocument();
    expect(screen.getByText("0.83")).toBeInTheDocument();
    expect(screen.getByText("0.71")).toBeInTheDocument();
  });

  it("names its toggle after its chart, so a page of them can be told apart (#101)", () => {
    render(
      <ChartCard title="Per class" tableColumns={COLUMNS} tableRows={ROWS}>
        <div>the chart</div>
      </ChartCard>
    );
    fireEvent.click(screen.getByRole("button", { name: "Show table for Per class" }));
    expect(screen.getByRole("button", { name: "Show chart for Per class" })).toBeInTheDocument();
  });

  it("goes back to the chart", () => {
    render(
      <ChartCard title="Per class" tableColumns={COLUMNS} tableRows={ROWS}>
        <div>the chart</div>
      </ChartCard>
    );
    fireEvent.click(screen.getByRole("button", { name: /show table/i }));
    fireEvent.click(screen.getByRole("button", { name: /show chart/i }));
    expect(screen.getByText("the chart")).toBeInTheDocument();
  });

  it("offers no toggle when there is no table to show", () => {
    render(
      <ChartCard title="Per class">
        <div>the chart</div>
      </ChartCard>
    );
    expect(screen.queryByRole("button", { name: /show table/i })).not.toBeInTheDocument();
  });

  it("renders a missing value as a dash rather than blank", () => {
    render(
      <ChartCard
        title="Per class"
        tableColumns={COLUMNS}
        tableRows={[{ label: "No", recall: null }]}
      >
        <div>the chart</div>
      </ChartCard>
    );
    fireEvent.click(screen.getByRole("button", { name: /show table/i }));
    expect(screen.getByText("—")).toBeInTheDocument();
  });

  it("shows the reading instruction and footnote", () => {
    render(
      <ChartCard title="Per class" description="How to read it." footnote="A caveat.">
        <div>the chart</div>
      </ChartCard>
    );
    expect(screen.getByText("How to read it.")).toBeInTheDocument();
    expect(screen.getByText("A caveat.")).toBeInTheDocument();
  });
});
