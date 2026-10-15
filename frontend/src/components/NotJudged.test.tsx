import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { NotJudged } from "./NotJudged";
import type { NotJudgedBlock } from "@/api/types";

const block = (sentence: string): NotJudgedBlock => ({
  status: "not_judged", what: "x", have: 1, need: 2, unit: "rows", sentence, because: null,
});

describe("NotJudged", () => {
  it("leads with the server's sentence, in the sky treatment that is not a failure colour", () => {
    render(<NotJudged block={block("Not enough new rows to judge: 10 rows, at least 30 needed.")} />);
    const box = screen.getByRole("status", { name: "Cannot judge" });
    expect(box).toHaveClass("border-sky-500/40");
    expect(box).not.toHaveClass("border-destructive/40");
    expect(screen.getByText("Not enough new rows to judge: 10 rows, at least 30 needed.")).toHaveClass("font-medium");
  });

  it("puts whatever the surface adds after the sentence", () => {
    render(<NotJudged block={block("Cannot judge: no probabilities.")}>The usual 0.50 is used.</NotJudged>);
    const box = screen.getByRole("status");
    expect(box.textContent).toBe("Cannot judge: no probabilities.The usual 0.50 is used.");
  });
});
