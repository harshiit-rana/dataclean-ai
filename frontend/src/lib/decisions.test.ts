import { describe, expect, it } from "vitest";
import { reasonsForRejectedCards } from "./decisions";

describe("reasonsForRejectedCards", () => {
  it("keeps the reason for a rejected card", () => {
    const result = reasonsForRejectedCards(
      { c1: "rejected" },
      { c1: "Those are our enterprise accounts, not errors." }
    );
    expect(result).toEqual({ c1: "Those are our enterprise accounts, not errors." });
  });

  it("drops a reason for a card that ended up approved", () => {
    // The real mistake this guards: reasons are typed while a card is rejected, and the text
    // is kept if it's un-rejected in case the user changes their mind again. Sending it
    // anyway would print the user's words in the report against a fix that was applied.
    const result = reasonsForRejectedCards(
      { c1: "approved" },
      { c1: "I changed my mind about this one." }
    );
    expect(result).toEqual({});
  });

  it("drops a reason for a card with no decision at all", () => {
    expect(reasonsForRejectedCards({}, { c1: "typed then abandoned" })).toEqual({});
  });

  it("drops blank and whitespace-only reasons", () => {
    const result = reasonsForRejectedCards(
      { c1: "rejected", c2: "rejected" },
      { c1: "   ", c2: "" }
    );
    expect(result).toEqual({});
  });

  it("keeps the exact text, including leading and trailing space", () => {
    // FR-30: verbatim. Trimming for the emptiness check must not trim what gets sent.
    const result = reasonsForRejectedCards({ c1: "rejected" }, { c1: "  keep me as typed  " });
    expect(result.c1).toBe("  keep me as typed  ");
  });

  it("handles a mix across several cards", () => {
    const result = reasonsForRejectedCards(
      { c1: "rejected", c2: "approved", c3: "rejected" },
      { c1: "reason one", c2: "stale", c3: "reason three" }
    );
    expect(result).toEqual({ c1: "reason one", c3: "reason three" });
  });
});
