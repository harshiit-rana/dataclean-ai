export type Decision = "approved" | "rejected";

/** Keeps only the reasons that belong to cards actually rejected, and drops blank ones.
 *
 * Extracted from the review page so it can be tested directly: it guards a real mistake.
 * Reasons are typed as cards are rejected, but a card can be un-rejected afterwards, and its
 * text is deliberately kept in case the user changes their mind again. Sending that text
 * anyway would print an explanation in the audit report against a fix that was applied --
 * words attributed to the user about a decision they didn't make.
 */
export function reasonsForRejectedCards(
  decisions: Record<string, Decision>,
  skipReasons: Record<string, string>
): Record<string, string> {
  return Object.fromEntries(
    Object.entries(skipReasons).filter(
      ([cardId, reason]) => decisions[cardId] === "rejected" && reason.trim()
    )
  );
}
