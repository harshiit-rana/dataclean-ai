"""One shape, and one sentence, for every place the app declines to judge.

The verdict, the drift check, the check against a new file's real answers and the cut-off each decline
to say anything when the evidence is thin, and each used to say so differently. They now share a block:

    {"status": "not_judged", "what", "have", "need", "unit", "sentence", "because"}

and a sentence pattern -- "Not enough {what} to judge: {have} {unit}, at least {need} needed." -- that
the page shows in one component. When the obstacle is not a count (a model with no probabilities has
nothing to put a cut-off on) `have` and `need` are None and the sentence is "Cannot judge: {because}."
"""

from __future__ import annotations


def not_judged(
    what: str | None = None,
    *,
    have: int | None = None,
    need: int | None = None,
    unit: str = "rows",
    because: str | None = None,
) -> dict:
    """The block for a state the app cannot judge. Counts when there are counts, a reason when not."""
    if what and have is not None and need is not None:
        sentence = f"Not enough {what} to judge: {have:,} {unit}, at least {need:,} needed."
    else:
        sentence = f"Cannot judge: {(because or 'there is not enough evidence').rstrip('.')}."
    return {
        "status": "not_judged",
        "what": what,
        "have": have,
        "need": need,
        "unit": unit,
        "sentence": sentence,
        "because": because,
    }
