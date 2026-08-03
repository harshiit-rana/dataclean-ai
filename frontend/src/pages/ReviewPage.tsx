import { useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { AgentStatusBar } from "@/components/review/AgentStatusBar";
import { IssueCountSummary } from "@/components/review/IssueCountSummary";
import { SuggestionCardView } from "@/components/review/SuggestionCardView";
import { useProfileStream } from "@/hooks/useProfileStream";

/** Page 2, PRD US-04/US-06: streams the Profiling + Explanation agents' progress, then lets
 * the user approve/reject each suggestion card. Submitting decisions (POST /api/resume)
 * happens once that endpoint is implemented -- see SuggestionCardView's docstring. */
export function ReviewPage() {
  const { sessionId } = useParams();
  const stream = useProfileStream(sessionId ?? null);
  const [decisions, setDecisions] = useState<Record<string, "approved" | "rejected">>({});

  const inProgressCardIds = useMemo(() => {
    const completed = new Set(stream.cards.map((c) => c.card_id));
    return Object.keys(stream.streamingCardText).filter((id) => !completed.has(id));
  }, [stream.cards, stream.streamingCardText]);

  const handleDecide = (cardId: string, decision: "approved" | "rejected") => {
    setDecisions((prev) => ({ ...prev, [cardId]: decision }));
  };

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Review suggestions</h1>
        <p className="text-muted-foreground">
          Approve or reject each fix. Nothing changes in your data until you approve it.
        </p>
      </div>

      <div className="flex items-center justify-between">
        <AgentStatusBar
          status={stream.status}
          currentAgent={stream.currentAgent}
          statusMessage={stream.statusMessage}
        />
        <IssueCountSummary counts={stream.issueCounts} />
      </div>

      {stream.status === "error" && (
        <p className="text-sm text-destructive">{stream.error}</p>
      )}

      <div className="flex flex-col gap-4">
        {stream.cards.map((card) => (
          <SuggestionCardView
            key={card.card_id}
            card={card}
            decision={decisions[card.card_id] ?? null}
            onDecide={handleDecide}
          />
        ))}

        {inProgressCardIds.map((cardId) => (
          <div
            key={cardId}
            className="rounded-lg border border-dashed border-border p-4 text-sm text-muted-foreground"
          >
            <p className="mb-1 font-medium">Writing suggestion...</p>
            <p className="whitespace-pre-wrap">{stream.streamingCardText[cardId]}</p>
          </div>
        ))}
      </div>

      {stream.status === "interrupted" && stream.cards.length === 0 && (
        <p className="text-sm text-muted-foreground">No issues found -- your data looks clean.</p>
      )}
    </div>
  );
}
