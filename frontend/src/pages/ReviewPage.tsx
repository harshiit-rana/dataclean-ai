import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { AgentStatusBar } from "@/components/review/AgentStatusBar";
import { ApprovalBar } from "@/components/review/ApprovalBar";
import { ColumnDetailPanel } from "@/components/review/ColumnDetailPanel";
import { DataTable } from "@/components/review/DataTable";
import { IssueCountSummary } from "@/components/review/IssueCountSummary";
import { SuggestionCardView } from "@/components/review/SuggestionCardView";
import { useProfileStream } from "@/hooks/useProfileStream";
import { ApiError, fetchPreview } from "@/api/client";
import type { ColumnIssue, PreviewResponse } from "@/api/types";

/** Page 2, PRD US-04/US-06: shows the uploaded data as a table (click a column header to
 * inspect what it holds and what was flagged in it), with the streamed suggestion cards
 * below for approve/reject. Submitting decisions (POST /api/resume) happens once that
 * endpoint is implemented -- see SuggestionCardView's docstring. */
export function ReviewPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const stream = useProfileStream(sessionId ?? null);
  const [decisions, setDecisions] = useState<Record<string, "approved" | "rejected">>({});
  // FR-30: the user's own words for why a fix was declined, carried through to the audit
  // report verbatim. Kept beside `decisions` rather than inside it so a reason typed, then
  // un-rejected, then re-rejected isn't lost.
  const [skipReasons, setSkipReasons] = useState<Record<string, string>>({});
  const [preview, setPreview] = useState<PreviewResponse | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    setPreviewError(null);
    fetchPreview(sessionId)
      .then((data) => {
        setPreview(data);
        setPreviewError(null);
      })
      .catch((err) => {
        setPreview(null);
        setPreviewError(
          err instanceof ApiError ? `${err.error}: ${err.detail}` : String(err?.message ?? err)
        );
      });
  }, [sessionId]);

  const issuesByColumn = useMemo(() => {
    const grouped: Record<string, ColumnIssue[]> = {};
    for (const issue of stream.issues) {
      (grouped[issue.column_name] ??= []).push(issue);
    }
    return grouped;
  }, [stream.issues]);

  const inProgressCardIds = useMemo(() => {
    const completed = new Set(stream.cards.map((c) => c.card_id));
    return Object.keys(stream.streamingCardText).filter((id) => !completed.has(id));
  }, [stream.cards, stream.streamingCardText]);

  const selectedColumnMeta = preview?.columns.find((c) => c.name === selectedColumn) ?? null;

  // Stable identities so the memoized DataTable and the suggestion cards don't re-render
  // on every streamed-text flush.
  const handleDecide = useCallback((cardId: string, decision: "approved" | "rejected") => {
    setDecisions((prev) => ({ ...prev, [cardId]: decision }));
  }, []);

  const handleReasonChange = useCallback((cardId: string, reason: string) => {
    setSkipReasons((prev) => ({ ...prev, [cardId]: reason }));
  }, []);

  const handleSelectColumn = useCallback((name: string) => {
    setSelectedColumn((cur) => (cur === name ? null : name));
  }, []);

  const decisionCounts = useMemo(() => {
    const values = stream.cards.map((c) => decisions[c.card_id]);
    return {
      approved: values.filter((d) => d === "approved").length,
      rejected: values.filter((d) => d === "rejected").length,
      undecided: values.filter((d) => d === undefined).length,
    };
  }, [stream.cards, decisions]);

  const handleSubmit = () => {
    if (!sessionId) return;
    // Only reasons for cards actually rejected -- a reason typed and then reversed would
    // otherwise show up in the report against a fix that was applied.
    const reasons = Object.fromEntries(
      Object.entries(skipReasons).filter(
        ([cardId, reason]) => decisions[cardId] === "rejected" && reason.trim()
      )
    );
    // The cleaning run streams on the Clean page, which starts it on mount.
    navigate(`/clean/${sessionId}`, { state: { decisions, skipReasons: reasons } });
  };

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6 p-8">
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

      {stream.status === "error" && <p className="text-sm text-destructive">{stream.error}</p>}

      {previewError && (
        <div className="rounded-md border border-destructive/50 p-3 text-sm text-destructive">
          <p className="font-medium">Couldn't load the data table</p>
          <p className="text-xs">{previewError}</p>
        </div>
      )}

      {preview && (
        <DataTable
          preview={preview}
          issuesByColumn={issuesByColumn}
          selectedColumn={selectedColumn}
          onSelectColumn={handleSelectColumn}
        />
      )}

      {preview && selectedColumnMeta && (
        <ColumnDetailPanel
          column={selectedColumnMeta}
          issues={issuesByColumn[selectedColumnMeta.name] ?? []}
          preview={preview}
          onClose={() => setSelectedColumn(null)}
        />
      )}

      <div className="flex flex-col gap-4">
        {stream.cards.map((card) => (
          <SuggestionCardView
            key={card.card_id}
            card={card}
            decision={decisions[card.card_id] ?? null}
            onDecide={handleDecide}
            reason={skipReasons[card.card_id] ?? ""}
            onReasonChange={handleReasonChange}
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

      <ApprovalBar
        total={stream.cards.length}
        approvedCount={decisionCounts.approved}
        rejectedCount={decisionCounts.rejected}
        undecidedCount={decisionCounts.undecided}
        submitting={false}
        disabled={stream.status !== "interrupted"}
        onSubmit={handleSubmit}
      />
    </div>
  );
}
