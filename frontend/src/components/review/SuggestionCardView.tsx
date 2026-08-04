import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { SuggestionCard } from "@/api/types";

interface SuggestionCardViewProps {
  card: SuggestionCard;
  decision: "approved" | "rejected" | null;
  onDecide: (cardId: string, decision: "approved" | "rejected") => void;
  /** Why this fix was rejected, in the user's own words (PRD FR-30). */
  reason: string;
  onReasonChange: (cardId: string, reason: string) => void;
}

/** PRD §12.2/US-04: renders one suggestion card (Found / Proposed Action / Reasoning /
 * Tradeoff) with per-card approve/reject controls, and — once rejected — a box for the
 * reason. Decisions and reasons are held in ReviewPage state and submitted together to
 * POST /api/resume. */
export const SuggestionCardView = memo(function SuggestionCardView({
  card,
  decision,
  onDecide,
  reason,
  onReasonChange,
}: SuggestionCardViewProps) {
  return (
    <Card className={cn(decision === "rejected" && "opacity-60")}>
      <CardHeader className="flex-row items-center justify-between">
        <CardTitle className="text-base">{card.column_name}</CardTitle>
        {decision && <Badge variant={decision === "approved" ? "low" : "default"}>{decision}</Badge>}
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        <div>
          <p className="font-medium text-muted-foreground">Found</p>
          <p>{card.found}</p>
        </div>
        <div>
          <p className="font-medium text-muted-foreground">Proposed action</p>
          <p>{card.proposed_action}</p>
        </div>
        <div>
          <p className="font-medium text-muted-foreground">Reasoning</p>
          <p>{card.reasoning}</p>
        </div>
        <div>
          <p className="font-medium text-muted-foreground">Tradeoff</p>
          <p>{card.tradeoff}</p>
        </div>

        {/* Written by the app, not the model: this fix was corrected or blocked because of
            the prediction target. Styled apart from the model's prose so it reads as a fact
            about what will run, not as more advice. */}
        {card.warning && (
          <div className="rounded-md border border-amber-500/40 bg-amber-500/10 p-3">
            <p className="font-medium text-amber-700 dark:text-amber-500">
              {card.operation ? "Adjusted for your prediction target" : "Blocked"}
            </p>
            <p className="mt-1 text-muted-foreground">{card.warning}</p>
          </div>
        )}

        <div className="mt-2 flex gap-2">
          <Button
            size="sm"
            variant={decision === "approved" ? "default" : "outline"}
            onClick={() => onDecide(card.card_id, "approved")}
          >
            Approve
          </Button>
          <Button
            size="sm"
            variant={decision === "rejected" ? "destructive" : "outline"}
            onClick={() => onDecide(card.card_id, "rejected")}
          >
            Reject
          </Button>
        </div>

        {/* FR-30: whatever is typed here is copied into the audit report word for word and
            is never sent to the model. Optional -- a declined fix is still recorded without
            one, just without the explanation. */}
        {decision === "rejected" && (
          <div className="flex flex-col gap-1.5">
            <label
              htmlFor={`reason-${card.card_id}`}
              className="text-sm font-medium text-muted-foreground"
            >
              Why are you declining this? (optional)
            </label>
            <Textarea
              id={`reason-${card.card_id}`}
              rows={2}
              placeholder="e.g. those outliers are real -- they're our enterprise accounts"
              value={reason}
              onChange={(e) => onReasonChange(card.card_id, e.target.value)}
            />
            <p className="text-xs text-muted-foreground">
              Copied into your audit report exactly as written.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
});
