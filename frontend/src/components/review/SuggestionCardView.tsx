import { memo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { SuggestionCard } from "@/api/types";

interface SuggestionCardViewProps {
  card: SuggestionCard;
  decision: "approved" | "rejected" | null;
  onDecide: (cardId: string, decision: "approved" | "rejected") => void;
}

/** PRD §12.2/US-04: renders one suggestion card (Found / Proposed Action / Reasoning /
 * Tradeoff) with per-card approve/reject controls. Decisions are held in ReviewPage state;
 * submitting them to POST /api/resume is wired up once that endpoint is implemented
 * (currently a 501 stub -- see backend/app/api/routes_resume.py). */
export const SuggestionCardView = memo(function SuggestionCardView({
  card,
  decision,
  onDecide,
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
      </CardContent>
    </Card>
  );
});
