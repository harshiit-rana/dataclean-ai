import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { ApiError, undoAction, type ActionLogEntry } from "@/api/client";
import { OPERATION_LABELS } from "@/lib/operations";

interface AppliedActionsProps {
  sessionId: string;
  actions: ActionLogEntry[];
  onChange: (actions: ActionLogEntry[]) => void;
}

/** The persisted action log with per-action undo (PRD US-10). Shown once the cleaning run
 * finishes, replacing the live progress view. */
export function AppliedActions({ sessionId, actions, onChange }: AppliedActionsProps) {
  const [undoing, setUndoing] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleUndo = async (actionId: string) => {
    setUndoing(actionId);
    setError(null);
    try {
      const result = await undoAction(sessionId, actionId);
      onChange(result.actions);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setUndoing(null);
    }
  };

  const applied = actions.filter((a) => a.status === "applied");
  const undone = actions.filter((a) => a.status === "undone");
  // Surfaced rather than hidden inside the "N skipped" count: a fix that was approved and
  // then didn't run is something the user should know about, and the issue it addressed is
  // still in their data.
  const failed = actions.filter((a) => a.status === "rejected" && a.failure_reason);

  return (
    <div className="flex flex-col gap-3">
      {error && <p className="text-sm text-destructive">{error}</p>}

      {applied.map((action) => (
        <Card key={action.action_id}>
          <CardContent className="flex items-center justify-between gap-3 p-4 text-sm">
            <div>
              <span className="font-medium">{action.column_name}</span>
              <span className="text-muted-foreground">
                {" "}
                — {OPERATION_LABELS[action.operation] ?? action.operation}
              </span>
            </div>
            <Button
              size="sm"
              variant="outline"
              disabled={undoing !== null}
              onClick={() => handleUndo(action.action_id)}
            >
              {undoing === action.action_id ? "Undoing..." : "Undo"}
            </Button>
          </CardContent>
        </Card>
      ))}

      {undone.map((action) => (
        <Card key={action.action_id} className="opacity-60">
          <CardContent className="flex items-center justify-between gap-3 p-4 text-sm">
            <div>
              <span className="font-medium">{action.column_name}</span>
              <span className="text-muted-foreground">
                {" "}
                — {OPERATION_LABELS[action.operation] ?? action.operation}
              </span>
            </div>
            <Badge>undone</Badge>
          </CardContent>
        </Card>
      ))}

      {failed.map((action) => (
        <Card key={action.action_id} className="border-amber-500/40 bg-amber-500/5">
          <CardContent className="flex flex-col gap-1 p-4 text-sm">
            <div>
              <span className="font-medium">{action.column_name}</span>
              <span className="text-muted-foreground">
                {" "}
                — {OPERATION_LABELS[action.operation] ?? action.operation}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">
              Couldn't be applied: {action.failure_reason}
            </p>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
