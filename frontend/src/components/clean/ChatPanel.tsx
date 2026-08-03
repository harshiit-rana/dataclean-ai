import { useState, type FormEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ApiError, sendChatMessage, type ActionLogEntry } from "@/api/client";

interface ChatPanelProps {
  sessionId: string;
  onActionsChanged: (actions: ActionLogEntry[]) => void;
}

interface Turn {
  role: "user" | "assistant";
  text: string;
  applied?: boolean;
}

/** PRD FR-37..40: further cleaning in plain English, after the suggestion-card pass.
 *
 * Instructions are mapped onto the same whitelisted operations the rest of the app uses, so
 * anything that can't be expressed that way is declined with an explanation rather than
 * approximated. */
export function ChatPanel({ sessionId, onActionsChanged }: ChatPanelProps) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault();
    const instruction = message.trim();
    if (!instruction || busy) return;

    setTurns((prev) => [...prev, { role: "user", text: instruction }]);
    setMessage("");
    setBusy(true);

    try {
      const result = await sendChatMessage(sessionId, instruction);
      setTurns((prev) => [
        ...prev,
        { role: "assistant", text: result.message, applied: result.applied },
      ]);
      if (result.applied) onActionsChanged(result.actions);
    } catch (err) {
      setTurns((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err),
          applied: false,
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Anything else?</CardTitle>
        <p className="text-sm text-muted-foreground">
          Describe another change in plain English — for example &ldquo;drop rows where salary
          is under 30000&rdquo; or &ldquo;rename hire_date to start_date&rdquo;.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {turns.length > 0 && (
          <div className="flex flex-col gap-2">
            {turns.map((turn, idx) => (
              <div
                key={idx}
                className={cn(
                  "rounded-md px-3 py-2 text-sm",
                  turn.role === "user"
                    ? "self-end bg-primary text-primary-foreground"
                    : turn.applied === false
                      ? "bg-muted text-muted-foreground"
                      : "bg-muted"
                )}
              >
                {turn.text}
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="e.g. drop rows where age is negative"
            disabled={busy}
          />
          <Button type="submit" disabled={busy || !message.trim()}>
            {busy ? "Working..." : "Send"}
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
