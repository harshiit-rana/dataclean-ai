import { useParams } from "react-router-dom";

/** Page 3, PRD §12.3/§13.1 -- STUB. Will show the action log as fixes are applied and host
 * the conversational follow-up chat (FR-37..40), once cleaning_agent and
 * conversational_loop are implemented and POST /api/resume + /api/chat are real. */
export function CleanPage() {
  const { sessionId } = useParams<{ sessionId: string }>();

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4 p-8">
      <h1 className="text-2xl font-semibold">Cleaning</h1>
      <p className="text-muted-foreground">
        Coming soon -- this page will show each fix being applied to session {sessionId} and
        let you give follow-up instructions in plain English.
      </p>
    </div>
  );
}
