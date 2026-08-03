import { useParams } from "react-router-dom";

/** Page 4, PRD §8.3/FR-41..44 -- STUB. Will preview the audit report and offer CSV + PDF
 * downloads via GET /api/export/{session_id}/{csv|pdf}, once audit_agent is implemented. */
export function ExportPage() {
  const { sessionId } = useParams<{ sessionId: string }>();

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-4 p-8">
      <h1 className="text-2xl font-semibold">Export</h1>
      <p className="text-muted-foreground">
        Coming soon -- this page will let you preview the audit report and download the
        cleaned CSV and PDF report for session {sessionId}.
      </p>
    </div>
  );
}
