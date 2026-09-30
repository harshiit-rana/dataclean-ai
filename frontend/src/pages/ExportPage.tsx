import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { buttonClasses } from "@/components/ui/button";
import { DownloadButton } from "@/components/ui/download-button";
import { Card, CardContent } from "@/components/ui/card";
import { ReadinessPanel } from "@/components/readiness/ReadinessPanel";
import { ApiError, fetchAuditReport } from "@/api/client";

/** Page 4, PRD §8.3 / FR-41..44: preview the audit report and download the artifacts.
 *
 * Everything served here is generated server-side into the session folder. The user's own
 * file is never touched. */
export function ExportPage() {
  const { sessionId } = useParams();
  const [markdown, setMarkdown] = useState<string | null>(null);
  const [hasPdf, setHasPdf] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    fetchAuditReport(sessionId)
      .then((data) => {
        setMarkdown(data.markdown);
        setHasPdf(data.has_pdf);
        setError(null);
      })
      .catch((err) =>
        setError(err instanceof ApiError ? err.detail : String(err?.message ?? err))
      );
  }, [sessionId]);

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Export</h1>
        <p className="text-muted-foreground">
          Your cleaned data and a report documenting every change — including the ones you
          declined, in your own words.
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <DownloadButton
          sessionId={sessionId!}
          path={`/api/export/${sessionId}/zip`}
          filename="dataclean_export.zip"
          onError={setError}
        >
          Download everything (.zip)
        </DownloadButton>
        <DownloadButton
          sessionId={sessionId!}
          path={`/api/export/${sessionId}/csv`}
          filename="cleaned.csv"
          variant="outline"
          onError={setError}
        >
          Cleaned CSV only
        </DownloadButton>
        {hasPdf && (
          <DownloadButton
            sessionId={sessionId!}
            path={`/api/export/${sessionId}/pdf`}
            filename="audit_report.pdf"
            variant="outline"
            onError={setError}
          >
            Report PDF only
          </DownloadButton>
        )}
        <Link to={`/apply/${sessionId}`} className={buttonClasses("outline")}>
          Use this on a new file
        </Link>
        <Link to="/" className={buttonClasses("ghost")}>
          Clean another file
        </Link>
      </div>

      {sessionId && <ReadinessPanel sessionId={sessionId} />}

      {error && (
        <div className="rounded-md border border-destructive/50 p-3 text-sm text-destructive">
          <p className="font-medium">Couldn't load the report</p>
          <p className="text-xs">{error}</p>
        </div>
      )}

      {!hasPdf && markdown && (
        <p className="rounded-md border border-border bg-muted p-3 text-xs text-muted-foreground">
          The PDF couldn't be generated in this environment, so the archive includes the
          report as markdown instead.
        </p>
      )}

      {markdown && (
        <Card>
          <CardContent className="p-6">
            <div className="report-preview text-sm">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{markdown}</ReactMarkdown>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
