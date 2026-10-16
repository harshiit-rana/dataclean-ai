import { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DownloadButton } from "@/components/ui/download-button";
import { fetchApplications } from "@/api/client";
import type { ApplicationSummary } from "@/api/types";

/** The files this model was used on before, newest first, with their downloads -- so results
 * stay reachable once the page that made them is closed. `refreshKey` changes after each new
 * application so the list includes it. */
export function PreviousApplications({ sessionId, refreshKey }: { sessionId: string; refreshKey?: string }) {
  const [applications, setApplications] = useState<ApplicationSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchApplications(sessionId)
      .then(setApplications)
      .catch(() => setApplications([]));
  }, [sessionId, refreshKey]);

  if (applications.length === 0) return null;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Files this model was used on</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2 text-sm">
        {applications.map((a) => (
          <div key={a.apply_id} className="flex flex-wrap items-center gap-2 border-b border-border/60 pb-2">
            <div className="flex-1">
              <span className="font-medium">{a.file_name ?? a.apply_id}</span>
              <span className="block text-xs text-muted-foreground">
                {a.at} · {a.rows_kept?.toLocaleString()} of {a.rows_in?.toLocaleString()} rows kept
                {a.rows_with_findings ? ` · ${a.rows_with_findings.toLocaleString()} with something to check` : ""}
                {a.drift_verdict ? ` · inputs: ${a.drift_verdict.replace(/_/g, " ")}` : ""}
              </span>
            </div>
            {a.downloads.predictions && (
              <DownloadButton
                sessionId={sessionId}
                path={a.downloads.predictions}
                filename={`predictions_${a.file_name ?? a.apply_id}`}
                variant="outline"
                onError={setError}
              >
                Predictions
              </DownloadButton>
            )}
            {a.downloads.cleaned && (
              <DownloadButton
                sessionId={sessionId}
                path={a.downloads.cleaned}
                filename={`cleaned_${a.file_name ?? a.apply_id}`}
                variant="outline"
                onError={setError}
              >
                Cleaned file
              </DownloadButton>
            )}
          </div>
        ))}
        {error && <p className="text-xs text-destructive">{error}</p>}
      </CardContent>
    </Card>
  );
}
