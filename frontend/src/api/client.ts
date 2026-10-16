// Thin fetch wrapper for the plain request/response endpoints. The SSE endpoint
// (POST /api/profile/{id}) is handled separately in api/sse.ts.

import type {
  ApplicationSummary,
  ApplyResult,
  Comparison,
  BusinessRule,
  CorrelationResult,
  ErrorResponse,
  FinaliseResult,
  GoalContext,
  GoalContextResponse,
  ImproveRequest,
  ExportAcceptance,
  ModelHistory,
  PredictionHistory,
  PromotionCriteria,
  PromotionDecision,
  Reproduction,
  RunPairComparison,
  VersionDiff,
  ModelPlan,
  ModelResult,
  PreviewResponse,
  ReadinessResponse,
  RulesResponse,
  TrainRequest,
  UploadResponse,
} from "./types";
import { sessionHeaders, storeSessionToken } from "@/lib/sessionToken";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public error: string,
    public detail: string,
    public status: number,
    public code?: string
  ) {
    super(detail);
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json()) as ErrorResponse;
    throw new ApiError(body.error, body.detail, response.status, body.code);
  }
  return response.json() as Promise<T>;
}

/** Every request scoped to a session goes through here, so none can be written without its
 * access token. The mirror of attaching the check at the router on the server. */
async function sessionFetch(
  sessionId: string,
  path: string,
  init: RequestInit = {}
): Promise<Response> {
  return fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { ...(init.headers ?? {}), ...sessionHeaders(sessionId) },
  });
}

/** Downloads a session file through fetch rather than a plain `<a download>` link.
 *
 * A link can't carry the access-token header, and putting the token in the query string
 * would write it into server logs, browser history and any referrer — exactly the places it
 * must not be. Fetching to a blob keeps it in the header, and has the side benefit that a
 * failure surfaces as a readable message instead of the browser rendering raw JSON. */
export async function downloadSessionFile(
  sessionId: string,
  path: string,
  filename: string
): Promise<void> {
  const response = await sessionFetch(sessionId, path);
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ErrorResponse | null;
    throw new ApiError(
      body?.error ?? "Download failed",
      body?.detail ?? `HTTP ${response.status}`,
      response.status
    );
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });
  const body = await handleResponse<UploadResponse>(response);
  // Kept for the rest of this session's requests; everything after upload needs it.
  storeSessionToken(body.session_id, body.session_token);
  return body;
}

/** Cleans a new CSV the way this session's file was cleaned, and scores it if a model was
 * trained. The session's token goes with it -- the new file is read against this session's
 * data and model. */
export async function applyToNewFile(sessionId: string, file: File): Promise<ApplyResult> {
  const formData = new FormData();
  formData.append("file", file);
  return handleResponse<ApplyResult>(
    await sessionFetch(sessionId, `/api/apply/${sessionId}`, { method: "POST", body: formData })
  );
}

/** The files this session's model was used on before, newest first. */
export async function fetchApplications(sessionId: string): Promise<ApplicationSummary[]> {
  const response = await sessionFetch(sessionId, `/api/apply/${sessionId}`);
  return (await handleResponse<{ applications: ApplicationSummary[] }>(response)).applications;
}

/** What the model in use predicted at training time and on each file since. */
export async function fetchPredictionHistory(sessionId: string): Promise<PredictionHistory> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/prediction-history`);
  return handleResponse<PredictionHistory>(response);
}

export async function fetchPreview(sessionId: string): Promise<PreviewResponse> {
  return handleResponse<PreviewResponse>(
    await sessionFetch(sessionId, `/api/preview/${sessionId}`)
  );
}

/** Recomputed server-side from the file as it currently stands, so it stays correct after an
 * undo or a conversational edit. Safe to re-fetch after any change. */
export async function fetchReadiness(sessionId: string): Promise<ReadinessResponse> {
  return handleResponse<ReadinessResponse>(
    await sessionFetch(sessionId, `/api/readiness/${sessionId}`)
  );
}

// --- Business rules ------------------------------------------------------------------

export async function fetchRules(sessionId: string): Promise<RulesResponse> {
  return handleResponse<RulesResponse>(await sessionFetch(sessionId, `/api/rules/${sessionId}`));
}

/** Replaces the session's rules with exactly this list. The caller always holds the full set,
 * so there is no add/remove -- partial updates would let client and server drift apart. */
export async function saveRules(
  sessionId: string,
  rules: BusinessRule[]
): Promise<RulesResponse> {
  const response = await sessionFetch(sessionId, `/api/rules/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ rules }),
  });
  return handleResponse<RulesResponse>(response);
}

// --- Model building ------------------------------------------------------------------

export async function fetchModelPlan(sessionId: string): Promise<ModelPlan> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/plan`);
  return handleResponse<ModelPlan>(response);
}

export async function trainModel(
  sessionId: string,
  request: TrainRequest
): Promise<ModelResult> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<ModelResult>(response);
}

/** Searches for a better model and returns an ordinary result with the trail attached.
 *
 * Slow by nature -- it fits a model many times over -- so callers should say so rather than
 * leaving a spinner with no explanation. */
export async function improveModel(
  sessionId: string,
  request: ImproveRequest
): Promise<ModelResult> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/improve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<ModelResult>(response);
}

/** Opens the sealed final holdout and scores the current model on it, once.
 *
 * A second call, for this model or another, returns the stored result with a note rather than
 * evaluating again: a second model scored on rows the first result may have informed is not an
 * independent evaluation. */
export async function finaliseModel(sessionId: string): Promise<FinaliseResult> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/finalise`, {
    method: "POST",
  });
  return handleResponse<FinaliseResult>(response);
}

/** The last evaluation, so returning to the page doesn't mean retraining. 404 when nothing
 * has been trained yet, which callers treat as "no model", not an error. */
export async function fetchModelResult(sessionId: string): Promise<ModelResult | null> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/result`);
  if (response.status === 404) return null;
  return handleResponse<ModelResult>(response);
}

/** The run history, newest first, a page at a time: `before` asks for the entries older than
 * that run. */
export async function fetchModelHistory(
  sessionId: string,
  page: { before?: string; limit?: number } = {}
): Promise<ModelHistory> {
  const params = new URLSearchParams();
  if (page.before) params.set("before", page.before);
  if (page.limit) params.set("limit", String(page.limit));
  const query = params.toString();
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/history${query ? `?${query}` : ""}`
  );
  return handleResponse<ModelHistory>(response);
}

/** Makes a stored run the model in use again. Refused (409, code "data_changed") when the data
 * has changed since it was trained: its scores would describe rows that are no longer the data. */
export async function restoreRun(sessionId: string, runId: string): Promise<ModelResult> {
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/runs/${encodeURIComponent(runId)}/restore`,
    { method: "POST" }
  );
  return handleResponse<ModelResult>(response);
}

/** Re-runs a recorded run's configuration on the same data and says whether its numbers come back. */
export async function reproduceRun(sessionId: string, runId: string): Promise<Reproduction> {
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/runs/${encodeURIComponent(runId)}/reproduce`,
    { method: "POST" }
  );
  return handleResponse<Reproduction>(response);
}

/** Run `a` against run `b`: what differed and whether the score moved beyond the noise. */
export async function compareRuns(sessionId: string, a: string, b: string): Promise<RunPairComparison> {
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/runs/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}`
  );
  return handleResponse<RunPairComparison>(response);
}

/** What changed between two versions of the session's data. */
export async function fetchVersionDiff(sessionId: string, from: number, to: number): Promise<VersionDiff> {
  const response = await sessionFetch(
    sessionId,
    `/api/session/${sessionId}/versions/diff?from=${from}&to=${to}`
  );
  return handleResponse<VersionDiff>(response);
}

/** Runs the exported scripts and checks the rebuilt model agrees with the app's on every held-out row. */
export async function verifyExport(sessionId: string): Promise<ExportAcceptance> {
  const response = await sessionFetch(sessionId, `/api/export/${sessionId}/verify`, { method: "POST" });
  return handleResponse<ExportAcceptance>(response);
}

/** The last such check of the model in use. */
export async function fetchExportVerification(sessionId: string): Promise<ExportAcceptance> {
  const response = await sessionFetch(sessionId, `/api/export/${sessionId}/verify`);
  return handleResponse<ExportAcceptance>(response);
}

/** Starts a comparison of the model in use (or `championRunId`) with a challenger, on criteria fixed now. */
export async function startComparison(
  sessionId: string,
  challengerRunId: string,
  criteria: PromotionCriteria,
  championRunId?: string
): Promise<Comparison> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/comparisons`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ challenger_run_id: challengerRunId, champion_run_id: championRunId, criteria }),
  });
  return handleResponse<Comparison>(response);
}

/** Scores both models on the same rows of a labelled file and checks the criteria. */
export async function evaluateComparison(sessionId: string, comparisonId: string, file: File): Promise<Comparison> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/comparisons/${encodeURIComponent(comparisonId)}/evaluate`,
    { method: "POST", body: formData }
  );
  return handleResponse<Comparison>(response);
}

/** Promotes or rejects the challenger; the reason is stored as written. */
export async function decideComparison(
  sessionId: string,
  comparisonId: string,
  decision: "promote" | "reject",
  reason: string,
  override: boolean
): Promise<PromotionDecision> {
  const response = await sessionFetch(
    sessionId,
    `/api/model/${sessionId}/comparisons/${encodeURIComponent(comparisonId)}/decision`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reason: reason || null, override }),
    }
  );
  return handleResponse<PromotionDecision>(response);
}

export async function fetchDecisions(sessionId: string): Promise<PromotionDecision[]> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/decisions`);
  return (await handleResponse<{ decisions: PromotionDecision[] }>(response)).decisions;
}

/** Puts back the champion the last promotion replaced. */
export async function rollBack(sessionId: string, reason: string): Promise<PromotionDecision> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/rollback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason: reason || null }),
  });
  return handleResponse<PromotionDecision>(response);
}

export async function fetchModelCard(sessionId: string): Promise<string> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/card`);
  const body = await handleResponse<{ markdown: string }>(response);
  return body.markdown;
}

export async function fetchFeatureCorrelations(
  sessionId: string
): Promise<CorrelationResult> {
  const response = await sessionFetch(sessionId, `/api/model/${sessionId}/correlations`);
  return handleResponse<CorrelationResult>(response);
}

export interface AuditReportResponse {
  session_id: string;
  markdown: string;
  has_pdf: boolean;
}

export async function fetchAuditReport(sessionId: string): Promise<AuditReportResponse> {
  const response = await sessionFetch(sessionId, `/api/export/${sessionId}/report`);
  return handleResponse<AuditReportResponse>(response);
}

export interface ActionLogEntry {
  action_id: string;
  card_id: string | null;
  column_name: string;
  operation: string;
  /** "superseded" means it was approved and an earlier approved fix had already dealt with
   * what it addressed — not a failure. */
  status: "applied" | "rejected" | "undone" | "superseded";
  /** The user's own words, verbatim (FR-30). */
  rejection_reason: string | null;
  /** Why the app couldn't run it — kept separate so it's never shown as the user's words. */
  failure_reason: string | null;
  before_stats: Record<string, number> | null;
  after_stats: Record<string, number> | null;
}

export async function fetchActionLog(sessionId: string): Promise<{ actions: ActionLogEntry[] }> {
  const response = await sessionFetch(sessionId, `/api/actions/${sessionId}`);
  return handleResponse<{ actions: ActionLogEntry[] }>(response);
}

export async function undoAction(
  sessionId: string,
  actionId: string
): Promise<{ actions: ActionLogEntry[]; report_regenerated: boolean }> {
  const response = await sessionFetch(sessionId, `/api/undo/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action_id: actionId }),
  });
  return handleResponse<{ actions: ActionLogEntry[]; report_regenerated: boolean }>(response);
}

export interface ChatResult {
  applied: boolean;
  message: string;
  actions: ActionLogEntry[];
}

export async function sendChatMessage(
  sessionId: string,
  message: string
): Promise<ChatResult> {
  const response = await sessionFetch(sessionId, `/api/chat/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return handleResponse<ChatResult>(response);
}

export async function submitGoalContext(
  sessionId: string,
  goal: GoalContext
): Promise<GoalContextResponse> {
  const response = await sessionFetch(sessionId, `/api/goal/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(goal),
  });
  return handleResponse<GoalContextResponse>(response);
}
