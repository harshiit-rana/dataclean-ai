// Thin fetch wrapper for the plain request/response endpoints. The SSE endpoint
// (POST /api/profile/{id}) is handled separately in api/sse.ts.

import type {
  CorrelationResult,
  ErrorResponse,
  GoalContext,
  GoalContextResponse,
  ModelPlan,
  ModelResult,
  PreviewResponse,
  ReadinessResponse,
  TrainRequest,
  UploadResponse,
} from "./types";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public error: string,
    public detail: string,
    public status: number
  ) {
    super(detail);
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = (await response.json()) as ErrorResponse;
    throw new ApiError(body.error, body.detail, response.status);
  }
  return response.json() as Promise<T>;
}

export async function uploadCsv(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/upload`, {
    method: "POST",
    body: formData,
  });
  return handleResponse<UploadResponse>(response);
}

export async function fetchPreview(sessionId: string): Promise<PreviewResponse> {
  const response = await fetch(`${API_BASE_URL}/api/preview/${sessionId}`);
  return handleResponse<PreviewResponse>(response);
}

/** Recomputed server-side from the file as it currently stands, so it stays correct after an
 * undo or a conversational edit. Safe to re-fetch after any change. */
export async function fetchReadiness(sessionId: string): Promise<ReadinessResponse> {
  const response = await fetch(`${API_BASE_URL}/api/readiness/${sessionId}`);
  return handleResponse<ReadinessResponse>(response);
}

// --- Model building ------------------------------------------------------------------

export async function fetchModelPlan(sessionId: string): Promise<ModelPlan> {
  const response = await fetch(`${API_BASE_URL}/api/model/${sessionId}/plan`);
  return handleResponse<ModelPlan>(response);
}

export async function trainModel(
  sessionId: string,
  request: TrainRequest
): Promise<ModelResult> {
  const response = await fetch(`${API_BASE_URL}/api/model/${sessionId}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
  return handleResponse<ModelResult>(response);
}

/** The last evaluation, so returning to the page doesn't mean retraining. 404 when nothing
 * has been trained yet, which callers treat as "no model", not an error. */
export async function fetchModelResult(sessionId: string): Promise<ModelResult | null> {
  const response = await fetch(`${API_BASE_URL}/api/model/${sessionId}/result`);
  if (response.status === 404) return null;
  return handleResponse<ModelResult>(response);
}

export async function fetchModelCard(sessionId: string): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/api/model/${sessionId}/card`);
  const body = await handleResponse<{ markdown: string }>(response);
  return body.markdown;
}

export async function fetchFeatureCorrelations(
  sessionId: string
): Promise<CorrelationResult> {
  const response = await fetch(`${API_BASE_URL}/api/model/${sessionId}/correlations`);
  return handleResponse<CorrelationResult>(response);
}

export interface AuditReportResponse {
  session_id: string;
  markdown: string;
  has_pdf: boolean;
}

export async function fetchAuditReport(sessionId: string): Promise<AuditReportResponse> {
  const response = await fetch(`${API_BASE_URL}/api/export/${sessionId}/report`);
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
  const response = await fetch(`${API_BASE_URL}/api/actions/${sessionId}`);
  return handleResponse<{ actions: ActionLogEntry[] }>(response);
}

export async function undoAction(
  sessionId: string,
  actionId: string
): Promise<{ actions: ActionLogEntry[]; report_regenerated: boolean }> {
  const response = await fetch(`${API_BASE_URL}/api/undo/${sessionId}`, {
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
  const response = await fetch(`${API_BASE_URL}/api/chat/${sessionId}`, {
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
  const response = await fetch(`${API_BASE_URL}/api/goal/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(goal),
  });
  return handleResponse<GoalContextResponse>(response);
}
