// Thin fetch wrapper for the plain request/response endpoints. The SSE endpoint
// (POST /api/profile/{id}) is handled separately in api/sse.ts.

import type {
  ErrorResponse,
  GoalContext,
  GoalContextResponse,
  PreviewResponse,
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
