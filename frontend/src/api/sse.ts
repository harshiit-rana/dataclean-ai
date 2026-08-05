// POST-based SSE via @microsoft/fetch-event-source -- native EventSource can only do GET,
// and POST /api/profile/{session_id} is what starts the graph (PRD §10.1/§10.2).

import { fetchEventSource } from "@microsoft/fetch-event-source";
import { API_BASE_URL } from "./client";
import { sessionHeaders } from "@/lib/sessionToken";
import type { ProfileStreamEvent } from "./types";

export interface ProfileStreamHandlers {
  onEvent: (event: ProfileStreamEvent) => void;
  onError: (error: Error) => void;
  onClose?: () => void;
}

export function streamProfile(
  sessionId: string,
  handlers: ProfileStreamHandlers
): AbortController {
  return streamPost(`${API_BASE_URL}/api/profile/${sessionId}`, undefined, handlers, sessionId);
}

/** Submits approve/reject decisions and streams the cleaning run that follows. */
export function streamResume(
  sessionId: string,
  decisions: {
    approved_action_ids: string[];
    skipped_action_ids: string[];
    skip_reasons: Record<string, string>;
  },
  handlers: ProfileStreamHandlers
): AbortController {
  return streamPost(`${API_BASE_URL}/api/resume/${sessionId}`, decisions, handlers, sessionId);
}

function streamPost(
  url: string,
  body: unknown,
  { onEvent, onError, onClose }: ProfileStreamHandlers,
  sessionId: string
): AbortController {
  const controller = new AbortController();

  fetchEventSource(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...sessionHeaders(sessionId) },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: controller.signal,
    async onopen(response) {
      if (!response.ok) {
        // Surface the server's plain-English {error, detail} rather than a bare status code.
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          if (body?.detail) detail = body.detail;
        } catch {
          // non-JSON error body -- keep the status code
        }
        throw new Error(detail);
      }
    },
    onmessage(msg) {
      if (!msg.data) return;
      try {
        onEvent(JSON.parse(msg.data) as ProfileStreamEvent);
      } catch {
        // malformed frame -- ignore rather than crash the stream
      }
    },
    onerror(err) {
      onError(err instanceof Error ? err : new Error(String(err)));
      throw err; // stop fetch-event-source's automatic retry; caller decides what's next
    },
    onclose() {
      onClose?.();
    },
  }).catch((err) => {
    onError(err instanceof Error ? err : new Error(String(err)));
  });

  return controller;
}
