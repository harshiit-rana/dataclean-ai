// POST-based SSE via @microsoft/fetch-event-source -- native EventSource can only do GET,
// and POST /api/profile/{session_id} is what starts the graph (PRD §10.1/§10.2).

import { fetchEventSource } from "@microsoft/fetch-event-source";
import { API_BASE_URL } from "./client";
import type { ProfileStreamEvent } from "./types";

export interface ProfileStreamHandlers {
  onEvent: (event: ProfileStreamEvent) => void;
  onError: (error: Error) => void;
  onClose?: () => void;
}

export function streamProfile(
  sessionId: string,
  { onEvent, onError, onClose }: ProfileStreamHandlers
): AbortController {
  const controller = new AbortController();

  fetchEventSource(`${API_BASE_URL}/api/profile/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal: controller.signal,
    async onopen(response) {
      if (!response.ok) {
        throw new Error(`Failed to start profiling stream: ${response.status}`);
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
