// POST-based SSE via @microsoft/fetch-event-source -- native EventSource can only do GET,
// and POST /api/profile/{session_id} is what starts the graph (PRD §10.1/§10.2).

import { fetchEventSource } from "@microsoft/fetch-event-source";
import { API_BASE_URL } from "./client";
import { sessionHeaders } from "@/lib/sessionToken";
import type { ImproveRequest, ImproveStreamEvent, ProfileStreamEvent } from "./types";

export interface StreamHandlers<T> {
  onEvent: (event: T) => void;
  onError: (error: Error) => void;
  onClose?: () => void;
}

export type ProfileStreamHandlers = StreamHandlers<ProfileStreamEvent>;

/** A stream the server refused before it opened, with the status it refused with. The status
 * is what lets a page tell "already done -- go here instead" from "something broke". */
export class StreamRequestError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
    this.name = "StreamRequestError";
  }
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

/** Runs the improvement loop and reports each round as it happens.
 *
 * The search keeps running on the server if this connection drops, and saves its result when
 * it finishes -- so a dropped stream means "come back for it", not "it failed". */
export function streamImprove(
  sessionId: string,
  request: ImproveRequest,
  handlers: StreamHandlers<ImproveStreamEvent>
): AbortController {
  return streamPost(`${API_BASE_URL}/api/model/${sessionId}/improve/stream`, request, handlers, sessionId);
}

export function streamPost<T>(
  url: string,
  body: unknown,
  { onEvent, onError, onClose }: StreamHandlers<T>,
  sessionId: string
): AbortController {
  const controller = new AbortController();

  fetchEventSource(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...sessionHeaders(sessionId) },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: controller.signal,
    // fetch-event-source's default is to abort the request when the tab is hidden and send
    // it again when the tab comes back. Every stream here is a POST that *does* something --
    // start profiling, apply approved fixes, run a search -- so switching tabs and back used
    // to re-send it. Keep the one request open instead.
    openWhenHidden: true,
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
        throw new StreamRequestError(detail, response.status);
      }
    },
    onmessage(msg) {
      if (!msg.data) return;
      try {
        onEvent(JSON.parse(msg.data) as T);
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
