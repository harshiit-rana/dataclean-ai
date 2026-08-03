import { useEffect, useRef, useState } from "react";
import { streamProfile } from "@/api/sse";
import type { ColumnIssue, ProfileStreamEvent, SuggestionCard } from "@/api/types";

interface ProfileStreamState {
  status: "idle" | "streaming" | "interrupted" | "error";
  currentAgent: string | null;
  statusMessage: string;
  issueCounts: { high: number; medium: number; low: number } | null;
  issues: ColumnIssue[];
  cards: SuggestionCard[];
  streamingCardText: Record<string, string>;
  error: string | null;
}

const initialState: ProfileStreamState = {
  status: "idle",
  currentAgent: null,
  statusMessage: "",
  issueCounts: null,
  issues: [],
  cards: [],
  streamingCardText: {},
  error: null,
};

/** Drives Page 2 (Review). Starts POST /api/profile/{sessionId} and reduces the SSE event
 * stream (PRD §10.2) into UI-friendly state: streamed card text while a card is still being
 * written, then the finalized card once card_complete arrives. */
/** How often buffered token chunks are flushed to React state, in ms. A fast provider
 * emits well over a thousand card_chunk events in a few seconds; setting state on every
 * one re-renders the whole review page (data table included) faster than React can paint
 * and locks the tab up. Batching to ~15fps keeps the text visibly streaming while cutting
 * renders by two orders of magnitude. */
const CHUNK_FLUSH_MS = 66;

export function useProfileStream(sessionId: string | null) {
  const [state, setState] = useState<ProfileStreamState>(initialState);
  const controllerRef = useRef<AbortController | null>(null);
  const pendingChunksRef = useRef<Record<string, string>>({});
  const flushTimerRef = useRef<number | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    setState({ ...initialState, status: "streaming" });
    pendingChunksRef.current = {};

    const flushChunks = () => {
      flushTimerRef.current = null;
      const pending = pendingChunksRef.current;
      if (Object.keys(pending).length === 0) return;
      pendingChunksRef.current = {};

      setState((s) => {
        const merged = { ...s.streamingCardText };
        for (const [cardId, text] of Object.entries(pending)) {
          merged[cardId] = (merged[cardId] ?? "") + text;
        }
        return { ...s, streamingCardText: merged };
      });
    };

    const scheduleFlush = () => {
      if (flushTimerRef.current !== null) return;
      flushTimerRef.current = window.setTimeout(flushChunks, CHUNK_FLUSH_MS);
    };

    const handleEvent = (event: ProfileStreamEvent) => {
      // Buffered separately from the reducer below so a burst of tokens costs one render.
      if (event.event === "card_chunk") {
        pendingChunksRef.current[event.card_id] =
          (pendingChunksRef.current[event.card_id] ?? "") + event.chunk;
        scheduleFlush();
        return;
      }

      // Any non-chunk event must land after the text that preceded it (e.g. card_complete
      // replaces the streaming placeholder), so drain the buffer first.
      flushChunks();

      setState((s) => {
        switch (event.event) {
          case "agent_start":
            return { ...s, currentAgent: event.agent, statusMessage: event.message };
          case "profiling_complete":
            return {
              ...s,
              issueCounts: { high: event.high, medium: event.medium, low: event.low },
              issues: event.issues ?? [],
            };
          case "card_complete":
            return { ...s, cards: [...s.cards, event.card] };
          case "interrupt_reached":
            return { ...s, status: "interrupted" };
          case "error":
            return { ...s, status: "error", error: event.detail };
          default:
            return s;
        }
      });
    };

    controllerRef.current = streamProfile(sessionId, {
      onEvent: handleEvent,
      onError: (error) => setState((s) => ({ ...s, status: "error", error: error.message })),
      onClose: flushChunks,
    });

    return () => {
      controllerRef.current?.abort();
      if (flushTimerRef.current !== null) {
        window.clearTimeout(flushTimerRef.current);
        flushTimerRef.current = null;
      }
    };
  }, [sessionId]);

  return state;
}
