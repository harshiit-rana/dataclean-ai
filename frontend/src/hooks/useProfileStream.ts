import { useEffect, useRef, useState } from "react";
import { streamProfile } from "@/api/sse";
import type { ProfileStreamEvent, SuggestionCard } from "@/api/types";

interface ProfileStreamState {
  status: "idle" | "streaming" | "interrupted" | "error";
  currentAgent: string | null;
  statusMessage: string;
  issueCounts: { high: number; medium: number; low: number } | null;
  cards: SuggestionCard[];
  streamingCardText: Record<string, string>;
  error: string | null;
}

const initialState: ProfileStreamState = {
  status: "idle",
  currentAgent: null,
  statusMessage: "",
  issueCounts: null,
  cards: [],
  streamingCardText: {},
  error: null,
};

/** Drives Page 2 (Review). Starts POST /api/profile/{sessionId} and reduces the SSE event
 * stream (PRD §10.2) into UI-friendly state: streamed card text while a card is still being
 * written, then the finalized card once card_complete arrives. */
export function useProfileStream(sessionId: string | null) {
  const [state, setState] = useState<ProfileStreamState>(initialState);
  const controllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    setState(initialState);
    setState((s) => ({ ...s, status: "streaming" }));

    const handleEvent = (event: ProfileStreamEvent) => {
      setState((s) => {
        switch (event.event) {
          case "agent_start":
            return { ...s, currentAgent: event.agent, statusMessage: event.message };
          case "profiling_complete":
            return {
              ...s,
              issueCounts: { high: event.high, medium: event.medium, low: event.low },
            };
          case "card_chunk":
            return {
              ...s,
              streamingCardText: {
                ...s.streamingCardText,
                [event.card_id]: (s.streamingCardText[event.card_id] ?? "") + event.chunk,
              },
            };
          case "card_complete":
            return { ...s, cards: [...s.cards, event.card] };
          case "all_cards_complete":
            return s;
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
    });

    return () => controllerRef.current?.abort();
  }, [sessionId]);

  return state;
}
