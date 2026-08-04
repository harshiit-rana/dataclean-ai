import { useCallback, useRef, useState } from "react";
import { streamResume } from "@/api/sse";
import type { ColumnStats, ProfileStreamEvent } from "@/api/types";

export interface AppliedAction {
  action_id: string;
  operation: string;
  column_name: string;
  before_stats?: ColumnStats;
  after_stats?: ColumnStats;
}

interface CleaningRunState {
  status: "idle" | "running" | "done" | "error";
  actions: AppliedAction[];
  appliedCount: number;
  skippedCount: number;
  /** Approved fixes an earlier approved fix had already dealt with. Counted apart from
   * skipped so a run that went exactly right doesn't look like it partly failed. */
  supersededCount: number;
  error: string | null;
}

const initialState: CleaningRunState = {
  status: "idle",
  actions: [],
  appliedCount: 0,
  skippedCount: 0,
  supersededCount: 0,
  error: null,
};

/** Submits approve/reject decisions to POST /api/resume and reduces the resulting SSE
 * stream (action_start / action_complete / all_actions_complete) into UI state. */
export function useCleaningRun(sessionId: string | null) {
  const [state, setState] = useState<CleaningRunState>(initialState);
  const controllerRef = useRef<AbortController | null>(null);

  const start = useCallback(
    (decisions: Record<string, "approved" | "rejected">, skipReasons: Record<string, string>) => {
      if (!sessionId) return;

      const approved = Object.entries(decisions)
        .filter(([, d]) => d === "approved")
        .map(([id]) => id);
      const skipped = Object.entries(decisions)
        .filter(([, d]) => d === "rejected")
        .map(([id]) => id);

      setState({ ...initialState, status: "running" });

      const handleEvent = (event: ProfileStreamEvent) => {
        setState((s) => {
          switch (event.event) {
            case "action_start":
              return {
                ...s,
                actions: [
                  ...s.actions,
                  {
                    action_id: event.action_id,
                    operation: event.operation,
                    column_name: event.column_name,
                  },
                ],
              };
            case "action_complete":
              return {
                ...s,
                actions: s.actions.map((a) =>
                  a.action_id === event.action_id
                    ? { ...a, before_stats: event.before_stats, after_stats: event.after_stats }
                    : a
                ),
              };
            case "all_actions_complete":
              return {
                ...s,
                status: "done",
                appliedCount: event.actions_applied,
                skippedCount: event.actions_skipped,
                supersededCount: event.actions_superseded ?? 0,
              };
            case "error":
              return { ...s, error: event.detail };
            default:
              return s;
          }
        });
      };

      controllerRef.current = streamResume(
        sessionId,
        {
          approved_action_ids: approved,
          skipped_action_ids: skipped,
          skip_reasons: skipReasons,
        },
        {
          onEvent: handleEvent,
          onError: (error) => setState((s) => ({ ...s, status: "error", error: error.message })),
          // The stream ending without all_actions_complete means the run stopped early.
          onClose: () =>
            setState((s) => (s.status === "running" ? { ...s, status: "done" } : s)),
        }
      );
    },
    [sessionId]
  );

  return { ...state, start };
}
