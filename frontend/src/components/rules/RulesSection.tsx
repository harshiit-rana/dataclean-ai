import { useCallback, useEffect, useState } from "react";
import { fetchRules, saveRules } from "@/api/client";
import { RulesPanel } from "./RulesPanel";
import type { BusinessRule, RulesResponse } from "@/api/types";

/** Fetches the rules for a session and saves what the user approves.
 *
 * Split from `RulesPanel` so the panel stays a pure function of its props -- the decisions it
 * presents are the interesting part and they are worth testing without a network in the way.
 *
 * `refreshKey` exists for the same reason it does on the readiness panel: undo and the chat
 * box both rewrite cleaned.csv, and the proposals are computed from that file. A rule
 * suggested on the basis of rows that have since been dropped is a question about data that
 * no longer exists. */
export function RulesSection({
  sessionId,
  refreshKey = 0,
  onRulesChanged,
}: {
  sessionId: string;
  refreshKey?: number;
  /** Fires after a successful save so the readiness verdict can be recomputed -- approved
   * rules become checks, and the verdict is stale the moment one is added. */
  onRulesChanged?: () => void;
}) {
  const [data, setData] = useState<RulesResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchRules(sessionId)
      .then((response) => {
        if (!cancelled) setData(response);
      })
      .catch(() => {
        // A failure here must not take the page down with it. Rules are an addition to the
        // cleaning flow, not a precondition for it.
        if (!cancelled) setData(null);
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, refreshKey]);

  const handleChange = useCallback(
    async (rules: BusinessRule[]) => {
      setBusy(true);
      setError(null);
      try {
        setData(await saveRules(sessionId, rules));
        onRulesChanged?.();
      } catch (err) {
        setError(err instanceof Error ? err.message : "That rule could not be saved.");
      } finally {
        setBusy(false);
      }
    },
    [sessionId, onRulesChanged]
  );

  if (!data) return null;

  return (
    <div className="flex flex-col gap-2">
      <RulesPanel
        rules={data.rules}
        proposed={data.proposed}
        checks={data.checks}
        onChange={handleChange}
        busy={busy}
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
