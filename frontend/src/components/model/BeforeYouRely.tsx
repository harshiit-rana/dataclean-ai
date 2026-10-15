import { useState } from "react";
import { cn } from "@/lib/utils";
import type { ModelCaution } from "@/api/types";

/** How many cautions lead the box. The rest are one click away. */
export const LEADING_CAUTIONS = 3;

const SEVERITY_LABEL: Record<ModelCaution["severity"], string> = {
  high: "Matters most",
  medium: "Worth knowing",
  low: "Context",
};

/** What to know before relying on this result, straight under the verdict.
 *
 * The cautions are the model card's own, written in code and ranked on the server, so the page and
 * the card say the same words. The top three show; the count of the rest says how many are hidden. */
export function BeforeYouRely({ cautions }: { cautions: ModelCaution[] | null | undefined }) {
  const [showAll, setShowAll] = useState(false);
  if (!cautions?.length) return null;

  const shown = showAll ? cautions : cautions.slice(0, LEADING_CAUTIONS);
  const hidden = cautions.length - LEADING_CAUTIONS;

  return (
    <section
      aria-label="Before you rely on this"
      className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/5 p-3"
    >
      <h3 className="text-sm font-medium">Before you rely on this</h3>
      <ul className="mt-2 flex flex-col gap-2">
        {shown.map((caution) => (
          <li key={caution.id} className="text-sm">
            <span
              className={cn(
                "mr-2 text-xs font-medium uppercase tracking-wide",
                caution.severity === "high" ? "text-amber-700 dark:text-amber-400" : "text-muted-foreground",
              )}
            >
              {SEVERITY_LABEL[caution.severity]}
            </span>
            {caution.text}
          </li>
        ))}
      </ul>
      {hidden > 0 && (
        <button
          type="button"
          className="mt-2 text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
          aria-expanded={showAll}
          onClick={() => setShowAll((open) => !open)}
        >
          {showAll ? "Show fewer" : `Show all ${cautions.length} (${hidden} more)`}
        </button>
      )}
    </section>
  );
}
