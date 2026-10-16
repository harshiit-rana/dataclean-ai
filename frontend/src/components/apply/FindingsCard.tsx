import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ApplyFindings } from "@/api/types";
import { cn } from "@/lib/utils";

/** The new file in three questions, kept apart because each answer means something different:
 * inputs that moved do not mean worse predictions, and stable inputs do not mean the model still
 * works. Performance is said only when the file has answers. */

const MOVED = new Set(["shifted", "different", "changed"]);

function Finding({ label, verdict, headline }: { label: string; verdict: string; headline: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</span>
      <span className={cn("text-sm", MOVED.has(verdict) && "text-amber-700 dark:text-amber-500")}>{headline}</span>
    </div>
  );
}

export function FindingsCard({ findings }: { findings: ApplyFindings }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What this file shows</CardTitle>
        <p className="text-sm text-muted-foreground">{findings.summary}</p>
      </CardHeader>
      <CardContent className="grid gap-3 sm:grid-cols-3">
        <Finding label="Inputs" verdict={findings.inputs.verdict} headline={findings.inputs.headline} />
        <Finding label="Predictions" verdict={findings.predictions.verdict} headline={findings.predictions.headline} />
        <Finding label="Performance" verdict={findings.performance.verdict} headline={findings.performance.headline} />
      </CardContent>
    </Card>
  );
}
