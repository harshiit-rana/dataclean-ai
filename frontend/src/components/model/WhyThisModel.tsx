import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ExplanationStatement, ModelExplanation } from "@/api/types";

const SECTION_ORDER: ExplanationStatement["section"][] = [
  "how_chosen",
  "alternatives",
  "constraints",
  "improvement",
];

function formatEvidence(value: number | string | boolean): string {
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3);
  return value;
}

function Statement({ statement, lead = false }: { statement: ExplanationStatement; lead?: boolean }) {
  return (
    <li
      className={cn(
        "text-sm",
        (lead || statement.tone === "warning") && "rounded-md border border-amber-500/40 bg-amber-500/5 p-3",
        !lead && statement.tone === "notable" && "text-foreground",
      )}
    >
      <p className={cn(lead && "font-medium")}>{statement.text}</p>
      {statement.evidence.length > 0 && (
        <p className="mt-1 text-xs text-muted-foreground">
          {statement.evidence.map((e) => `${e.label} ${formatEvidence(e.value)}`).join(" · ")}
        </p>
      )}
    </li>
  );
}

/** Why this model: how it was chosen, why the others were not, and what constrained the choice.
 *
 * Every sentence is written on the server from fields of the result, and carries the numbers it
 * rests on. A choice that is not clearly better than its closest rival leads the panel. */
export function WhyThisModel({ explanation }: { explanation: ModelExplanation | null | undefined }) {
  if (!explanation?.statements?.length) return null;
  const lead = explanation.statements.find((s) => s.id === explanation.lead_id);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Why this model?</CardTitle>
        <p className="text-sm text-muted-foreground">
          Written from the numbers in this result, not by a language model. Each sentence shows what
          it rests on.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {lead && (
          <ul>
            <Statement statement={lead} lead />
          </ul>
        )}
        {SECTION_ORDER.map((key) => {
          const statements = explanation.statements.filter((s) => s.section === key && s.id !== lead?.id);
          if (statements.length === 0) return null;
          return (
            <section key={key} aria-label={explanation.sections[key]}>
              <h4 className="mb-2 text-sm font-medium">{explanation.sections[key]}</h4>
              <ul className="flex flex-col gap-2">
                {statements.map((statement) => (
                  <Statement key={statement.id} statement={statement} />
                ))}
              </ul>
            </section>
          );
        })}
      </CardContent>
    </Card>
  );
}
