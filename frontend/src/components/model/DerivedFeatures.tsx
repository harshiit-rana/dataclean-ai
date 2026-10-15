import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { FeatureRecipe } from "@/api/types";

interface DerivedFeaturesProps {
  recipes: FeatureRecipe[];
  chosen: string[];
  onChange: (next: string[]) => void;
}

const KIND_LABELS: Record<FeatureRecipe["kind"], string> = {
  date_parts: "From a date",
  text_length: "From free text",
  missing_flag: "From the blanks",
  frequent_categories: "Keeping the common values",
  target_encoding: "Encoded by the average answer",
  interaction: "Two columns combined",
};

/** Inputs the model can use, built from columns it otherwise couldn't.
 *
 * A date column used to be thrown away wholesale for having "too many values", and free
 * text was misfiled as an identifier. Tuning can only move a score as far as the columns
 * allow; these widen what the columns can offer.
 *
 * Offered with the reason attached, and off until ticked. They were on by default until the
 * churn sample, where signup_date's parts duplicated a tenure column already in the file and
 * made the default model worse. An extra input is a change, and a change is the user's to
 * opt into -- the improvement loop tries them either way and reports whether they helped. */
export function DerivedFeatures({ recipes, chosen, onChange }: DerivedFeaturesProps) {
  if (recipes.length === 0) return null;

  const toggle = (id: string) =>
    onChange(chosen.includes(id) ? chosen.filter((c) => c !== id) : [...chosen, id]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Inputs we can build from your columns</CardTitle>
        <p className="text-sm text-muted-foreground">
          Some columns can&apos;t be used as they stand — a date is a different value on
          nearly every row, and a paragraph of text isn&apos;t a category. These turn them
          into something a model can learn from. They&apos;re off until you tick them;
          &ldquo;Improve it&rdquo; tries them either way and tells you whether they helped.
          Each one is rebuilt by the exported script.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {recipes.map((recipe) => {
          const isChosen = chosen.includes(recipe.id);
          return (
            <label
              key={recipe.id}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-md border border-border p-3 transition-colors hover:bg-muted/40",
                !isChosen && "bg-muted/30"
              )}
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={isChosen}
                onChange={() => toggle(recipe.id)}
              />
              <span className="flex-1 text-sm">
                <span className={cn("font-medium", !isChosen && "text-muted-foreground")}>
                  {recipe.column}
                </span>
                <span className="ml-2 text-xs text-muted-foreground">
                  {KIND_LABELS[recipe.kind]}
                </span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  {recipe.reason}
                </span>
                {recipe.outputs.length > 0 && (
                  <span className="mt-1 flex flex-wrap gap-1">
                    {recipe.outputs.map((output) => (
                      <code
                        key={output}
                        className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground"
                      >
                        {output}
                      </code>
                    ))}
                  </span>
                )}
              </span>
            </label>
          );
        })}
      </CardContent>
    </Card>
  );
}
