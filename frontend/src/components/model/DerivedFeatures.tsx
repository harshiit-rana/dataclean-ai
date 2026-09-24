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
};

/** Inputs the model can use, built from columns it otherwise couldn't.
 *
 * The improvement loop is what made this necessary: on the churn sample the settings search
 * found nothing worth keeping, which means the ceiling is the columns, not the settings. A
 * `signup_date` was being thrown away wholesale when tenure derived from it is obviously
 * predictive of churn.
 *
 * Ticked by default with the reason attached, and untickable — the same shape as the
 * suggestion cards and the business rules. A list of things the app quietly decided to
 * compute would be the same defect the dropped-columns list already fixed. */
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
          into something a model can learn from. Each one is rebuilt by the exported
          script, so nothing here traps the model in this app.
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
