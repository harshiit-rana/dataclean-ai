import { useId } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { ClippingOffer } from "@/api/types";

interface ClipOutliersProps {
  offers: ClippingOffer[];
  chosen: string[];
  onChange: (next: string[]) => void;
}

/** Inputs with values far outside the rest, offered for clipping.
 *
 * Cleaning can cap outliers too, but it computes the limit from every row, test rows
 * included, before any split -- so with a prediction target named it no longer proposes it.
 * This does the same job inside the model, learning the limits from the training rows only.
 *
 * Off until ticked, with the evidence attached: clipping changes what the model sees, and a
 * change is the user's to opt into. Offered only where some value is far out, because a
 * list padded with columns that have nothing to clip is how people learn to tick without
 * reading. */
export function ClipOutliers({ offers, chosen, onChange }: ClipOutliersProps) {
  const titleId = useId();
  if (offers.length === 0) return null;

  const toggle = (column: string) =>
    onChange(chosen.includes(column) ? chosen.filter((c) => c !== column) : [...chosen, column]);

  return (
    <Card role="group" aria-labelledby={titleId}>
      <CardHeader>
        <CardTitle id={titleId} className="text-base">Tame extreme values</CardTitle>
        <p className="text-sm text-muted-foreground">
          A few values in these columns sit far from all the others, and a model can be thrown
          by them. Ticking one pulls such values back to the edge of the usual range — the
          limits are worked out from the training rows only, so nothing about the held-back
          rows shapes them, and no row is removed. They&apos;re off until you tick them, and
          the exported script does the same.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {offers.map((offer) => {
          const isChosen = chosen.includes(offer.column);
          return (
            <label
              key={offer.column}
              className={cn(
                "flex cursor-pointer items-start gap-3 rounded-md border border-border p-3 transition-colors hover:bg-muted/40",
                !isChosen && "bg-muted/30"
              )}
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={isChosen}
                onChange={() => toggle(offer.column)}
              />
              <span className="flex-1 text-sm">
                <span className={cn("font-medium", !isChosen && "text-muted-foreground")}>
                  {offer.column}
                </span>
                <span className="mt-0.5 block text-xs text-muted-foreground">{offer.reason}</span>
              </span>
            </label>
          );
        })}
      </CardContent>
    </Card>
  );
}
