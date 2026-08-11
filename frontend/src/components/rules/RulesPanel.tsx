import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { BusinessRule, ReadinessCheck } from "@/api/types";

/** Facts about the data that no statistic can find.
 *
 * Everything else in the app reasons from the numbers in the file, which structurally cannot
 * find the largest category of real-world error: `age = -5` is not an outlier, it is
 * impossible, and a 150% discount sits comfortably inside the interquartile range of a column
 * that runs 0 to 200 because half the rows are wrong.
 *
 * The split on this panel is the design. Proposals are guesses and do nothing; rules are what
 * the user approved. Nothing crosses from one list to the other without a click, because a
 * rule applied unread can quietly mark a fifth of someone's rows as errors — which is the
 * exact failure this app exists to prevent. */

function confidenceWord(confidence: number | null): string {
  if (confidence == null) return "";
  if (confidence >= 0.9) return "Very likely";
  if (confidence >= 0.75) return "Likely";
  return "Possible";
}

function ProposedRule({
  rule,
  onApprove,
  onDismiss,
  busy,
}: {
  rule: BusinessRule;
  onApprove: () => void;
  onDismiss: () => void;
  busy: boolean;
}) {
  return (
    <li className="rounded-md border border-border p-3">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-medium">{rule.sentence}</span>
        <span className="text-xs uppercase tracking-wide text-muted-foreground">
          {confidenceWord(rule.confidence)}
        </span>
      </div>
      {/* The evidence, not the guess. "We think age can't be negative" is an opinion; "age
          holds 3 negative values" is a fact the user can check in a second. */}
      <p className="mt-1 text-sm text-muted-foreground">{rule.rationale}</p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <Button size="sm" onClick={onApprove} disabled={busy}>
          Use this rule
        </Button>
        <Button variant="ghost" size="sm" onClick={onDismiss} disabled={busy}>
          Not a rule
        </Button>
        <code className="text-xs text-muted-foreground">{rule.expression}</code>
      </div>
    </li>
  );
}

const STATUS_STYLES: Record<string, { mark: string; className: string }> = {
  failed: { mark: "Broken", className: "border-destructive/40 bg-destructive/5" },
  warning: { mark: "Check the rule", className: "border-amber-500/40 bg-amber-500/5" },
  not_run: { mark: "Not checked", className: "border-border bg-muted/40" },
  passed: { mark: "Holds", className: "border-border" },
};

function ActiveRule({
  rule,
  check,
  onRemove,
  busy,
}: {
  rule: BusinessRule;
  check: ReadinessCheck | undefined;
  onRemove: () => void;
  busy: boolean;
}) {
  const style = STATUS_STYLES[check?.status ?? "not_run"] ?? STATUS_STYLES.not_run;
  return (
    <li className={cn("rounded-md border p-3", style.className)}>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="font-medium">{rule.sentence}</span>
        <span className="text-xs uppercase tracking-wide text-muted-foreground">
          {style.mark}
        </span>
      </div>
      {check && <p className="mt-1 text-sm">{check.finding}</p>}
      {check?.evidence && (
        <p className="mt-1 text-xs text-muted-foreground">{check.evidence}</p>
      )}
      {check?.remedy && <p className="mt-1 text-xs">{check.remedy}</p>}
      <div className="mt-2 flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onRemove} disabled={busy}>
          Remove
        </Button>
        <code className="text-xs text-muted-foreground">{rule.expression}</code>
      </div>
    </li>
  );
}

export function RulesPanel({
  rules,
  proposed,
  checks,
  onChange,
  busy = false,
}: {
  rules: BusinessRule[];
  proposed: BusinessRule[];
  checks: ReadinessCheck[];
  /** Called with the complete new set. The parent saves it and re-renders from the response. */
  onChange: (rules: BusinessRule[]) => void;
  busy?: boolean;
}) {
  // Dismissals live only in this component. A rule the user rejected should stop being
  // offered for as long as they are looking at the page, but persisting it would mean
  // silently never mentioning a real problem again on a later visit -- and the proposals are
  // recomputed from the file each time, so a dismissal that outlived the data it was about
  // would be a decision made against numbers that no longer exist.
  const [dismissed, setDismissed] = useState<string[]>([]);
  const offered = proposed.filter((rule) => !dismissed.includes(rule.name));
  const checkFor = (rule: BusinessRule) => checks.find((c) => c.name === rule.name);
  const broken = checks.filter((c) => c.status === "failed").length;

  if (rules.length === 0 && offered.length === 0) return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">What this data is allowed to contain</CardTitle>
        <p className="text-sm text-muted-foreground">
          Rules catch what statistics can&apos;t. A negative age isn&apos;t an unusual value,
          it&apos;s an impossible one — and no outlier check will ever flag it.
        </p>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 text-sm">
        {rules.length > 0 && (
          <div>
            <p className="font-medium">
              In force{broken > 0 && ` — ${broken} broken`}
            </p>
            <ul className="mt-2 flex flex-col gap-2">
              {rules.map((rule) => (
                <ActiveRule
                  key={rule.name}
                  rule={rule}
                  check={checkFor(rule)}
                  busy={busy}
                  onRemove={() => onChange(rules.filter((r) => r.name !== rule.name))}
                />
              ))}
            </ul>
          </div>
        )}

        {offered.length > 0 && (
          <div>
            <p className="font-medium">Suggested</p>
            {/* "Nothing here is in force" read as a claim about the whole panel once rules
                started appearing in force directly above it. */}
            <p className="text-xs text-muted-foreground">
              None of these are in force yet. They are guesses from the column names and the
              values in the file — approve the ones that are actually true of your data.
            </p>
            <ul className="mt-2 flex flex-col gap-2">
              {offered.map((rule) => (
                <ProposedRule
                  key={rule.name}
                  rule={rule}
                  busy={busy}
                  onApprove={() => onChange([...rules, rule])}
                  onDismiss={() => setDismissed((d) => [...d, rule.name])}
                />
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
