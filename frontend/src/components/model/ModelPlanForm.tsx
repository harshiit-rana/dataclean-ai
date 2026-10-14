import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Select } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { ClipOutliers } from "@/components/model/ClipOutliers";
import { DerivedFeatures } from "@/components/model/DerivedFeatures";
import {
  OperatingPointForm,
  operatingPointIsComplete,
} from "@/components/model/OperatingPointForm";
import type { ModelPlan, OperatingPointRequest, TrainRequest } from "@/api/types";

const SPLIT_NAMES: Record<"random" | "time" | "group", string> = {
  random: "A random split",
  time: "A split by date",
  group: "Keeping related rows together",
};

/** The fractions offered, in five-point steps, always including the recommendation. */
function testSizeOptions(recommended?: number): number[] {
  const steps = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4];
  if (recommended != null && !steps.some((s) => Math.abs(s - recommended) < 1e-9)) {
    steps.push(recommended);
    steps.sort((a, b) => a - b);
  }
  return steps;
}

const EXCLUSION_LABELS: Record<string, string> = {
  leakage: "Gives away the answer",
  identifier: "Looks like an ID",
  personal: "A person's name or contact details",
};

interface ModelPlanFormProps {
  plan: ModelPlan;
  onTrain: (request: TrainRequest) => void;
  training: boolean;
}

/** The decisions that matter in a model build all happen before any fitting: which columns
 * count as inputs, how the data is split, which models to try. Putting them behind a single
 * "train" button would make them defaults nobody chose, so they are laid out here with the
 * app's reasoning attached and everything adjustable. */
export function ModelPlanForm({ plan, onTrain, training }: ModelPlanFormProps) {
  const [excluded, setExcluded] = useState<string[]>(plan.excluded_columns);
  const [models, setModels] = useState<string[]>(plan.candidate_models.map((m) => m.name));
  // Pre-selects what the app recommends, with its reasons shown; a recommendation that names a
  // column the form cannot offer (it is not in the lists below) falls back to a random split.
  const advice = plan.recommended_split ?? null;
  const usableAdvice =
    advice &&
    (advice.strategy === "random" ||
      (advice.strategy === "time" && advice.column != null && plan.date_columns.includes(advice.column)) ||
      (advice.strategy === "group" && advice.column != null && (plan.group_columns ?? []).includes(advice.column)))
      ? advice
      : null;
  const [splitStrategy, setSplitStrategy] = useState<"random" | "time" | "group">(
    usableAdvice?.strategy ?? "random"
  );
  const [recipes, setRecipes] = useState<string[]>(plan.feature_recipes ?? []);
  const [clip, setClip] = useState<string[]>(plan.features.clipped ?? []);
  const [keepCopies, setKeepCopies] = useState<boolean>(plan.keep_identical_rows_together ?? false);
  const [sealed, setSealed] = useState<boolean>(plan.sealed_holdout ?? false);
  // Pre-filled with the recommendation; the user can change it, and what they chose is what is
  // sent. A sealed holdout has fixed parts, so the choice is set aside while it is on.
  const recommended = plan.recommended_test_size ?? null;
  const [testSize, setTestSize] = useState<number>(recommended?.fraction ?? 0.2);
  const sealedOn = Boolean(plan.sealed_holdout_offer?.available) && sealed;
  const [timeColumn, setTimeColumn] = useState<string>(
    (usableAdvice?.strategy === "time" ? usableAdvice.column : null) ?? plan.date_columns[0] ?? ""
  );
  // Defaults to the most-repeated column, which is the one the readiness panel names.
  const [groupColumn, setGroupColumn] = useState<string>(
    (usableAdvice?.strategy === "group" ? usableAdvice.column : null) ?? plan.group_columns?.[0] ?? ""
  );
  const chosenColumn =
    splitStrategy === "time" ? timeColumn : splitStrategy === "group" ? groupColumn : null;
  const overrides =
    usableAdvice != null &&
    (splitStrategy !== usableAdvice.strategy || chosenColumn !== (usableAdvice.column ?? null));
  // Only a two-class target has a line to draw. Regression has no cut-off, and with three
  // or more answers "flag it or don't" stops being the question.
  const twoClass = plan.task_type === "classification" && (plan.class_labels?.length ?? 0) === 2;
  const [operatingPoint, setOperatingPoint] = useState<OperatingPointRequest>({
    objective: "default",
    cost_of_missing: 1,
    cost_of_false_alarm: 1,
    target_rate: null,
    positive_label: plan.class_labels?.[plan.class_labels.length - 1] ?? null,
  });
  const tradeOffReady = !twoClass || operatingPointIsComplete(operatingPoint);

  // The server refuses to train when readiness is blocked, so offering the button anyway
  // turns a verdict the app already reached into an error message the user has to provoke.
  const blocked = plan.blockers.length > 0;
  const trainable = plan.trainable_row_count ?? plan.row_count;

  // A recipe reads the column it derives from, so unticking that column has to take the
  // recipe with it -- otherwise a column excluded for giving away the answer comes straight
  // back in through a missing flag. The server refuses it too; this keeps the form honest.
  const offeredRecipes = (plan.suggested_features ?? []).filter(
    (recipe) => !excluded.includes(recipe.column)
  );
  const liveRecipes = recipes.filter((id) => offeredRecipes.some((r) => r.id === id));

  // Same rule for clipping: a column left out of the model has nothing to clip.
  const offeredClipping = (plan.suggested_clipping ?? []).filter(
    (offer) => !excluded.includes(offer.column)
  );
  const liveClip = clip.filter((column) => offeredClipping.some((o) => o.column === column));

  const usable = [
    ...plan.features.numeric,
    ...plan.features.categorical,
    ...(plan.features.frequent ?? []),
  ];
  const candidates = [...usable, ...excluded].filter(
    (column, index, all) => all.indexOf(column) === index
  );
  const suggestionFor = (column: string) =>
    plan.suggested_exclusions.find((s) => s.column === column);

  const toggle = (list: string[], value: string) =>
    list.includes(value) ? list.filter((v) => v !== value) : [...list, value];

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Which columns should the model use?</CardTitle>
          <p className="text-sm text-muted-foreground">
            Everything is included unless you say otherwise. Columns we think should be left
            out are already unticked, with the reason — untick our suggestion if you disagree.
          </p>
          {plan.exclusions_basis && plan.suggested_exclusions.length > 0 && (
            <p className="text-xs text-muted-foreground">{plan.exclusions_basis}</p>
          )}
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {candidates.map((column) => {
            const suggestion = suggestionFor(column);
            const isExcluded = excluded.includes(column);
            return (
              <label
                key={column}
                className={cn(
                  "flex cursor-pointer items-start gap-3 rounded-md border p-3 transition-colors",
                  isExcluded ? "border-border bg-muted/40" : "border-border hover:bg-muted/40",
                  suggestion && !isExcluded && "border-amber-500/50"
                )}
              >
                <input
                  type="checkbox"
                  className="mt-0.5"
                  checked={!isExcluded}
                  onChange={() => setExcluded((prev) => toggle(prev, column))}
                />
                <span className="flex-1 text-sm">
                  <span className={cn("font-medium", isExcluded && "text-muted-foreground")}>
                    {column}
                  </span>
                  {suggestion && (
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      <span className="font-medium text-amber-700 dark:text-amber-500">
                        {EXCLUSION_LABELS[suggestion.kind] ?? "Suggested to leave out"}
                        {" — "}
                      </span>
                      {suggestion.reason}
                    </span>
                  )}
                </span>
              </label>
            );
          })}

          {plan.features.dropped.length > 0 && (
            <div className="mt-2 rounded-md border border-dashed border-border p-3">
              <p className="text-xs font-medium text-muted-foreground">
                Left out automatically — these can&apos;t be used as model inputs
              </p>
              <ul className="mt-1 flex flex-col gap-1 text-xs text-muted-foreground">
                {plan.features.dropped
                  .filter((d) => !excluded.includes(d.column))
                  .map((d) => (
                    <li key={d.column}>
                      <span className="font-medium">{d.column}</span> — {d.reason}
                    </li>
                  ))}
              </ul>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">How should the data be split?</CardTitle>
          <p className="text-sm text-muted-foreground">
            Some rows are held back so the model can be scored on data it has never seen.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          {usableAdvice && (
            <div
              className="rounded-md border border-sky-500/40 bg-sky-500/5 p-3 text-sm"
              data-testid="split-recommendation"
            >
              <p className="font-medium">
                Recommended: {SPLIT_NAMES[usableAdvice.strategy]}
                {usableAdvice.column ? ` on ${usableAdvice.column}` : ""}
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {usableAdvice.confidence} confidence
                </span>
              </p>
              <ul className="mt-1 list-disc pl-5 text-xs text-muted-foreground">
                {usableAdvice.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
              {overrides && (
                <p className="mt-2 text-xs">
                  You chose {SPLIT_NAMES[splitStrategy].toLowerCase()}
                  {chosenColumn ? ` on ${chosenColumn}` : ""} instead of the recommendation. That is
                  allowed, and the run will record that you did.
                </p>
              )}
            </div>
          )}
          {plan.split_options.map((option) => {
            const disabled =
              (option.value === "time" && plan.date_columns.length === 0) ||
              (option.value === "group" && (plan.group_columns?.length ?? 0) === 0);
            return (
              <label
                key={option.value}
                className={cn(
                  "flex cursor-pointer items-start gap-3 rounded-md border border-border p-3",
                  disabled && "cursor-not-allowed opacity-50"
                )}
              >
                <input
                  type="radio"
                  name="split"
                  className="mt-0.5"
                  checked={splitStrategy === option.value}
                  disabled={disabled}
                  onChange={() => setSplitStrategy(option.value)}
                />
                <span className="flex-1 text-sm">
                  <span className="font-medium">{option.label}</span>
                  {usableAdvice?.strategy === option.value && (
                    <span className="ml-2 rounded bg-sky-500/15 px-1.5 py-0.5 text-xs font-normal text-sky-800 dark:text-sky-200">
                      recommended
                    </span>
                  )}
                  <span className="mt-0.5 block text-xs text-muted-foreground">
                    {!disabled
                      ? option.description
                      : option.value === "time"
                        ? "No date column in this data to order the rows by."
                        : "No column in this data looks like one thing appearing in many rows."}
                  </span>
                </span>
              </label>
            );
          })}

          {splitStrategy === "time" && plan.date_columns.length > 0 && (
            <Select value={timeColumn} onChange={(e) => setTimeColumn(e.target.value)}>
              {plan.date_columns.map((column) => (
                <option key={column} value={column}>
                  Order by {column}
                </option>
              ))}
            </Select>
          )}

          <div className="flex flex-col gap-2 rounded-md border border-border p-3 text-sm">
            <label className="flex flex-wrap items-center gap-3">
              <span className="font-medium">How much to hold back</span>
              <Select
                aria-label="Fraction held back for testing"
                value={String(testSize)}
                disabled={sealedOn}
                onChange={(e) => setTestSize(Number(e.target.value))}
              >
                {testSizeOptions(recommended?.fraction).map((fraction) => (
                  <option key={fraction} value={fraction}>
                    {Math.round(fraction * 100)}%
                    {recommended && fraction === recommended.fraction ? " (recommended)" : ""}
                  </option>
                ))}
              </Select>
            </label>
            {sealedOn ? (
              <p className="text-xs text-muted-foreground">
                With a sealed holdout the three parts are fixed at 70 / 15 / 15, so this is not used.
              </p>
            ) : (
              recommended && (
                <p
                  className={cn(
                    "text-xs",
                    recommended.feasible
                      ? "text-muted-foreground"
                      : "rounded-md border border-amber-500/40 bg-amber-500/10 p-2 text-amber-900 dark:text-amber-200"
                  )}
                >
                  {recommended.reason}
                  {testSize !== recommended.fraction &&
                    ` You chose ${Math.round(testSize * 100)}% instead of the recommended ${Math.round(recommended.fraction * 100)}%.`}
                </p>
              )
            )}
          </div>

          {plan.sealed_holdout_offer && (
            <label
              className={cn(
                "flex items-start gap-3 rounded-md border border-border p-3",
                plan.sealed_holdout_offer.available ? "cursor-pointer" : "cursor-not-allowed opacity-60"
              )}
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={sealed && plan.sealed_holdout_offer.available}
                disabled={!plan.sealed_holdout_offer.available}
                onChange={() => setSealed((on) => !on)}
              />
              <span className="flex-1 text-sm">
                <span className="font-medium">Keep a sealed final holdout</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  {plan.sealed_holdout_offer.available
                    ? `Sets aside ${plan.sealed_holdout_offer.final_rows?.toLocaleString()} rows before anything is decided. They are never scored while you iterate — only an explicit "open the sealed rows" does it, once — so the number you report has not been chosen against. Training uses about ${plan.sealed_holdout_offer.train_rows?.toLocaleString()} rows and the score you see while iterating is measured on ${plan.sealed_holdout_offer.development_rows?.toLocaleString()}. It only counts if you take it before your first look.`
                    : `Not available for this file. ${plan.sealed_holdout_offer.reason ?? ""}`}
                </span>
              </span>
            </label>
          )}

          {splitStrategy === "random" && plan.identical_rows && (
            <label className="flex cursor-pointer items-start gap-3 rounded-md border border-amber-500/50 p-3">
              <input
                type="checkbox"
                className="mt-0.5"
                checked={keepCopies}
                onChange={() => setKeepCopies((on) => !on)}
              />
              <span className="flex-1 text-sm">
                <span className="font-medium">Keep identical rows together</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  {plan.identical_rows.copies.toLocaleString()} rows (
                  {Math.round(plan.identical_rows.share * 100)}%) are exact copies of another row,
                  answer included. A copy in the test set is a question the model has already
                  been shown the answer to, which flatters every score. Ticked, all copies of a
                  row go to the same side. If they are genuine repeats rather than duplicated
                  records, untick it.
                </span>
              </span>
            </label>
          )}

          {splitStrategy === "group" && (plan.group_columns?.length ?? 0) > 0 && (
            <div className="flex flex-col gap-1">
              <Select value={groupColumn} onChange={(e) => setGroupColumn(e.target.value)}>
                {plan.group_columns.map((column) => (
                  <option key={column} value={column}>
                    Group by {column}
                  </option>
                ))}
              </Select>
              {/* Said here because it is a consequence of the choice, not of the column
                  picker above it -- and finding out afterwards that an input silently
                  vanished is the kind of surprise this whole page exists to avoid. */}
              <p className="text-xs text-muted-foreground">
                {groupColumn} won&apos;t be used as an input. Every group in the test set is
                one the model has never seen, so it could only ever memorise them.
              </p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Which models should we try?</CardTitle>
          <p className="text-sm text-muted-foreground">
            All of them by default. They train on the same split, so the scores are
            comparable.
          </p>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {plan.candidate_models.map((model) => (
            <label
              key={model.name}
              className="flex cursor-pointer items-start gap-3 rounded-md border border-border p-3 transition-colors hover:bg-muted/40"
            >
              <input
                type="checkbox"
                className="mt-0.5"
                checked={models.includes(model.name)}
                onChange={() => setModels((prev) => toggle(prev, model.name))}
              />
              <span className="flex-1 text-sm">
                <span className="font-medium">{model.label}</span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  {model.description}
                </span>
              </span>
            </label>
          ))}
        </CardContent>
      </Card>

      <DerivedFeatures recipes={offeredRecipes} chosen={liveRecipes} onChange={setRecipes} />

      <ClipOutliers offers={offeredClipping} chosen={liveClip} onChange={setClip} />

      {twoClass && (
        <OperatingPointForm
          classLabels={plan.class_labels ?? []}
          value={operatingPoint}
          onChange={setOperatingPoint}
        />
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button
          onClick={() =>
            onTrain({
              excluded_columns: excluded,
              model_names: models,
              split_strategy: splitStrategy,
              time_column: splitStrategy === "time" ? timeColumn : null,
              group_column: splitStrategy === "group" ? groupColumn : null,
              include_deep_diagnostics: true,
              operating_point: twoClass ? operatingPoint : null,
              feature_recipes: liveRecipes,
              clip_outliers: liveClip,
              keep_identical_rows_together:
                splitStrategy === "random" && Boolean(plan.identical_rows) && keepCopies,
              sealed_holdout: Boolean(plan.sealed_holdout_offer?.available) && sealed,
              test_size: sealedOn ? null : testSize,
            })
          }
          disabled={training || models.length === 0 || blocked || !tradeOffReady}
        >
          {training ? "Training..." : `Train ${models.length} model${models.length === 1 ? "" : "s"}`}
        </Button>
        <span className="text-sm text-muted-foreground">
          {usable.length - excluded.filter((c) => usable.includes(c)).length} columns in
          {liveRecipes.length > 0 && ` plus ${liveRecipes.length} derived`}
          {liveClip.length > 0 && `, ${liveClip.length} clipped`},{" "}
          {/* Rows that can be trained on, not rows in the file. Printing the file size here
              said "4,407 rows" under a button that would have fitted on 58. Where the two
              differ, both are shown -- the gap is the point. */}
          {trainable.toLocaleString()} rows
          {trainable !== plan.row_count && ` of ${plan.row_count.toLocaleString()}`}.
          {blocked && " Fix what's blocking this above before training."}
          {!blocked && !tradeOffReady && " Give the cut-off trade-off a number first."}
        </span>
      </div>
    </div>
  );
}
