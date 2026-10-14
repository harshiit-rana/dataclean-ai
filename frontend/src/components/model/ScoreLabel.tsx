import { cn } from "@/lib/utils";
import type { EvaluationStatus } from "@/api/types";

/** The three kinds of number this app shows, which are not interchangeable.
 *
 * - `exploratory`: scored on cross-validated folds of the training rows. Safe to compare and
 *   to choose by; not an independent estimate of how the model will do.
 * - `held_out`: scored once on rows the model never saw and nothing was chosen against. The
 *   estimate to report -- while the rows have not been reused.
 * - `new_data`: scored on newly uploaded rows that did not exist when the model was built.
 *
 * They used to be presented with similar typography and no consistent label, so people
 * compared numbers that meant different things: a fold score that had been maximised over
 * several candidates beside a held-out score that had not. */
export type ScoreKind = "exploratory" | "held_out" | "new_data";

const KINDS: Record<ScoreKind, { label: string; hint: string }> = {
  exploratory: {
    label: "Exploratory · training folds",
    hint:
      "Measured on cross-validated folds of the training rows. Safe to compare and choose " +
      "by, but not an independent estimate of how the model will do.",
  },
  held_out: {
    label: "Held-out evaluation",
    hint:
      "Measured once, on rows the model never saw and nothing was chosen against. The " +
      "estimate to report, as long as those rows have not been used over and over.",
  },
  new_data: {
    label: "New data",
    hint: "Measured on newly uploaded rows that did not exist when the model was built.",
  },
};

interface ScoreLabelProps {
  kind: ScoreKind;
  /** For `held_out`: how the held-out rows have been used. Absent on results from before
   * the ledger, which show the plain label rather than a count they never recorded. */
  evaluation?: Pick<EvaluationStatus, "look_count" | "status"> | null;
  className?: string;
}

export function ScoreLabel({ kind, evaluation, className }: ScoreLabelProps) {
  const base = KINDS[kind];
  const reused = kind === "held_out" && evaluation && evaluation.status !== "independent";
  const label = reused
    ? `${base.label} · used ${evaluation.look_count} times`
    : base.label;
  const hint = reused
    ? `${base.hint} These rows have now been used ${evaluation.look_count} times, so this ` +
      "score is not independent."
    : base.hint;

  return (
    <span
      title={hint}
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-medium leading-none",
        kind === "exploratory" && "border-sky-500/40 bg-sky-500/10 text-sky-800 dark:text-sky-300",
        kind === "held_out" &&
          !reused &&
          "border-emerald-500/40 bg-emerald-500/10 text-emerald-800 dark:text-emerald-300",
        reused &&
          (evaluation.status === "exhausted"
            ? "border-destructive/50 bg-destructive/10 text-destructive"
            : "border-amber-500/50 bg-amber-500/10 text-amber-900 dark:text-amber-300"),
        kind === "new_data" &&
          "border-violet-500/40 bg-violet-500/10 text-violet-800 dark:text-violet-300",
        className
      )}
    >
      {label}
      {/* The title is a mouse affordance; this keeps the one-line explanation for everyone. */}
      <span className="sr-only">. {hint}</span>
    </span>
  );
}

interface HeldOutUseProps {
  evaluation?: EvaluationStatus;
  split: {
    strategy: "random" | "time" | "group";
    time_column: string | null;
    group_column: string | null;
  };
}

function splitRule(split: HeldOutUseProps["split"]): string {
  if (split.strategy === "time") return `split by ${split.time_column}`;
  if (split.strategy === "group") return `kept whole by ${split.group_column}`;
  return "random split";
}

/** How many times the held-out rows have been evaluated, next to the score it qualifies.
 *
 * The ledger only changes behaviour if people see it. Once is stated quietly -- it is the
 * normal case and a warning on every first run would teach people to ignore the real ones.
 * From the second look it is a warning that says what to do. A result from before the ledger
 * has no count and shows nothing, rather than "1": that would present a score of unknown
 * history as the first look. */
export function HeldOutUse({ evaluation, split }: HeldOutUseProps) {
  if (!evaluation) return null;

  const times = evaluation.look_count === 1 ? "once" : `${evaluation.look_count} times`;
  const rule = splitRule(split);

  if (evaluation.status === "independent" && !evaluation.looks_on_other_splits) {
    return (
      <p className="mt-2 text-xs text-muted-foreground">
        Evaluated on these held-out rows once ({rule}).
      </p>
    );
  }

  const exhausted = evaluation.status === "exhausted";
  return (
    <div
      className={cn(
        "mt-3 rounded-md border p-3 text-sm",
        exhausted
          ? "border-destructive/40 bg-destructive/5"
          : "border-amber-500/40 bg-amber-500/5"
      )}
    >
      <p className="font-medium">
        Evaluated on these held-out rows {times} ({rule}).
      </p>
      {evaluation.message && <p className="mt-1 text-muted-foreground">{evaluation.message}</p>}
      <details className="mt-2 text-xs text-muted-foreground">
        <summary className="cursor-pointer font-medium text-foreground">
          Why does using the same rows again make the score optimistic?
        </summary>
        <p className="mt-1">
          Every score has some luck in it. If you retrain, read the score, change something
          and retrain, the version you keep is the one whose luck was best on these particular
          rows — so the number you end up with is the best of several draws, and a new set of
          rows would score it lower. The model wasn&apos;t trained on these rows, but your
          decisions were shaped by them. The fix is a measurement nothing was chosen against:
          use the model on new labelled rows (&ldquo;Use this on a new file&rdquo;).
        </p>
      </details>
    </div>
  );
}
