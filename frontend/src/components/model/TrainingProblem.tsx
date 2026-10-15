import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import type { ModelPlan } from "@/api/types";

/** The reason codes the server attaches to a refused or failed training. Stable: see the docstring of
 * `TrainingFailedError` in routes_model. A code this page does not know is shown as the plain message. */
export type TrainingErrorCode =
  | "data_blocked"
  | "target_missing"
  | "target_unusable"
  | "too_few_labelled_rows"
  | "no_usable_columns"
  | "no_model_trained"
  | "split_unusable"
  | "too_large";

const TITLES: Record<TrainingErrorCode, string> = {
  data_blocked: "This data can't support a model yet",
  target_missing: "The column to predict is gone",
  target_unusable: "The column to predict has nothing to learn from",
  too_few_labelled_rows: "Too few rows have an answer",
  no_usable_columns: "No columns are left to learn from",
  no_model_trained: "No model could be trained on these columns",
  split_unusable: "The rows can't be divided as asked",
  too_large: "This file is too big to train on here",
};

const MAX_DROPPED_SHOWN = 8;

interface Props {
  code?: string | null;
  detail: string;
  plan: ModelPlan | null;
  sessionId: string;
  /** Opens the column and split chooser. */
  onOpenSetup: () => void;
}

/** A refused or failed training, turned into next steps.
 *
 * The server's message explains what went wrong; the code says which way out is worth offering. A
 * code this page does not recognise -- or none -- falls back to the plain message in the box this page
 * has always used, so a new server-side reason can never leave the user with nothing. */
export function TrainingProblem({ code, detail, plan, sessionId, onOpenSetup }: Props) {
  if (!code || !(code in TITLES)) {
    return (
      <div className="rounded-md border border-destructive/50 p-3 text-sm">
        <p className="font-medium text-destructive">Training didn&apos;t run</p>
        <p className="mt-1 text-muted-foreground">{detail}</p>
      </div>
    );
  }

  const known = code as TrainingErrorCode;
  const dropped = plan?.features.dropped ?? [];

  return (
    <div role="alert" className="rounded-md border border-destructive/50 p-3 text-sm">
      <p className="font-medium text-destructive">{TITLES[known]}</p>
      <p className="mt-1 text-muted-foreground">{detail}</p>

      {known === "data_blocked" && (
        <Link to={`/clean/${sessionId}`} className="mt-3 inline-block text-primary underline">
          See the failing readiness checks
        </Link>
      )}

      {known === "too_few_labelled_rows" && (
        <>
          <p className="mt-2">
            A model needs enough rows that carry an answer to learn from and still keep some back to
            test on. Add more labelled rows, or check how many rows are labelled in the readiness
            checks.
          </p>
          <Link to={`/clean/${sessionId}`} className="mt-2 inline-block text-primary underline">
            See the readiness checks
          </Link>
        </>
      )}

      {known === "no_usable_columns" && (
        <>
          {dropped.length > 0 && (
            <div className="mt-2">
              <p>Why each column was left out:</p>
              <ul className="mt-1 list-disc pl-5 text-muted-foreground">
                {dropped.slice(0, MAX_DROPPED_SHOWN).map((entry) => (
                  <li key={entry.column}>
                    <span className="font-medium text-foreground">{entry.column}</span>: {entry.reason}
                  </li>
                ))}
              </ul>
              {dropped.length > MAX_DROPPED_SHOWN && (
                <p className="mt-1 text-xs text-muted-foreground">
                  and {dropped.length - MAX_DROPPED_SHOWN} more in the column chooser.
                </p>
              )}
            </div>
          )}
          <Button variant="outline" size="sm" className="mt-3" onClick={onOpenSetup}>
            Open the column chooser
          </Button>
        </>
      )}

      {(known === "target_missing" || known === "target_unusable") && (
        <Link to="/" className="mt-3 inline-block text-primary underline">
          Start again and choose a different column to predict
        </Link>
      )}

      {known === "split_unusable" && (
        <Button variant="outline" size="sm" className="mt-3" onClick={onOpenSetup}>
          Change the split
        </Button>
      )}

      {known === "too_large" && (
        <p className="mt-2">
          Training is limited so that one request cannot run for as long as it likes. Train on a
          smaller sample of the file, or ask whoever runs this server to raise the limit.
        </p>
      )}

      {known === "no_model_trained" && (
        <Button variant="outline" size="sm" className="mt-3" onClick={onOpenSetup}>
          Change the columns or models
        </Button>
      )}
    </div>
  );
}
