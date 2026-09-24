import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button, buttonClasses } from "@/components/ui/button";
import { DownloadButton } from "@/components/ui/download-button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ImprovementTrail } from "@/components/model/ImprovementTrail";
import { ModelPlanForm } from "@/components/model/ModelPlanForm";
import { ModelResults } from "@/components/model/ModelResults";
import { RunHistory } from "@/components/model/RunHistory";
import {
  ApiError,
  fetchModelCard,
  fetchModelHistory,
  fetchModelPlan,
  fetchModelResult,
  improveModel,
  trainModel,
} from "@/api/client";
import type { ModelHistory, ModelPlan, ModelResult, TrainRequest } from "@/api/types";

/** Page 5: build a model from the cleaned data.
 *
 * Three states rather than a wizard -- review the plan, watch it train, read the results --
 * because adjusting the columns and retraining is the normal loop, not an exception. The
 * plan stays reachable the whole time. */
export function ModelPage() {
  const { sessionId } = useParams();
  const [plan, setPlan] = useState<ModelPlan | null>(null);
  const [result, setResult] = useState<ModelResult | null>(null);
  const [card, setCard] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [improving, setImproving] = useState(false);
  // Kept as text, not a number: an empty box means "as good as you can get", and coercing
  // that to 0 would read as a target of zero.
  const [goal, setGoal] = useState("");
  const [lastRequest, setLastRequest] = useState<TrainRequest | null>(null);
  const [loadingCard, setLoadingCard] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [showPlan, setShowPlan] = useState(true);
  const [history, setHistory] = useState<ModelHistory | null>(null);

  const load = useCallback(async () => {
    if (!sessionId) return;
    try {
      const [nextPlan, existing, runs] = await Promise.all([
        fetchModelPlan(sessionId),
        fetchModelResult(sessionId).catch(() => null),
        // A failure here must not take the page down: the history is a record of what was
        // tried, not something the page needs in order to work.
        fetchModelHistory(sessionId).catch(() => null),
      ]);
      setPlan(nextPlan);
      setResult(existing);
      setHistory(runs);
      setShowPlan(existing === null);
      setPlanError(null);
    } catch (err) {
      setPlanError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    }
  }, [sessionId]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleTrain = async (request: TrainRequest) => {
    if (!sessionId) return;
    setTraining(true);
    setError(null);
    setCard(null);
    setLastRequest(request);
    try {
      const trained = await trainModel(sessionId, request);
      setResult(trained);
      setHistory(await fetchModelHistory(sessionId).catch(() => history));
      setShowPlan(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setTraining(false);
    }
  };

  /** Reuses whatever columns and split the last train used, so "improve it" means "improve
   * this", not "start again with different inputs and see if the number moves". */
  const handleImprove = async () => {
    if (!sessionId) return;
    const parsed = Number.parseFloat(goal);
    const target = goal.trim() !== "" && Number.isFinite(parsed) ? parsed : null;

    setImproving(true);
    setError(null);
    setCard(null);
    try {
      const improved = await improveModel(sessionId, {
        excluded_columns: lastRequest?.excluded_columns ?? plan?.excluded_columns ?? [],
        model_names: lastRequest?.model_names ?? [],
        split_strategy: lastRequest?.split_strategy ?? "random",
        time_column: lastRequest?.time_column ?? null,
        // Without this, "improve" after a grouped train fell back to a grouped split with
        // no column to group on, and the server refused it.
        group_column: lastRequest?.group_column ?? null,
        operating_point: lastRequest?.operating_point ?? null,
        // The loop's fourth round may add these or drop them; what it is handed is where it
        // starts from, not what it has to end with.
        feature_recipes: lastRequest?.feature_recipes ?? plan?.feature_recipes ?? [],
        include_deep_diagnostics: true,
        goal: target,
      });
      setResult(improved);
      setHistory(await fetchModelHistory(sessionId).catch(() => history));
      setShowPlan(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setImproving(false);
    }
  };

  const handleShowCard = async () => {
    if (!sessionId || card) return;
    setLoadingCard(true);
    try {
      setCard(await fetchModelCard(sessionId));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setLoadingCard(false);
    }
  };

  if (planError) {
    return (
      <div className="mx-auto flex max-w-3xl flex-col gap-4 p-8">
        <h1 className="text-2xl font-semibold">Build a model</h1>
        <div className="rounded-md border border-destructive/50 p-4 text-sm">
          <p className="font-medium text-destructive">Can&apos;t build a model for this session</p>
          <p className="mt-1 text-muted-foreground">{planError}</p>
        </div>
        <Link to="/" className="text-sm text-primary underline">
          Start a new session
        </Link>
      </div>
    );
  }

  if (!plan) {
    return (
      <div className="mx-auto max-w-3xl p-8 text-sm text-muted-foreground">
        Working out what can be built...
      </div>
    );
  }

  return (
    <div className="mx-auto flex max-w-4xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">Build a model</h1>
        <p className="text-muted-foreground">
          Predicting <span className="font-medium text-foreground">{plan.target_column}</span> from
          your cleaned data. Nothing here changes the data — you can retrain as often as you
          like.
        </p>
      </div>

      {plan.blockers.length > 0 && (
        <Card className="border-destructive/40 bg-destructive/5">
          <CardHeader>
            <CardTitle className="text-base">This data can&apos;t support a model yet</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            <ul className="list-disc pl-5 text-muted-foreground">
              {plan.blockers.map((blocker) => (
                <li key={blocker}>{blocker}</li>
              ))}
            </ul>
            <Link
              to={`/clean/${sessionId}`}
              className="mt-3 inline-block text-sm text-primary underline"
            >
              Go back and keep cleaning
            </Link>
          </CardContent>
        </Card>
      )}

      {error && (
        <div className="rounded-md border border-destructive/50 p-3 text-sm">
          <p className="font-medium text-destructive">Training didn&apos;t run</p>
          <p className="mt-1 text-muted-foreground">{error}</p>
        </div>
      )}

      {result && (
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => setShowPlan((open) => !open)}>
            {showPlan ? "Hide setup" : "Change columns or retrain"}
          </Button>
          <DownloadButton
            sessionId={sessionId!}
            path={`/api/model/${sessionId}/script`}
            filename="train_model.py"
            variant="outline"
            size="sm"
            onError={setError}
          >
            Download train_model.py
          </DownloadButton>
          <DownloadButton
            sessionId={sessionId!}
            path={`/api/model/${sessionId}/download`}
            filename="model.joblib"
            variant="outline"
            size="sm"
            onError={setError}
          >
            Download the fitted model
          </DownloadButton>
          <Link to={`/export/${sessionId}`} className={buttonClasses("ghost", "sm")}>
            Export everything
          </Link>
        </div>
      )}

      {(showPlan || !result) && (
        <ModelPlanForm plan={plan} onTrain={handleTrain} training={training} />
      )}

      {training && (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            Training and scoring each model, then cross-validating the best one. This usually
            takes a few seconds — longer on bigger files.
          </CardContent>
        </Card>
      )}

      {result && !training && !improving && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Try to do better</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3 text-sm">
            <p className="text-muted-foreground">
              Searches model settings and the other model types, judging each one on
              cross-validated folds of the training rows. The test set stays sealed until the
              end — which is what stops a long search from finding a configuration that suits
              those particular held-out rows and calling it an improvement.
            </p>
            <div className="flex flex-wrap items-center gap-2">
              <label htmlFor="goal" className="text-muted-foreground">
                Stop at
              </label>
              <input
                id="goal"
                type="text"
                inputMode="decimal"
                value={goal}
                onChange={(event) => setGoal(event.target.value)}
                placeholder="e.g. 0.85"
                className="h-9 w-28 rounded-md border border-border bg-transparent px-3 text-sm tabular-nums"
              />
              <Button onClick={handleImprove} disabled={improving}>
                Improve it
              </Button>
              <span className="text-xs text-muted-foreground">
                Leave it blank to get as far as it can.
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {improving && (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            Searching. It fits the model many times over — once per setting per fold — so this
            takes appreciably longer than a single train. It will stop on its own, either when
            it reaches your target or when another round stops being worth it.
          </CardContent>
        </Card>
      )}

      {result?.improvement && !training && !improving && (
        <ImprovementTrail trail={result.improvement} />
      )}

      {result && !training && !improving && <ModelResults result={result} />}

      {history && !training && !improving && (
        <RunHistory runs={history.runs} currentFingerprint={history.current_fingerprint} />
      )}

      {result && !training && !improving && (
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-base">Model card</CardTitle>
            {!card && (
              <Button variant="outline" size="sm" onClick={handleShowCard} disabled={loadingCard}>
                {loadingCard ? "Writing..." : "Generate"}
              </Button>
            )}
          </CardHeader>
          <CardContent>
            {card ? (
              <div className="report-preview text-sm">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{card}</ReactMarkdown>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                A written summary of what was built, what the numbers mean, and what not to
                trust about it — suitable for sharing. Included in the export bundle.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
