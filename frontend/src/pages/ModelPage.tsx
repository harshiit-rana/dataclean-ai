import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button, buttonClasses } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ModelPlanForm } from "@/components/model/ModelPlanForm";
import { ModelResults } from "@/components/model/ModelResults";
import {
  API_BASE_URL,
  ApiError,
  fetchModelCard,
  fetchModelPlan,
  fetchModelResult,
  trainModel,
} from "@/api/client";
import type { ModelPlan, ModelResult, TrainRequest } from "@/api/types";

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
  const [loadingCard, setLoadingCard] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [planError, setPlanError] = useState<string | null>(null);
  const [showPlan, setShowPlan] = useState(true);

  const load = useCallback(async () => {
    if (!sessionId) return;
    try {
      const [nextPlan, existing] = await Promise.all([
        fetchModelPlan(sessionId),
        fetchModelResult(sessionId).catch(() => null),
      ]);
      setPlan(nextPlan);
      setResult(existing);
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
    try {
      const trained = await trainModel(sessionId, request);
      setResult(trained);
      setShowPlan(false);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setTraining(false);
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
          <a
            href={`${API_BASE_URL}/api/model/${sessionId}/script`}
            download
            className={buttonClasses("outline", "sm")}
          >
            Download train_model.py
          </a>
          <a
            href={`${API_BASE_URL}/api/model/${sessionId}/download`}
            download
            className={buttonClasses("outline", "sm")}
          >
            Download the fitted model
          </a>
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

      {result && !training && <ModelResults result={result} />}

      {result && !training && (
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
