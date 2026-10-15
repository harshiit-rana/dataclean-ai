import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button, buttonClasses } from "@/components/ui/button";
import { DownloadButton } from "@/components/ui/download-button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ImprovementTrail } from "@/components/model/ImprovementTrail";
import { LiveImprovement } from "@/components/model/LiveImprovement";
import { ModelPlanForm, type Preselect } from "@/components/model/ModelPlanForm";
import { ModelResults } from "@/components/model/ModelResults";
import { RunHistory } from "@/components/model/RunHistory";
import { TrainingProblem } from "@/components/model/TrainingProblem";
import {
  ApiError,
  fetchModelCard,
  fetchModelHistory,
  fetchModelPlan,
  fetchModelResult,
  finaliseModel,
  compareRuns,
  fetchVersionDiff,
  reproduceRun,
  restoreRun,
  trainModel,
} from "@/api/client";
import { StreamRequestError, streamImprove } from "@/api/sse";
import type {
  ImprovementRound,
  CalibrationMethod,
  ImproveRequest,
  ModelHistory,
  ModelPlan,
  ModelResult,
  TrainRequest,
} from "@/api/types";

/** The request that produced a stored result, rebuilt from the result itself.
 *
 * "Improve it" means improve *this* -- same columns, same split, same trade-off -- and the
 * request that trained it only lives in component state, which a refresh discards. Falling
 * back to the plan's defaults after a reload quietly turned a grouped run into a random one.
 * The stored result records everything that matters, so it is the fallback. */
export function requestFromResult(result: ModelResult): TrainRequest {
  const point = result.operating_point;
  return {
    excluded_columns: result.features.excluded ?? [],
    model_names: [],
    split_strategy: result.split.strategy,
    time_column: result.split.time_column,
    group_column: result.split.group_columns?.[0] ?? result.split.group_column,
    group_columns: result.split.group_columns ?? [],
    include_deep_diagnostics: true,
    operating_point:
      point && point.objective !== "default"
        ? {
            objective: point.objective,
            cost_of_missing: point.cost_of_missing,
            cost_of_false_alarm: point.cost_of_false_alarm,
            target_rate: point.target_rate,
            positive_label: point.positive_label,
          }
        : null,
    feature_recipes: (result.features.derived ?? []).map((recipe) => recipe.id),
    clip_outliers: result.features.clipped ?? [],
    keep_identical_rows_together: result.split.keep_duplicates_together ?? false,
    sealed_holdout: result.split.sealed ?? false,
    calibration: result.calibration?.method ?? "none",
    // A sealed holdout has fixed parts, so the fraction is not part of what is rebuilt.
    test_size: result.split.sealed ? null : (result.split.requested_test_size ?? null),
    time_gap_days: result.split.strategy === "time" ? result.split.gap_days || null : null,
  };
}

/** Page 5: build a model from the cleaned data.
 *
 * Three states rather than a wizard -- review the plan, watch it train, read the results --
 * because adjusting the columns and retraining is the normal loop, not an exception. The
 * plan stays reachable the whole time. */
export function ModelPage() {
  const { sessionId } = useParams();
  // A link from a readiness check pre-fills the form (a column to leave out, a grouped split). It only
  // pre-fills: the user reviews the form and presses train.
  const [searchParams] = useSearchParams();
  const preselect: Preselect = {
    exclude: searchParams.getAll("exclude"),
    split: searchParams.get("split") === "group" ? "group" : undefined,
    group: searchParams.get("group") ?? undefined,
  };
  const hasPreselect = (preselect.exclude?.length ?? 0) > 0 || preselect.split === "group";
  const [plan, setPlan] = useState<ModelPlan | null>(null);
  const [result, setResult] = useState<ModelResult | null>(null);
  const [card, setCard] = useState<string | null>(null);
  const [training, setTraining] = useState(false);
  const [improving, setImproving] = useState(false);
  // Kept as text, not a number: an empty box means "as good as you can get", and coercing
  // that to 0 would read as a target of zero.
  const [goal, setGoal] = useState("");
  // Minutes, as text: empty means no limit. Sent as whole seconds, between one second and an hour.
  const [timeLimit, setTimeLimit] = useState("");
  const [lastRequest, setLastRequest] = useState<TrainRequest | null>(null);
  const [loadingCard, setLoadingCard] = useState(false);
  const [error, setErrorMessage] = useState<string | null>(null);
  // The server's reason code for a refused or failed training, when it gave one. Set together with the
  // message so a later error from somewhere else cannot inherit an earlier one's way out.
  const [errorCode, setErrorCode] = useState<string | null>(null);
  const setError = useCallback((message: string | null, code: string | null = null) => {
    setErrorMessage(message);
    setErrorCode(code);
  }, []);
  const [planError, setPlanError] = useState<string | null>(null);
  const [showPlan, setShowPlan] = useState(true);
  const [history, setHistory] = useState<ModelHistory | null>(null);
  // The improvement loop, as it runs. Cleared at the start of each search.
  const [liveRounds, setLiveRounds] = useState<ImprovementRound[]>([]);
  const [liveStage, setLiveStage] = useState<string | null>(null);
  const streamRef = useRef<AbortController | null>(null);

  // Leaving the page stops listening. The search itself carries on server-side and saves
  // its result, so coming back finds it.
  useEffect(() => () => streamRef.current?.abort(), []);

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
      // The form is shown when there is nothing to read yet, or when a link asked for it to be set up.
      setShowPlan(existing === null || hasPreselect);
      setPlanError(null);
    } catch (err) {
      setPlanError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    }
  }, [sessionId, hasPreselect]);

  useEffect(() => {
    void load();
  }, [load]);

  // Opening the sealed holdout is an explicit act, done at most once per model, so it has its
  // own small state rather than riding on training's.
  const [finalising, setFinalising] = useState(false);
  const [finaliseError, setFinaliseError] = useState<string | null>(null);
  const [finaliseNote, setFinaliseNote] = useState<string | null>(null);

  const handleFinalise = async () => {
    if (!sessionId) return;
    setFinalising(true);
    setFinaliseError(null);
    try {
      const opened = await finaliseModel(sessionId);
      setFinaliseNote(opened.for_this_model ? null : (opened.note ?? null));
      const refreshed = await fetchModelResult(sessionId).catch(() => null);
      if (refreshed) setResult(refreshed);
    } catch (err) {
      setFinaliseError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setFinalising(false);
    }
  };

  // Bringing back an earlier run: the result, the card and the history all follow the pointer.
  const [restoring, setRestoring] = useState<string | null>(null);
  const [restoreError, setRestoreError] = useState<string | null>(null);

  const handleRestore = async (runId: string) => {
    if (!sessionId) return;
    setRestoring(runId);
    setRestoreError(null);
    try {
      const restored = await restoreRun(sessionId, runId);
      setResult(restored);
      setCard(null);
      setHistory(await fetchModelHistory(sessionId).catch(() => history));
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setRestoreError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setRestoring(null);
    }
  };

  // Older pages of the history, appended below the newest. The server works out the comparisons on
  // the whole history, so a page's edge does not cut a run off from the one before it.
  const [loadingOlder, setLoadingOlder] = useState(false);
  const handleLoadOlder = async () => {
    if (!sessionId || !history?.next_before) return;
    setLoadingOlder(true);
    try {
      const older = await fetchModelHistory(sessionId, { before: history.next_before });
      setHistory({
        ...history,
        runs: [...history.runs, ...older.runs],
        comparisons: { ...(history.comparisons ?? {}), ...(older.comparisons ?? {}) },
        older_count: older.older_count,
        next_before: older.next_before,
      });
    } catch {
      // The newest page is still shown; the button stays for another try.
    } finally {
      setLoadingOlder(false);
    }
  };

  // A re-run of a recorded configuration: its outcome is kept on the run's history entry.
  const [reproducing, setReproducing] = useState<string | null>(null);
  const handleReproduce = async (runId: string) => {
    if (!sessionId) return;
    setReproducing(runId);
    setRestoreError(null);
    try {
      await reproduceRun(sessionId, runId);
      setHistory(await fetchModelHistory(sessionId).catch(() => history));
    } catch (err) {
      setRestoreError(err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err));
    } finally {
      setReproducing(null);
    }
  };

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
      setError(
        err instanceof ApiError ? err.detail : String((err as Error)?.message ?? err),
        err instanceof ApiError ? (err.code ?? null) : null
      );
    } finally {
      setTraining(false);
    }
  };

  /** Reuses whatever columns and split the last train used, so "improve it" means "improve
   * this", not "start again with different inputs and see if the number moves". Streamed:
   * each round appears as it finishes, and the page says what it is on as each one starts. */
  const handleImprove = (calibration?: CalibrationMethod) => {
    if (!sessionId) return;
    const parsed = Number.parseFloat(goal);
    const target = goal.trim() !== "" && Number.isFinite(parsed) ? parsed : null;
    const minutes = Number.parseFloat(timeLimit);
    const limitSeconds =
      timeLimit.trim() !== "" && Number.isFinite(minutes) && minutes > 0
        ? Math.min(3600, Math.max(1, Math.round(minutes * 60)))
        : null;
    const base: TrainRequest | null = lastRequest ?? (result ? requestFromResult(result) : null);

    const request: ImproveRequest = {
      excluded_columns: base?.excluded_columns ?? plan?.excluded_columns ?? [],
      model_names: base?.model_names ?? [],
      split_strategy: base?.split_strategy ?? "random",
      time_column: base?.time_column ?? null,
      // Without this, "improve" after a grouped train fell back to a grouped split with
      // no column to group on, and the server refused it.
      group_column: base?.group_column ?? null,
      operating_point: base?.operating_point ?? null,
      // The loop's fourth round may add these or drop them; what it is handed is where it
      // starts from, not what it has to end with.
      feature_recipes: base?.feature_recipes ?? plan?.feature_recipes ?? [],
      // Part of what is being improved, not something the loop searches over: a run that
      // clipped its inputs should be improved with them clipped.
      clip_outliers: base?.clip_outliers ?? plan?.features.clipped ?? [],
      keep_identical_rows_together:
        base?.keep_identical_rows_together ?? plan?.keep_identical_rows_together ?? false,
      sealed_holdout: base?.sealed_holdout ?? plan?.sealed_holdout ?? false,
      calibration: calibration ?? base?.calibration ?? "none",
      test_size: base?.test_size ?? null,
      time_gap_days: base?.time_gap_days ?? null,
      include_deep_diagnostics: true,
      goal: target,
      time_limit_seconds: limitSeconds,
    };

    setImproving(true);
    setError(null);
    setCard(null);
    setLiveRounds([]);
    setLiveStage(null);

    let finished = false;
    const finish = () => {
      finished = true;
      setImproving(false);
      setLiveStage(null);
    };

    streamRef.current?.abort();
    streamRef.current = streamImprove(sessionId, request, {
      onEvent: (event) => {
        if (event.type === "stage") {
          setLiveStage(event.description);
        } else if (event.type === "round") {
          setLiveRounds((previous) => [...previous, event.round]);
        } else if (event.type === "result") {
          finish();
          setResult(event.result);
          setShowPlan(false);
          void fetchModelHistory(sessionId)
            .then(setHistory)
            .catch(() => undefined);
          window.scrollTo({ top: 0, behavior: "smooth" });
        } else if (event.type === "error") {
          finish();
          setError(event.detail, event.code ?? null);
        }
      },
      onError: (err) => {
        if (finished) return;
        finish();
        setError(err.message, err instanceof StreamRequestError ? (err.code ?? null) : null);
      },
      onClose: () => {
        if (finished) return;
        // The connection closed without a result or an error. The search is still running
        // on the server and saves its result when it finishes, so this is "come back for
        // it", not "it failed" -- and saying it failed would send the user to start a
        // second search that competes with the first.
        finish();
        setError(
          "Lost the connection to the search. It keeps running on the server and its result " +
            "will be here when you refresh the page in a minute or two."
        );
      },
    });
  };

  /** Retrains the model that is on screen with its probabilities calibrated (or not). A model the
   * improvement loop tuned is rebuilt by the loop again, since a plain train would lose its
   * settings and give a different model. Either way it is a new run, and a new look at the
   * held-out rows, which the card says before it is clicked. */
  const handleCalibrate = (method: CalibrationMethod) => {
    if (!result) return;
    if (result.improvement) {
      handleImprove(method);
      return;
    }
    const base: TrainRequest = lastRequest ?? requestFromResult(result);
    void handleTrain({ ...base, calibration: method });
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
        <TrainingProblem
          code={errorCode}
          detail={error}
          plan={plan}
          sessionId={sessionId!}
          onOpenSetup={() => {
            setShowPlan(true);
            window.scrollTo({ top: 0, behavior: "smooth" });
          }}
        />
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
          <Link to={`/apply/${sessionId}`} className={buttonClasses("outline", "sm")}>
            Predict on a new file
          </Link>
          <Link to={`/export/${sessionId}`} className={buttonClasses("ghost", "sm")}>
            Export everything
          </Link>
        </div>
      )}

      {(showPlan || !result) && (
        <ModelPlanForm plan={plan} onTrain={handleTrain} training={training} preselect={preselect} />
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
              <Button onClick={() => handleImprove()} disabled={improving}>
                Improve it
              </Button>
              <span className="text-xs text-muted-foreground">
                Leave it blank to get as far as it can.
              </span>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <label htmlFor="time-limit" className="text-muted-foreground">
                Spend at most
              </label>
              <input
                id="time-limit"
                type="text"
                inputMode="decimal"
                value={timeLimit}
                onChange={(event) => setTimeLimit(event.target.value)}
                placeholder="no limit"
                className="h-9 w-28 rounded-md border border-border bg-transparent px-3 text-sm tabular-nums"
              />
              <span className="text-muted-foreground">minutes</span>
              <span className="text-xs text-muted-foreground">
                Bounds the search; scoring the result on the held-out rows comes after.
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {improving && <LiveImprovement rounds={liveRounds} stage={liveStage} />}

      {result?.improvement && !training && !improving && (
        <ImprovementTrail trail={result.improvement} />
      )}

      {result && !training && !improving && (
        <ModelResults
          result={result}
          onCalibrate={handleCalibrate}
          calibrating={training || improving}
          onFinalise={handleFinalise}
          finalising={finalising}
          finaliseError={finaliseError}
          finaliseNote={finaliseNote}
        />
      )}

      {history && !training && !improving && (
        <RunHistory
          runs={history.runs}
          currentFingerprint={history.current_fingerprint}
          comparisons={history.comparisons}
          currentRunId={history.current_run_id}
          storedRunIds={history.stored_run_ids}
          onRestore={handleRestore}
          restoring={restoring}
          restoreError={restoreError}
          olderCount={history.older_count ?? 0}
          onLoadOlder={handleLoadOlder}
          loadingOlder={loadingOlder}
          currentDatasetVersion={history.current_dataset_version ?? null}
          fetchDiff={sessionId ? (from, to) => fetchVersionDiff(sessionId, from, to) : undefined}
          fetchComparison={sessionId ? (a, b) => compareRuns(sessionId, a, b) : undefined}
          onReproduce={handleReproduce}
          reproducing={reproducing}
        />
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
