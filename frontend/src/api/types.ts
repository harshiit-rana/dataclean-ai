// Mirrors backend/app/models/api_schemas.py and sse_events.py. PRD §9, §10.

export type UseCase = "ml_training" | "reporting" | "analysis" | "pipeline_validation";
export type CleaningAggression = "conservative" | "moderate" | "thorough";
export type Severity = "high" | "medium" | "low";
export type TaskType = "auto" | "classification" | "regression";
export type IssueType =
  | "missing_values"
  | "duplicates"
  | "type_mismatch"
  | "outliers"
  | "inconsistent_format"
  | "high_cardinality"
  // Only detectable when a target column is set -- see backend services/ml_readiness.py.
  | "target_missing"
  | "target_imbalance"
  | "target_leakage";

export interface GoalContext {
  use_case: UseCase;
  expected_output: string;
  critical_columns: string[];
  cleaning_aggression: CleaningAggression;
  /** Empty/null means the user isn't building a model; all target-aware behaviour is off. */
  target_column: string | null;
  task_type: TaskType;
}

export interface UploadResponse {
  session_id: string;
  /** Proof this browser uploaded the file; sent back on every session-scoped request. */
  session_token: string;
  file_name: string;
  row_count: number;
  column_count: number;
  columns: string[];
}

export interface GoalContextResponse {
  session_id: string;
  accepted: boolean;
}

export interface ColumnIssue {
  issue_id: string;
  column_name: string;
  issue_type: IssueType;
  severity: Severity;
  description: string;
  stats: Record<string, unknown>;
  priority_score: number;
}

export interface SuggestionCard {
  card_id: string;
  issue_id: string;
  column_name: string;
  found: string;
  proposed_action: string;
  reasoning: string;
  tradeoff: string;
  /** null when the model didn't return a usable whitelisted operation, meaning this card
   * is informational and can't be executed. */
  operation: string | null;
  operation_params: Record<string, string>;
  /** Written in code, never by the model: the fix was corrected or blocked because it would
   * have damaged the prediction target, or it bakes a whole-file statistic into the data. */
  warning: string | null;
  /** What kind of correction the warning describes. Absent on cards saved before it existed. */
  warning_kind?: "adjusted" | "blocked" | "left_to_model" | null;
}

export interface ReadinessResponse {
  session_id: string;
  target_column: string | null;
  task_type: string | null;
  row_count: number; // rows in the file
  trainable_row_count: number; // rows carrying an answer -- what would be modelled
  column_count: number;
  verdict: "ready" | "warnings" | "blocked" | "not_applicable";
  blockers: string[];
  warnings: string[];
  notes: string[];
  target_summary: Record<string, unknown> | null;
  remaining_issues: { high: number; medium: number; low: number; total: number };
  checks: ReadinessCheck[];
  check_counts: ReadinessCheckCounts;
}

/** One question asked of the data, and the answer with its working.
 *
 * `not_run` is the status that earns this its place: "we did not look" and "we looked and it
 * was fine" are different statements, and without this distinction they appear identically
 * as silence. */
export interface ReadinessCheck {
  name: string;
  label: string;
  status: "passed" | "warning" | "failed" | "not_run";
  finding: string;
  evidence: string;
  /** What would change the answer. Empty on a pass. */
  remedy: string;
  /** The check whose failure made this one impossible. Set only when status is not_run. */
  blocked_by: string | null;
  severity: "blocker" | "warning" | null;
}

/** One statement about what the data is allowed to contain.
 *
 * A closed set of kinds rather than an expression, mirroring the backend: a rule is data, not
 * code. `confidence` and `rationale` are present only on proposals -- once a rule is approved
 * it is no longer a guess, it is something the user told us. */
export interface BusinessRule {
  kind: "range" | "comparison" | "arithmetic";
  column: string;
  label: string;
  minimum: number | null;
  maximum: number | null;
  other_column: string | null;
  comparison: "lte" | "lt" | "gte" | "gt" | null;
  multiplier_column: string | null;
  result_column: string | null;
  confidence: number | null;
  rationale: string;
  /** Derived server-side so the client renders without recomputing. */
  name: string;
  sentence: string;
  expression: string;
}

export interface RulesResponse {
  session_id: string;
  /** In force. These are checked on every readiness call. */
  rules: BusinessRule[];
  /** Guesses. These do nothing at all until the user approves them. */
  proposed: BusinessRule[];
  /** What the approved rules currently find, in the same shape as every other check. */
  checks: ReadinessCheck[];
}

export interface ReadinessCheckCounts {
  passed: number;
  warning: number;
  failed: number;
  not_run: number;
  total: number;
}

export interface ErrorResponse {
  error: string;
  detail: string;
}

export interface PreviewColumn {
  name: string;
  dtype: string;
  null_count: number;
  unique_count: number;
  is_critical: boolean;
  is_target: boolean;
}

export interface PreviewResponse {
  session_id: string;
  columns: PreviewColumn[];
  rows: Record<string, string | number | boolean | null>[];
  total_rows: number;
  total_columns: number;
  preview_row_count: number;
}

// --- Model building ----------------------------------------------------------------------

export interface FeatureBreakdown {
  numeric: string[];
  categorical: string[];
  /** High-cardinality columns kept under a frequent_categories recipe: the most common
   * values get an input each and the rest are pooled, encoded inside the pipeline. */
  frequent?: string[];
  dropped: { column: string; reason: string }[];
  excluded?: string[];
  /** The derived inputs this run used. */
  derived?: FeatureRecipe[];
  /** Numeric inputs whose extreme values were clipped to limits learned from the training
   * rows. Empty unless the user ticked them. */
  clipped?: string[];
}

/** A numeric input with values far outside the rest, offered for clipping. A proposal only:
 * nothing is clipped until the column is ticked and sent back as `clip_outliers`. */
export interface ClippingOffer {
  column: string;
  /** Written from the values in the column -- how many, the largest, where the rest sit. */
  reason: string;
  evidence: {
    far_out: number;
    count: number;
    share: number;
    low: number;
    high: number;
    lowest: number | null;
    highest: number | null;
    typical_min: number;
    typical_max: number;
  };
}

/** One derived input: a column the model could not use, turned into one it can.
 *
 * A proposal until its id is sent back with a train request -- same shape as a suggestion
 * card or a business rule, and for the same reason. `outputs` is empty for
 * frequent_categories, which changes how a column is encoded rather than adding one. */
export interface FeatureRecipe {
  id: string;
  kind: "date_parts" | "text_length" | "missing_flag" | "frequent_categories";
  column: string;
  outputs: string[];
  reason: string;
  evidence: Record<string, unknown>;
}

export interface SuggestedExclusion {
  column: string;
  reason: string;
  kind: "leakage" | "identifier" | "personal";
  severity: Severity;
}

export interface CandidateModel {
  name: string;
  label: string;
  description: string;
  explains_features: boolean;
}

export interface SplitOption {
  value: "random" | "time" | "group";
  label: string;
  description: string;
}

export interface ModelPlan {
  session_id: string;
  target_column: string;
  task_type: "classification" | "regression";
  row_count: number; // rows in the file
  trainable_row_count: number; // rows carrying an answer -- what will be fitted on
  features: FeatureBreakdown;
  suggested_exclusions: SuggestedExclusion[];
  exclusions_basis?: string;
  excluded_columns: string[];
  candidate_models: CandidateModel[];
  split_options: SplitOption[];
  date_columns: string[];
  /** Candidates for a grouped split, most repeated first. */
  group_columns: string[];
  blockers: string[];
  warnings: string[];
  trained: boolean;
  /** The distinct answers, for a classification target -- so the user can say which one
   * counts as "a case" when stating a cut-off trade-off. */
  class_labels?: string[];
  /** Derived inputs this file could support, each with its evidence. Proposals only. */
  suggested_features?: FeatureRecipe[];
  /** The recipe ids the last run used, so the form reopens showing what was chosen. */
  feature_recipes?: string[];
  /** Numeric inputs with extreme values, each with the evidence. Offered, never pre-ticked. */
  suggested_clipping?: ClippingOffer[];
  /** Rows that are exact copies of another row -- inputs and answer -- among the default
   * inputs, or null when there are none. */
  identical_rows?: { copies: number; share: number; rows: number } | null;
  /** Whether the form should open with "keep identical rows together" ticked: on when copies
   * exist, or what the last run used. */
  keep_identical_rows_together?: boolean;
  /** Whether a sealed final holdout can be taken from this file, and the part sizes or why not. */
  sealed_holdout_offer?: {
    available: boolean;
    reason?: string | null;
    train_rows?: number;
    development_rows?: number;
    final_rows?: number;
  };
  /** Whether the last run sealed one, so the form reopens showing what was chosen. */
  sealed_holdout?: boolean;
}

/** How far each headline held-out metric could move on a different set of rows: the middle
 * `level` of `resamples` bootstrap resamples. Resampled by whole groups for a grouped split. */
export interface MetricIntervals {
  level: number;
  resamples: number;
  by: "rows" | "groups";
  metrics: Record<string, { low: number; high: number }>;
}

/** The model scored once on the sealed final holdout. */
export interface SealedEvaluation {
  metrics_intervals?: MetricIntervals | null;
  at: string | null;
  rows: number | null;
  metrics: Record<string, number>;
  baseline: Record<string, number | string> | null;
  assessment: { verdict: string; headline: string } | null;
  primary_metric: string | null;
  rows_seen_in_training: number;
}

/** What `POST /finalise` returns. */
export interface FinaliseResult extends SealedEvaluation {
  already_opened: boolean;
  /** False when the rows were opened earlier for a different model: this is that result. */
  for_this_model: boolean;
  opened_at: string | null;
  run_id: string | null;
  note?: string;
}

/** How the held-out rows had been used when a run was made (decision 0001). */
export interface EvaluationStatus {
  /** What number of look this was, counting itself. */
  look_count: number;
  /** The last look that is still only "reused"; the one after it is "exhausted". */
  limit: number;
  status: "independent" | "reused" | "exhausted";
  /** The qualification the user is shown, including any note about other split rules. */
  message: string;
  looks_on_other_splits: number;
  /** Fixes approved while cleaning that worked a value out over every row -- the held-out rows
   * included -- on columns this model uses. Undone fixes and unused columns are left out. */
  whole_file_statistics?: { operation: string; column: string }[];
  /** How the model has done on rows that did not exist when it was built, shown beside the
   * held-out score and never in place of it. Looked up when the result is read. */
  /** The sealed final holdout, once opened for this model. Looked up when the result is read. */
  final?: SealedEvaluation | null;
  new_data?: {
    count: number;
    latest: {
      at: string | null;
      apply_id: string | null;
      rows: number | null;
      metric: string | null;
      value: number | null;
      reported: number | null;
      overlap_excluded: number;
    };
  } | null;
}

/** One completed training run, small enough to keep twenty of.
 *
 * Not a stored result — there is no restoring an old run, and it deliberately holds only what
 * answers "did the thing I changed help?": the settings, the headline number, the verdict. */
export interface ModelRun {
  run_id: string;
  at: string;
  /** Identifies the data this ran against, so runs from before a cleaning change are visible
   * as such rather than compared as though nothing happened. */
  data_fingerprint: string;
  target_column: string | null;
  task: string | null;
  best_model: string | null;
  model_label: string | null;
  models_tried: string[];
  primary_metric: string | null;
  metrics: Record<string, number>;
  baseline: Record<string, number | string>;
  verdict: string | null;
  headline: string | null;
  split: {
    strategy: string | null;
    time_column: string | null;
    group_column: string | null;
    train_rows: number | null;
    test_rows: number | null;
  };
  feature_count: number;
  excluded_columns: string[];
  tuned: boolean;
  tuned_settings: Record<string, unknown>;
  stopped_because: string | null;
  /** Which derived inputs this run used -- two runs on the same columns with different
   * derived inputs are the comparison the loop's fourth round exists to make. */
  feature_recipes?: string[];
  /** Which inputs had their extreme values clipped. */
  clipped_columns?: string[];
  /** The primary metric's 95% interval for this run, when one could be computed. */
  metric_interval?: { metric: string; low: number; high: number } | null;
  /** How many times the held-out rows had been used, and what that made the score. Absent
   * on runs from before the ledger existed: shown as "not recorded", never "independent". */
  evaluation?: { look_count: number; status: "independent" | "reused" | "exhausted" } | null;
  /** Where the cut-off sat. Two runs with the same AUC and different recall are the same
   * model at different lines, and the table has to be able to say so. */
  operating_point?: {
    objective: string | null;
    description: string | null;
    threshold: number | null;
    positive_label: string | null;
    observed: CutoffMetrics | null;
  } | null;
}

export interface ModelHistory {
  session_id: string;
  /** Newest first. */
  runs: ModelRun[];
  current_fingerprint: string;
}

/** Where the cut-off should sit, stated as a trade-off rather than a number.
 *
 * The number is chosen server-side on out-of-fold predictions over the training rows. The
 * client never sends a threshold: picking one off the test-set sweep would be selecting on
 * the test set in miniature. */
export type OperatingObjective = "default" | "cost" | "min_recall" | "min_precision" | "f1";

export interface OperatingPointRequest {
  objective: OperatingObjective;
  cost_of_missing?: number;
  cost_of_false_alarm?: number;
  target_rate?: number | null;
  positive_label?: string | null;
}

export interface TrainRequest {
  excluded_columns: string[];
  model_names: string[];
  split_strategy: "random" | "time" | "group";
  time_column: string | null;
  /** The column saying which rows belong to the same entity, for a grouped split. */
  group_column?: string | null;
  include_deep_diagnostics: boolean;
  /** Two-class targets only; ignored otherwise. */
  operating_point?: OperatingPointRequest | null;
  /** Ids from the plan's suggested_features. Re-proposed server-side rather than trusted. */
  feature_recipes?: string[];
  /** Numeric inputs to clip to limits learned from the training rows, inside the pipeline.
   * Names that are not numeric inputs are ignored by the server. */
  clip_outliers?: string[];
  /** Put exact copies of a row on one side of a random split. Ignored by the other splits. */
  keep_identical_rows_together?: boolean;
  /** Take a final partition out first, to open once with `finalise`. */
  sealed_holdout?: boolean;
}

/** The confusion matrix at one cut-off, as counts and rates. */
export interface CutoffMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
  flagged: number;
  caught: number;
  missed: number;
  false_alarms: number;
  rows: number;
}

/** The cut-off the chosen model was reported at, and how it was arrived at. */
export interface OperatingPointResult {
  objective: OperatingObjective;
  description: string;
  cost_of_missing: number;
  cost_of_false_alarm: number;
  target_rate: number | null;
  positive_label: string;
  threshold: number;
  chosen_on: string;
  folds: number;
  /** False when a floor was asked for that no cut-off clears; `note` says what was set. */
  reachable: boolean;
  note: string | null;
  /** What the training folds promised at this line. Null at the default. */
  expected: CutoffMetrics | null;
  /** What the sealed test set delivered at this line. */
  observed: CutoffMetrics;
  /** The test set at 0.50, for comparison. */
  at_default: CutoffMetrics;
}

export interface RocPoint {
  fpr: number;
  tpr: number;
  threshold: number;
}
export interface PrecisionRecallPoint {
  recall: number;
  precision: number;
  threshold: number;
}
export interface CalibrationPoint {
  predicted: number;
  actual: number;
  count: number;
  bin_start: number;
  bin_end: number;
}
export interface ThresholdRow {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  flagged: number;
  caught: number;
  missed: number;
}
export interface PerClassRow {
  label: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}
export interface ConfusionCell {
  actual: string;
  predicted: string;
  count: number;
  share_of_actual: number;
  correct: boolean;
}
export interface ClassCount {
  label: string;
  count: number;
}
export interface PredictedVsActualPoint {
  actual: number;
  predicted: number;
}
export interface ResidualPoint {
  predicted: number;
  residual: number;
}
export interface ResidualBar {
  centre: number;
  count: number;
}
export interface ErrorByRangeRow {
  range: string;
  mean_absolute_error: number;
  mean_error: number;
  count: number;
}

export interface ModelDiagnostics {
  roc_curve?: RocPoint[];
  precision_recall_curve?: PrecisionRecallPoint[];
  calibration?: CalibrationPoint[];
  thresholds?: ThresholdRow[];
  per_class?: PerClassRow[];
  confusion_cells?: ConfusionCell[];
  class_distribution?: ClassCount[];
  predicted_vs_actual?: PredictedVsActualPoint[];
  residuals?: ResidualPoint[];
  residual_histogram?: ResidualBar[];
  error_by_range?: ErrorByRangeRow[];
}

export interface ColumnImportance {
  column: string;
  importance: number;
}
export interface PermutationImportance {
  column: string;
  importance: number;
  std: number;
  /** The metric whose loss this is: the same one the cross-validation spread uses. */
  metric?: string;
}
export interface CrossValidation {
  metric: string;
  folds: number[];
  mean: number;
  std: number;
  spread: number;
}
export interface LearningCurvePoint {
  training_rows: number;
  training_score: number;
  validation_score: number;
  /** The metric scored: roc_auc for two classes, accuracy for more, r2 for a number. */
  metric?: string;
}
export interface CorrelationResult {
  columns: string[];
  cells: { row: string; column: string; correlation: number }[];
  strong_pairs: { a: string; b: string; correlation: number }[];
}

export interface CandidateResult {
  model: string;
  label: string;
  description: string;
  metrics: Record<string, number | string | number[][] | string[] | null>;
  feature_importance: { feature: string; importance: number }[];
  column_importance: ColumnImportance[];
  diagnostics: ModelDiagnostics;
  failed: string | null;
  /** The cross-validated score on the training rows that the model was *chosen* on, and how
   * much it moved between folds. Null when the candidate could not be cross-validated, and
   * absent on results from before models were chosen on folds. */
  cv_score?: number | null;
  cv_std?: number | null;
  /** 95% intervals for the headline held-out metrics. Present for the chosen model only, and
   * absent when the test set is too small for an interval to mean anything. */
  metrics_intervals?: MetricIntervals | null;
}

/** How the winning model was chosen. The held-out rows are never part of it. */
export interface ModelSelection {
  chosen: string;
  method: "cross_validation" | "only_candidate" | "default_order";
  /** The metric the folds were scored on (roc_auc, accuracy or r2). */
  metric: string | null;
  folds: number;
  scores: Record<string, { score: number; std: number; folds: number }>;
  /** One sentence saying what the choice was based on, written in code. */
  chosen_on: string;
  /** "one_standard_error" when the simplest model within one standard error of the best was
   * kept instead of the raw highest scorer; "highest_score" otherwise. */
  rule?: "one_standard_error" | "highest_score";
  /** The candidate with the highest raw fold score, which is not always the one kept. */
  highest_scorer?: string;
  standard_error?: number;
  /** The sentence saying why the simpler model was kept, or null when the rule changed nothing. */
  rule_note?: string | null;
}

export interface ModelResult {
  task: "classification" | "regression";
  target_column: string;
  features: FeatureBreakdown;
  split: {
    strategy: "random" | "time" | "group";
    time_column: string | null;
    group_column: string | null;
    train_rows: number;
    test_rows: number;
    /** Entities on each side. Row counts alone hide what matters about a grouped split:
     * 800 training rows drawn from 4 customers is not 800 independent observations. */
    train_groups: number | null;
    test_groups: number | null;
    /** Test rows that are exact copies of a training row, answer included. */
    test_rows_seen_in_training?: number;
    /** True when exact copies were kept on one side of the split. */
    keep_duplicates_together?: boolean;
    /** True when a final partition was set aside before anything was decided. */
    sealed?: boolean;
    final_rows?: number;
    final_fraction?: number;
    /** How many looks at the held-out rows had been taken before the seal was made. Zero means
     * the rows were sealed before anything was decided. */
    sealed_after_looks?: number;
    /** Why a sealed holdout was asked for and could not be made. */
    sealed_unavailable_reason?: string;
  };
  baseline: Record<string, number | string>;
  primary_metric: string;
  results: CandidateResult[];
  best_model: string;
  /** Absent on results from before models were chosen on training folds. */
  selection?: ModelSelection;
  assessment: {
    verdict: "useful" | "marginal" | "no_better_than_guessing" | "not_enough_test_data";
    headline: string;
    /** True when the headline carries a note that the held-out rows have been reused. */
    qualified_by_reuse?: boolean;
  };
  /** Names this run in the saved result, the run history and the test-set ledger. */
  run_id?: string;
  /** How the held-out rows had been used when this was made. Absent on older results. */
  evaluation?: EvaluationStatus;
  deep_diagnostics: {
    skipped?: string | null;
    cross_validation?: CrossValidation | null;
    learning_curve?: LearningCurvePoint[] | null;
    permutation_importance?: PermutationImportance[] | null;
  };
  feature_correlations: CorrelationResult;
  /** Present only when the result came from the improvement loop rather than a single train. */
  improvement?: ImprovementTrail | null;
  /** Where the cut-off sits, for a two-class model. Null for regression and multi-class. */
  operating_point?: OperatingPointResult | null;
}

/** One attempt at making the model better, kept whether or not it worked.
 *
 * The rejected rounds are the point: "we tried tuning it and tuning did not help" says
 * something about the data that a silent success would not. */
export interface ImprovementRound {
  name: string;
  label: string;
  what_it_tried: string;
  score: number | null;
  std: number | null;
  gain: number | null;
  accepted: boolean;
  outcome: string;
  detail: Record<string, unknown>;
}

export interface ImprovementTrail {
  rounds: ImprovementRound[];
  /** What the rounds were judged on: roc_auc, accuracy or r2. */
  metric: string;
  goal: number | null;
  stopped_because: "goal_reached" | "goal_out_of_reach" | "no_further_gains";
  /** Cross-validated, from the training rows only -- not the headline number, which is
   * measured once on rows nothing was selected against. */
  cv_score: number;
  cv_std: number;
  settings: Record<string, unknown>;
}

export interface ImproveRequest extends TrainRequest {
  goal?: number | null;
  budget?: number;
}

/** What POST /api/model/{id}/improve/stream sends, one per `data:` line. */
export type ImproveStreamEvent =
  /** A round -- or the final refit, `name: "rebuilding"` -- is starting. */
  | { type: "stage"; name: string; description: string }
  /** A round finished, kept or not. Same shape as the result's improvement.rounds. */
  | { type: "round"; round: ImprovementRound }
  /** Done: the same payload POST /improve returns, already saved server-side. */
  | { type: "result"; result: ModelResult }
  | { type: "error"; detail: string };

// --- SSE event payloads, PRD §10.2 -------------------------------------------------------

export interface AgentStartEvent {
  event: "agent_start";
  agent: string;
  message: string;
}

export interface ProfilingCompleteEvent {
  event: "profiling_complete";
  issue_count: number;
  high: number;
  medium: number;
  low: number;
  issues: ColumnIssue[];
}

export interface CardChunkEvent {
  event: "card_chunk";
  card_id: string;
  chunk: string;
}

export interface CardCompleteEvent {
  event: "card_complete";
  card: SuggestionCard;
}

export interface AllCardsCompleteEvent {
  event: "all_cards_complete";
  total_cards: number;
}

export interface InterruptReachedEvent {
  event: "interrupt_reached";
  session_id: string;
}

export interface ErrorEvent {
  event: "error";
  error: string;
  detail: string;
}

export interface ColumnStats {
  row_count: number;
  null_count?: number;
  unique_value_count?: number;
  min?: number;
  max?: number;
  mean?: number;
  std?: number;
}

export interface ActionStartEvent {
  event: "action_start";
  action_id: string;
  operation: string;
  column_name: string;
}

export interface ActionCompleteEvent {
  event: "action_complete";
  action_id: string;
  before_stats: ColumnStats;
  after_stats: ColumnStats;
}

export interface AllActionsCompleteEvent {
  event: "all_actions_complete";
  actions_applied: number;
  actions_skipped: number;
  /** Approved fixes an earlier approved fix had already made unnecessary. */
  actions_superseded?: number;
}

export type ProfileStreamEvent =
  | AgentStartEvent
  | ProfilingCompleteEvent
  | CardChunkEvent
  | CardCompleteEvent
  | AllCardsCompleteEvent
  | InterruptReachedEvent
  | ActionStartEvent
  | ActionCompleteEvent
  | AllActionsCompleteEvent
  | ErrorEvent;

/** One of the session's fixes, run against the new file. */
export interface AppliedStep {
  action_id: string;
  operation: string;
  column: string | null;
  status: "applied" | "skipped" | "failed";
  note: string | null;
  rows_before: number;
  rows_after: number;
  dropped_count: number;
  /** The statistic the session learned and reused: a fill value, or outlier limits. */
  learned: Record<string, number | string> | null;
}

export interface DriftColumn {
  column: string;
  kind: "numeric" | "categorical";
  status: "stable" | "shifted" | "different" | "not_judged";
  psi?: number;
  outside_training_range?: number;
  training_range?: [number, number];
  training_median?: number;
  new_median?: number;
  unseen_share?: number;
  unseen_examples?: string[];
  note?: string;
}

export interface DriftReport {
  verdict: "similar" | "shifted" | "different" | "not_enough_rows";
  headline: string;
  columns: DriftColumn[];
}

/** POST /api/apply/{session_id}: a finished session used on a new file. */
export interface ApplyResult {
  session_id: string;
  apply_id: string;
  file_name: string;
  rows_in: number;
  rows_kept: number;
  steps: AppliedStep[];
  /** First few rows removed by a step, 1-based data rows. */
  dropped: { row: number; reason: string }[];
  dropped_count: number;
  notes: string[];
  drift: DriftReport | null;
  predictions: {
    count: number;
    columns: string[];
    preview: Record<string, string | number | null>[];
    cutoff: number | null;
  } | null;
  answers: {
    rows: number;
    metric: string | null;
    reported: number | null;
    now: number | null;
    headline: string;
    /** Labelled rows in the file that the model was built from, left out of the check. */
    overlap_excluded?: number;
    /** True when this file was recorded as a new independent evaluation of the model. */
    counted_as_independent?: boolean;
    /** True when this exact file had already been counted: one piece of evidence, not two. */
    already_counted?: boolean;
  } | null;
  downloads: { cleaned: string; predictions?: string };
}
