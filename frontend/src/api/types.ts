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
}

export interface ReadinessResponse {
  session_id: string;
  target_column: string | null;
  task_type: string | null;
  row_count: number;
  column_count: number;
  verdict: "ready" | "warnings" | "blocked" | "not_applicable";
  blockers: string[];
  warnings: string[];
  notes: string[];
  target_summary: Record<string, unknown> | null;
  remaining_issues: { high: number; medium: number; low: number; total: number };
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
  dropped: { column: string; reason: string }[];
  excluded?: string[];
}

export interface SuggestedExclusion {
  column: string;
  reason: string;
  kind: "leakage" | "identifier";
  severity: Severity;
}

export interface CandidateModel {
  name: string;
  label: string;
  description: string;
  explains_features: boolean;
}

export interface SplitOption {
  value: "random" | "time";
  label: string;
  description: string;
}

export interface ModelPlan {
  session_id: string;
  target_column: string;
  task_type: "classification" | "regression";
  row_count: number;
  features: FeatureBreakdown;
  suggested_exclusions: SuggestedExclusion[];
  excluded_columns: string[];
  candidate_models: CandidateModel[];
  split_options: SplitOption[];
  date_columns: string[];
  blockers: string[];
  warnings: string[];
  trained: boolean;
}

export interface TrainRequest {
  excluded_columns: string[];
  model_names: string[];
  split_strategy: "random" | "time";
  time_column: string | null;
  include_deep_diagnostics: boolean;
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
}

export interface ModelResult {
  task: "classification" | "regression";
  target_column: string;
  features: FeatureBreakdown;
  split: {
    strategy: "random" | "time";
    time_column: string | null;
    train_rows: number;
    test_rows: number;
  };
  baseline: Record<string, number | string>;
  primary_metric: string;
  results: CandidateResult[];
  best_model: string;
  assessment: { verdict: "useful" | "marginal" | "no_better_than_guessing"; headline: string };
  deep_diagnostics: {
    skipped?: string | null;
    cross_validation?: CrossValidation | null;
    learning_curve?: LearningCurvePoint[] | null;
    permutation_importance?: PermutationImportance[] | null;
  };
  feature_correlations: CorrelationResult;
}

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
