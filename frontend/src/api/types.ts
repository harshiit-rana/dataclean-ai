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
