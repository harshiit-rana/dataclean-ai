// Mirrors backend/app/models/api_schemas.py and sse_events.py. PRD §9, §10.

export type UseCase = "ml_training" | "reporting" | "analysis" | "pipeline_validation";
export type CleaningAggression = "conservative" | "moderate" | "thorough";
export type Severity = "high" | "medium" | "low";
export type IssueType =
  | "missing_values"
  | "duplicates"
  | "type_mismatch"
  | "outliers"
  | "inconsistent_format"
  | "high_cardinality";

export interface GoalContext {
  use_case: UseCase;
  expected_output: string;
  critical_columns: string[];
  cleaning_aggression: CleaningAggression;
}

export interface UploadResponse {
  session_id: string;
  file_name: string;
  row_count: number;
  column_count: number;
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

export type ProfileStreamEvent =
  | AgentStartEvent
  | ProfilingCompleteEvent
  | CardChunkEvent
  | CardCompleteEvent
  | AllCardsCompleteEvent
  | InterruptReachedEvent
  | ErrorEvent;
