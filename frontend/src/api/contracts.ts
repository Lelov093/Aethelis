export type WorldInstanceStatus = "active" | "archived";
export type SaveReason = "initial" | "automatic" | "manual" | "suspend" | "close";

export interface PlayerProfile {
  id: string;
  principal_id: string;
  display_name: string;
  locale: string;
  accessibility_preferences: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface AvailableWorldContent {
  world_definition_id: string;
  world_name: string;
  content_version_id: string;
  package_id: string;
  default_locale: string;
  supported_locales: string[];
}

export interface SavePointView {
  id: string;
  world_instance_id: string;
  name: string;
  world_version: number;
  reason: SaveReason;
  location_name: string | null;
  created_at: string;
}

export interface SavePoint {
  id: string;
  world_instance_id: string;
  world_version: number;
  snapshot_id: string;
  content_version_id: string;
  play_session_id: string | null;
  command_id: string | null;
  name: string | null;
  reason: SaveReason;
  created_at: string;
}

export interface WorldTimelineView {
  id: string;
  name: string;
  status: WorldInstanceStatus;
  world_name: string;
  world_version: number;
  location_name: string | null;
  latest_save: SavePointView;
  forked_from_world_instance_id: string | null;
  forked_from_save_point_id: string | null;
  updated_at: string;
}

export interface WorldInstance {
  id: string;
  name: string;
  status: WorldInstanceStatus;
  current_world_version: number;
  forked_from_world_instance_id: string | null;
  forked_from_save_point_id: string | null;
}

export interface PlaySession {
  id: string;
  world_instance_id: string;
  player_profile_id: string;
  status: "active" | "suspended" | "closed";
  entry_world_version: number;
  last_observed_world_version: number;
}

export interface VisibleEntityView {
  id: string;
  name: string;
  summary: string;
  status: string;
}

export interface VisibleResourceView {
  id: string;
  name: string;
  summary: string;
  quantity: number;
}

export interface ContextualActionView {
  action_id: string;
  label: string;
  location_id: string | null;
  target_id: string | null;
  command_required: boolean;
}

export interface SceneView {
  world_instance_id: string;
  world_version: number;
  world_turn: number;
  elapsed_minutes: number;
  location_id: string | null;
  location_name: string | null;
  visible_entities: VisibleEntityView[];
  visible_resources: VisibleResourceView[];
  public_facts: string[];
  contextual_actions: ContextualActionView[];
  content_version_id: string;
  supports_free_dialogue: boolean;
  supports_world_narrative: boolean;
  recommended_content_version_id: string | null;
}

export interface MapLocationView {
  id: string;
  name: string;
  summary: string;
  accessibility_label: string;
  is_current: boolean;
  is_reachable: boolean;
}

export interface MapView {
  world_instance_id: string;
  world_version: number;
  current_location_id: string | null;
  locations: MapLocationView[];
}

export interface JournalView {
  world_instance_id: string;
  world_version: number;
  entries: string[];
  confirmed_facts: string[];
  observations: string[];
  current_objectives: string[];
  resources: JournalResourceView[];
  opportunities: OpportunityView[];
  situation: SituationView;
  knowledge: JournalKnowledgeView[];
  relationships: JournalRelationshipView[];
  commitments: JournalCommitmentView[];
  outcomes: JournalOutcomeView[];
  world_responses?: JournalWorldResponseView[];
  world_activities?: JournalActivityView[];
  dialogue_interactions?: DialogueInteractionView[];
}

export interface SituationView {
  phase: "unstable" | "contained" | "repaired";
  title: string;
  summary: string;
  completed_steps: number;
  total_steps: number;
  recovery_guidance: string[];
}

export interface DialogueExchangeView {
  id: string;
  input_kind: "preset" | "free";
  player_text: string;
  response_text: string;
  requested_effect_status: string;
  visible_effects: string[];
  committed_event_id: string;
}

export interface DialogueInteractionView {
  id: string;
  target_kind: "character" | "world_narrative";
  target_id: string | null;
  target_name: string;
  contains_free_expression: boolean;
  exchanges: DialogueExchangeView[];
}

export interface JournalActivityView {
  id: string;
  turn: number;
  actor_names: string[];
  activity_kind: string;
  summary: string;
}

export interface JournalKnowledgeView {
  id: string;
  kind: "confirmed_fact" | "rumor";
  statement: string;
  source_name: string;
  confidence: string;
}

export interface JournalRelationshipView {
  character_id: string;
  character_name: string;
  trust: number;
  standing_label: string;
  interaction_count: number;
}

export interface JournalResourceView {
  id: string;
  name: string;
  summary: string;
  kind: string;
  quantity: number;
  custody_label: string;
  source_resource_id: string | null;
  is_player_owned: boolean;
}

export interface JournalCommitmentView {
  id: string;
  counterparty_name: string;
  description: string;
  status: "active" | "fulfilled" | "broken";
}

export interface JournalOutcomeView {
  id: string;
  outcome_type: "ending" | "stable_continuation";
  title: string;
  description: string;
}

export interface JournalWorldResponseView {
  id: string;
  actor_name: string;
  response_kind: "civic_support" | "social_withdrawal";
  summary: string;
}

export interface OpportunityView {
  id: string;
  title: string;
  description: string;
  location_id: string;
  location_name: string;
  action_id: string;
  target_id: string | null;
  is_at_location: boolean;
  is_completed: boolean;
  is_optional: boolean;
}

export interface ResumeSummaryView {
  world_instance_id: string;
  world_version: number;
  world_name: string;
  location_name: string | null;
  last_save_reason: string;
  visible_resource_count: number;
  resumable_session_id: string | null;
}

export type PlayerCommandStatus =
  | "submitted"
  | "interpreting"
  | "needs_clarification"
  | "ready_for_governance"
  | "verifying"
  | "committed"
  | "rejected"
  | "projecting"
  | "completed"
  | "cancelled"
  | "failed";

export interface CommandResultView {
  command_id: string;
  status: PlayerCommandStatus;
  message: string;
  source_world_version: number;
  resulting_world_version: number | null;
  snapshot_id: string | null;
  consequences: string[];
  available_actions: string[];
  created_at: string;
}

export interface CommandReceipt {
  command: {
    id: string;
    action_id: string | null;
    status: PlayerCommandStatus;
    cancellation_requested: boolean;
    updated_at: string;
  };
  execution: {
    attempt_count: number;
    max_attempts: number;
    error_code: string | null;
    error_message: string | null;
    retryable: boolean;
    parsed_intent?: {
      normalized_action: string;
      target_ids: string[];
      confidence: number;
      missing_fields: string[];
      safety_classification: string;
    } | null;
  };
  status_url: string;
  result: CommandResultView | null;
}

export interface ContextualCommandInput {
  player_profile_id: string;
  play_session_id: string;
  action_id: string;
  actor_id: string;
  target_ids: string[];
  location_id: string | null;
  expected_world_version: number;
  locale: string;
  dialogue_interaction_id?: string;
}

export interface NaturalLanguageCommandInput {
  player_profile_id: string;
  play_session_id: string;
  text: string;
  actor_id: string;
  target_ids: string[];
  target_hints: Record<string, string>;
  location_id: string | null;
  expected_world_version: number;
  locale: string;
  dialogue_interaction_id?: string;
}

export interface ResumeState {
  world_instance: WorldInstance;
  latest_save_point: {
    id: string;
    name: string | null;
    world_version: number;
    reason: SaveReason;
  };
  play_session: PlaySession | null;
}

export interface ProblemDetail {
  type: string;
  title: string;
  status: number;
  code: string;
  detail: string;
}
