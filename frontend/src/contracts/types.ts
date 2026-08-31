/**
 * The wire shapes `vantage.dashboard.api.DashboardApi` actually returns.
 *
 * Every interface here was written against a captured response, not against the
 * demo fixtures. The previous version described a different API entirely --
 * `start_time` where the server sends `first_seen`, `"ACTIVE"` where it sends
 * `"active"`, a `severity_breakdown` whose five field names shared none with the
 * seven the server publishes -- which is why the page only ever worked in demo
 * mode. Types that describe fixtures instead of a server are worse than no
 * types: they make the compiler agree with the fiction.
 *
 * Two conventions run through all of it:
 *
 * - `available: false` plus a `reason` is how the server says a subsystem is not
 *   attached. It is not the same as an empty result, and the UI must not render
 *   it as one -- "the floor is clear" and "nothing is watching the floor" are
 *   different facts.
 * - Anything the server may genuinely not know is optional or nullable here, so
 *   the compiler forces a decision at the point of use rather than letting a
 *   `?? 0.94` invent it.
 */

import type { IncidentState, Metric, Severity } from './vocabulary';

/** Common head of every response. */
export interface Availability {
  available: boolean;
  /** Present when `available` is false: why, in a sentence, for the operator. */
  reason?: string;
}

// ---------------------------------------------------------------------------
// live
// ---------------------------------------------------------------------------

export interface LiveEntity {
  entity_id: string;
  label: string;
  motion: string;
  /** Heights per second, as the state estimator reports it. Not m/s. */
  speed: number;
  posture: string | null;
  activities: string[];
  zones: string[];
  /** Resolved name, or null when identity is off or nobody was recognised. */
  identity: string | null;
}

export interface LiveEvent {
  timestamp: number;
  severity: Severity;
  summary: string;
  rule: string;
  zone: string | null;
}

export interface LiveStats {
  fps: number;
  /** Human-readable name of what is being watched. */
  source: string;
  /**
   * Frames the ingestion queue had to drop. Single-camera runs only: the
   * facility pipeline paces each camera independently and has no single queue
   * to drop from, so it sends the fields below instead. Optional rather than
   * defaulted, so a panel renders an em dash rather than a confident zero.
   */
  dropped?: number;
  active_cameras?: number;
  total_entities?: number;
}

/** One entry of `StageRegistry.to_dict()`. Field names are load-bearing. */
export interface StageHealth {
  name: string;
  calls: number;
  failures: number;
  failure_rate: number;
  /** True once the circuit has opened. The stage is no longer being called. */
  disabled: boolean;
  last_error: string | null;
  worst_streak: number;
  error_types: Record<string, number>;
}

export interface LiveResponse extends Availability {
  frame_index?: number;
  captured_at?: number;
  age_s?: number;
  entities?: LiveEntity[];
  events?: LiveEvent[];
  stats?: LiveStats;
  health?: Record<string, StageHealth>;
  has_frame?: boolean;
  viewers?: number;
}

// ---------------------------------------------------------------------------
// events / history
// ---------------------------------------------------------------------------

export interface StoredEvent {
  id: number;
  timestamp: number;
  rule: string;
  severity: Severity;
  summary: string;
  entity_id: string | null;
  identity: string | null;
  zone: string | null;
  evidence: Record<string, unknown>;
}

export interface EventsResponse extends Availability {
  count?: number;
  events?: StoredEvent[];
}

export interface StoredObservation {
  timestamp: number;
  entity_id: string;
  identity: string | null;
  entity_type: string;
  motion: string;
  speed: number;
  posture: string | null;
  zones: string[];
  activities: string[];
}

export interface ObservationsResponse extends Availability {
  count?: number;
  observations?: StoredObservation[];
}

// ---------------------------------------------------------------------------
// analytics
// ---------------------------------------------------------------------------

export interface AnalyticsBucket {
  /** Unix seconds at the start of the bucket. */
  start: number;
  value: number;
  samples: number;
  /**
   * A measured zero rather than an absence. `samples === 0 && !known_zero` is
   * the one case the chart must draw as a gap, not as a floor.
   */
  known_zero: boolean;
}

export interface AnalyticsAnomaly {
  start: number;
  observed: number;
  expected: number;
  direction: 'above' | 'below';
  score: number;
  severity: string;
}

export interface AnalyticsResponse extends Availability {
  metric?: Metric;
  label?: string;
  interval_s?: number;
  /** Fraction of buckets in the window that hold a reading, 0..1. */
  coverage?: number;
  buckets?: AnalyticsBucket[];
  anomalies?: AnalyticsAnomaly[];
  /** False when no slot had enough history to compare against. */
  anomalies_available?: boolean;
  anomalies_reason?: string;
  judged?: number;
  unjudged?: number;
}

// ---------------------------------------------------------------------------
// entity timeline (store-backed; works on a single camera)
// ---------------------------------------------------------------------------

export interface TimelineSegment {
  start_time: number;
  end_time: number;
  duration_s: number;
  motion: string | null;
  mean_speed: number;
  posture: string | null;
  zones: string[];
  activities: string[];
  identity: string | null;
  observation_count: number;
}

export interface TimelineEvent {
  timestamp: number;
  rule: string;
  severity: Severity;
  summary: string;
  zone: string | null;
  evidence: Record<string, unknown>;
}

export interface EntityTimelineResponse extends Availability {
  found?: boolean;
  entity_id?: string;
  camera_id?: string;
  identity?: string | null;
  first_seen?: number;
  last_seen?: number;
  total_duration_s?: number;
  summary?: string;
  segments?: TimelineSegment[];
  events?: TimelineEvent[];
  message?: string;
}

// ---------------------------------------------------------------------------
// incidents
// ---------------------------------------------------------------------------

export interface IncidentTimelineEntry {
  entry_id: string;
  timestamp: number;
  event_id: number | string | null;
  event_type: string;
  camera_id: string;
  entities: string[];
  objects: string[];
  zone: string | null;
  summary: string;
  evidence_ref: {
    rule?: string;
    severity?: Severity;
    clip_url?: string | null;
    evidence?: Record<string, unknown>;
  };
}

/** `IncidentSeverityBreakdown.to_dict()`. */
export interface IncidentSeverityBreakdown {
  highest_event_severity: Severity;
  corroborating_event_count: number;
  involved_entity_count: number;
  restricted_zone_factor: number;
  escalation_factor: number;
  final_severity: Severity;
  severity_score: number;
}

/**
 * `IncidentCorrelationBreakdown.to_dict()` -- why the most recent event joined
 * this incident. Null on an incident that was spawned rather than attached to.
 */
export interface IncidentCorrelationBreakdown {
  entity_overlap_score: number;
  temporal_proximity_score: number;
  spatial_zone_score: number;
  relationship_score: number;
  behavior_scene_score: number;
  continuity_penalty: number;
  positive_score: number;
  total_correlation_score: number;
  decision: 'attach' | 'correlation_candidate' | 'new_incident';
  explanation: string;
}

export interface CorrelationCandidate {
  related_incident_id: string;
  score: number;
  explanation: string;
  timestamp: number;
}

export interface CanonicalIncident {
  incident_id: string;
  title: string;
  state: IncidentState;
  severity: Severity;
  severity_breakdown: IncidentSeverityBreakdown;
  first_seen: number;
  last_seen: number;
  duration_s: number;
  cameras: string[];
  zones: string[];
  involved_entities: string[];
  involved_objects: string[];
  event_count: number;
  timeline: IncidentTimelineEntry[];
  relationship_links: Array<Record<string, unknown>>;
  correlation_candidates: CorrelationCandidate[];
  merge_candidates: string[];
  evidence_dossier: Record<string, unknown>;
  correlation_breakdown: IncidentCorrelationBreakdown | null;
}

export interface IncidentsResponse extends Availability {
  count?: number;
  incidents?: CanonicalIncident[];
}

export interface IncidentDetailResponse extends Availability {
  found?: boolean;
  incident?: CanonicalIncident;
  incident_id?: string;
  message?: string;
}

// ---------------------------------------------------------------------------
// relationships
// ---------------------------------------------------------------------------

export interface RelationshipScoreBreakdown {
  co_occurrence_contribution: number;
  proximity_contribution: number;
  following_contribution: number;
  duration_contribution: number;
  total_raw_score: number;
  active_decayed_score: number;
  decay_factor: number;
}

export interface EntityRelationship {
  entity_a: string;
  entity_b: string;
  active_strength: number;
  historical_score: number;
  score_breakdown: RelationshipScoreBreakdown;
  primary_derived_pattern: string | null;
  first_observed: number;
  last_observed: number;
  co_occurrence_count: number;
  proximity_count: number;
  following_count: number;
  total_interaction_duration_s: number;
  evidence_summary: string;
  recent_signals: string[];
}

export interface RelationshipsListResponse extends Availability {
  count?: number;
  relationships?: EntityRelationship[];
}

export interface RelationshipNode {
  id: string;
  degree: number;
  max_strength: number;
}

export interface RelationshipEdge {
  source: string;
  target: string;
  active_strength: number;
  historical_score: number;
  pattern: string | null;
  co_occurrence_count: number;
  proximity_count: number;
  following_count: number;
  evidence_summary?: string;
  score_breakdown?: RelationshipScoreBreakdown;
}

export interface RelationshipGraph {
  timestamp: number;
  total_nodes: number;
  total_edges: number;
  nodes: RelationshipNode[];
  edges: RelationshipEdge[];
}

export interface RelationshipGraphResponse extends Availability {
  graph?: RelationshipGraph;
}

// ---------------------------------------------------------------------------
// multi-camera surfaces (facility pipeline only)
// ---------------------------------------------------------------------------

export interface EntitySnapshot {
  global_id: string;
  label: string;
  identity?: {
    canonical_id?: string;
    confidence?: number;
    appearance_prototypes_count?: number;
    first_seen_ts?: number;
    last_seen_ts?: number;
  };
  spatial?: { camera_id?: string; zone?: string | null; recent_cameras?: string[] };
  kinematics?: {
    speed_mps?: number;
    speed_h_s?: number;
    bearing_deg?: number | null;
    motion_state?: string;
    posture?: string | null;
  };
  activity?: { current_action?: string; dwell_time_s?: number };
  behavior?: { behaviors?: string[]; confidence?: number; evidence?: string };
  relationships?: {
    active_relationships?: Array<{
      other_entity: string;
      active_strength: number;
      pattern: string;
      evidence_summary?: string;
    }>;
  };
  journey?: { camera_sequence?: string[]; dwell_times?: Record<string, number> };
  incidents?: { incident_ids?: string[]; recent_incident_count?: number };
}

export interface EntitiesResponse extends Availability {
  count?: number;
  stats?: { total_entities?: number; named_entities?: number; global_associated?: number };
  entities?: EntitySnapshot[];
}

export interface EntityDetailResponse extends Availability {
  found?: boolean;
  entity?: EntitySnapshot;
}

/** `TransientInteractionEdge`, as `SceneGraphSnapshot.to_dict` emits it. */
export interface SceneInteractionEdge {
  source: string;
  target: string;
  relation: string;
  /** Separation in normalised image space, not metres. */
  distance: number;
  confidence: number;
  evidence: string;
}

export interface CollectiveBehavior {
  type: string;
  entities: string[];
  centroid: [number, number];
  confidence: number;
  evidence: string;
}

export interface UnattendedObject {
  object_id: string;
  label: string;
  owner_id: string | null;
  source: string;
  confidence: number;
  unattended_dwell_s: number;
  /** Owner's distance from the object, normalised to the frame. */
  owner_distance_norm: number;
}

export interface SceneGraphCamera {
  camera_id: string;
  timestamp: number;
  entity_count: number;
  active_edges: SceneInteractionEdge[];
  collective_behaviors: CollectiveBehavior[];
  unattended_objects: UnattendedObject[];
}

export interface SceneResponse extends Availability {
  camera_count?: number;
  cameras?: Record<string, SceneGraphCamera>;
  found?: boolean;
  scene?: SceneGraphCamera | null;
}

export interface TwinRoom {
  id: string;
  name: string;
  /** `[x0, z0, x1, z1]` on the ground plane, metres. */
  bounds: [number, number, number, number];
  floor_color: string;
  wall_color: string;
}

export interface TwinCamera {
  camera_id: string;
  name: string;
  /** `[x, y, z]` metres. */
  position: [number, number, number];
  yaw_deg: number;
  pitch_deg: number;
  fov_deg: number;
  range_m: number;
  color: string;
}

export interface TwinZone {
  zone_id: string;
  name: string;
  camera_id: string;
  zone_type: string;
  /** Ground-plane polygon, `[x, z]` pairs in metres. */
  polygon_3d: Array<[number, number]>;
  height_m: number;
  color: string;
  severity: string;
  occupancy: number;
}

export interface TwinEntity {
  entity_id: string;
  label: string;
  camera_id: string;
  /** `[x, y, z]` metres; `y` is always 0 -- entities sit on the floor. */
  position: [number, number, number];
  velocity: [number, number, number];
  speed: number;
  bearing_deg: number | null;
  motion: string;
  posture: string;
  last_seen: number;
}

export interface TwinResponse extends Availability {
  facility?: {
    width_m: number;
    depth_m: number;
    height_m: number;
    rooms: TwinRoom[];
    /** `[x0, z0, x1, z1, height]` per wall segment. */
    walls: Array<[number, number, number, number, number]>;
  };
  cameras?: TwinCamera[];
  zones?: TwinZone[];
  entities?: TwinEntity[];
  /** Entity id -> recent `[x, y, z]` waypoints. */
  trails?: Record<string, Array<[number, number, number]>>;
  timestamp?: number;
}

export interface RadarZone {
  camera_id: string;
  name: string;
  /** `[x0, y0, x1, y1]` of the camera's ground footprint, metres. */
  rect: [number, number, number, number];
  origin: [number, number];
}

export interface RadarEntity {
  id: string;
  label: string;
  camera: string;
  x: number;
  y: number;
  motion: string;
  speed: number;
  activity: string | null;
  trail: Array<[number, number]>;
}

export interface RadarResponse extends Availability {
  timestamp?: number;
  zones?: RadarZone[];
  entities?: RadarEntity[];
  active_count?: number;
}

export interface CamerasResponse extends Availability {
  /** False on a single-camera run: the roster is set by the CLI, not the page. */
  managed?: boolean;
  count?: number;
  cameras?: Array<{
    camera_id: string;
    name: string;
    uri: string | null;
    status?: string;
  }>;
}

export interface ZonesResponse extends Availability {
  count?: number;
  zones?: Array<Record<string, unknown>>;
}

// ---------------------------------------------------------------------------
// search / stats
// ---------------------------------------------------------------------------

export interface SearchResult {
  id: number | string;
  timestamp: number;
  camera_id: string;
  rule: string;
  severity: Severity;
  summary: string;
  entity_id: string | null;
  zone: string | null;
  score: number;
  evidence_clip?: string | null;
}

export interface SearchResponse extends Availability {
  query?: string;
  total?: number;
  results?: SearchResult[];
  parsed_intent?: {
    camera: string | null;
    severity: string | null;
    rule: string | null;
    entity: string | null;
    expanded_concepts: string[];
  };
}

export interface SystemStatsResponse extends Availability {
  camera_id?: string;
  uptime_s?: number;
  live?: boolean;
  store?: { events: number; observations: number; bytes: number } | null;
  schema_version?: number;
}
