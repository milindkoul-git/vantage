/**
 * Turning API records into the shapes the dossiers render.
 *
 * The previous version of `buildIncidentEvidenceViewModel` ignored its argument
 * and returned a fixed table -- five weighted factors, three penalties, and
 * sentences like "Observation gap 38s (<180s timeout)" and "Re-ID confidence
 * 0.94" -- for every incident it was ever given. That table was the incident
 * drawer's headline "explainability" panel, and none of it had been measured.
 *
 * The real numbers exist. `IncidentCorrelationBreakdown` is what the correlator
 * computed when it decided this event belonged to this incident, and
 * `IncidentSeverityBreakdown` is how the severity was arrived at. Both are on
 * the incident. What follows only rearranges them, and where the backend has not
 * recorded one, the caller is told so rather than handed a plausible-looking
 * substitute.
 */

import type {
  CanonicalIncident,
  EntityRelationship,
  IncidentCorrelationBreakdown,
  RelationshipEdge,
} from '../../contracts/types';
import type { Severity } from '../../contracts/vocabulary';

export interface Factor {
  name: string;
  /** The correlator's configured weight for this factor. */
  weight: number;
  /** What it scored, 0..1. */
  score: number;
  /** weight x score, i.e. what it actually put into the total. */
  contribution: number;
  why: string;
}

export interface IncidentEvidenceViewModel {
  incidentId: string;
  title: string;
  severity: Severity;
  state: string;
  durationSeconds: number;
  involvedEntities: string[];
  cameras: string[];
  zones: string[];
  eventCount: number;
  /** Null when this incident was opened rather than joined -- there was nothing to correlate against. */
  correlation: {
    factors: Factor[];
    penalty: number;
    positive: number;
    total: number;
    decision: IncidentCorrelationBreakdown['decision'];
    explanation: string;
  } | null;
  severityFactors: Array<{ name: string; value: string }>;
  severityScore: number;
}

/**
 * The weights in `IncidentCorrelatorConfig`. Mirrored rather than sent because
 * they are configuration, not per-incident data; the scores beside them are the
 * measured half. If the config is retuned these need retuning with it, which is
 * why each one names the config key it comes from.
 */
const FACTOR_WEIGHTS = {
  entity_overlap_score: { weight: 0.35, name: 'Entity overlap', why: 'incidents.entity_overlap_weight' },
  temporal_proximity_score: {
    weight: 0.2,
    name: 'Temporal proximity',
    why: 'incidents.temporal_proximity_weight',
  },
  spatial_zone_score: { weight: 0.15, name: 'Spatial continuity', why: 'incidents.spatial_zone_weight' },
  relationship_score: { weight: 0.15, name: 'Known association', why: 'incidents.relationship_weight' },
  behavior_scene_score: { weight: 0.15, name: 'Behaviour match', why: 'incidents.behavior_scene_weight' },
} as const;

export function buildIncidentEvidenceViewModel(
  incident: CanonicalIncident,
): IncidentEvidenceViewModel {
  const breakdown = incident.correlation_breakdown;
  const severity = incident.severity_breakdown;

  return {
    incidentId: incident.incident_id,
    title: incident.title,
    severity: incident.severity,
    state: incident.state,
    durationSeconds: incident.duration_s,
    involvedEntities: incident.involved_entities,
    cameras: incident.cameras,
    zones: incident.zones,
    eventCount: incident.event_count,
    correlation: breakdown
      ? {
          factors: (Object.keys(FACTOR_WEIGHTS) as Array<keyof typeof FACTOR_WEIGHTS>).map((key) => {
            const meta = FACTOR_WEIGHTS[key];
            const score = breakdown[key];
            return {
              name: meta.name,
              weight: meta.weight,
              score,
              contribution: score * meta.weight,
              why: meta.why,
            };
          }),
          penalty: breakdown.continuity_penalty,
          positive: breakdown.positive_score,
          total: breakdown.total_correlation_score,
          decision: breakdown.decision,
          explanation: breakdown.explanation,
        }
      : null,
    severityFactors: severity
      ? [
          { name: 'Worst constituent event', value: severity.highest_event_severity },
          { name: 'Corroborating events', value: String(severity.corroborating_event_count) },
          { name: 'Entities involved', value: String(severity.involved_entity_count) },
          { name: 'Restricted-zone factor', value: severity.restricted_zone_factor.toFixed(2) },
          { name: 'Escalation factor', value: severity.escalation_factor.toFixed(2) },
        ]
      : [],
    severityScore: severity?.severity_score ?? 0,
  };
}

export interface RelationshipEvidenceViewModel {
  entityA: string;
  entityB: string;
  activeStrength: number;
  historicalScore: number;
  pattern: string | null;
  /** Each contribution as a share of the raw total, so the bars sum to 100%. */
  contributions: Array<{ name: string; value: number; share: number }>;
  decayFactor: number;
  coOccurrenceCount: number;
  proximityCount: number;
  followingCount: number;
  interactionSeconds: number;
  evidenceSummary: string;
}

export function buildRelationshipEvidenceViewModel(
  relationship: EntityRelationship,
): RelationshipEvidenceViewModel {
  const b = relationship.score_breakdown;
  const parts = [
    { name: 'Co-occurrence', value: b.co_occurrence_contribution },
    { name: 'Proximity', value: b.proximity_contribution },
    { name: 'Following', value: b.following_contribution },
    { name: 'Duration', value: b.duration_contribution },
  ];
  // Share of the total that was actually scored. Dividing by a constant would
  // make four zero contributions render as four full bars.
  const raw = parts.reduce((sum, part) => sum + part.value, 0);
  return {
    entityA: relationship.entity_a,
    entityB: relationship.entity_b,
    activeStrength: relationship.active_strength,
    historicalScore: relationship.historical_score,
    pattern: relationship.primary_derived_pattern,
    contributions: parts.map((part) => ({
      ...part,
      share: raw > 0 ? part.value / raw : 0,
    })),
    decayFactor: b.decay_factor,
    coOccurrenceCount: relationship.co_occurrence_count,
    proximityCount: relationship.proximity_count,
    followingCount: relationship.following_count,
    interactionSeconds: relationship.total_interaction_duration_s,
    evidenceSummary: relationship.evidence_summary,
  };
}

/** Graph edges carry a thinner record than the list endpoint; keep them apart. */
export function edgeLabel(edge: RelationshipEdge): string {
  const pattern = edge.pattern ?? 'associated';
  return `${pattern.replace(/_/g, ' ')} · ${edge.active_strength.toFixed(2)}`;
}
