"""Explainable Multi-Factor Incident Correlator with Negative Continuity Penalties."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vantage.incident.config import IncidentCorrelatorConfig
from vantage.incident.models import (
    CanonicalIncident,
    IncidentCorrelationBreakdown,
    IncidentCorrelationDecision,
)
from vantage.relationship.tracker import PersistentRelationshipTracker


class IncidentCorrelator:
    """Evaluates whether an incoming event belongs to an existing incident using multi-factor evidence."""

    def __init__(
        self,
        config: IncidentCorrelatorConfig | None = None,
        relationship_tracker: PersistentRelationshipTracker | None = None,
    ) -> None:
        self.config = config or IncidentCorrelatorConfig()
        self.relationship_tracker = relationship_tracker

    def evaluate_correlation(
        self,
        event: dict[str, Any],
        incident: CanonicalIncident,
        now: float,
    ) -> IncidentCorrelationBreakdown:
        """Compute explainable correlation breakdown between an event and an existing incident."""
        entity_id = event.get("entity_id")
        related_id = event.get("related_id")
        camera_id = event.get("camera_id", "default")
        zone = event.get("zone")
        ev_time = float(event.get("timestamp") or event.get("capture_wall") or now)
        evidence = event.get("evidence", {}) or {}

        # 1. Entity Overlap Score (Positive Weight: 0.35)
        s_ent = 0.0
        ent_reasons = []
        if entity_id and entity_id in incident.involved_entities:
            s_ent = 1.0
            ent_reasons.append(f"shared entity '{entity_id}'")
        elif related_id and related_id in incident.involved_entities:
            s_ent = 0.6
            ent_reasons.append(f"related entity '{related_id}'")

        # 2. Temporal Proximity Score (Positive Weight: 0.20)
        dt = max(0.0, ev_time - incident.last_seen)
        s_time = max(0.0, 1.0 - (dt / self.config.temporal_window_s))

        # 3. Spatial / Zone Score (Supporting Evidence Only - Positive Weight: 0.15)
        s_space = 0.0
        if zone and zone in incident.zones:
            s_space = 0.7  # Strong zone continuity, but never dominates alone
        elif camera_id in incident.cameras:
            s_space = 0.5
        else:
            s_space = 0.2

        # 4. Relationship Context Score (Positive Weight: 0.15)
        s_rel = 0.0
        if self.relationship_tracker and entity_id:
            max_rel_strength = 0.0
            for inc_ent in incident.involved_entities:
                if inc_ent != entity_id:
                    rel = self.relationship_tracker.get_relationship(
                        entity_id, inc_ent, now=now
                    )
                    if rel and rel.active_strength > max_rel_strength:
                        max_rel_strength = rel.active_strength
            s_rel = max_rel_strength

        # 5. Behavioral / Scene Continuity (Positive Weight: 0.15)
        str(event.get("rule", ""))
        s_behav = 0.0
        if s_ent > 0.0 or s_rel > 0.3:
            s_behav = 0.8
        else:
            s_behav = 0.1

        # Sum Positive Contributions
        pos_score = (
            self.config.entity_overlap_weight * s_ent
            + self.config.temporal_proximity_weight * s_time
            + self.config.spatial_zone_weight * s_space
            + self.config.relationship_weight * s_rel
            + self.config.behavior_scene_weight * s_behav
        )

        # 6. Negative Continuity Penalties (Contradictory Evidence)
        penalties = 0.0
        penalty_reasons = []

        # Implausible Camera Transition: different camera within < 2.0s
        last_cam = incident.timeline[-1].camera_id if incident.timeline else None
        if last_cam and last_cam != camera_id and dt < 2.0:
            penalties += self.config.impossible_speed_penalty
            penalty_reasons.append("physically implausible camera transition (<2s)")

        # Extended Temporal Gap (> 180s)
        if dt > 180.0:
            penalties += self.config.temporal_gap_penalty
            penalty_reasons.append(f"unexplained temporal gap ({dt:.1f}s)")

        # Degraded Identity / Coasting Track
        if evidence.get("is_coasting") or (
            evidence.get("identity_confidence") is not None
            and float(evidence["identity_confidence"]) < 0.40
        ):
            penalties += self.config.degraded_identity_penalty
            penalty_reasons.append("degraded identity confidence")

        total_score = max(0.0, pos_score - penalties)

        # 7. Decision Bands
        if total_score >= self.config.attach_threshold:
            decision = IncidentCorrelationDecision.ATTACH
        elif total_score >= self.config.candidate_threshold:
            decision = IncidentCorrelationDecision.CORRELATION_CANDIDATE
        else:
            decision = IncidentCorrelationDecision.NEW_INCIDENT

        # Build Explanation
        expl_parts = []
        if ent_reasons:
            expl_parts.extend(ent_reasons)
        if dt <= 30.0:
            expl_parts.append(f"occurred {dt:.1f}s after prior incident event")
        if zone and zone in incident.zones:
            expl_parts.append(f"same zone '{zone}'")
        elif camera_id in incident.cameras:
            expl_parts.append(f"same camera '{camera_id}'")
        if s_rel > 0.3:
            expl_parts.append(f"persistent relationship link (strength: {s_rel:.2f})")
        if penalty_reasons:
            expl_parts.append(f"penalties: {', '.join(penalty_reasons)}")

        explanation = (
            f"Correlation {total_score:.2f} ({decision.value}): " + "; ".join(expl_parts)
            if expl_parts
            else f"Correlation {total_score:.2f} ({decision.value})"
        )

        return IncidentCorrelationBreakdown(
            entity_overlap_score=s_ent,
            temporal_proximity_score=s_time,
            spatial_zone_score=s_space,
            relationship_score=s_rel,
            behavior_scene_score=s_behav,
            continuity_penalty=penalties,
            positive_score=pos_score,
            total_correlation_score=total_score,
            decision=decision,
            explanation=explanation,
        )

    def find_best_incident(
        self,
        event: dict[str, Any],
        active_incidents: Sequence[CanonicalIncident],
        now: float,
    ) -> tuple[CanonicalIncident | None, IncidentCorrelationBreakdown | None]:
        """Find the best matching active or quiescent incident for an incoming event."""
        best_inc: CanonicalIncident | None = None
        best_breakdown: IncidentCorrelationBreakdown | None = None
        highest_score = -1.0

        for inc in active_incidents:
            # Candidate gating: only evaluate active or quiescent incidents within resolution timeout
            if now - inc.last_seen > self.config.resolution_timeout_s:
                continue

            breakdown = self.evaluate_correlation(event, inc, now)
            if breakdown.total_correlation_score > highest_score:
                highest_score = breakdown.total_correlation_score
                best_inc = inc
                best_breakdown = breakdown

        return best_inc, best_breakdown

    def check_merge_candidates(
        self,
        inc_a: CanonicalIncident,
        inc_b: CanonicalIncident,
        now: float,
    ) -> tuple[bool, float, str]:
        """Evaluate if two distinct active incidents should be flagged as merge candidates."""
        if inc_a.incident_id == inc_b.incident_id:
            return False, 0.0, ""

        dt = abs(inc_a.last_seen - inc_b.last_seen)
        if dt > self.config.temporal_window_s:
            return False, 0.0, "Temporal gap too large"

        shared_entities = inc_a.involved_entities.intersection(inc_b.involved_entities)
        max_rel = 0.0
        if self.relationship_tracker:
            for ea in inc_a.involved_entities:
                for eb in inc_b.involved_entities:
                    if ea != eb:
                        rel = self.relationship_tracker.get_relationship(ea, eb, now=now)
                        if rel and rel.active_strength > max_rel:
                            max_rel = rel.active_strength

        # Merge score
        score = 0.0
        reasons = []
        if shared_entities:
            score += 0.50
            reasons.append(f"shared entities: {', '.join(shared_entities)}")
        if max_rel > 0.30:
            score += 0.35 * max_rel
            reasons.append(f"active relationship link ({max_rel:.2f})")
        if inc_a.cameras.intersection(inc_b.cameras):
            score += 0.15
            reasons.append("overlapping camera presence")

        is_merge_cand = score >= self.config.merge_threshold
        summary = (
            f"Merge candidate score {score:.2f}: " + "; ".join(reasons)
            if reasons
            else "No merge overlap"
        )
        return is_merge_cand, score, summary
