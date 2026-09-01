"""Which anonymous entities keep appearing together, scored over a session.

Thread-safe, LRU-bounded, and gated: comparing every pair every frame is
quadratic in a crowd, which is exactly the scene this is for.

Three gates decide what is even considered - a scene-graph edge, an association
already in memory, or simple proximity. The third used to read "pair everything
when there are five entities or fewer", and that cliff was the whole behaviour on
a single camera, because the first gate needs a scene graph that pipeline does
not build and the second cannot seed itself. Below six entities every pair was a
candidate, including two fragments of one person left by an id switch; at six and
above, nothing ever was.

MEASURED on real footage: a corridor holding one or two people produced 34
associations, all between fragments of the same person, while a pedestrian street
with 24 people in frame at once produced none at all, permanently. The subsystem
switched itself off in precisely the scene it exists for.

The gate is a bound on work, not a judgement about people. Measured at 0.06,
0.10, 0.15 and 0.25 of the frame across three clips, the number of pairs tracked
scales with it - a dense street runs 87 to 519 - while the number scoring above
0.20 stays at exactly one throughout. What is real is decided by the scorer.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Sequence

from vantage.events.contracts import EventCandidate
from vantage.relationship.config import FollowingDetectorConfig, RelationshipScoringConfig
from vantage.relationship.following import FollowingPatternDetector
from vantage.relationship.models import (
    EntityRelationship,
    ProximityBasis,
    RelationshipSignal,
    RelationshipSignalType,
)
from vantage.relationship.scoring import RelationshipScorer
from vantage.scene.graph import SceneGraphSnapshot


class PersistentRelationshipTracker:
    """Manages bipartite relationship lifecycles, candidate gating, and long-horizon persistence."""

    def __init__(
        self,
        scoring_config: RelationshipScoringConfig | None = None,
        following_config: FollowingDetectorConfig | None = None,
        max_relationships: int = 1000,
        cadence_s: float = 1.0,
        proximity_gate: float = 0.25,
        max_candidate_pairs: int = 400,
    ) -> None:
        self.scoring_config = scoring_config or RelationshipScoringConfig()
        self.following_config = following_config or FollowingDetectorConfig()
        self.max_relationships = max_relationships
        self.cadence_s = cadence_s
        if not 0.0 < proximity_gate <= 1.5:
            raise ValueError(
                f"proximity_gate is a fraction of the frame diagonal and must be in "
                f"(0, 1.5]; got {proximity_gate}"
            )
        self.proximity_gate = proximity_gate
        self.max_candidate_pairs = max_candidate_pairs

        self.scorer = RelationshipScorer(self.scoring_config)
        self.following_detector = FollowingPatternDetector(self.following_config)

        self._lock = threading.RLock()
        self._relationships: OrderedDict[tuple[str, str], EntityRelationship] = OrderedDict()
        self._last_evaluated_time: dict[str, float] = {}  # camera_id -> last_eval_time
        self._active_pair_dwells: dict[tuple[str, str], float] = {}  # pair -> start_time

    def _get_or_create(self, entity_a: str, entity_b: str, now: float) -> EntityRelationship:
        """Retrieve or initialize an undirected entity pair relationship."""
        # Canonical undirected pair ordering
        pair = (min(entity_a, entity_b), max(entity_a, entity_b))
        if pair in self._relationships:
            self._relationships.move_to_end(pair)
            return self._relationships[pair]

        # Enforce LRU cap
        if len(self._relationships) >= self.max_relationships:
            self._relationships.popitem(last=False)

        breakdown, pattern, summary = self.scorer.evaluate(
            co_occurrence_count=1,
            proximity_count=0,
            following_count=0,
            total_duration_s=0.0,
            last_observed=now,
            now=now,
        )

        rel = EntityRelationship(
            entity_a=pair[0],
            entity_b=pair[1],
            active_strength=breakdown.active_decayed_score,
            historical_score=breakdown.total_raw_score,
            score_breakdown=breakdown,
            primary_derived_pattern=pattern,
            first_observed=now,
            last_observed=now,
            co_occurrence_count=1,
            proximity_count=0,
            following_count=0,
            total_interaction_duration_s=0.0,
            signals=[],
            evidence_summary=summary,
        )
        self._relationships[pair] = rel
        return rel

    def process_frame(
        self,
        camera_id: str,
        active_entities: Sequence[
            tuple[str, float, float, float, float | None]
        ],  # (id, x, y, speed, bearing)
        scene_graph: SceneGraphSnapshot | None,
        entity_trajectories: dict[str, Sequence[tuple[float, float, float, float | None]]]
        | None,
        now: float,
        proximity_basis: ProximityBasis = ProximityBasis.NORMALIZED_IMAGE_SPACE,
    ) -> list[EventCandidate]:
        """Cadenced, gated relationship processing for active entities in a camera scene."""
        candidates: list[EventCandidate] = []
        if len(active_entities) < 2:
            return candidates

        with self._lock:
            # 1. Candidate Pair Gating (Prevent O(N^2) explosion)
            candidate_pairs: set[tuple[str, str]] = set()

            # Gate A: Pairs connected by transient scene graph interaction edges
            if scene_graph and scene_graph.active_edges:
                for edge in scene_graph.active_edges:
                    p = (
                        min(edge.source_id, edge.target_id),
                        max(edge.source_id, edge.target_id),
                    )
                    candidate_pairs.add(p)

            # Gate B: Pairs with existing active relationship in memory
            entity_ids = [e[0] for e in active_entities]
            for i in range(len(entity_ids)):
                for j in range(i + 1, len(entity_ids)):
                    p = (min(entity_ids[i], entity_ids[j]), max(entity_ids[i], entity_ids[j]))
                    if p in self._relationships:
                        candidate_pairs.add(p)

            # Gate C: entities close enough to each other to be together.
            #
            # This used to read `if len(entity_ids) <= 5`, and the cliff was the
            # whole behaviour. Below six entities every pair was a candidate,
            # including two fragments of one person left by an id switch; at six
            # and above nothing was, because Gate A needs a scene graph the
            # single-camera pipeline does not build and Gate B can only match
            # pairs that already exist. Measured: a clip with one or two people
            # produced 34 pairs, and a crowded street with 24 people at once
            # produced none, permanently.
            #
            # Proximity is the question that was meant to be asked. Distance is
            # in frame-diagonal fractions - the same normalised space the caller
            # supplies positions in - so it means the same thing at any
            # resolution.
            gate = self.proximity_gate
            positions = {e[0]: (e[1], e[2]) for e in active_entities}
            for i in range(len(entity_ids)):
                if len(candidate_pairs) >= self.max_candidate_pairs:
                    break
                ax, ay = positions[entity_ids[i]]
                for j in range(i + 1, len(entity_ids)):
                    bx, by = positions[entity_ids[j]]
                    if math.hypot(ax - bx, ay - by) > gate:
                        continue
                    candidate_pairs.add(
                        (
                            min(entity_ids[i], entity_ids[j]),
                            max(entity_ids[i], entity_ids[j]),
                        )
                    )

            # A hard ceiling on the work one frame can ask for. A packed
            # concourse is exactly when this must not become the frame budget.
            if len(candidate_pairs) > self.max_candidate_pairs:
                candidate_pairs = set(sorted(candidate_pairs)[: self.max_candidate_pairs])

            # 2. Process Gated Pairs
            for id_a, id_b in candidate_pairs:
                rel = self._get_or_create(id_a, id_b, now)
                prior_following = rel.following_count
                prior_proximity = rel.proximity_count

                # Co-occurrence registration (every evaluation)
                rel.co_occurrence_count += 1
                rel.last_observed = now

                # Check proximity from scene graph edges
                is_near = False
                near_dist = 1.0
                if scene_graph:
                    for edge in scene_graph.active_edges:
                        if (edge.source_id == id_a and edge.target_id == id_b) or (
                            edge.source_id == id_b and edge.target_id == id_a
                        ):
                            is_near = True
                            near_dist = edge.distance_norm
                            break

                if is_near:
                    rel.proximity_count += 1
                    dwell_start = self._active_pair_dwells.setdefault((id_a, id_b), now)
                    rel.total_interaction_duration_s += max(0.1, now - dwell_start)
                    self._active_pair_dwells[(id_a, id_b)] = now

                    sig = RelationshipSignal(
                        signal_type=RelationshipSignalType.RECURRENT_PROXIMITY,
                        timestamp=now,
                        camera_id=camera_id,
                        zone_id=None,
                        strength=0.85,
                        duration_s=round(rel.total_interaction_duration_s, 2),
                        proximity_basis=proximity_basis,
                        distance_metric=round(near_dist, 3),
                        evidence={"separation_norm": round(near_dist, 3)},
                    )
                    rel.signals.append(sig)
                else:
                    self._active_pair_dwells.pop((id_a, id_b), None)

                # Check following trajectory alignment
                if (
                    entity_trajectories
                    and id_a in entity_trajectories
                    and id_b in entity_trajectories
                ):
                    is_fol, fol_sig = self.following_detector.evaluate_trajectories(
                        id_a,
                        entity_trajectories[id_a],
                        id_b,
                        entity_trajectories[id_b],
                        camera_id=camera_id,
                        now=now,
                        proximity_basis=proximity_basis,
                    )
                    if not is_fol:
                        # Check reverse direction
                        is_fol, fol_sig = self.following_detector.evaluate_trajectories(
                            id_b,
                            entity_trajectories[id_b],
                            id_a,
                            entity_trajectories[id_a],
                            camera_id=camera_id,
                            now=now,
                            proximity_basis=proximity_basis,
                        )

                    if is_fol and fol_sig is not None:
                        rel.following_count += 1
                        rel.signals.append(fol_sig)

                # Keep sliding window of signals capped at 20
                if len(rel.signals) > 20:
                    rel.signals = rel.signals[-20:]

                # Re-evaluate explainable score and pattern
                breakdown, pattern, summary = self.scorer.evaluate(
                    co_occurrence_count=rel.co_occurrence_count,
                    proximity_count=rel.proximity_count,
                    following_count=rel.following_count,
                    total_duration_s=rel.total_interaction_duration_s,
                    last_observed=rel.last_observed,
                    now=now,
                )
                rel.score_breakdown = breakdown
                rel.active_strength = breakdown.active_decayed_score
                rel.historical_score = breakdown.total_raw_score
                rel.primary_derived_pattern = pattern
                rel.evidence_summary = summary

                # Generate event candidates on new milestone triggers
                if rel.following_count >= 2 and prior_following < 2:
                    candidates.append(
                        EventCandidate(
                            rule="following_pattern",
                            severity="notice",
                            summary=f"Following pattern observed between {id_a} and {id_b}",
                            entity_id=id_b,
                            camera_id=camera_id,
                            wall_time=now,
                            evidence={
                                "entity_a": id_a,
                                "entity_b": id_b,
                                "active_strength": round(rel.active_strength, 3),
                                "following_count": rel.following_count,
                                "summary": summary,
                            },
                        )
                    )
                elif rel.proximity_count >= 5 and prior_proximity < 5:
                    candidates.append(
                        EventCandidate(
                            rule="recurring_proximity",
                            severity="info",
                            summary=f"Recurring proximity ({rel.proximity_count}x) between {id_a} and {id_b}",
                            entity_id=id_a,
                            camera_id=camera_id,
                            wall_time=now,
                            evidence={
                                "entity_a": id_a,
                                "entity_b": id_b,
                                "active_strength": round(rel.active_strength, 3),
                                "proximity_count": rel.proximity_count,
                                "summary": summary,
                            },
                        )
                    )

        return candidates

    def get_relationship(
        self, entity_a: str, entity_b: str, now: float | None = None
    ) -> EntityRelationship | None:
        """Lookup relationship between two specific entities with up-to-date decay."""
        pair = (min(entity_a, entity_b), max(entity_a, entity_b))
        with self._lock:
            rel = self._relationships.get(pair)
            if rel and now is not None:
                breakdown, _pattern, _summary = self.scorer.evaluate(
                    co_occurrence_count=rel.co_occurrence_count,
                    proximity_count=rel.proximity_count,
                    following_count=rel.following_count,
                    total_duration_s=rel.total_interaction_duration_s,
                    last_observed=rel.last_observed,
                    now=now,
                )
                rel.score_breakdown = breakdown
                rel.active_strength = breakdown.active_decayed_score
            return rel

    def get_relationships_for_entity(
        self,
        entity_id: str,
        min_strength: float = 0.0,
        now: float | None = None,
    ) -> list[EntityRelationship]:
        """Return all relationships involving an entity."""
        results: list[EntityRelationship] = []
        with self._lock:
            for rel in self._relationships.values():
                if rel.entity_a == entity_id or rel.entity_b == entity_id:
                    if now is not None:
                        breakdown, _, _ = self.scorer.evaluate(
                            co_occurrence_count=rel.co_occurrence_count,
                            proximity_count=rel.proximity_count,
                            following_count=rel.following_count,
                            total_duration_s=rel.total_interaction_duration_s,
                            last_observed=rel.last_observed,
                            now=now,
                        )
                        rel.active_strength = breakdown.active_decayed_score
                    if (
                        rel.active_strength >= min_strength
                        or rel.historical_score >= min_strength
                    ):
                        results.append(rel)
        results.sort(key=lambda r: (r.active_strength, r.historical_score), reverse=True)
        return results

    def get_all_relationships(
        self, min_strength: float = 0.0, now: float | None = None
    ) -> list[EntityRelationship]:
        """Return all active relationships across the facility."""
        with self._lock:
            all_rels = list(self._relationships.values())
            if now is not None:
                for rel in all_rels:
                    breakdown, _, _ = self.scorer.evaluate(
                        co_occurrence_count=rel.co_occurrence_count,
                        proximity_count=rel.proximity_count,
                        following_count=rel.following_count,
                        total_duration_s=rel.total_interaction_duration_s,
                        last_observed=rel.last_observed,
                        now=now,
                    )
                    rel.active_strength = breakdown.active_decayed_score
            filtered = [
                r
                for r in all_rels
                if r.active_strength >= min_strength or r.historical_score >= min_strength
            ]
            filtered.sort(key=lambda r: (r.active_strength, r.historical_score), reverse=True)
            return filtered

    def get_top_associates(
        self, entity_id: str, limit: int = 5, now: float | None = None
    ) -> list[str]:
        """Return top counterpart entity IDs ranked by active relationship strength."""
        rels = self.get_relationships_for_entity(entity_id, now=now)
        associates = []
        for r in rels[:limit]:
            associates.append(r.other_entity(entity_id))
        return associates
