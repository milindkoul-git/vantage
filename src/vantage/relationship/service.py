"""High-Level Relationship Service coordinating Memory, Storage Persistence, and Graph Queries."""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

from vantage.relationship.config import FollowingDetectorConfig, RelationshipScoringConfig
from vantage.relationship.models import (
    EntityRelationship,
)
from vantage.relationship.tracker import PersistentRelationshipTracker
from vantage.storage.sqlite_store import SqliteStore

log = logging.getLogger(__name__)


class RelationshipService:
    """Coordinates persistent relationship intelligence, SQLite storage synchronization, and API graph queries."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        scoring_config: RelationshipScoringConfig | None = None,
        following_config: FollowingDetectorConfig | None = None,
        auto_persist_interval_s: float = 30.0,
        camera_id: str = "multi_camera",
        proximity_gate: float = 0.25,
    ) -> None:
        self.store = store
        self.camera_id = camera_id
        self.tracker = PersistentRelationshipTracker(
            scoring_config=scoring_config,
            following_config=following_config,
            proximity_gate=proximity_gate,
        )
        self.auto_persist_interval_s = auto_persist_interval_s
        self._last_persist_time = time.time()
        self._lock = threading.Lock()

        # Hydrate from SQLite if store is attached
        if self.store:
            self._hydrate_from_store()

    def _hydrate_from_store(self) -> None:
        """Load persistent relationship graph from disk on startup."""
        if not self.store:
            return
        try:
            stored = self.store.relationships(limit=500)
            now = time.time()
            with self.tracker._lock:
                for row in stored:
                    a = row.get("entity_a", "")
                    b = row.get("entity_b_or_zone", "")
                    if not a or not b or a == b:
                        continue
                    pair = (min(a, b), max(a, b))
                    co_count = row.get("occurrence_count", 1)
                    first_seen = row.get("first_seen", now)
                    last_seen = row.get("last_seen", now)

                    breakdown, pattern, summary = self.tracker.scorer.evaluate(
                        co_occurrence_count=co_count,
                        proximity_count=max(0, co_count // 2),
                        following_count=0,
                        total_duration_s=0.0,
                        last_observed=last_seen,
                        now=now,
                    )
                    rel = EntityRelationship(
                        entity_a=pair[0],
                        entity_b=pair[1],
                        active_strength=breakdown.active_decayed_score,
                        historical_score=breakdown.total_raw_score,
                        score_breakdown=breakdown,
                        primary_derived_pattern=pattern,
                        first_observed=first_seen,
                        last_observed=last_seen,
                        co_occurrence_count=co_count,
                        proximity_count=max(0, co_count // 2),
                        following_count=0,
                        total_interaction_duration_s=0.0,
                        signals=[],
                        evidence_summary=summary,
                    )
                    self.tracker._relationships[pair] = rel
        except Exception as exc:
            log.warning("could not hydrate relationship graph from store: %s", exc)

    def persist_to_store(self) -> int:
        """Flush current in-memory relationship state to SQLite storage."""
        if not self.store:
            return 0

        records: list[dict[str, Any]] = []
        now = time.time()
        with self.tracker._lock:
            for rel in self.tracker._relationships.values():
                evidence_dict = {
                    "score_breakdown": rel.score_breakdown.to_dict(),
                    "primary_pattern": (
                        rel.primary_derived_pattern.value
                        if rel.primary_derived_pattern
                        else None
                    ),
                    "proximity_count": rel.proximity_count,
                    "following_count": rel.following_count,
                    "duration_s": round(rel.total_interaction_duration_s, 2),
                    "summary": rel.evidence_summary,
                }
                records.append(
                    {
                        "camera_id": self.camera_id,
                        "entity_a": rel.entity_a,
                        "entity_b_or_zone": rel.entity_b,
                        "relation_type": (
                            rel.primary_derived_pattern.value
                            if rel.primary_derived_pattern
                            else "co_occurrence"
                        ),
                        "first_seen": rel.first_observed,
                        "last_seen": rel.last_observed,
                        "occurrence_count": rel.co_occurrence_count,
                        "max_confidence_tier": rel.historical_score,
                        "evidence": json.dumps(evidence_dict),
                    }
                )

        count = self.store.write_relationships(records)
        self._last_persist_time = now
        return count

    def maybe_persist(self, now: float | None = None) -> int:
        """Flush to the store if the persist interval has elapsed.

        Called every frame by the run loop; the interval is what keeps that from
        rewriting the whole graph thirty times a second. ``auto_persist_interval_s``
        existed from the start and nothing ever consulted it, so a run that was
        killed rather than closed lost the entire session's graph.
        """
        current = now if now is not None else time.time()
        if self.store is None:
            return 0
        if current - self._last_persist_time < self.auto_persist_interval_s:
            return 0
        return self.persist_to_store()

    def get_graph_snapshot(
        self, min_strength: float = 0.0, now: float | None = None
    ) -> dict[str, Any]:
        """Return graph nodes and weighted edges formatted for visualization and analytics."""
        t_now = now or time.time()
        rels = self.tracker.get_all_relationships(min_strength=min_strength, now=t_now)

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        for r in rels:
            # Register nodes
            for e_id in (r.entity_a, r.entity_b):
                if e_id not in nodes:
                    nodes[e_id] = {"id": e_id, "degree": 0, "max_strength": 0.0}
                nodes[e_id]["degree"] += 1
                nodes[e_id]["max_strength"] = max(
                    nodes[e_id]["max_strength"], r.active_strength
                )

            edges.append(
                {
                    "source": r.entity_a,
                    "target": r.entity_b,
                    "active_strength": round(r.active_strength, 3),
                    "historical_score": round(r.historical_score, 3),
                    "pattern": r.primary_derived_pattern.value
                    if r.primary_derived_pattern
                    else None,
                    "co_occurrence_count": r.co_occurrence_count,
                    "proximity_count": r.proximity_count,
                    "following_count": r.following_count,
                    "duration_s": round(r.total_interaction_duration_s, 1),
                    "summary": r.evidence_summary,
                    "breakdown": r.score_breakdown.to_dict(),
                }
            )

        return {
            "timestamp": round(t_now, 2),
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": list(nodes.values()),
            "edges": edges,
        }
