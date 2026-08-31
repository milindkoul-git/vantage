"""Advanced Security Threat Rules: Tailgating, Wrong-Way Direction & Rapid Incursion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vantage.perception.contracts import BoundingBox
from vantage.state.contracts import EntityState


@dataclass
class PassageRecord:
    entity_id: str
    camera_id: str
    timestamp: float
    box: BoundingBox | None = None


class ThreatDetectionEngine:
    """Evaluates high-security threat patterns across multi-camera streams."""

    def __init__(self, tailgating_gap_s: float = 1.8) -> None:
        self.tailgating_gap_s = tailgating_gap_s
        self._doorway_passages: dict[
            str, list[PassageRecord]
        ] = {}  # camera_id -> list of passages
        self._threat_cooldowns: dict[
            tuple[str, str], float
        ] = {}  # (threat_type, entity_id) -> timestamp

    def evaluate_threats(
        self,
        camera_id: str,
        states: list[EntityState] | tuple[EntityState, ...],
        wall_time: float,
    ) -> list[dict[str, Any]]:
        """Evaluate tracks for security threats like tailgating, wrong-way, or rapid incursions."""
        threats: list[dict[str, Any]] = []

        # 1. Check for Tailgating near doorways/entries (e.g. cam_04_doorway)
        if "doorway" in camera_id.lower() or "entry" in camera_id.lower():
            passages = [
                p
                for p in self._doorway_passages.get(camera_id, [])
                if wall_time - p.timestamp <= 10.0
            ]

            for st in states:
                # If person is moving forward through doorway
                if st.speed > 0.15:
                    for prev in passages:
                        if prev.entity_id != st.entity_id:
                            dt = wall_time - prev.timestamp
                            if 0.1 <= dt <= self.tailgating_gap_s:
                                key = ("tailgating", st.entity_id)
                                if wall_time - self._threat_cooldowns.get(key, 0.0) > 25.0:
                                    self._threat_cooldowns[key] = wall_time
                                    threats.append(
                                        {
                                            "rule": "tailgating",
                                            "severity": "alert",
                                            "summary": f"Tailgating breach detected at {camera_id.upper()}: {st.entity_id.upper()} closely followed {prev.entity_id.upper()} ({dt:.1f}s gap)",
                                            "entity_id": st.entity_id,
                                            "related_id": prev.entity_id,
                                            "camera_id": camera_id,
                                            "evidence": {"gap_seconds": round(dt, 2)},
                                        }
                                    )
                    # Record this passage
                    passages.append(PassageRecord(st.entity_id, camera_id, wall_time))

            self._doorway_passages[camera_id] = passages

        # 2. Check for Wrong-Way directional flow in one-way corridor
        for st in states:
            bearing = st.bearing_deg
            if (
                "corridor" in camera_id.lower()
                and bearing is not None
                and (0.0 <= bearing <= 45.0 or 315.0 <= bearing <= 360.0)
                and st.speed > 0.45
                and st.dwell_s > 2.0
            ):
                # Moving backwards against corridor direction
                key = ("wrong_way_direction", st.entity_id)
                if wall_time - self._threat_cooldowns.get(key, 0.0) > 25.0:
                    self._threat_cooldowns[key] = wall_time
                    threats.append(
                        {
                            "rule": "wrong_way_direction",
                            "severity": "alert",
                            "summary": f"Wrong-Way transit detected in {camera_id.upper()} by {st.entity_id.upper()}",
                            "entity_id": st.entity_id,
                            "related_id": None,
                            "camera_id": camera_id,
                            "evidence": {"bearing_deg": round(bearing, 1)},
                        }
                    )

        return threats
