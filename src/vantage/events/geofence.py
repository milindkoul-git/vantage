"""Dynamic Polygon Geofence Engine & Spatial Rule Evaluation."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from vantage.spatial.geometry.coordinates import get_entity_foot_point
from vantage.spatial.geometry.polygon import Polygon

if TYPE_CHECKING:
    from vantage.events.zone_registry import ActiveZoneSnapshot
    from vantage.perception.contracts import BoundingBox
    from vantage.state.contracts import EntityState


class ZoneType(str, Enum):
    """Supported geofence spatial behavior archetypes."""

    EXCLUSION = "exclusion"  # Unauthorized entry into restricted boundary
    OCCUPANCY = "occupancy"  # Maximum distinct simultaneous entity limit
    DWELL = "dwell"  # Maximum dwell / loiter duration limit
    DIRECTIONAL = "directional"  # Allowed travel vector / one-way transit


@dataclass(frozen=True, slots=True)
class PolygonZone:
    """Immutable domain representation of an operator-defined polygonal geofence."""

    zone_id: str
    name: str
    camera_id: str  # specific camera id (e.g. 'cam_01_retail') or 'all'
    polygon: Polygon
    zone_type: ZoneType
    rule_config: dict[str, Any] = field(default_factory=dict)
    severity: str = "alert"  # "alert" | "notice" | "info"
    color: str = "#ff3b30"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "polygon": self.polygon.to_list(),
            "zone_type": self.zone_type.value,
            "rule_config": dict(self.rule_config),
            "severity": self.severity,
            "color": self.color,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PolygonZone:
        coords = data.get("polygon") or data.get("polygon_vertices") or []
        poly = Polygon.from_list(coords)
        z_type_str = str(data.get("zone_type", "exclusion")).lower()
        try:
            z_type = ZoneType(z_type_str)
        except ValueError:
            z_type = ZoneType.EXCLUSION

        return cls(
            zone_id=str(data["zone_id"]),
            name=str(data.get("name", data["zone_id"])),
            camera_id=str(data.get("camera_id", "all")),
            polygon=poly,
            zone_type=z_type,
            rule_config=dict(data.get("rule_config") or {}),
            severity=str(data.get("severity", "alert")),
            color=str(data.get("color", "#ff3b30")),
            updated_at=float(data.get("updated_at", time.time())),
        )


@dataclass(frozen=True, slots=True)
class GeofenceBreach:
    """Structured geofence rule violation finding compatible with Vantage event contracts."""

    rule: str
    severity: str
    summary: str
    entity_id: str
    camera_id: str
    zone_id: str
    zone_name: str
    wall_time: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "summary": self.summary,
            "entity_id": self.entity_id,
            "identity": self.entity_id.upper(),
            "camera_id": self.camera_id,
            "zone": self.zone_name.upper(),
            "zone_id": self.zone_id,
            "timestamp": self.wall_time,
            "evidence": self.evidence,
        }


class GeofenceEngine:
    """Evaluates tracked entities against active immutable zone snapshots with temporal state."""

    def __init__(self) -> None:
        # (zone_id, entity_id) -> first_entry_wall_time
        self._zone_dwell_entries: dict[tuple[str, str], float] = {}
        # (zone_id, entity_id) -> last_seen_wall_time (to survive brief tracker loss)
        self._zone_last_seen: dict[tuple[str, str], float] = {}
        # (zone_id, entity_id, rule_type) -> last_event_emit_time
        self._cooldowns: dict[tuple[str, str, str], float] = {}
        # zone_id -> last occupancy alert time
        self._occupancy_cooldowns: dict[str, float] = {}

    def evaluate_snapshot(
        self,
        snapshot: ActiveZoneSnapshot,
        camera_id: str,
        states: Sequence[EntityState],
        frame_width: int,
        frame_height: int,
        wall_time: float,
        boxes: dict[int, BoundingBox] | None = None,
    ) -> tuple[list[GeofenceBreach], dict[str, int]]:
        """Evaluate active zones in snapshot for a camera feed against tracked entities."""
        breaches: list[GeofenceBreach] = []
        occupancy_counts: dict[str, int] = {}

        zones = snapshot.get_zones_for_camera(camera_id)
        if not zones:
            return breaches, occupancy_counts

        # Clean up stale dwell records (> 5.0s absence)
        stale_keys = [
            k for k, last_t in self._zone_last_seen.items() if wall_time - last_t > 5.0
        ]
        for k in stale_keys:
            self._zone_last_seen.pop(k, None)
            self._zone_dwell_entries.pop(k, None)

        for zone in zones:
            entities_in_zone: list[EntityState] = []

            for st in states:
                # Find bounding box for entity
                b_box = None
                if boxes and st.track_id in boxes:
                    b_box = boxes[st.track_id]
                elif hasattr(st, "box"):
                    b_box = st.box

                if b_box is None:
                    continue

                # Foot contact point on ground plane
                foot = get_entity_foot_point(b_box, frame_width, frame_height, normalize=True)
                if zone.polygon.contains_point(foot):
                    entities_in_zone.append(st)

            occupancy_counts[zone.zone_id] = len(entities_in_zone)

            # 1. EXCLUSION RULE
            if zone.zone_type == ZoneType.EXCLUSION:
                for ent in entities_in_zone:
                    cooldown = float(zone.rule_config.get("cooldown_s", 20.0))
                    cd_key = (zone.zone_id, ent.entity_id, "exclusion")
                    if cd_key not in self._cooldowns or (
                        wall_time - self._cooldowns[cd_key] >= cooldown
                    ):
                        self._cooldowns[cd_key] = wall_time
                        breaches.append(
                            GeofenceBreach(
                                rule="geofence_exclusion",
                                severity=zone.severity,
                                summary=f"Exclusion breach in '{zone.name}': {ent.entity_id.upper()} entered restricted boundary",
                                entity_id=ent.entity_id,
                                camera_id=camera_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                wall_time=wall_time,
                                evidence={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.name,
                                    "zone_type": "exclusion",
                                    "speed": round(ent.speed, 2),
                                },
                            )
                        )

            # 2. OCCUPANCY CAPACITY RULE
            elif zone.zone_type == ZoneType.OCCUPANCY:
                max_occ = int(zone.rule_config.get("max_occupancy", 3))
                count = len(entities_in_zone)
                if count > max_occ:
                    cooldown = float(zone.rule_config.get("cooldown_s", 15.0))
                    if zone.zone_id not in self._occupancy_cooldowns or (
                        wall_time - self._occupancy_cooldowns[zone.zone_id] >= cooldown
                    ):
                        self._occupancy_cooldowns[zone.zone_id] = wall_time
                        lead_entity = (
                            entities_in_zone[0].entity_id if entities_in_zone else "unknown"
                        )
                        breaches.append(
                            GeofenceBreach(
                                rule="geofence_occupancy_limit",
                                severity=zone.severity,
                                summary=f"Max capacity exceeded in '{zone.name}': {count} entities present (limit: {max_occ})",
                                entity_id=lead_entity,
                                camera_id=camera_id,
                                zone_id=zone.zone_id,
                                zone_name=zone.name,
                                wall_time=wall_time,
                                evidence={
                                    "zone_id": zone.zone_id,
                                    "zone_name": zone.name,
                                    "current_occupancy": count,
                                    "max_occupancy": max_occ,
                                },
                            )
                        )

            # 3. DWELL / LOITERING RULE
            elif zone.zone_type == ZoneType.DWELL:
                max_dwell_s = float(zone.rule_config.get("max_dwell_s", 20.0))
                cooldown = float(zone.rule_config.get("cooldown_s", 30.0))

                for ent in entities_in_zone:
                    dwell_key = (zone.zone_id, ent.entity_id)
                    entry_t = self._zone_dwell_entries.get(dwell_key)
                    if entry_t is None:
                        entry_t = wall_time
                        self._zone_dwell_entries[dwell_key] = entry_t
                    self._zone_last_seen[dwell_key] = wall_time

                    dwell_duration = wall_time - entry_t
                    if dwell_duration >= max_dwell_s:
                        cd_key = (zone.zone_id, ent.entity_id, "dwell")
                        if cd_key not in self._cooldowns or (
                            wall_time - self._cooldowns[cd_key] >= cooldown
                        ):
                            self._cooldowns[cd_key] = wall_time
                            breaches.append(
                                GeofenceBreach(
                                    rule="geofence_dwell_violation",
                                    severity=zone.severity,
                                    summary=f"Dwell limit exceeded in '{zone.name}': {ent.entity_id.upper()} loitering ({dwell_duration:.1f}s, limit: {max_dwell_s:.0f}s)",
                                    entity_id=ent.entity_id,
                                    camera_id=camera_id,
                                    zone_id=zone.zone_id,
                                    zone_name=zone.name,
                                    wall_time=wall_time,
                                    evidence={
                                        "zone_id": zone.zone_id,
                                        "zone_name": zone.name,
                                        "dwell_s": round(dwell_duration, 1),
                                        "max_dwell_s": max_dwell_s,
                                    },
                                )
                            )

            # 4. DIRECTIONAL FLOW RULE
            elif zone.zone_type == ZoneType.DIRECTIONAL:
                allowed_deg = float(zone.rule_config.get("allowed_bearing_deg", 0.0))
                tolerance_deg = float(zone.rule_config.get("angular_tolerance_deg", 45.0))
                min_speed = float(zone.rule_config.get("min_speed", 0.20))
                cooldown = float(zone.rule_config.get("cooldown_s", 20.0))

                for ent in entities_in_zone:
                    bearing = ent.bearing_deg
                    if bearing is not None and ent.speed >= min_speed:
                        # Angular difference modulo 360
                        diff = abs((bearing - allowed_deg + 180) % 360 - 180)
                        if diff > tolerance_deg:
                            cd_key = (zone.zone_id, ent.entity_id, "directional")
                            if cd_key not in self._cooldowns or (
                                wall_time - self._cooldowns[cd_key] >= cooldown
                            ):
                                self._cooldowns[cd_key] = wall_time
                                breaches.append(
                                    GeofenceBreach(
                                        rule="geofence_direction_violation",
                                        severity=zone.severity,
                                        summary=f"Direction violation in '{zone.name}': {ent.entity_id.upper()} traveling against flow ({bearing:.0f}° vs allowed {allowed_deg:.0f}°)",
                                        entity_id=ent.entity_id,
                                        camera_id=camera_id,
                                        zone_id=zone.zone_id,
                                        zone_name=zone.name,
                                        wall_time=wall_time,
                                        evidence={
                                            "zone_id": zone.zone_id,
                                            "zone_name": zone.name,
                                            "actual_bearing_deg": round(bearing, 1),
                                            "allowed_bearing_deg": allowed_deg,
                                            "tolerance_deg": tolerance_deg,
                                            "speed": round(ent.speed, 2),
                                        },
                                    )
                                )

        return breaches, occupancy_counts
