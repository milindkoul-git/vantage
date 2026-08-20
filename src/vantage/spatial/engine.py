"""The spatial engine: tracks and poses in, a scene graph out.

Owns the clock and the assembly, the way the activity engine does, and for the
same reason: time is accumulated from the tracker's own elapsed values rather
than read from a clock, so a recorded source replays identically and a zone
dwell of thirty seconds means thirty seconds of *footage*.

Deliberately does not own the zones' meaning. A zone here is a named polygon
with a free-form ``kind``; whether ``restricted`` should raise an alarm is a
policy question, and policy belongs to the event engine of the next phase.
"""

from __future__ import annotations

from vantage.core.logging import get_logger
from vantage.pose.contracts import PoseResult
from vantage.spatial.analyzer import SpatialAnalyzer, SpatialParams, entity_spatial
from vantage.spatial.contracts import SpatialResult, Zone
from vantage.state.contracts import StateResult
from vantage.tracking.contracts import TrackingResult

log = get_logger(__name__)


class SpatialEngine:
    """Assigns zones and derives relations for each frame."""

    def __init__(self, analyzer: SpatialAnalyzer | None = None) -> None:
        self._analyzer = analyzer or SpatialAnalyzer()
        self._elapsed = 0.0

    @property
    def analyzer(self) -> SpatialAnalyzer:
        return self._analyzer

    @property
    def zones(self) -> tuple[Zone, ...]:
        return self._analyzer.zones

    @property
    def elapsed_s(self) -> float:
        return self._elapsed

    def update(
        self,
        tracking: TrackingResult,
        pose: PoseResult | None = None,
        state: "StateResult | None" = None,
    ) -> SpatialResult:
        """Advance by ``tracking.elapsed_s`` and report the scene graph.

        ``state`` is optional but materially changes what can be claimed:
        without it, interaction is only reported when a wrist landmark confirms
        a reach, because proximity alone cannot tell lingering from passing.
        """
        elapsed = max(0.0, tracking.elapsed_s)
        self._elapsed += elapsed

        poses = pose.by_track() if pose is not None else {}
        motion = (
            {entity.track_id: entity.motion for entity in state} if state is not None else {}
        )
        tracks = tracking.tracks

        zones = self._analyzer.assign_zones(tracks, tracking.frame_size, self._elapsed)
        relations, considered = self._analyzer.relations(
            tracks, poses, motion, self._elapsed, elapsed
        )

        # Symmetric relations are generated once per pair already, but a
        # deduplicating pass keeps that a property of the result rather than a
        # property of how the loop happens to be written.
        unique: dict[tuple[str, int, int], object] = {}
        deduped = []
        for relation in relations:
            if relation.key in unique:
                continue
            unique[relation.key] = relation
            deduped.append(relation)

        return SpatialResult(
            entities=tuple(
                entity_spatial(track, zones.get(track.track_id, ())) for track in tracks
            ),
            relations=tuple(deduped),
            source_id=tracking.source_id,
            frame_index=tracking.frame_index,
            capture_wall=tracking.capture_wall,
            elapsed_s=elapsed,
            zones_defined=len(self._analyzer.zones),
            pose_available=pose is not None,
            state_available=state is not None,
            metadata={
                "entities_paired": considered,
                "entities_total": len(tracks),
                "elapsed_total_s": round(self._elapsed, 2),
            },
        )

    def reset(self) -> None:
        self._analyzer.reset()
        self._elapsed = 0.0


def build_spatial_engine(config=None) -> SpatialEngine:
    """Construct from a :class:`~vantage.config.schema.SpatialConfig`."""
    if config is None:
        return SpatialEngine()
    zones = tuple(
        Zone(
            name=zone.name,
            points=tuple((float(x), float(y)) for x, y in zone.points),
            kind=zone.kind,
        )
        for zone in config.zones
    )
    engine = SpatialEngine(
        SpatialAnalyzer(
            zones,
            SpatialParams(
                near_distance=config.near_distance,
                near_hysteresis=config.near_hysteresis,
                approach_rate=config.approach_rate,
                approach_window_s=config.approach_window_s,
                interact_distance=config.interact_distance,
                interact_s=config.interact_s,
                reach_confidence=config.reach_confidence,
                zone_event_hold_s=config.zone_event_hold_s,
                max_entities=config.max_entities,
                history=config.history,
            ),
        )
    )
    if zones:
        log.info(
            "spatial zones loaded",
            extra={"vantage_fields": {"zones": ", ".join(z.name for z in zones)}},
        )
    return engine
