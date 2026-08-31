"""Entity Context Manager: Thread-safe repository of all active entities."""

from __future__ import annotations

import threading
import time
from typing import Any

from vantage.core.logging import get_logger
from vantage.entity.context import EntityContext
from vantage.entity.contracts import EntitySnapshot, IdentityLevel
from vantage.perception.contracts import BoundingBox

log = get_logger(__name__)


class EntityContextManager:
    """Manages active entity contexts across single and multi-camera pipelines.

    Thread-safe, lock-minimized, and produces immutable `EntitySnapshot` collections
    for real-time dashboard telemetry without blocking worker loops.
    """

    def __init__(self, *, prune_timeout_s: float = 120.0) -> None:
        self._lock = threading.Lock()
        self._entities: dict[str, EntityContext] = {}  # global_id -> EntityContext
        self._local_to_global: dict[
            tuple[str, int], str
        ] = {}  # (camera_id, track_id) -> global_id
        self._prune_timeout_s = prune_timeout_s
        self._last_prune = time.time()

    def get_or_create(
        self,
        global_id: str,
        label: str,
        camera_id: str,
        track_id: int,
        box: BoundingBox,
        wall_time: float,
    ) -> EntityContext:
        """Retrieve existing EntityContext or register a new one."""
        with self._lock:
            # Map local key to global
            self._local_to_global[(camera_id, track_id)] = global_id

            if global_id not in self._entities:
                ctx = EntityContext(
                    global_id=global_id,
                    label=label,
                    initial_camera=camera_id,
                    initial_track_id=track_id,
                    initial_box=box,
                    wall_time=wall_time,
                )
                self._entities[global_id] = ctx
                log.debug(
                    f"Registered new entity context '{global_id}' ({label}) on {camera_id}"
                )
            else:
                ctx = self._entities[global_id]

            return ctx

    def get_by_local(self, camera_id: str, track_id: int) -> EntityContext | None:
        """Lookup entity context by camera and local track ID."""
        with self._lock:
            gid = self._local_to_global.get((camera_id, track_id))
            if gid:
                return self._entities.get(gid)
            return None

    def get_by_global(self, global_id: str) -> EntityContext | None:
        """Lookup entity context by global ID."""
        with self._lock:
            return self._entities.get(global_id)

    def get_snapshot(self, global_id: str) -> EntitySnapshot | None:
        """Get an immutable snapshot of a specific entity."""
        ctx = self.get_by_global(global_id)
        if ctx:
            return ctx.to_snapshot()
        return None

    def get_active_snapshots(self, active_within_s: float = 30.0) -> list[EntitySnapshot]:
        """Return immutable snapshots of all recently seen entities.

        Designed for sub-millisecond dashboard API responses.
        """
        now = time.time()
        with self._lock:
            contexts = list(self._entities.values())

        if not contexts:
            return []

        latest_seen = max(ctx.last_seen_wall for ctx in contexts)
        snapshots = []
        for ctx in contexts:
            if (now - ctx.last_seen_wall <= active_within_s) or (
                latest_seen - ctx.last_seen_wall <= active_within_s
            ):
                snapshots.append(ctx.to_snapshot())

        # Sort by most recently active first
        snapshots.sort(key=lambda s: s.timestamp, reverse=True)
        return snapshots

    def prune_stale(self, now: float | None = None) -> int:
        """Remove entities that have not been seen for longer than `prune_timeout_s`."""
        t_now = now or time.time()
        removed = 0
        with self._lock:
            stale_gids = [
                gid
                for gid, ctx in self._entities.items()
                if t_now - ctx.last_seen_wall > self._prune_timeout_s
            ]
            for gid in stale_gids:
                del self._entities[gid]
                removed += 1

            # Clean local-to-global mappings
            self._local_to_global = {
                k: v for k, v in self._local_to_global.items() if v in self._entities
            }

        if removed:
            log.debug(f"Pruned {removed} stale entity contexts from memory.")
        return removed

    def stats(self) -> dict[str, Any]:
        """Summary metrics of active entity contexts."""
        with self._lock:
            total = len(self._entities)
            named = sum(
                1
                for ctx in self._entities.values()
                if ctx.identity_level == IdentityLevel.NAMED_CONFIRMED
            )
            global_assoc = sum(
                1
                for ctx in self._entities.values()
                if ctx.identity_level == IdentityLevel.GLOBAL_ASSOCIATED
            )

        return {
            "total_entities": total,
            "named_entities": named,
            "global_associated": global_assoc,
        }
