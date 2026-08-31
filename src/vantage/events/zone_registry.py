"""Thread-Safe In-Memory Zone Registry & Immutable Snapshot Manager."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from vantage.events.geofence import PolygonZone

log = logging.getLogger(__name__)


class ActiveZoneSnapshot:
    """Immutable snapshot of active geofence zones observed by camera workers."""

    __slots__ = ("_by_camera", "_created_at", "_version", "_zones")

    def __init__(self, zones: dict[str, PolygonZone], version: int = 1) -> None:
        self._zones: dict[str, PolygonZone] = dict(zones)
        by_cam: dict[str, list[PolygonZone]] = {}
        for z in zones.values():
            # 'all' means applied to all camera views
            c_key = z.camera_id
            if c_key not in by_cam:
                by_cam[c_key] = []
            by_cam[c_key].append(z)

        self._by_camera: dict[str, tuple[PolygonZone, ...]] = {
            k: tuple(v) for k, v in by_cam.items()
        }
        self._version: int = version
        self._created_at: float = time.time()

    @property
    def version(self) -> int:
        return self._version

    @property
    def created_at(self) -> float:
        return self._created_at

    def get_zone(self, zone_id: str) -> PolygonZone | None:
        return self._zones.get(zone_id)

    def list_all_zones(self) -> list[PolygonZone]:
        return list(self._zones.values())

    def get_zones_for_camera(self, camera_id: str) -> tuple[PolygonZone, ...]:
        """Return all zones matching this specific camera or global 'all' zones."""
        specific = self._by_camera.get(camera_id, ())
        global_zones = self._by_camera.get("all", ())
        if specific and global_zones:
            return specific + global_zones
        return specific or global_zones

    def count(self) -> int:
        return len(self._zones)


class ZoneRegistry:
    """Manages dynamic geofence configuration with atomic immutable snapshot publication.

    Hot-path Performance Contract:
    ------------------------------
    Camera workers call `registry.get_snapshot()` on each frame. This is a single
    reference read of an immutable object (0 disk I/O, 0 locks in read path).

    When an operator creates, edits, or deletes a zone:
    1. The zone is validated.
    2. Persisted to SQLite synchronously.
    3. A new immutable `ActiveZoneSnapshot` is instantiated.
    4. The active snapshot reference is atomically updated.

    Workers immediately see the new complete snapshot on their next frame without
    observing partial mutations or race conditions.
    """

    def __init__(self, store: Any | None = None) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._version_counter: int = 1
        self._active_snapshot: ActiveZoneSnapshot = ActiveZoneSnapshot({}, version=1)

        # Benchmarking / Metrics
        self.last_update_latency_ms: float = 0.0
        self.total_updates: int = 0

        # Load persisted zones at startup
        if self._store:
            self.load_persisted_zones()

    def get_snapshot(self) -> ActiveZoneSnapshot:
        """Fetch the current active immutable zone snapshot (sub-microsecond read)."""
        return self._active_snapshot

    def load_persisted_zones(self) -> ActiveZoneSnapshot:
        """Load stored zones from SQLite and publish initial active snapshot."""
        with self._lock:
            loaded_zones: dict[str, PolygonZone] = {}
            if self._store and hasattr(self._store, "list_zones"):
                try:
                    persisted = self._store.list_zones()
                    for z_dict in persisted:
                        try:
                            z = PolygonZone.from_dict(z_dict)
                            loaded_zones[z.zone_id] = z
                        except Exception as err:
                            log.warning(f"Skipping corrupted persisted zone: {err}")
                    log.info(
                        f"ZoneRegistry loaded {len(loaded_zones)} zones from SQLite store."
                    )
                except Exception as exc:
                    log.error(f"Failed to load persisted zones from store: {exc}")

            self._version_counter += 1
            self._active_snapshot = ActiveZoneSnapshot(
                loaded_zones, version=self._version_counter
            )
            return self._active_snapshot

    def save_zone(self, zone: PolygonZone) -> ActiveZoneSnapshot:
        """Validate, persist, and atomically publish a zone creation or update."""
        t_start = time.perf_counter()

        # 1. Validate geometry
        valid, msg = zone.polygon.is_valid()
        if not valid:
            raise ValueError(f"Invalid polygon geometry for zone '{zone.zone_id}': {msg}")

        with self._lock:
            # 2. Persist to storage
            if self._store and hasattr(self._store, "save_zone"):
                try:
                    self._store.save_zone(zone.to_dict())
                except Exception as exc:
                    log.error(f"Failed to persist zone {zone.zone_id} to store: {exc}")

            # 3. Create new snapshot dictionary
            current_zones = dict(self._active_snapshot._zones)
            current_zones[zone.zone_id] = zone

            # 4. Atomically swap snapshot
            self._version_counter += 1
            new_snapshot = ActiveZoneSnapshot(current_zones, version=self._version_counter)
            self._active_snapshot = new_snapshot

            self.last_update_latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.total_updates += 1
            log.info(
                f"Published zone snapshot v{new_snapshot.version} with {new_snapshot.count()} zones "
                f"(update latency: {self.last_update_latency_ms:.2f}ms)"
            )
            return new_snapshot

    def delete_zone(self, zone_id: str) -> ActiveZoneSnapshot:
        """Remove a zone, delete from storage, and atomically publish updated snapshot."""
        t_start = time.perf_counter()
        with self._lock:
            if self._store and hasattr(self._store, "delete_zone"):
                try:
                    self._store.delete_zone(zone_id)
                except Exception as exc:
                    log.error(f"Failed to delete zone {zone_id} from store: {exc}")

            current_zones = dict(self._active_snapshot._zones)
            current_zones.pop(zone_id, None)

            self._version_counter += 1
            new_snapshot = ActiveZoneSnapshot(current_zones, version=self._version_counter)
            self._active_snapshot = new_snapshot

            self.last_update_latency_ms = (time.perf_counter() - t_start) * 1000.0
            self.total_updates += 1
            log.info(
                f"Published zone snapshot v{new_snapshot.version} after deleting '{zone_id}' "
                f"(update latency: {self.last_update_latency_ms:.2f}ms)"
            )
            return new_snapshot
