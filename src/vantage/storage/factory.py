"""Resolving configuration into a store, a writer and a recorder."""

from __future__ import annotations

from pathlib import Path

from vantage.core.logging import get_logger
from vantage.storage.recorder import Recorder
from vantage.storage.sqlite_store import SqliteStore
from vantage.storage.writer import StoreWriter

log = get_logger(__name__)


def open_store(path: str | Path, read_only: bool = False) -> SqliteStore:
    """Open the store at ``path``. Used by the CLI to query a live run's file."""
    return SqliteStore(path, read_only=read_only)


def build_storage(config, camera_id: str) -> tuple[SqliteStore, StoreWriter, Recorder]:
    """Construct the whole write path from a StorageConfig.

    Returned as a triple rather than one object because their lifetimes differ:
    the recorder is used per frame, the writer is flushed and closed at
    shutdown, and the store outlives both if the CLI is reading from it.
    """
    store = SqliteStore(config.path)
    writer = StoreWriter(
        store,
        batch_size=config.batch_size,
        flush_interval_s=config.flush_interval_s,
        observation_queue=config.observation_queue,
        event_queue=config.event_queue,
    )
    recorder = Recorder(
        writer,
        camera_id=camera_id,
        observation_interval=config.observation_interval,
        store_observations=config.store_observations,
    )
    log.info(
        "storage ready",
        extra={
            "vantage_fields": {
                "path": str(store.path),
                "schema_version": store.schema_version,
                "observation_interval": config.observation_interval,
                "observations": config.store_observations,
            }
        },
    )
    return store, writer, recorder
