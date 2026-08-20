"""Observation and event storage.

SQLite, chosen on the same grounds as every other dependency decision here: it
is in the standard library, needs no server, and is entirely adequate for the
load. :class:`~vantage.storage.contracts.Store` is a Protocol so a Postgres
backend can be added for the multi-camera phase without touching anything that
writes.

Three separations that matter:

* **The run loop never writes to disk.** It enqueues; a background thread
  batches and commits. A slow disk must not become dropped frames.
* **Events and observations are queued separately.** Observations are
  continuous and may be dropped; events are rare, already filtered by a rule,
  and a dropped one is logged as an error.
* **Volume is controlled by sampling, not by overflow.** Chosen sampling is
  reproducible; what a full queue discards depends on when the disk was busy.
"""

from vantage.storage.contracts import (
    Query,
    RecordKind,
    Store,
    StoredEvent,
    StoredObservation,
    Timeline,
    WriteStats,
)
from vantage.storage.schema import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "Query",
    "RecordKind",
    "Recorder",
    "SqliteStore",
    "Store",
    "StoreWriter",
    "StoredEvent",
    "StoredObservation",
    "Timeline",
    "WriteStats",
    "build_storage",
    "open_store",
]


def __getattr__(name: str):
    if name == "SqliteStore":
        from vantage.storage.sqlite_store import SqliteStore

        return SqliteStore
    if name == "StoreWriter":
        from vantage.storage.writer import StoreWriter

        return StoreWriter
    if name == "Recorder":
        from vantage.storage.recorder import Recorder

        return Recorder
    if name in ("build_storage", "open_store"):
        from vantage.storage import factory

        return getattr(factory, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
