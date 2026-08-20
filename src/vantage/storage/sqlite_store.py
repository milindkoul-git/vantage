"""The SQLite store: batched writes, indexed reads, bounded retention."""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.storage.contracts import Query, StoredEvent, StoredObservation
from vantage.storage.schema import (
    SCHEMA_VERSION,
    connect,
    initialise,
    like_term,
    unwrap_list,
)

log = get_logger(__name__)

_EVENT_COLUMNS = (
    "timestamp",
    "camera_id",
    "rule",
    "severity",
    "summary",
    "entity_id",
    "identity",
    "related_id",
    "zone",
    "frame_index",
    "elapsed_s",
    "evidence",
)

_OBSERVATION_COLUMNS = (
    "timestamp",
    "camera_id",
    "entity_id",
    "identity",
    "entity_type",
    "motion",
    "speed",
    "posture",
    "zones",
    "activities",
    "frame_index",
    "elapsed_s",
)


class SqliteStore:
    """A file-backed store with one connection **per thread**.

    SQLite refuses to let a connection cross threads, and this store is used
    from two: the caller creates it, then a background writer thread does the
    inserts. The first version shared one connection and failed on the first
    batch with "SQLite objects created in a thread can only be used in that same
    thread" - which the functional test caught immediately and no unit test
    would have, because unit tests do not cross threads.

    ``check_same_thread=False`` was the tempting one-word fix and is wrong: it
    silences the guard without making anything safe, leaving two threads sharing
    one transaction state. A connection per thread is genuinely safe, and WAL is
    what lets those connections coexist without blocking each other.
    """

    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self._path = Path(path).expanduser()
        self._read_only = read_only
        if not read_only:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        elif not self._path.is_file():
            raise VantageError(
                f"no store at {self._path}. Runs write one only when "
                "storage.enabled is set (or --store is passed)."
            )

        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._closed = False

        try:
            self._version = initialise(self._connect(), time.time())
        except VantageError:
            # Already actionable - a version mismatch or a missing migration.
            # Re-wrapping would bury the explanation inside a vaguer one.
            raise
        except sqlite3.DatabaseError as exc:
            raise VantageError(f"could not open store at {self._path}: {exc}") from exc

        log.debug(
            "store opened",
            extra={
                "vantage_fields": {
                    "path": str(self._path),
                    "schema_version": self._version,
                }
            },
        )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def schema_version(self) -> int:
        return self._version

    def _connect(self) -> sqlite3.Connection:
        """This thread's connection, opened on first use."""
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = connect(str(self._path))
            self._local.connection = connection
            # Tracked so close() can shut all of them, not just the caller's.
            with self._lock:
                self._connections.append(connection)
        return connection

    def _require(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("store has been closed")
        return self._connect()

    # -- writing ---------------------------------------------------------

    def write_events(self, records: list[dict[str, Any]]) -> int:
        """Insert a batch of events. One transaction for the whole batch."""
        if not records:
            return 0
        rows = [
            tuple(
                json.dumps(record.get("evidence") or {})
                if column == "evidence"
                else record.get(column)
                for column in _EVENT_COLUMNS
            )
            for record in records
        ]
        return self._insert("events", _EVENT_COLUMNS, rows)

    def write_observations(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        rows = [
            tuple(record.get(column) for column in _OBSERVATION_COLUMNS) for record in records
        ]
        return self._insert("observations", _OBSERVATION_COLUMNS, rows)

    def write_heartbeats(self, records: list[dict[str, Any]]) -> int:
        """Record that a camera was alive at these moments.

        The cheapest row in the schema, and the one that makes analytics
        honest: without it, an empty hour is indistinguishable from an hour the
        recorder spent dead, and no amount of cleverness over the observation
        rows can recover the difference.
        """
        if not records:
            return 0
        rows = [(r["camera_id"], float(r["timestamp"])) for r in records]
        return self._insert("heartbeat", ("camera_id", "timestamp"), rows)

    def heartbeats(self, since: float, until: float) -> list[float]:
        """Every heartbeat timestamp in a window, ascending."""
        rows = (
            self._require()
            .execute(
                "SELECT timestamp FROM heartbeat WHERE timestamp >= ? AND timestamp < ? "
                "ORDER BY timestamp",
                (since, until),
            )
            .fetchall()
        )
        return [float(row["timestamp"]) for row in rows]

    def _insert(self, table: str, columns: tuple[str, ...], rows: list[tuple]) -> int:
        connection = self._require()
        placeholders = ", ".join("?" for _ in columns)
        statement = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
        # One explicit transaction per batch. With isolation_level=None the
        # connection is in autocommit, so without this each row would be its own
        # transaction and its own fsync - the difference between hundreds of
        # rows a second and tens of thousands.
        connection.execute("BEGIN")
        try:
            connection.executemany(statement, rows)
        except sqlite3.Error:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return len(rows)

    # -- reading ---------------------------------------------------------

    def events(self, query: Query) -> list[StoredEvent]:
        where, params = _conditions(
            query,
            {
                "entity_id": query.entity_id,
                "rule": query.rule,
                "severity": query.severity,
                "zone": query.zone,
            },
        )
        order = "DESC" if query.newest_first else "ASC"
        sql = (
            "SELECT * FROM events"
            + (f" WHERE {where}" if where else "")
            + f" ORDER BY timestamp {order}, id {order} LIMIT ?"
        )
        rows = self._require().execute(sql, (*params, query.limit)).fetchall()
        return [_event_from_row(row) for row in rows]

    def observations(self, query: Query) -> list[StoredObservation]:
        extra = {"entity_id": query.entity_id, "entity_type": query.entity_type}
        where, params = _conditions(query, extra)
        if query.zone:
            # LIKE against the delimiter-wrapped column, so "till" cannot match
            # "till_annexe". See the note in schema.py.
            where = f"{where} AND zones LIKE ?" if where else "zones LIKE ?"
            params = (*params, like_term(query.zone))
        order = "DESC" if query.newest_first else "ASC"
        sql = (
            "SELECT * FROM observations"
            + (f" WHERE {where}" if where else "")
            + f" ORDER BY timestamp {order}, id {order} LIMIT ?"
        )
        rows = self._require().execute(sql, (*params, query.limit)).fetchall()
        return [_observation_from_row(row) for row in rows]

    def timeline(self, entity_id: str, limit: int = 500) -> list[StoredEvent]:
        """Every event for one entity, oldest first - a readable history."""
        return self.events(Query(entity_id=entity_id, limit=limit, newest_first=False))

    # -- housekeeping ----------------------------------------------------

    def counts(self) -> dict[str, int]:
        connection = self._require()
        result: dict[str, int] = {}
        for table in ("events", "observations"):
            row = connection.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            result[table] = int(row["n"])
        span = connection.execute(
            "SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM observations"
        ).fetchone()
        if span and span["lo"] is not None:
            result["span_s"] = int(span["hi"] - span["lo"])
        result["bytes"] = self.size_bytes()
        return result

    def size_bytes(self) -> int:
        """Total on-disk size, including the WAL sidecars.

        The main file alone is misleading under WAL: recently written rows live
        in ``-wal`` until a checkpoint, so a store holding five hundred rows
        reported 4096 bytes - the size of an empty database. Anyone using this
        to decide whether to prune would have concluded there was nothing there.
        """
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self._path) + suffix)
            if candidate.is_file():
                total += candidate.stat().st_size
        return total

    def prune(self, before: float) -> dict[str, int]:
        """Delete records older than ``before``. Returns rows removed per table.

        Retention is not optional in practice. A camera producing 120
        observation rows a second fills ten million rows a day, and a store that
        only grows is a disk-full outage with a long fuse. Events are pruned on
        the same call but are typically kept far longer - see
        :class:`~vantage.config.schema.StorageConfig`.
        """
        connection = self._require()
        removed: dict[str, int] = {}
        connection.execute("BEGIN")
        try:
            # Heartbeats are pruned with everything else. They are tiny, but
            # one a minute is half a million rows a year, and a table that only
            # the retention policy forgot is the one that fills the disk.
            for table in ("observations", "events", "heartbeat"):
                cursor = connection.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?", (before,)
                )
                removed[table] = cursor.rowcount
        except sqlite3.Error:
            connection.execute("ROLLBACK")
            raise
        connection.execute("COMMIT")
        return removed

    def vacuum(self) -> None:
        """Reclaim space after a large prune.

        Separate from :meth:`prune` on purpose: VACUUM rewrites the whole file
        and holds a lock for as long as that takes, which is not something to do
        implicitly in the middle of a live run.
        """
        self._require().execute("VACUUM")

    def close(self) -> None:
        """Close every thread's connection, not merely the caller's."""
        with self._lock:
            self._closed = True
            connections, self._connections = self._connections, []
        for connection in connections:
            # Closing a connection owned by a thread that has already gone can
            # fail; there is nothing to recover and nothing to report.
            with contextlib.suppress(sqlite3.Error):
                connection.close()
        self._local = threading.local()

    def __enter__(self) -> SqliteStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _conditions(query: Query, equals: dict[str, str | None]) -> tuple[str, tuple]:
    """Build the WHERE clause. Parameterised throughout - never interpolated."""
    clauses: list[str] = []
    params: list[Any] = []
    if query.since is not None:
        clauses.append("timestamp >= ?")
        params.append(query.since)
    if query.until is not None:
        clauses.append("timestamp <= ?")
        params.append(query.until)
    if query.camera_id:
        clauses.append("camera_id = ?")
        params.append(query.camera_id)
    for column, value in equals.items():
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    return " AND ".join(clauses), tuple(params)


def _event_from_row(row: sqlite3.Row) -> StoredEvent:
    try:
        evidence = json.loads(row["evidence"]) if row["evidence"] else {}
    except json.JSONDecodeError:
        # A row written by something that did not encode evidence as JSON.
        # Surfaced rather than crashing the whole query: one malformed row
        # should not make the other ten thousand unreadable.
        evidence = {"_unparsed": row["evidence"]}
    return StoredEvent(
        id=row["id"],
        timestamp=row["timestamp"],
        camera_id=row["camera_id"],
        rule=row["rule"],
        severity=row["severity"],
        summary=row["summary"],
        entity_id=row["entity_id"],
        identity=row["identity"],
        related_id=row["related_id"],
        zone=row["zone"],
        frame_index=row["frame_index"],
        elapsed_s=row["elapsed_s"],
        evidence=evidence,
    )


def _observation_from_row(row: sqlite3.Row) -> StoredObservation:
    return StoredObservation(
        id=row["id"],
        timestamp=row["timestamp"],
        camera_id=row["camera_id"],
        entity_id=row["entity_id"],
        identity=row["identity"],
        entity_type=row["entity_type"],
        motion=row["motion"],
        speed=row["speed"],
        posture=row["posture"],
        zones=", ".join(unwrap_list(row["zones"])) or None,
        activities=", ".join(unwrap_list(row["activities"])) or None,
        frame_index=row["frame_index"],
        elapsed_s=row["elapsed_s"],
    )


__all__ = ["SCHEMA_VERSION", "SqliteStore"]
