"""Where enrolments and the audit trail live.

A separate database from the observation store, and that is a deliberate
boundary rather than an accident of implementation. The observation store is
bulk telemetry that is pruned on a schedule and might reasonably be shipped
somewhere for analysis. This file contains biometric templates. Keeping them
apart means the two can be handled, backed up, permissioned and deleted
differently, and it makes "delete the biometrics" a single file rather than a
DELETE against a table with ten million rows of something else in it.

The audit trail records rejections as well as matches. A trail that logged only
successes could not answer "did this system look at me and decide it did not
know me", which is a question someone is entitled to ask of a thing that watches
a room.
"""

from __future__ import annotations

import array
import contextlib
import sqlite3
import threading
import time
from pathlib import Path

from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.identity.contracts import (
    EMBEDDING_DIM,
    AuditAction,
    AuditRecord,
    Enrollment,
)

log = get_logger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS identity_schema (
    version INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS identities (
    name TEXT PRIMARY KEY,
    template BLOB NOT NULL,
    samples INTEGER NOT NULL,
    enrolled_at REAL NOT NULL,
    note TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS identity_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,
    name TEXT NOT NULL,
    entity_id TEXT,
    similarity REAL,
    detail TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_audit_time ON identity_audit (timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_name ON identity_audit (name, timestamp);
"""


class IdentityStore:
    """Enrolments and audit records, in their own SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        self._closed = False
        try:
            connection = self._connect()
            connection.executescript(_SCHEMA)
            row = connection.execute(
                "SELECT MAX(version) AS version FROM identity_schema"
            ).fetchone()
            if row is None or row["version"] is None:
                connection.execute(
                    "INSERT INTO identity_schema (version, applied_at) VALUES (?, ?)",
                    (SCHEMA_VERSION, time.time()),
                )
            elif row["version"] > SCHEMA_VERSION:
                raise VantageError(
                    f"identity database schema {row['version']} is newer than this "
                    f"build understands ({SCHEMA_VERSION})"
                )
        except sqlite3.DatabaseError as exc:
            raise VantageError(
                f"could not open the identity store at {self._path}: {exc}"
            ) from exc

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(str(self._path), timeout=5.0, isolation_level=None)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            self._local.connection = connection
            with self._lock:
                self._connections.append(connection)
        return connection

    def _require(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("identity store has been closed")
        return self._connect()

    # -- enrolments -------------------------------------------------------

    def enroll(self, enrollment: Enrollment) -> None:
        """Add or replace one enrolment, and record that it happened."""
        connection = self._require()
        blob = array.array("f", enrollment.template).tobytes()
        connection.execute(
            "INSERT OR REPLACE INTO identities (name, template, samples, enrolled_at, note) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                enrollment.name,
                blob,
                enrollment.samples,
                enrollment.enrolled_at,
                enrollment.note,
            ),
        )
        self.audit(
            AuditRecord(
                action=AuditAction.ENROLLED,
                name=enrollment.name,
                timestamp=enrollment.enrolled_at,
                detail=f"{enrollment.samples} samples",
            )
        )

    def revoke(self, name: str) -> bool:
        """Delete an enrolment. Returns whether one existed.

        The template row is removed outright rather than flagged. A "deleted"
        biometric that is still in the file is not deleted, and this is the one
        table in the project where that distinction is not a matter of taste.
        The audit entry recording the revocation stays, because erasing the
        record of a deletion would defeat the point of having a trail.
        """
        connection = self._require()
        cursor = connection.execute("DELETE FROM identities WHERE name = ?", (name,))
        removed = cursor.rowcount > 0
        if removed:
            self.audit(
                AuditRecord(
                    action=AuditAction.REVOKED,
                    name=name,
                    timestamp=time.time(),
                    detail="template deleted",
                )
            )
        return removed

    def load(self) -> list[Enrollment]:
        rows = (
            self._require()
            .execute(
                "SELECT name, template, samples, enrolled_at, note FROM identities ORDER BY name"
            )
            .fetchall()
        )
        loaded: list[Enrollment] = []
        for row in rows:
            values = array.array("f")
            try:
                values.frombytes(row["template"])
            except (ValueError, TypeError):
                # A blob whose length is not a multiple of the item size raises
                # rather than producing a short array, so a single corrupt row
                # made the entire gallery unloadable - everyone unrecognisable
                # because of one bad record.
                log.warning(
                    "skipping an unreadable enrolment",
                    extra={
                        "vantage_fields": {"name": row["name"], "reason": "corrupt template"}
                    },
                )
                continue
            if len(values) != EMBEDDING_DIM:
                log.warning(
                    "skipping a malformed enrolment",
                    extra={
                        "vantage_fields": {
                            "name": row["name"],
                            "dimensions": len(values),
                            "expected": EMBEDDING_DIM,
                        }
                    },
                )
                continue
            loaded.append(
                Enrollment(
                    name=row["name"],
                    template=tuple(values),
                    samples=row["samples"],
                    enrolled_at=row["enrolled_at"],
                    note=row["note"] or "",
                )
            )
        return loaded

    def names(self) -> list[str]:
        rows = self._require().execute("SELECT name FROM identities ORDER BY name").fetchall()
        return [row["name"] for row in rows]

    # -- audit ------------------------------------------------------------

    def audit(self, record: AuditRecord) -> None:
        self._require().execute(
            "INSERT INTO identity_audit (timestamp, action, name, entity_id, similarity, detail) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.timestamp,
                record.action.value,
                record.name,
                record.entity_id,
                record.similarity,
                record.detail,
            ),
        )

    def audit_trail(
        self, since: float | None = None, name: str | None = None, limit: int = 100
    ) -> list[AuditRecord]:
        clauses: list[str] = []
        params: list[object] = []
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if name:
            clauses.append("name = ?")
            params.append(name)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = (
            self._require()
            .execute(
                f"SELECT * FROM identity_audit {where} ORDER BY timestamp DESC, id DESC LIMIT ?",
                (*params, limit),
            )
            .fetchall()
        )
        return [
            AuditRecord(
                action=AuditAction(row["action"]),
                name=row["name"],
                timestamp=row["timestamp"],
                detail=row["detail"] or "",
                entity_id=row["entity_id"],
                similarity=row["similarity"],
            )
            for row in rows
        ]

    def counts(self) -> dict[str, int]:
        connection = self._require()
        return {
            "identities": int(
                connection.execute("SELECT COUNT(*) AS n FROM identities").fetchone()["n"]
            ),
            "audit_records": int(
                connection.execute("SELECT COUNT(*) AS n FROM identity_audit").fetchone()["n"]
            ),
        }

    def close(self) -> None:
        with self._lock:
            self._closed = True
            connections, self._connections = self._connections, []
        for connection in connections:
            with contextlib.suppress(sqlite3.Error):
                connection.close()
        self._local = threading.local()

    def __enter__(self) -> IdentityStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
