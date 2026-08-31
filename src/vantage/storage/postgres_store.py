"""PostgreSQL / TimescaleDB store implementation of the Store protocol.

Provides scalable multi-camera time-series storage for observations, events,
heartbeats, and persistent relationships.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vantage.core.logging import get_logger
from vantage.storage.contracts import Query, StoredEvent, StoredObservation

log = get_logger(__name__)

POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at DOUBLE PRECISION NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL,
    timestamp DOUBLE PRECISION NOT NULL,
    camera_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    entity_id TEXT,
    identity TEXT,
    related_id TEXT,
    zone TEXT,
    frame_index BIGINT NOT NULL,
    elapsed_s DOUBLE PRECISION NOT NULL,
    evidence JSONB,
    PRIMARY KEY (id, timestamp)
);

CREATE TABLE IF NOT EXISTS observations (
    id BIGSERIAL,
    timestamp DOUBLE PRECISION NOT NULL,
    camera_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    identity TEXT,
    entity_type TEXT NOT NULL,
    motion TEXT,
    speed DOUBLE PRECISION,
    posture TEXT,
    zones TEXT,
    activities TEXT,
    frame_index BIGINT NOT NULL,
    elapsed_s DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (id, timestamp)
);

CREATE TABLE IF NOT EXISTS heartbeat (
    camera_id TEXT NOT NULL,
    timestamp DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (camera_id, timestamp)
);

CREATE TABLE IF NOT EXISTS relationship_graph (
    id BIGSERIAL PRIMARY KEY,
    camera_id TEXT NOT NULL,
    entity_a TEXT NOT NULL,
    entity_b_or_zone TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    first_seen DOUBLE PRECISION NOT NULL,
    last_seen DOUBLE PRECISION NOT NULL,
    occurrence_count BIGINT NOT NULL,
    max_confidence_tier DOUBLE PRECISION NOT NULL,
    evidence JSONB
);

-- TimescaleDB hypertables (optional extension)
-- SELECT create_hypertable('observations', 'timestamp', if_not_exists => TRUE);
-- SELECT create_hypertable('events', 'timestamp', if_not_exists => TRUE);
"""


class PostgresStore:
    """A PostgreSQL store satisfying the :class:`Store` protocol."""

    def __init__(self, connection_or_pool: Any) -> None:
        self._pool = connection_or_pool
        self._closed = False

    def write_events(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        statement = """
        INSERT INTO events (
            timestamp, camera_id, rule, severity, summary, entity_id, identity,
            related_id, zone, frame_index, elapsed_s, evidence
        ) VALUES (
            %(timestamp)s, %(camera_id)s, %(rule)s, %(severity)s, %(summary)s, %(entity_id)s, %(identity)s,
            %(related_id)s, %(zone)s, %(frame_index)s, %(elapsed_s)s, %(evidence)s
        )
        """
        return self._execute_batch(statement, records)

    def write_observations(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        statement = """
        INSERT INTO observations (
            timestamp, camera_id, entity_id, identity, entity_type, motion, speed,
            posture, zones, activities, frame_index, elapsed_s
        ) VALUES (
            %(timestamp)s, %(camera_id)s, %(entity_id)s, %(identity)s, %(entity_type)s, %(motion)s, %(speed)s,
            %(posture)s, %(zones)s, %(activities)s, %(frame_index)s, %(elapsed_s)s
        )
        """
        return self._execute_batch(statement, records)

    def write_heartbeats(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        statement = """
        INSERT INTO heartbeat (camera_id, timestamp)
        VALUES (%(camera_id)s, %(timestamp)s)
        ON CONFLICT DO NOTHING
        """
        return self._execute_batch(statement, records)

    def write_relationships(self, records: list[dict[str, Any]]) -> int:
        if not records:
            return 0
        statement = """
        INSERT INTO relationship_graph (
            camera_id, entity_a, entity_b_or_zone, relation_type,
            first_seen, last_seen, occurrence_count, max_confidence_tier, evidence
        ) VALUES (
            %(camera_id)s, %(entity_a)s, %(entity_b_or_zone)s, %(relation_type)s,
            %(first_seen)s, %(last_seen)s, %(occurrence_count)s, %(max_confidence_tier)s, %(evidence)s
        )
        """
        return self._execute_batch(statement, records)

    def heartbeats(self, since: float, until: float) -> list[float]:
        query = "SELECT timestamp FROM heartbeat WHERE timestamp >= %s AND timestamp <= %s ORDER BY timestamp"
        rows = self._execute_query(query, (since, until))
        return [float(r[0]) for r in rows]

    def events(self, query: Query) -> list[StoredEvent]:
        clauses, params = _build_conditions(query)
        order = "DESC" if query.newest_first else "ASC"
        sql = f"SELECT * FROM events WHERE {clauses} ORDER BY timestamp {order} LIMIT {query.limit}"
        rows = self._execute_query(sql, params)
        return [_event_from_pg_row(r) for r in rows]

    def observations(self, query: Query) -> list[StoredObservation]:
        clauses, params = _build_conditions(query)
        order = "DESC" if query.newest_first else "ASC"
        sql = f"SELECT * FROM observations WHERE {clauses} ORDER BY timestamp {order} LIMIT {query.limit}"
        rows = self._execute_query(sql, params)
        return [_obs_from_pg_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        c_events = self._execute_query("SELECT COUNT(*) FROM events", ())[0][0]
        c_obs = self._execute_query("SELECT COUNT(*) FROM observations", ())[0][0]
        return {"events": int(c_events), "observations": int(c_obs)}

    def prune(self, before: float) -> dict[str, int]:
        del_ev = self._execute_mutation("DELETE FROM events WHERE timestamp < %s", (before,))
        del_obs = self._execute_mutation(
            "DELETE FROM observations WHERE timestamp < %s", (before,)
        )
        return {"events": del_ev, "observations": del_obs}

    def close(self) -> None:
        self._closed = True
        if hasattr(self._pool, "close"):
            self._pool.close()

    def _execute_batch(self, statement: str, records: list[dict[str, Any]]) -> int:
        if hasattr(self._pool, "executemany"):
            self._pool.executemany(statement, records)
            return len(records)
        return len(records)

    def _execute_query(self, query: str, params: Sequence[Any]) -> list[Any]:
        if hasattr(self._pool, "execute"):
            cursor = self._pool.execute(query, params)
            if hasattr(cursor, "fetchall"):
                return cursor.fetchall()
        return []

    def _execute_mutation(self, statement: str, params: tuple) -> int:
        if hasattr(self._pool, "execute"):
            cursor = self._pool.execute(statement, params)
            return getattr(cursor, "rowcount", 0)
        return 0


def _build_conditions(query: Query) -> tuple[str, list[Any]]:
    clauses = ["1=1"]
    params: list[Any] = []
    if query.since is not None:
        clauses.append("timestamp >= %s")
        params.append(query.since)
    if query.until is not None:
        clauses.append("timestamp <= %s")
        params.append(query.until)
    if query.camera_id:
        clauses.append("camera_id = %s")
        params.append(query.camera_id)
    if query.entity_id:
        clauses.append("entity_id = %s")
        params.append(query.entity_id)
    if query.rule:
        clauses.append("rule = %s")
        params.append(query.rule)
    if query.severity:
        clauses.append("severity = %s")
        params.append(query.severity)
    return " AND ".join(clauses), params


def _event_from_pg_row(row: Any) -> StoredEvent:
    if isinstance(row, dict):
        d = row
    else:
        # Tuple
        return StoredEvent(
            id=row[0],
            timestamp=row[1],
            camera_id=row[2],
            rule=row[3],
            severity=row[4],
            summary=row[5],
            entity_id=row[6],
            identity=row[7],
            related_id=row[8],
            zone=row[9],
            frame_index=row[10],
            elapsed_s=row[11],
            evidence=row[12] if isinstance(row[12], dict) else {},
        )
    return StoredEvent(
        id=d["id"],
        timestamp=d["timestamp"],
        camera_id=d["camera_id"],
        rule=d["rule"],
        severity=d["severity"],
        summary=d["summary"],
        entity_id=d["entity_id"],
        identity=d["identity"],
        related_id=d["related_id"],
        zone=d["zone"],
        frame_index=d["frame_index"],
        elapsed_s=d["elapsed_s"],
        evidence=d["evidence"] if isinstance(d.get("evidence"), dict) else {},
    )


def _obs_from_pg_row(row: Any) -> StoredObservation:
    if isinstance(row, dict):
        d = row
    else:
        return StoredObservation(
            id=row[0],
            timestamp=row[1],
            camera_id=row[2],
            entity_id=row[3],
            identity=row[4],
            entity_type=row[5],
            motion=row[6],
            speed=row[7],
            posture=row[8],
            zones=row[9],
            activities=row[10],
            frame_index=row[11],
            elapsed_s=row[12],
        )
    return StoredObservation(
        id=d["id"],
        timestamp=d["timestamp"],
        camera_id=d["camera_id"],
        entity_id=d["entity_id"],
        identity=d["identity"],
        entity_type=d["entity_type"],
        motion=d["motion"],
        speed=d["speed"],
        posture=d["posture"],
        zones=d["zones"],
        activities=d["activities"],
        frame_index=d["frame_index"],
        elapsed_s=d["elapsed_s"],
    )
