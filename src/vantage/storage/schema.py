"""The database schema, and how it changes.

SQLite, and why
---------------
The specification offers SQLite or PostgreSQL. SQLite wins here on the same
grounds that decided every other dependency question in this project: it is in
the standard library, it needs no server, and it is entirely adequate for the
load. A single camera producing 120 observation rows a second is ten million a
day, which SQLite handles without complaint given the right pragmas and an index
on the column people actually filter by.

PostgreSQL becomes the right answer when several cameras write to one store, or
when the query load outgrows one process. That is a multi-camera concern -
Phase 12 - and :class:`~vantage.storage.contracts.Store` is a Protocol so it can
be added then without touching anything that writes.

Migrations, from the first version
-----------------------------------
There are two versions now and a ``schema_version`` table to record it. That looks
like ceremony for a schema nobody has changed yet, and it is exactly the thing
that is impossible to add later: by the time a migration is needed there are
databases in the field with no version marker, and no way to tell what shape
they are.

Denormalisation, deliberately
-----------------------------
``zones`` and ``activities`` are comma-joined strings rather than join tables. An
entity is typically in zero or one zone and doing one thing; a join table would
triple the write cost of the highest-volume table in the system to make a query
nobody has asked for slightly cleaner. Filtering uses ``LIKE`` on a delimited
string, which is why the values are stored wrapped in the delimiter - so that
searching for ``,till,`` cannot match ``,till_annexe,``.
"""

from __future__ import annotations

import sqlite3

from vantage.core.errors import VantageError

SCHEMA_VERSION = 2

DELIMITER = ","
"""Wraps every element of a denormalised list column.

Stored as ``,till,lobby,`` rather than ``till,lobby``, so a ``LIKE '%,till,%'``
cannot match ``till_annexe``. Without the wrapping, a zone whose name is a
prefix of another silently matches it, and the failure looks like a data problem
rather than a query one.
"""

_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    camera_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    severity TEXT NOT NULL,
    summary TEXT NOT NULL,
    entity_id TEXT,
    identity TEXT,
    related_id TEXT,
    zone TEXT,
    frame_index INTEGER NOT NULL,
    elapsed_s REAL NOT NULL,
    evidence TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    camera_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    identity TEXT,
    entity_type TEXT NOT NULL,
    motion TEXT,
    speed REAL,
    posture TEXT,
    zones TEXT,
    activities TEXT,
    frame_index INTEGER NOT NULL,
    elapsed_s REAL NOT NULL
);
"""

_INDEXES = """
-- Time is the first filter in essentially every query, and it is also what
-- retention prunes on, so both paths want it.
CREATE TABLE IF NOT EXISTS heartbeat (
    camera_id TEXT NOT NULL,
    timestamp REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_events_time ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_time ON observations (timestamp);

-- "What happened to person_17" and "show me the alerts" are the two questions
-- an operator actually asks. Both are compound with time, because a timeline is
-- always bounded.
CREATE INDEX IF NOT EXISTS idx_events_entity ON events (entity_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events (severity, timestamp);
CREATE INDEX IF NOT EXISTS idx_events_rule ON events (rule, timestamp);
CREATE INDEX IF NOT EXISTS idx_observations_entity ON observations (entity_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_heartbeat_time ON heartbeat (timestamp);
"""

_PRAGMAS = (
    # Write-ahead logging: readers do not block the writer, which matters
    # because the CLI queries the same file a live run is writing to.
    ("journal_mode", "WAL"),
    # NORMAL rather than FULL. FULL fsyncs on every commit, which at batch rates
    # dominates the write cost; NORMAL under WAL risks losing only the last
    # transaction on an OS crash, and the last transaction is a second of
    # observations rather than anything irreplaceable.
    ("synchronous", "NORMAL"),
    ("foreign_keys", "ON"),
    # Give SQLite room to wait out a concurrent writer instead of raising
    # "database is locked" at the caller.
    ("busy_timeout", "5000"),
)


def connect(path: str) -> sqlite3.Connection:
    """Open a connection with the pragmas this workload needs."""
    connection = sqlite3.connect(path, timeout=5.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    for name, value in _PRAGMAS:
        connection.execute(f"PRAGMA {name}={value}")
    return connection


def initialise(connection: sqlite3.Connection, now: float) -> int:
    """Create the schema if absent and return the version on disk.

    Raises:
        VantageError: the database was written by a newer version of this
            software. Refusing is the point - a schema this code does not
            understand may have columns it will silently ignore, and a store
            that quietly discards fields is worse than one that will not open.
    """
    connection.executescript(_TABLES)
    connection.executescript(_INDEXES)

    row = connection.execute("SELECT MAX(version) AS version FROM schema_version").fetchone()
    current = row["version"] if row and row["version"] is not None else None

    if current is None:
        connection.execute(
            "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
            (SCHEMA_VERSION, now),
        )
        return SCHEMA_VERSION

    if current > SCHEMA_VERSION:
        # VantageError rather than RuntimeError: the CLI prints these as a
        # message instead of a traceback, and "your database is from a newer
        # build" is something a user can act on.
        raise VantageError(
            f"database schema version {current} is newer than this build "
            f"understands ({SCHEMA_VERSION}). Refusing to open it: a newer schema "
            "may carry columns this code would silently ignore."
        )

    if current < SCHEMA_VERSION:
        _migrate(connection, current, now)
    return SCHEMA_VERSION


def _migrate(connection: sqlite3.Connection, from_version: int, now: float) -> None:
    """Apply migrations in order.

    Empty at version 1, which is the correct amount of migration code for a
    schema that has never changed. The dispatch exists so the first real change
    adds a function rather than an architecture.
    """
    steps: dict[int, str] = {
        # v2: the heartbeat table.
        #
        # Analytics needs to tell "nobody was there" apart from "nothing was
        # recording", and no arrangement of the observation rows can settle it:
        # an office with a nine-hour overnight quiet period and an office whose
        # recorder died at 21:00 produce byte-identical tables. Inferring it from
        # neighbouring buckets works for short gaps and cannot work for long
        # ones, because the honest answer is that the rows do not contain it.
        #
        # A row saying "this camera was alive at this moment" does contain it,
        # and it costs one insert a minute.
        2: """
        CREATE TABLE IF NOT EXISTS heartbeat (
            camera_id TEXT NOT NULL,
            timestamp REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_heartbeat_time ON heartbeat (timestamp);
        """,
    }
    for version in range(from_version + 1, SCHEMA_VERSION + 1):
        statement = steps.get(version)
        if statement is None:
            raise VantageError(
                f"no migration defined from schema version {version - 1} to {version}"
            )
        connection.executescript(statement)
    connection.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
        (SCHEMA_VERSION, now),
    )


def wrap_list(values: list[str] | tuple[str, ...] | None) -> str | None:
    """Join a list for a denormalised column, delimiter-wrapped for safe LIKE."""
    if not values:
        return None
    return DELIMITER + DELIMITER.join(values) + DELIMITER


def unwrap_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split(DELIMITER) if part)


def like_term(value: str) -> str:
    """The LIKE pattern that matches one element of a wrapped list exactly."""
    return f"%{DELIMITER}{value}{DELIMITER}%"
