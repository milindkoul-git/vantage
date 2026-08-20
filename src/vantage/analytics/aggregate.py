"""Turning rows into buckets, in the database rather than in Python.

Every other read path in this project returns rows and lets the caller work on
them, which is right when the caller wants rows. Analytics never does. A month
of observations at the shipped sampling rate is millions of rows whose only use
here is to be counted, and fetching them to count them in a loop would move
tens of megabytes through Python to produce a few hundred numbers.

So aggregation is a ``GROUP BY``. The bucket index is computed in SQL as an
integer division against a fixed origin, which keeps the timestamp comparison
in the query sargable - the ``idx_observations_time`` index still does the range
scan, and only the grouping is arithmetic.

The one thing not done in SQL
-----------------------------
Bucket *labelling* - "which hour of which weekday is this" - happens in Python
on the handful of resulting buckets, not in SQL on every row. SQLite can do it
with ``strftime(..., 'localtime')``, but that reads the process timezone at
query time and gets daylight saving wrong at exactly the boundary where it
matters. A few hundred timezone-aware conversions in Python are free and
correct; a few million in SQL would be neither.

Origin alignment, and the limit of it
-------------------------------------
Buckets are anchored to a caller-supplied origin, normally local midnight at
the start of the window, so an "hourly" series has boundaries on the hour rather
than on whatever minute the query happened to run. Across a daylight-saving
transition inside a single window, the alignment shifts by an hour for buckets
after the change. That is a real limitation and is not worked around: doing so
would mean variable-width buckets, and a bucket whose width changes silently is
worse than a boundary that moves once a year.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

from vantage.analytics.contracts import Bucket, Metric, Series
from vantage.core.errors import VantageError
from vantage.core.logging import get_logger
from vantage.storage.schema import like_term

log = get_logger(__name__)

_EPSILON = 1e-6
"""Nudge used to make the right edge of a window exclusive."""

MAX_BUCKETS = 20_000
"""Ceiling on how many buckets one query may produce.

Not a suggestion. A five-second interval over a year is six million buckets,
every one of which becomes an object, and the request that asks for it is
almost always a mistyped interval rather than a real intention.
"""

_TABLE_FOR = {
    Metric.ENTITIES: "observations",
    Metric.OBSERVATIONS: "observations",
    Metric.MEAN_SPEED: "observations",
    Metric.MOVING_FRACTION: "observations",
    Metric.EVENTS: "events",
}

_VALUE_EXPRESSION = {
    Metric.ENTITIES: "COUNT(DISTINCT entity_id)",
    Metric.OBSERVATIONS: "COUNT(*)",
    Metric.EVENTS: "COUNT(*)",
    Metric.MEAN_SPEED: "AVG(speed)",
    # SUM of a boolean over COUNT, rather than AVG of a CASE, so that rows with
    # a NULL motion are counted in the denominator. They are observations that
    # genuinely happened and simply had no motion state; excluding them would
    # report a fraction of a population that quietly excluded its own zeros.
    Metric.MOVING_FRACTION: "CAST(SUM(CASE WHEN motion = 'moving' THEN 1 ELSE 0 END) AS REAL) "
    "/ COUNT(*)",
}


def local_midnight(timestamp: float) -> float:
    """The most recent local midnight at or before ``timestamp``.

    The default bucket origin. Without it an "hourly" series started at 14:37
    has buckets running :37 to :37, and every hour-of-day comparison against it
    is comparing two different hours blended together.
    """
    local = datetime.fromtimestamp(timestamp, tz=UTC).astimezone()
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.timestamp()


def next_local_midnight(timestamp: float) -> float:
    """Local midnight strictly after ``timestamp``, DST included.

    Computed by adding a day to the local date and re-resolving the offset,
    rather than by adding 86400 seconds, because on a transition day the two
    differ by an hour and the second one lands at 23:00 or 01:00.
    """
    local = datetime.fromtimestamp(timestamp, tz=UTC).astimezone()
    midnight = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight + timedelta(days=1)).timestamp()


def bucket_series(
    connection: sqlite3.Connection,
    metric: Metric,
    *,
    since: float,
    until: float,
    interval_s: float,
    camera_id: str | None = None,
    entity_type: str | None = None,
    zone: str | None = None,
    origin: float | None = None,
) -> Series:
    """Aggregate one metric into contiguous, equal-width buckets.

    Empty intervals come back as empty buckets rather than being omitted, so the
    result is a real time series that can be indexed by position and compared
    against another of the same shape.
    """
    if interval_s <= 0:
        raise VantageError(f"bucket interval must be positive, got {interval_s}")
    if until <= since:
        raise VantageError(f"analysis window is empty or inverted: since={since} until={until}")

    anchor = local_midnight(since) if origin is None else origin
    # Snap the window outward to whole buckets. A partial bucket at either end
    # holds a fraction of an interval's traffic and would read as a quiet period
    # in every chart and as an anomaly in every detector.
    first_index = _index_of(since, anchor, interval_s)
    # ``until`` is exclusive, so the last bucket is the one holding the final
    # instant before it. Indexing ``until`` directly appends an extra empty
    # bucket whenever the window ends exactly on a boundary - which is the
    # normal case, since windows are built from whole hours and days. That
    # bucket then counts against coverage and dilutes every average, so a clean
    # 24-hour query reported 25 buckets and 96% coverage.
    last_index = _index_of(until - _EPSILON, anchor, interval_s)
    if last_index < first_index:
        last_index = first_index
    count = last_index - first_index + 1
    if count > MAX_BUCKETS:
        raise VantageError(
            f"{count} buckets requested (interval {interval_s:g}s over "
            f"{(until - since) / 3600:.1f}h). The ceiling is {MAX_BUCKETS}; "
            "widen the interval or narrow the window."
        )

    table = _TABLE_FOR[metric]
    expression = _VALUE_EXPRESSION[metric]

    clauses = ["timestamp >= ?", "timestamp < ?"]
    window_start = anchor + first_index * interval_s
    window_end = anchor + (last_index + 1) * interval_s
    params: list[object] = [window_start, window_end]

    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)
    if entity_type:
        if table != "observations":
            raise VantageError(f"entity_type does not apply to {metric.value}")
        clauses.append("entity_type = ?")
        params.append(entity_type)
    if zone:
        if table == "observations":
            # Delimiter-wrapped LIKE, so "till" cannot match "till_annexe".
            clauses.append("zones LIKE ?")
            params.append(like_term(zone))
        else:
            clauses.append("zone = ?")
            params.append(zone)

    sql = (
        f"SELECT CAST((timestamp - ?) / ? AS INTEGER) AS bucket, "
        f"{expression} AS value, COUNT(*) AS samples "
        f"FROM {table} WHERE {' AND '.join(clauses)} "
        "GROUP BY bucket ORDER BY bucket"
    )
    rows = connection.execute(sql, (anchor, interval_s, *params)).fetchall()

    found = {int(row["bucket"]): row for row in rows}
    buckets: list[Bucket] = []
    for index in range(first_index, last_index + 1):
        row = found.get(index)
        start = anchor + index * interval_s
        if row is None:
            buckets.append(Bucket(start=start, width_s=interval_s, value=0.0, samples=0))
            continue
        # AVG over a column that is entirely NULL returns NULL, not 0: rows
        # existed, they simply had no speed. Reporting 0.0 would be a claim that
        # nothing moved, which is a different statement from having no data.
        raw = row["value"]
        buckets.append(
            Bucket(
                start=start,
                width_s=interval_s,
                value=float(raw) if raw is not None else 0.0,
                samples=int(row["samples"]),
            )
        )

    series = Series(
        metric=metric,
        buckets=tuple(buckets),
        interval_s=interval_s,
        since=window_start,
        until=window_end,
        camera_id=camera_id,
        metadata={
            "origin": anchor,
            "requested_since": since,
            "requested_until": until,
            "zone": zone,
            "entity_type": entity_type,
        },
    )
    log.debug(
        "aggregated series",
        extra={
            "vantage_fields": {
                "metric": metric.value,
                "buckets": len(series),
                "coverage": round(series.coverage, 3),
                "interval_s": interval_s,
            }
        },
    )
    return series


def _index_of(timestamp: float, origin: float, interval_s: float) -> int:
    """Floor division that stays correct left of the origin.

    Python's ``//`` already floors toward negative infinity, which is what is
    wanted; ``int()`` on the quotient would truncate toward zero and put every
    pre-origin timestamp one bucket too high.
    """
    return int((timestamp - origin) // interval_s)
