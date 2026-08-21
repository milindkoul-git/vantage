"""The JSON payloads, built without reference to how they are served.

Kept apart from the HTTP layer on purpose. Every function here takes a dict of
string parameters and returns a dict of primitives, so swapping
``http.server`` for an ASGI framework later is a transport change rather than a
rewrite - which is the whole argument for having started with the standard
library.

Read-only, deliberately
-----------------------
There is no endpoint that changes anything: no configuration writes, no rule
edits, no pruning. The dashboard serves live footage with no authentication, so
every write endpoint would be an unauthenticated write endpoint. Pruning and
configuration stay on the CLI, where the operator is by definition the person at
the machine.
"""

from __future__ import annotations

import time
from typing import Any

from vantage.dashboard.live import LiveFeed
from vantage.storage.contracts import Query
from vantage.storage.query_cli import parse_duration

MAX_BUCKETS = 2000
"""Ceiling on chart buckets in one analytics response.

The same reasoning as MAX_LIMIT: a five-second interval over a year is six
million points, and no browser is going to draw them.
"""

MAX_LIMIT = 500
"""Ceiling on any single response.

Not a suggestion: without it, a request for a million rows would build a
million-element list in memory and serialise it, which is a denial of service
anyone with the URL could perform by accident.
"""


class DashboardApi:
    """Answers the dashboard's questions from the store and the live feed."""

    def __init__(
        self,
        store: Any | None = None,
        feed: LiveFeed | None = None,
        *,
        camera_id: str = "camera_01",
    ) -> None:
        self._store = store
        self._feed = feed
        self._camera_id = camera_id
        self._started = time.time()

    def handle(self, route: str, params: dict[str, str]) -> dict[str, Any]:
        """Dispatch one ``/api/...`` request."""
        handlers = {
            "live": self.live,
            "events": self.events,
            "observations": self.observations,
            "timeline": self.timeline,
            "stats": self.stats,
            "analytics": self.analytics,
        }
        handler = handlers.get(route)
        if handler is None:
            raise ValueError(f"no API route {route!r}; available: {sorted(handlers)}")
        return handler(params)

    # -- live -------------------------------------------------------------

    def live(self, params: dict[str, str]) -> dict[str, Any]:
        if self._feed is None:
            return {
                "available": False,
                # Said plainly rather than returning empty data, which a viewer
                # cannot distinguish from a quiet scene.
                "reason": "no pipeline attached; this dashboard is reading history only",
            }
        snapshot = self._feed.snapshot().to_dict()
        snapshot["available"] = True
        snapshot["has_frame"] = self._feed.has_frame
        snapshot["viewers"] = self._feed.viewers
        return snapshot

    # -- history ----------------------------------------------------------

    def events(self, params: dict[str, str]) -> dict[str, Any]:
        # Parameters are validated before the store is consulted. A malformed
        # request is malformed whether or not there is anything to query, and
        # checking the store first returned 200 with an explanation for
        # "?limit=lots" - which tells the caller nothing about their mistake.
        query = self._query(params)
        if self._store is None:
            return _no_store()
        rows = self._store.events(query)
        return {
            "available": True,
            "count": len(rows),
            "events": [
                {
                    "id": row.id,
                    "timestamp": row.timestamp,
                    "rule": row.rule,
                    "severity": row.severity,
                    "summary": row.summary,
                    "entity_id": row.entity_id,
                    "identity": row.identity,
                    "zone": row.zone,
                    "evidence": row.evidence,
                }
                for row in rows
            ],
        }

    def observations(self, params: dict[str, str]) -> dict[str, Any]:
        query = self._query(params)
        if self._store is None:
            return _no_store()
        rows = self._store.observations(query)
        return {
            "available": True,
            "count": len(rows),
            "observations": [
                {
                    "timestamp": row.timestamp,
                    "entity_id": row.entity_id,
                    "identity": row.identity,
                    "entity_type": row.entity_type,
                    "motion": row.motion,
                    "speed": row.speed,
                    "posture": row.posture,
                    "zones": row.zones,
                    "activities": row.activities,
                }
                for row in rows
            ],
        }

    def timeline(self, params: dict[str, str]) -> dict[str, Any]:
        entity = params.get("entity", "").strip()
        if not entity:
            raise ValueError("timeline needs an ?entity= parameter")
        limit = _limit(params)
        if self._store is None:
            return _no_store()
        rows = self._store.timeline(entity, limit=limit)
        return {
            "available": True,
            "entity_id": entity,
            "count": len(rows),
            "span_s": round(rows[-1].timestamp - rows[0].timestamp, 1)
            if len(rows) > 1
            else 0.0,
            "events": [
                {
                    "timestamp": row.timestamp,
                    "rule": row.rule,
                    "severity": row.severity,
                    "summary": row.summary,
                    "zone": row.zone,
                }
                for row in rows
            ],
        }

    def analytics(self, params: dict[str, str]) -> dict[str, Any]:
        """Bucketed history and anomalies, for the trend panel.

        The dashboard drew no charts until Phase 11 existed, and the honest
        reason is that there was nothing true to draw: a chart of the last
        thirty seconds of frame rate is decoration. This returns the same
        buckets ``vantage analytics`` prints, so the picture and the command
        line cannot disagree.

        Coverage travels with the answer. A chart of a week that only holds four
        hours of data looks identical to a quiet week unless the page is told
        which it is looking at.
        """
        from vantage.analytics.contracts import Metric
        from vantage.analytics.engine import AnalyticsEngine, AnalyticsParams

        if self._store is None:
            return _no_store()

        try:
            metric = Metric(params.get("metric", "entities"))
        except ValueError:
            raise ValueError(
                f"unknown metric {params.get('metric')!r}; "
                f"available: {', '.join(m.value for m in Metric)}"
            ) from None

        span = parse_duration(params.get("since", "24h"))
        interval = parse_duration(params.get("interval", "1h"))
        if span <= 0 or interval <= 0:
            raise ValueError("since and interval must be positive durations")
        if span / interval > MAX_BUCKETS:
            raise ValueError(
                f"that window would produce more than {MAX_BUCKETS} buckets; "
                "widen the interval or shorten the window"
            )

        until = time.time()
        engine = AnalyticsEngine(self._store, params=AnalyticsParams(interval_s=interval))
        series = engine.series(metric, since=until - span, until=until)

        payload: dict[str, Any] = {
            "available": True,
            "metric": metric.value,
            "label": metric.label,
            "interval_s": series.interval_s,
            "coverage": round(series.coverage, 4),
            "buckets": [
                {
                    "start": b.start,
                    "value": round(b.value, 4),
                    "samples": b.samples,
                    "known_zero": b.known_zero,
                }
                for b in series
            ],
            "anomalies": [],
        }

        # Anomalies need a baseline learned from history *before* the window,
        # which a young database does not have. Absent is reported as absent
        # rather than as an empty list meaning "all clear".
        try:
            result = engine.analyse(metric, since=until - span, until=until)
        except Exception as exc:  # pragma: no cover - a store too young to judge
            payload["anomalies_available"] = False
            payload["anomalies_reason"] = str(exc)
            return payload

        payload["anomalies_available"] = result.judged > 0
        payload["judged"] = result.judged
        payload["unjudged"] = result.skipped_untrained
        if result.judged == 0:
            payload["anomalies_reason"] = (
                "no slot has enough history behind it yet, so nothing was compared"
            )
        payload["anomalies"] = [
            {
                "start": a.bucket.start,
                "observed": round(a.observed, 3),
                "expected": round(a.expected, 3),
                "direction": a.direction.value,
                "score": round(a.score, 2),
                "severity": a.severity,
            }
            for a in sorted(result.anomalies, key=lambda a: -a.score)[:50]
        ]
        return payload

    def stats(self, params: dict[str, str]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_id": self._camera_id,
            "uptime_s": round(time.time() - self._started, 1),
            "live": self._feed is not None,
        }
        if self._store is not None:
            payload["store"] = self._store.counts()
            payload["schema_version"] = self._store.schema_version
        else:
            payload["store"] = None
        return payload

    # -- helpers ----------------------------------------------------------

    def _limit_for_test(self, params: dict[str, str]) -> int:
        """The limit rule, exposed so a test asserts the real one.

        A test that reimplemented the cap would pass while the server used a
        different number, which is the failure mode a shared helper exists to
        prevent.
        """
        return _limit(params)

    def _query(self, params: dict[str, str]) -> Query:
        since = None
        if params.get("since"):
            since = time.time() - parse_duration(params["since"])
        return Query(
            since=since,
            entity_id=params.get("entity") or None,
            rule=params.get("rule") or None,
            severity=params.get("severity") or None,
            zone=params.get("zone") or None,
            limit=_limit(params),
        )


def _limit(params: dict[str, str]) -> int:
    raw = params.get("limit")
    if not raw:
        return 50
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"limit must be a whole number, got {raw!r}") from None
    if value < 1:
        raise ValueError("limit must be at least 1")
    return min(value, MAX_LIMIT)


def _no_store() -> dict[str, Any]:
    return {
        "available": False,
        "reason": (
            "no store: this run was started without --store, so nothing is being "
            "recorded to query"
        ),
        "count": 0,
        "events": [],
        "observations": [],
    }
