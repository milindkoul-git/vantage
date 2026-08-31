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
from vantage.incident.models import decode_dossier
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

    @property
    def _graph_store(self) -> Any | None:
        """The store, when it is one that keeps incidents and relationships."""
        from vantage.storage.contracts import IntelligenceStore

        if self._store is not None and isinstance(self._store, IntelligenceStore):
            return self._store
        return None

    def __init__(
        self,
        store: Any | None = None,
        feed: LiveFeed | None = None,
        radar_map: Any | None = None,
        spatial_twin: Any | None = None,
        zone_registry: Any | None = None,
        pipeline: Any | None = None,
        incident_service: Any | None = None,
        relationship_service: Any | None = None,
        *,
        camera_id: str = "camera_01",
    ) -> None:
        self._store = store
        self._feed = feed
        self._radar = radar_map
        self._spatial_twin = spatial_twin
        self._zone_registry = zone_registry
        self._pipeline = pipeline
        # Incidents and relationships are produced by the single-camera app as
        # well as by the facility pipeline, so they are taken as services in
        # their own right. Probing a pipeline object with ``hasattr`` made the
        # two paths behave differently for no reason and made "this build has
        # no incident correlation" indistinguishable from "nothing happened".
        self._incident_service = (
            incident_service
            if incident_service is not None
            else getattr(pipeline, "incident_service", None)
        )
        self._relationship_service = (
            relationship_service
            if relationship_service is not None
            else getattr(pipeline, "relationship_service", None)
        )
        self._relationship_tracker = getattr(
            self._relationship_service, "tracker", None
        ) or getattr(pipeline, "relationship_tracker", None)
        self._camera_id = camera_id
        self._started = time.time()
        from vantage.ingestion.connectors.discovery import CameraDiscoveryService
        from vantage.ingestion.connectors.manager import CameraConnectorManager

        self._discovery = CameraDiscoveryService()
        self._connector_manager = CameraConnectorManager()
        from vantage.search.semantic import SemanticEventSearch

        self._search: SemanticEventSearch | None = (
            SemanticEventSearch(store=store) if store is not None else None
        )

    def handle(self, route: str, params: dict[str, str]) -> dict[str, Any]:
        """Dispatch one ``/api/...`` request."""
        handlers = {
            "live": self.live,
            "events": self.events,
            "observations": self.observations,
            "timeline": self.timeline,
            "entity_timeline": self.entity_timeline,
            "relationships": self.relationships,
            "relationships/graph": self.relationships_graph,
            "incidents": self.incidents,
            "incident": self.incident_detail,
            "incident/timeline": self.incident_timeline_api,
            "incident/dossier": self.incident_dossier_api,
            "stats": self.stats,
            "analytics": self.analytics,
            "search": self.search,
            "radar": self.radar,
            "twin": self.twin,
            "zones": self.zones,
            "cameras": self.cameras,
            "cameras/discover": self.cameras_discover,
            "cameras/presets": self.cameras_presets,
            "entities": self.entities,
            "entity": self.entities,
            "scene": self.scene,
        }
        handler = handlers.get(route)
        if handler is None:
            raise ValueError(f"no API route {route!r}; available: {sorted(handlers)}")
        return handler(params)

    def handle_post(self, route: str, data: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one POST ``/api/...`` request."""
        if route == "zones":
            return self.save_zone_api(data)
        if route == "cameras/test":
            return self.cameras_test_api(data)
        if route == "cameras/connect":
            return self.cameras_connect_api(data)
        raise ValueError(f"no POST handler for API route {route!r}")

    def handle_delete(self, route: str, params: dict[str, str]) -> dict[str, Any]:
        """Dispatch one DELETE ``/api/...`` request."""
        if route == "zones":
            return self.delete_zone_api(params)
        if route == "cameras":
            return self.cameras_disconnect_api(params)
        raise ValueError(f"no DELETE handler for API route {route!r}")

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

    def entity_timeline(self, params: dict[str, str]) -> dict[str, Any]:
        """Project complete entity story across time."""
        entity = params.get("entity", "").strip()
        if not entity:
            raise ValueError("entity_timeline needs an ?entity= parameter")
        if self._store is None:
            return _no_store()
        from vantage.storage.entity_timeline import build_entity_timeline

        timeline = build_entity_timeline(self._store, entity)
        if timeline is None:
            return {
                "available": True,
                "found": False,
                "entity_id": entity,
                "message": f"no observations recorded for {entity}",
            }
        data = timeline.to_dict()
        data["available"] = True
        data["found"] = True
        return data

    def relationships(self, params: dict[str, str]) -> dict[str, Any]:
        """Query persistent relationship graph edges and explainable dossiers."""
        entity = params.get("entity", "").strip() or params.get("entity_id", "").strip() or None
        min_conf = float(params.get("min_confidence", params.get("min_strength", "0.0")))
        limit = _limit(params)

        # 1. Prefer live in-memory tracker if pipeline attached
        if self._relationship_tracker is not None:
            now = time.time()
            if entity:
                rels = self._relationship_tracker.get_relationships_for_entity(
                    entity, min_strength=min_conf, now=now
                )
            else:
                rels = self._relationship_tracker.get_all_relationships(
                    min_strength=min_conf, now=now
                )
            rel_dicts = [r.to_dict() for r in rels[:limit]]
            return {
                "available": True,
                "count": len(rel_dicts),
                "relationships": rel_dicts,
            }

        # 2. Fallback to the store
        if self._store is None:
            return _no_store()
        graph_store = self._graph_store
        if graph_store is None:
            return _no_graph_store("a relationship graph")

        edges = graph_store.relationships(
            entity_id=entity,
            min_confidence=min_conf,
            limit=limit,
        )
        return {
            "available": True,
            "count": len(edges),
            "relationships": edges,
        }

    def relationships_graph(self, params: dict[str, str]) -> dict[str, Any]:
        """Query full facility relationship graph for node-link visualization."""
        min_conf = float(params.get("min_confidence", params.get("min_strength", "0.0")))
        if self._relationship_service is not None:
            return {
                "available": True,
                "graph": self._relationship_service.get_graph_snapshot(
                    min_strength=min_conf, now=time.time()
                ),
            }

        # Fallback to reconstructing graph from SQLite store
        if self._store is None:
            return _no_store()

        graph_store = self._graph_store
        if graph_store is None:
            return _no_graph_store("a relationship graph")

        edges = graph_store.relationships(min_confidence=min_conf, limit=500)
        nodes: dict[str, dict[str, Any]] = {}
        graph_edges: list[dict[str, Any]] = []

        for e in edges:
            src = e.get("entity_a", "")
            tgt = e.get("entity_b_or_zone", "")
            if src and tgt:
                for n_id in (src, tgt):
                    if n_id not in nodes:
                        nodes[n_id] = {"id": n_id, "degree": 0}
                    nodes[n_id]["degree"] += 1
                graph_edges.append(
                    {
                        "source": src,
                        "target": tgt,
                        "active_strength": e.get("max_confidence_tier", 0.5),
                        "pattern": e.get("relation_type", "co_occurrence"),
                        "co_occurrence_count": e.get("occurrence_count", 1),
                    }
                )

        return {
            "available": True,
            "graph": {
                "timestamp": round(time.time(), 2),
                "total_nodes": len(nodes),
                "total_edges": len(graph_edges),
                "nodes": list(nodes.values()),
                "edges": graph_edges,
            },
        }

    def incidents(self, params: dict[str, str]) -> dict[str, Any]:
        """Query situational incidents with optional status or entity filtering."""
        state = params.get("status") or params.get("state") or None
        entity = params.get("entity") or params.get("entity_id") or None
        limit = _limit(params)

        # 1. Prefer live in-memory IncidentService
        if self._incident_service is not None:
            incs = self._incident_service.get_incidents(
                state=state, entity_id=entity, limit=limit, now=time.time()
            )
            return {
                "available": True,
                "count": len(incs),
                "incidents": [i.to_dict() for i in incs],
            }

        # 2. Fallback to SQLite store
        if self._store is None:
            return _no_store()

        graph_store = self._graph_store
        if graph_store is None:
            return _no_graph_store("an incident log")
        stored = graph_store.incidents(state=state, limit=limit)
        results: list[dict[str, Any]] = []
        for row in stored:
            results.append(decode_dossier(row))
        return {
            "available": True,
            "count": len(results),
            "incidents": results,
        }

    def incident_detail(self, params: dict[str, str]) -> dict[str, Any]:
        """Query a single incident by ID."""
        inc_id = params.get("id") or params.get("incident_id") or ""
        if not inc_id:
            raise ValueError("incident detail query needs an ?id= parameter")

        if self._incident_service is not None:
            inc = self._incident_service.get_incident(inc_id)
            if inc:
                return {"available": True, "found": True, "incident": inc.to_dict()}

        graph_store = self._graph_store
        if graph_store is not None:
            row = graph_store.get_incident(inc_id)
            if row:
                return {"available": True, "found": True, "incident": decode_dossier(row)}

        return {
            "available": True,
            "found": False,
            "incident_id": inc_id,
            "message": "Incident not found",
        }

    def incident_timeline_api(self, params: dict[str, str]) -> dict[str, Any]:
        """Query chronological timeline for a specific incident."""
        detail = self.incident_detail(params)
        if not detail.get("found"):
            return detail
        inc_data = detail.get("incident", {})
        timeline = inc_data.get("timeline", [])
        return {
            "available": True,
            "incident_id": inc_data.get("incident_id"),
            "count": len(timeline),
            "timeline": timeline,
        }

    def incident_dossier_api(self, params: dict[str, str]) -> dict[str, Any]:
        """Query complete evidence dossier and relationship links for an incident."""
        detail = self.incident_detail(params)
        if not detail.get("found"):
            return detail
        inc_data = detail.get("incident", {})
        return {
            "available": True,
            "incident_id": inc_data.get("incident_id"),
            "severity_breakdown": inc_data.get("severity_breakdown"),
            "correlation_breakdown": inc_data.get("correlation_breakdown"),
            "evidence_dossier": inc_data.get("evidence_dossier"),
            "relationship_links": inc_data.get("relationship_links"),
            "correlation_candidates": inc_data.get("correlation_candidates"),
            "merge_candidates": inc_data.get("merge_candidates"),
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
            "available": True,
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

    def search(self, params: dict[str, str]) -> dict[str, Any]:
        q = params.get("q", "")
        if self._search is None:
            return {
                "available": False,
                "reason": "no store attached; there is no history to search",
                "query": q,
                "total": 0,
                "results": [],
            }
        found = dict(self._search.search(q, limit=_limit(params)))
        found["available"] = True
        return found

    def radar(self, params: dict[str, str]) -> dict[str, Any]:
        """Overhead radar state, or an honest refusal.

        An empty radar and an absent one are different facts. Reporting both as
        ``{"entities": []}`` tells a viewer the floor is clear when nothing is
        actually watching it.
        """
        if self._radar is None:
            return {
                "available": False,
                "reason": "no radar map attached; this needs the multi-camera pipeline",
                "zones": [],
                "entities": [],
                "active_count": 0,
            }
        state = dict(self._radar.get_radar_state())
        state["available"] = True
        return state

    def twin(self, params: dict[str, str]) -> dict[str, Any]:
        """Return 3D spatial facility mesh, camera frustums, and real-time 3D entities."""
        if self._spatial_twin is None:
            return {
                "available": False,
                "reason": "no spatial twin attached; this needs a calibrated facility model",
                "facility": {},
                "cameras": [],
                "zones": [],
                "entities": [],
                "trails": {},
            }
        occupancies = (
            getattr(self._pipeline, "_zone_occupancies", {})
            if self._pipeline is not None
            else {}
        )
        state = dict(self._spatial_twin.get_digital_twin_state(live_occupancies=occupancies))
        state["available"] = True
        return state

    def zones(self, params: dict[str, str]) -> dict[str, Any]:
        """Return active geofence zones from the registry."""
        if self._zone_registry is None:
            return {
                "available": False,
                "count": 0,
                "zones": [],
                "reason": "no zone registry attached",
            }
        snapshot = self._zone_registry.get_snapshot()
        camera_id = params.get("camera_id")
        if camera_id:
            zones_list = list(snapshot.get_zones_for_camera(camera_id))
        else:
            zones_list = snapshot.list_all_zones()

        return {
            "available": True,
            "count": len(zones_list),
            "version": snapshot.version,
            "last_update_latency_ms": round(
                getattr(self._zone_registry, "last_update_latency_ms", 0.0), 2
            ),
            "zones": [z.to_dict() for z in zones_list],
        }

    def save_zone_api(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate and create or update a polygonal geofence zone."""
        if self._zone_registry is None:
            raise ValueError("No zone registry attached to dashboard API")

        from vantage.events.geofence import PolygonZone

        try:
            zone = PolygonZone.from_dict(data)
        except Exception as exc:
            raise ValueError(f"Malformed zone payload: {exc}") from None

        snapshot = self._zone_registry.save_zone(zone)
        return {
            "status": "ok",
            "action": "saved",
            "zone_id": zone.zone_id,
            "version": snapshot.version,
            "total_zones": snapshot.count(),
            "propagation_latency_ms": round(self._zone_registry.last_update_latency_ms, 2),
        }

    def delete_zone_api(self, params: dict[str, str]) -> dict[str, Any]:
        """Delete a geofence zone by ID."""
        if self._zone_registry is None:
            raise ValueError("No zone registry attached to dashboard API")

        zone_id = params.get("id") or params.get("zone_id")
        if not zone_id:
            raise ValueError("Missing 'id' parameter for zone deletion")

        snapshot = self._zone_registry.delete_zone(zone_id)
        return {
            "status": "ok",
            "action": "deleted",
            "zone_id": zone_id,
            "version": snapshot.version,
            "total_zones": snapshot.count(),
            "propagation_latency_ms": round(self._zone_registry.last_update_latency_ms, 2),
        }

    # -- camera connectors ------------------------------------------------

    def cameras(self, params: dict[str, str]) -> dict[str, Any]:
        """List active streaming cameras in the multi-camera pipeline."""
        if self._pipeline is None or not hasattr(self._pipeline, "workers"):
            # A single-camera run has exactly one feed and no connector manager
            # behind it. Report the one that exists and say the roster is not
            # editable, rather than inventing a row with a made-up URI.
            live = self._feed is not None
            return {
                "available": True,
                "managed": False,
                "reason": "single-camera run; the camera roster is set by the CLI, not the dashboard",
                "count": 1 if live else 0,
                "cameras": (
                    [
                        {
                            "camera_id": self._camera_id,
                            "name": self._camera_id.replace("_", " ").title(),
                            "uri": None,
                            "status": "streaming" if self._feed.has_frame else "starting",
                        }
                    ]
                    if live and self._feed is not None
                    else []
                ),
            }

        cams = []
        for cam_id, worker in self._pipeline.workers.items():
            cams.append(
                {
                    "camera_id": cam_id,
                    "name": cam_id.replace("_", " ").title(),
                    "uri": getattr(worker, "source_path", "live"),
                    "status": "streaming" if getattr(worker, "_running", True) else "stopped",
                }
            )
        return {"available": True, "managed": True, "count": len(cams), "cameras": cams}

    def cameras_discover(self, params: dict[str, str]) -> dict[str, Any]:
        """Probe for locally connected USB webcams and DirectShow video capture devices."""
        max_idx = int(params.get("max", 4))
        webcams = self._discovery.discover_local_webcams(max_indices=max_idx)
        return {
            "available": True,
            "count": len(webcams),
            "webcams": [w.to_dict() for w in webcams],
            "presets": self._discovery.get_presets(),
        }

    def cameras_presets(self, params: dict[str, str]) -> dict[str, Any]:
        """Return RTSP path presets by vendor (Hikvision, Dahua, Axis, Reolink, Tapo, etc.)."""
        return {"available": True, "presets": self._discovery.get_presets()}

    def cameras_test_api(self, data: dict[str, Any]) -> dict[str, Any]:
        """Test opening a camera URI (RTSP stream, USB webcam, or video) with low timeout."""
        uri = data.get("uri", "").strip()
        if not uri:
            raise ValueError("Missing 'uri' field in camera test payload")
        return self._discovery.test_camera_connection(uri=uri)

    def cameras_connect_api(self, data: dict[str, Any]) -> dict[str, Any]:
        """Dynamically attach a camera to the live multi-camera pipeline and 3D digital twin."""
        if self._pipeline is None:
            raise ValueError("No multi-camera pipeline attached to dashboard API")

        camera_id = data.get("camera_id", "").strip()
        uri = data.get("uri", "").strip()
        name = data.get("name", "").strip() or camera_id

        if not camera_id:
            raise ValueError("Missing 'camera_id' field")
        if not uri:
            raise ValueError("Missing 'uri' field")

        return self._connector_manager.attach_camera(
            pipeline=self._pipeline,
            camera_id=camera_id,
            name=name,
            uri=uri,
            mount_x=float(data.get("mount_x", 20.0)),
            mount_y=float(data.get("mount_y", 3.5)),
            mount_z=float(data.get("mount_z", 12.0)),
            yaw_deg=float(data.get("yaw_deg", 0.0)),
            pitch_deg=float(data.get("pitch_deg", -25.0)),
            fov_deg=float(data.get("fov_deg", 70.0)),
            range_m=float(data.get("range_m", 16.0)),
            color=data.get("color", "#00e5ff"),
        )

    def cameras_disconnect_api(self, params: dict[str, str]) -> dict[str, Any]:
        """Dynamically detach a camera from the pipeline and 3D twin."""
        if self._pipeline is None:
            raise ValueError("No multi-camera pipeline attached to dashboard API")

        camera_id = params.get("id") or params.get("camera_id")
        if not camera_id:
            raise ValueError("Missing 'id' parameter for camera detachment")

        return self._connector_manager.detach_camera(
            pipeline=self._pipeline,
            camera_id=camera_id,
        )

    # -- canonical entity intelligence read models ------------------------

    def entities(self, params: dict[str, str]) -> dict[str, Any]:
        """Canonical entity intelligence read model."""
        if self._pipeline is None or not hasattr(self._pipeline, "entity_manager"):
            return {
                "available": False,
                "reason": "No multi-camera entity manager active",
                "count": 0,
                "entities": [],
            }

        em = self._pipeline.entity_manager
        entity_id = params.get("id") or params.get("global_id")
        if entity_id:
            snap = em.get_snapshot(entity_id)
            if not snap:
                return {"available": True, "found": False, "entity": None}
            return {"available": True, "found": True, "entity": snap.to_dict()}

        active_within = float(params.get("active_within", "45.0"))
        snapshots = em.get_active_snapshots(active_within_s=active_within)
        stats = em.stats()

        return {
            "available": True,
            "count": len(snapshots),
            "stats": stats,
            "entities": [s.to_dict() for s in snapshots],
        }

    # -- transient scene intelligence -------------------------------------

    def scene(self, params: dict[str, str]) -> dict[str, Any]:
        """Transient scene intelligence read model."""
        if self._pipeline is None or not hasattr(self._pipeline, "scene_graphs"):
            return {
                "available": False,
                "reason": "No multi-camera scene graph active",
                "cameras": {},
            }

        cam_id = params.get("camera_id") or params.get("camera")
        if cam_id:
            sg = self._pipeline.scene_graphs.get(cam_id)
            if not sg or not sg.last_snapshot:
                return {"available": True, "found": False, "scene": None}
            return {"available": True, "found": True, "scene": sg.last_snapshot.to_dict()}

        cam_snapshots = {}
        for c_id, sg in self._pipeline.scene_graphs.items():
            if sg.last_snapshot:
                cam_snapshots[c_id] = sg.last_snapshot.to_dict()

        return {
            "available": True,
            "camera_count": len(cam_snapshots),
            "cameras": cam_snapshots,
        }

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


def _no_graph_store(what: str) -> dict[str, Any]:
    """The store exists but cannot hold this.

    Distinct from :func:`_no_store`. "You did not pass --store" and "your store
    backend does not keep an incident log" are different problems with different
    fixes, and collapsing them into one message sends the operator looking for
    the wrong thing.
    """
    return {
        "available": False,
        "reason": (
            f"the configured store does not keep {what}; that needs a SQLite store, "
            "and this run is using a backend without it"
        ),
        "count": 0,
    }


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
