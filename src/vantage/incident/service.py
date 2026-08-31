"""High-Level Incident Service coordinating Ingestion, State Machines, Persistence, and Dossiers."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from vantage.events.contracts import EventCandidate
from vantage.incident.config import IncidentCorrelatorConfig
from vantage.incident.correlator import IncidentCorrelator
from vantage.incident.models import (
    CanonicalIncident,
    IncidentCorrelationDecision,
    IncidentSeverityBreakdown,
    IncidentState,
    IncidentTimelineEntry,
)
from vantage.incident.timeline import IncidentTimelineManager
from vantage.relationship.tracker import PersistentRelationshipTracker
from vantage.storage.sqlite_store import SqliteStore

log = logging.getLogger(__name__)


class IncidentService:
    """Coordinates lifecycle, correlation, severity, timeline dossiers, and persistence for incidents."""

    def __init__(
        self,
        store: SqliteStore | None = None,
        config: IncidentCorrelatorConfig | None = None,
        relationship_tracker: PersistentRelationshipTracker | None = None,
        max_active_memory_incidents: int = 500,
    ) -> None:
        self.store = store
        self.config = config or IncidentCorrelatorConfig()
        self.relationship_tracker = relationship_tracker
        self.max_active_memory_incidents = max_active_memory_incidents

        self.correlator = IncidentCorrelator(
            config=self.config,
            relationship_tracker=self.relationship_tracker,
        )

        self._lock = threading.RLock()
        self._incidents: OrderedDict[str, CanonicalIncident] = OrderedDict()

        if self.store:
            self._hydrate_from_store()

    def _compute_severity(
        self,
        events: Sequence[dict[str, Any]],
        zones: set[str],
        entities: set[str],
    ) -> tuple[str, IncidentSeverityBreakdown]:
        """Compute explainable incident severity anchored in constituent event history."""
        sev_rank = {"info": 1, "notice": 2, "alert": 3}
        highest_rank = 1
        highest_str = "info"

        for ev in events:
            s_str = str(ev.get("severity", "info")).lower()
            rank = sev_rank.get(s_str, 1)
            if rank > highest_rank:
                highest_rank = rank
                highest_str = s_str

        corroboration_count = len(events)
        entity_count = len(entities)
        has_restricted_zone = any("restricted" in z.lower() for z in zones)
        zone_factor = 0.20 if has_restricted_zone else 0.0

        # Base score from highest event
        base_score = (
            0.25 if highest_str == "info" else (0.60 if highest_str == "notice" else 0.85)
        )
        corroboration_bonus = min(0.15, (corroboration_count - 1) * 0.03)

        # Escalation bonus if severe alert occurs after lower warnings
        has_escalation = False
        if len(events) >= 2:
            first_sev = sev_rank.get(str(events[0].get("severity", "info")).lower(), 1)
            last_sev = sev_rank.get(str(events[-1].get("severity", "info")).lower(), 1)
            if last_sev > first_sev:
                has_escalation = True
        escalation_factor = 0.10 if has_escalation else 0.0

        total_sev_score = min(
            1.0, base_score + corroboration_bonus + zone_factor + escalation_factor
        )
        final_sev = (
            "alert"
            if total_sev_score >= 0.75
            else ("notice" if total_sev_score >= 0.50 else "info")
        )

        breakdown = IncidentSeverityBreakdown(
            highest_event_severity=highest_str,
            corroborating_event_count=corroboration_count,
            involved_entity_count=entity_count,
            restricted_zone_factor=zone_factor,
            escalation_factor=escalation_factor,
            final_severity=final_sev,
            severity_score=total_sev_score,
        )
        return final_sev, breakdown

    def _create_timeline_entry(
        self, event: dict[str, Any], now: float
    ) -> IncidentTimelineEntry:
        """Construct structured timeline entry with complete provenance linking."""
        entry_id = f"tle_{uuid.uuid4().hex[:8]}"
        ev_time = float(event.get("timestamp") or event.get("capture_wall") or now)
        ev_id = event.get("id") or event.get("event_id")
        rule = str(event.get("rule", "observation"))
        camera_id = str(event.get("camera_id", "default"))
        entity_id = event.get("entity_id")
        related_id = event.get("related_id")
        zone = event.get("zone")
        summary = str(event.get("summary") or f"Event {rule} observed on {camera_id}")

        entities = []
        if entity_id:
            entities.append(str(entity_id))
        if related_id and related_id not in entities:
            entities.append(str(related_id))

        objects = []
        evidence = event.get("evidence", {}) or {}
        if "object_id" in evidence:
            objects.append(str(evidence["object_id"]))

        evidence_ref = {
            "rule": rule,
            "severity": event.get("severity", "info"),
            "clip_url": evidence.get("clip_url") or event.get("clip_url"),
            "evidence": evidence,
        }

        return IncidentTimelineEntry(
            entry_id=entry_id,
            timestamp=ev_time,
            event_id=ev_id,
            event_type=rule,
            camera_id=camera_id,
            entities=tuple(entities),
            objects=tuple(objects),
            zone=zone,
            summary=summary,
            evidence_ref=evidence_ref,
        )

    def ingest_event(
        self,
        event: dict[str, Any],
        now: float | None = None,
    ) -> tuple[CanonicalIncident, IncidentCorrelationDecision, list[EventCandidate]]:
        """Ingest a new perception/system event, correlate into incident state, and return event candidates."""
        current_time = float(now or event.get("timestamp") or time.time())
        candidates: list[EventCandidate] = []

        with self._lock:
            # 1. Advance state machine on active incidents
            self._advance_lifecycle(current_time)

            # 2. Find best matching active incident
            active_pool = [
                inc
                for inc in self._incidents.values()
                if inc.state in (IncidentState.ACTIVE, IncidentState.QUIESCENT)
            ]
            best_inc, breakdown = self.correlator.find_best_incident(
                event, active_pool, current_time
            )

            decision = (
                breakdown.decision if breakdown else IncidentCorrelationDecision.NEW_INCIDENT
            )
            target_incident: CanonicalIncident

            if decision == IncidentCorrelationDecision.ATTACH and best_inc is not None:
                # High Confidence: Attach to existing incident
                target_incident = best_inc
                prior_sev = target_incident.severity
                if breakdown is not None:
                    target_incident.correlation_breakdown = breakdown.to_dict()

                entry = self._create_timeline_entry(event, current_time)
                IncidentTimelineManager.add_entry(target_incident.timeline, entry)

                target_incident.events.append(event)
                target_incident.last_seen = max(target_incident.last_seen, entry.timestamp)
                target_incident.state = IncidentState.ACTIVE

                if entry.camera_id:
                    target_incident.cameras.add(entry.camera_id)
                if entry.zone:
                    target_incident.zones.add(entry.zone)
                for ent in entry.entities:
                    target_incident.involved_entities.add(ent)
                for obj in entry.objects:
                    target_incident.involved_objects.add(obj)

                # Recompute severity
                sev_str, sev_breakdown = self._compute_severity(
                    target_incident.events,
                    target_incident.zones,
                    target_incident.involved_entities,
                )
                target_incident.severity = sev_str
                target_incident.severity_breakdown = sev_breakdown

                # Check for incident escalation milestone
                if sev_str == "alert" and prior_sev != "alert":
                    candidates.append(
                        EventCandidate(
                            rule="incident_escalation",
                            severity="alert",
                            summary=f"Incident {target_incident.incident_id} escalated to ALERT severity ({target_incident.title})",
                            entity_id=event.get("entity_id"),
                            camera_id=event.get("camera_id", "multi_camera"),
                            wall_time=current_time,
                            evidence={
                                "incident_id": target_incident.incident_id,
                                "event_count": len(target_incident.events),
                                "involved_entities": list(target_incident.involved_entities),
                                "summary": target_incident.title,
                            },
                        )
                    )

                self._incidents.move_to_end(target_incident.incident_id)

            elif (
                decision == IncidentCorrelationDecision.CORRELATION_CANDIDATE
                and best_inc is not None
                and breakdown is not None
            ):
                # Ambiguous: Spawn new incident, but record correlation link on both
                target_incident = self._spawn_new_incident(event, current_time)
                candidate_record_a = {
                    "related_incident_id": target_incident.incident_id,
                    "score": round(breakdown.total_correlation_score, 3),
                    "explanation": breakdown.explanation,
                    "timestamp": current_time,
                }
                candidate_record_b = {
                    "related_incident_id": best_inc.incident_id,
                    "score": round(breakdown.total_correlation_score, 3),
                    "explanation": breakdown.explanation,
                    "timestamp": current_time,
                }
                best_inc.correlation_candidates.append(candidate_record_a)
                target_incident.correlation_candidates.append(candidate_record_b)

            else:
                # Low Confidence / New Situation: Spawn new incident
                target_incident = self._spawn_new_incident(event, current_time)

            # 3. Check for Merge Candidates between active incidents
            for other_inc in list(self._incidents.values()):
                if (
                    other_inc.incident_id != target_incident.incident_id
                    and other_inc.state == IncidentState.ACTIVE
                ):
                    is_merge, m_score, m_summary = self.correlator.check_merge_candidates(
                        target_incident, other_inc, current_time
                    )
                    if is_merge:
                        if other_inc.incident_id not in target_incident.merge_candidates:
                            target_incident.merge_candidates.append(other_inc.incident_id)
                        if target_incident.incident_id not in other_inc.merge_candidates:
                            other_inc.merge_candidates.append(target_incident.incident_id)
                        candidates.append(
                            EventCandidate(
                                rule="incident_merge_candidate",
                                severity="notice",
                                summary=f"Merge candidate detected between Incident {target_incident.incident_id} and {other_inc.incident_id}",
                                entity_id=event.get("entity_id"),
                                camera_id=event.get("camera_id", "multi_camera"),
                                wall_time=current_time,
                                evidence={
                                    "incident_a": target_incident.incident_id,
                                    "incident_b": other_inc.incident_id,
                                    "merge_score": round(m_score, 3),
                                    "summary": m_summary,
                                },
                            )
                        )

            # 4. Enforce memory cap without losing persistence
            if len(self._incidents) > self.max_active_memory_incidents:
                # Evict oldest resolved or quiescent incident
                for inc_id, inc in list(self._incidents.items()):
                    if inc.state in (IncidentState.RESOLVED, IncidentState.EXPIRED):
                        self._persist_single_incident(inc)
                        del self._incidents[inc_id]
                        break

            return target_incident, decision, candidates

    def _spawn_new_incident(
        self, initial_event: dict[str, Any], now: float
    ) -> CanonicalIncident:
        """Create a new CanonicalIncident from an initial event."""
        inc_id = f"inc_{uuid.uuid4().hex[:12]}"
        rule = str(initial_event.get("rule", "Situation"))
        camera_id = str(initial_event.get("camera_id", "default"))
        entity_id = initial_event.get("entity_id")
        title = f"{rule.replace('_', ' ').title()} on {camera_id}"
        if entity_id:
            title += f" involving {entity_id}"

        entry = self._create_timeline_entry(initial_event, now)
        timeline = [entry]

        cameras = {entry.camera_id} if entry.camera_id else set()
        zones = {entry.zone} if entry.zone else set()
        entities = set(entry.entities)
        objects = set(entry.objects)

        sev_str, sev_breakdown = self._compute_severity([initial_event], zones, entities)

        inc = CanonicalIncident(
            incident_id=inc_id,
            title=title,
            state=IncidentState.ACTIVE,
            severity=sev_str,
            severity_breakdown=sev_breakdown,
            first_seen=entry.timestamp,
            last_seen=entry.timestamp,
            cameras=cameras,
            zones=zones,
            involved_entities=entities,
            involved_objects=objects,
            timeline=timeline,
            events=[initial_event],
            relationship_links=[],
            correlation_candidates=[],
            merge_candidates=[],
            evidence_dossier={
                "initial_rule": rule,
                "camera_sequence": [camera_id],
            },
        )
        self._incidents[inc_id] = inc
        return inc

    def _advance_lifecycle(self, now: float) -> None:
        """Update incident lifecycle state based on inactivity timeouts."""
        for inc in self._incidents.values():
            if inc.state == IncidentState.RESOLVED or inc.state == IncidentState.EXPIRED:
                continue
            dt = max(0.0, now - inc.last_seen)
            if dt >= self.config.resolution_timeout_s:
                inc.state = IncidentState.RESOLVED
            elif dt >= self.config.quiescent_timeout_s:
                inc.state = IncidentState.QUIESCENT

    def get_incidents(
        self,
        state: str | None = None,
        entity_id: str | None = None,
        limit: int = 50,
        now: float | None = None,
    ) -> list[CanonicalIncident]:
        """Query in-memory incidents with optional state and entity filters."""
        current_time = now or time.time()
        with self._lock:
            self._advance_lifecycle(current_time)
            results: list[CanonicalIncident] = []
            for inc in self._incidents.values():
                if state and inc.state.value != state.lower():
                    continue
                if entity_id and entity_id not in inc.involved_entities:
                    continue
                results.append(inc)
            results.sort(key=lambda i: (i.last_seen, i.first_seen), reverse=True)
            return results[:limit]

    def get_incident(self, incident_id: str) -> CanonicalIncident | None:
        """Lookup a single incident by ID from memory or SQLite storage."""
        with self._lock:
            if incident_id in self._incidents:
                return self._incidents[incident_id]

        if self.store:
            row = self.store.get_incident(incident_id)
            if row:
                return self._deserialize_incident(row)
        return None

    def _persist_single_incident(self, inc: CanonicalIncident) -> None:
        """Save a single incident record to SQLite store."""
        if not self.store:
            return
        try:
            record = {
                "incident_id": inc.incident_id,
                "title": inc.title,
                "state": inc.state.value,
                "severity": inc.severity,
                "first_seen": inc.first_seen,
                "last_seen": inc.last_seen,
                "cameras": json.dumps(sorted(inc.cameras)),
                "zones": json.dumps(sorted(inc.zones)),
                "entities": json.dumps(sorted(inc.involved_entities)),
                "event_count": len(inc.events),
                "dossier_json": json.dumps(inc.to_dict()),
                "updated_at": time.time(),
            }
            self.store.write_incidents([record])
        except Exception as exc:
            log.warning("error writing incident to store: %s", exc)

    def persist_to_store(self) -> int:
        """Flush all in-memory incidents to SQLite."""
        if not self.store:
            return 0
        records = []
        with self._lock:
            for inc in self._incidents.values():
                records.append(
                    {
                        "incident_id": inc.incident_id,
                        "title": inc.title,
                        "state": inc.state.value,
                        "severity": inc.severity,
                        "first_seen": inc.first_seen,
                        "last_seen": inc.last_seen,
                        "cameras": json.dumps(sorted(inc.cameras)),
                        "zones": json.dumps(sorted(inc.zones)),
                        "entities": json.dumps(sorted(inc.involved_entities)),
                        "event_count": len(inc.events),
                        "dossier_json": json.dumps(inc.to_dict()),
                        "updated_at": time.time(),
                    }
                )
        return self.store.write_incidents(records)

    def _hydrate_from_store(self) -> None:
        """Load open/quiescent incidents from SQLite on startup."""
        if not self.store:
            return
        try:
            stored = self.store.incidents(limit=100)
            with self._lock:
                for row in stored:
                    inc = self._deserialize_incident(row)
                    if inc and inc.incident_id not in self._incidents:
                        self._incidents[inc.incident_id] = inc
        except Exception as exc:
            log.warning("could not hydrate incidents from store: %s", exc)

    def _deserialize_incident(self, row: dict[str, Any]) -> CanonicalIncident | None:
        """Reconstruct CanonicalIncident from SQLite row."""
        try:
            dossier = json.loads(row.get("dossier_json", "{}"))
            timeline = [
                IncidentTimelineEntry(
                    entry_id=t["entry_id"],
                    timestamp=t["timestamp"],
                    event_id=t.get("event_id"),
                    event_type=t["event_type"],
                    camera_id=t["camera_id"],
                    entities=tuple(t.get("entities", ())),
                    objects=tuple(t.get("objects", ())),
                    zone=t.get("zone"),
                    summary=t["summary"],
                    evidence_ref=t.get("evidence_ref", {}),
                )
                for t in dossier.get("timeline", [])
            ]
            sev_b = dossier.get("severity_breakdown", {})
            breakdown = IncidentSeverityBreakdown(
                highest_event_severity=sev_b.get("highest_event_severity", "info"),
                corroborating_event_count=sev_b.get("corroborating_event_count", 1),
                involved_entity_count=sev_b.get("involved_entity_count", 1),
                restricted_zone_factor=sev_b.get("restricted_zone_factor", 0.0),
                escalation_factor=sev_b.get("escalation_factor", 0.0),
                final_severity=sev_b.get("final_severity", row.get("severity", "info")),
                severity_score=sev_b.get("severity_score", 0.5),
            )
            return CanonicalIncident(
                incident_id=row["incident_id"],
                title=row["title"],
                state=IncidentState(row["state"]),
                severity=row["severity"],
                severity_breakdown=breakdown,
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                cameras=set(json.loads(row.get("cameras", "[]"))),
                zones=set(json.loads(row.get("zones", "[]"))),
                involved_entities=set(json.loads(row.get("entities", "[]"))),
                involved_objects=set(dossier.get("involved_objects", [])),
                timeline=timeline,
                events=dossier.get("events", []),
                relationship_links=dossier.get("relationship_links", []),
                correlation_candidates=dossier.get("correlation_candidates", []),
                merge_candidates=dossier.get("merge_candidates", []),
                evidence_dossier=dossier.get("evidence_dossier", {}),
                correlation_breakdown=dossier.get("correlation_breakdown"),
            )
        except Exception as exc:
            log.warning("error deserializing incident: %s", exc)
            return None
