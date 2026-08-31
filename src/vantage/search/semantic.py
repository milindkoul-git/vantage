"""Incident & Event Search Engine with Structured Intent Parsing & Security Event Ontology Expansion."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from vantage.incident.models import decode_dossier
from vantage.storage.contracts import Query
from vantage.storage.sqlite_store import SqliteStore


@dataclass
class SearchResultItem:
    """One ranked search result matching an operator query."""

    id: int | str
    timestamp: float
    camera_id: str
    rule: str
    severity: str
    summary: str
    entity_id: str | None
    zone: str | None
    score: float
    evidence_clip: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id,
            "rule": self.rule,
            "severity": self.severity,
            "summary": self.summary,
            "entity_id": self.entity_id,
            "zone": self.zone,
            "score": round(self.score, 2),
            "evidence_clip": self.evidence_clip,
        }


@dataclass
class ParsedSearchIntent:
    """Structured query filters parsed from natural language text."""

    camera_filter: str | None = None
    severity_filter: str | None = None
    rule_filter: str | None = None
    entity_filter: str | None = None
    raw_tokens: set[str] = None  # type: ignore[assignment]
    expanded_concepts: set[str] = None  # type: ignore[assignment]


class EventOntologyExpander:
    """Expands natural language operator queries into security concept clusters and synonyms."""

    CONCEPT_MAP: ClassVar[dict[str, set[str]]] = {
        "fall": {"sudden_collapse", "falling", "prone", "floor", "collapse", "drop"},
        "fell": {"sudden_collapse", "falling", "prone", "floor", "collapse"},
        "collapsed": {"sudden_collapse", "falling", "prone", "floor"},
        "collapse": {"sudden_collapse", "falling", "prone", "floor"},
        "fight": {"erratic_high_energy_motion", "group_convergence", "aggressive", "running"},
        "scuffle": {"erratic_high_energy_motion", "group_convergence", "near"},
        "pacing": {"erratic_pacing", "loitering", "dwell", "moving"},
        "nervous": {"erratic_pacing", "abrupt_direction_reversal"},
        "crouch": {"crouching_dwell", "crouching", "sitting"},
        "crouching": {"crouching_dwell", "crouching"},
        "hiding": {"crouching_dwell", "crouching", "loitering"},
        "crowd": {"group_convergence", "high_crowd_density", "convergence"},
        "gathering": {"group_convergence", "high_crowd_density", "near"},
        "scatter": {"group_dispersion", "running", "dispersion"},
        "panic": {"group_dispersion", "erratic_high_energy_motion", "running"},
        "bag": {
            "unattended_object_dwell",
            "backpack",
            "suitcase",
            "handbag",
            "carrying_baggage",
        },
        "luggage": {"unattended_object_dwell", "suitcase", "backpack", "carrying_baggage"},
        "abandoned": {"unattended_object_dwell", "unattended"},
        "unattended": {"unattended_object_dwell", "unattended"},
        "tailgate": {"tailgating", "tailgate_approach", "near"},
        "tailgating": {"tailgating", "tailgate_approach", "near"},
        "reversal": {"abrupt_direction_reversal", "wrong_way_direction"},
        "wrong way": {"wrong_way_direction", "abrupt_direction_reversal"},
        "breach": {"exclusion_breach", "zone_entry"},
        "intrusion": {"exclusion_breach", "zone_entry"},
        "following": {"following_pattern", "tailing", "shadowing", "near"},
        "tailing": {"following_pattern", "near"},
        "shadowing": {"following_pattern", "near"},
        "together": {
            "recurring_proximity",
            "recurrent_interaction",
            "co_occurrence",
            "group_association",
        },
        "companions": {"recurring_proximity", "recurrent_interaction", "co_occurrence"},
        "associates": {
            "recurring_proximity",
            "recurrent_interaction",
            "co_occurrence",
            "group_association",
        },
        "co-occurrence": {"recurring_proximity", "co_occurrence"},
        "meeting": {"recurrent_interaction", "recurring_proximity", "near"},
        "rendezvous": {"recurrent_interaction", "recurring_proximity", "near"},
    }

    def expand(self, tokens: set[str]) -> set[str]:
        """Expand tokens using the domain ontology map."""
        expanded = set(tokens)
        for token in tokens:
            if token in self.CONCEPT_MAP:
                expanded.update(self.CONCEPT_MAP[token])
        return expanded


class StructuredQueryParser:
    """Parses natural-language queries into structured security filters and ontology terms."""

    CAMERAS: ClassVar[list[str]] = [
        "retail",
        "crosswalk",
        "corridor",
        "doorway",
        "lobby",
        "gate",
        "view_a",
        "view_b",
    ]
    SEVERITIES: ClassVar[dict[str, str]] = {
        "alert": "alert",
        "critical": "alert",
        "danger": "alert",
        "notice": "notice",
        "warning": "notice",
        "info": "info",
    }
    RULES: ClassVar[dict[str, str]] = {
        "tailgating": "tailgating",
        "wrong_way": "wrong_way_direction",
        "wrong way": "wrong_way_direction",
        "loitering": "loitering",
        "loiter": "loitering",
        "handover": "cross_camera_handover",
        "transition": "cross_camera_handover",
        "exclusion": "exclusion_breach",
        "breach": "exclusion_breach",
        "entry": "zone_entry",
        "fall": "sudden_collapse",
        "collapse": "sudden_collapse",
        "pacing": "erratic_pacing",
        "unattended": "unattended_object_dwell",
        "convergence": "group_convergence",
        "dispersion": "group_dispersion",
    }

    def __init__(self) -> None:
        self.expander = EventOntologyExpander()

    def parse(self, query_text: str) -> ParsedSearchIntent:
        q_lower = query_text.lower().strip()
        tokens = set(re.findall(r"\w+", q_lower))

        cam_filter = next((c for c in self.CAMERAS if c in q_lower), None)
        sev_filter = next(
            (target for term, target in self.SEVERITIES.items() if term in q_lower), None
        )
        rule_filter = next(
            (target for term, target in self.RULES.items() if term in q_lower), None
        )

        entity_match = re.search(r"\b(global_\w+|\w+_\d+)\b", q_lower)
        entity_filter = entity_match.group(1) if entity_match else None

        expanded = self.expander.expand(tokens)

        return ParsedSearchIntent(
            camera_filter=cam_filter,
            severity_filter=sev_filter,
            rule_filter=rule_filter,
            entity_filter=entity_filter,
            raw_tokens=tokens,
            expanded_concepts=expanded,
        )


class SemanticRetriever(Protocol):
    """Protocol seam for future vector/embedding-based semantic retrieval."""

    def retrieve(self, query_text: str, limit: int = 50) -> list[SearchResultItem]: ...


class LexicalRetriever:
    """Ranks stored events based on token overlap, ontology expansion, and structured filters."""

    def __init__(self, store: SqliteStore) -> None:
        self.store = store

    def retrieve(self, intent: ParsedSearchIntent, limit: int = 50) -> list[SearchResultItem]:
        events_query = Query(since=time.time() - 7 * 86400, limit=200)
        rows = self.store.events(events_query)

        ranked: list[SearchResultItem] = []
        for row in rows:
            score = 0.0
            row_text = f"{row.summary} {row.rule} {row.severity} {row.camera_id} {row.entity_id} {row.zone}".lower()
            row_tokens = set(re.findall(r"\w+", row_text))

            # 1. Exact raw token overlap score (weight = 2.0)
            overlap = intent.raw_tokens.intersection(row_tokens)
            score += len(overlap) * 2.0

            # 2. Ontology concept expansion match (weight = 3.0)
            if intent.expanded_concepts:
                concept_overlap = (
                    intent.expanded_concepts.intersection(row_tokens) - intent.raw_tokens
                )
                score += len(concept_overlap) * 3.0

            # 3. Structured filter bonuses
            if intent.camera_filter and intent.camera_filter in row.camera_id.lower():
                score += 5.0
            if intent.severity_filter and intent.severity_filter == row.severity.lower():
                score += 5.0
            if intent.rule_filter and intent.rule_filter in row.rule.lower():
                score += 6.0
            if intent.entity_filter and intent.entity_filter in (row.entity_id or "").lower():
                score += 8.0

            if score > 0:
                clip = (
                    f"/api/evidence/{row.id}.mp4"
                    if row.severity in ("alert", "notice")
                    else None
                )
                ranked.append(
                    SearchResultItem(
                        id=row.id,
                        timestamp=row.timestamp,
                        camera_id=row.camera_id,
                        rule=row.rule,
                        severity=row.severity,
                        summary=row.summary,
                        entity_id=row.entity_id,
                        zone=row.zone,
                        score=score,
                        evidence_clip=clip,
                    )
                )

        ranked.sort(key=lambda r: (r.score, r.timestamp, str(r.id)), reverse=True)
        return ranked[:limit]


class IncidentSearch:
    """Answers natural-language security questions against the observation & event store."""

    def __init__(self, store: SqliteStore | None = None) -> None:
        self.store = store
        self.parser = StructuredQueryParser()
        self.retriever = LexicalRetriever(store) if store else None

    def search(self, query_text: str, limit: int = 50) -> dict[str, Any]:
        """Execute a natural-language search query."""
        if not self.store or not self.retriever:
            return {
                "query": query_text,
                "total": 0,
                "results": [],
                "reason": "No store attached",
            }

        q_lower = query_text.lower().strip()
        if not q_lower:
            return {"query": query_text, "total": 0, "results": []}

        intent = self.parser.parse(query_text)
        top_results = self.retriever.retrieve(intent, limit=limit)

        return {
            "query": query_text,
            "total": len(top_results),
            "parsed_intent": {
                "camera": intent.camera_filter,
                "severity": intent.severity_filter,
                "rule": intent.rule_filter,
                "entity": intent.entity_filter,
                "expanded_concepts": list(intent.expanded_concepts)
                if intent.expanded_concepts
                else [],
            },
            "results": [r.to_dict() for r in top_results],
        }

    def search_incidents(self, query_text: str, limit: int = 20) -> dict[str, Any]:
        """Search canonical situational incidents across entities, cameras, and ontology concepts."""
        if not self.store:
            return {
                "query": query_text,
                "total": 0,
                "incidents": [],
                "reason": "No store attached",
            }

        intent = self.parser.parse(query_text)
        stored_incs = self.store.incidents(limit=200)

        matched: list[dict[str, Any]] = []
        for inc_row in stored_incs:
            dossier = decode_dossier(inc_row)

            score = 0.0
            # Entity match
            if intent.entity_filter:
                ents = inc_row.get("entities", "")
                if intent.entity_filter.lower() in ents.lower():
                    score += 0.50

            # Camera match
            if intent.camera_filter:
                cams = inc_row.get("cameras", "")
                if intent.camera_filter.lower() in cams.lower():
                    score += 0.30

            # Severity filter / match
            if intent.severity_filter and inc_row.get("severity") == intent.severity_filter:
                score += 0.25

            # Concept / Rule matching
            title = inc_row.get("title", "").lower()
            for concept in intent.expanded_concepts:
                if concept in title:
                    score += 0.20

            if score > 0.0 or not intent.raw_tokens:
                dossier["search_relevance"] = round(score, 2)
                matched.append(dossier or inc_row)

        matched.sort(
            key=lambda x: (x.get("search_relevance", 0.0), x.get("last_seen", 0.0)),
            reverse=True,
        )
        return {
            "query": query_text,
            "total": len(matched[:limit]),
            "parsed_intent": {
                "camera": intent.camera_filter,
                "severity": intent.severity_filter,
                "rule": intent.rule_filter,
                "entity": intent.entity_filter,
            },
            "incidents": matched[:limit],
        }


# Backward compatibility alias
SemanticEventSearch = IncidentSearch
