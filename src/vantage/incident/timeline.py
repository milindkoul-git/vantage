"""Chronological Timeline Management, Canonical Event Deduplication, and Provenance Linking."""

from __future__ import annotations

import hashlib

from vantage.incident.models import IncidentTimelineEntry


class IncidentTimelineManager:
    """Maintains strictly ordered, deduplicated timeline observations for an incident."""

    @staticmethod
    def _entry_fingerprint(
        camera_id: str,
        event_type: str,
        entity_id: str | None,
        timestamp: float,
        summary: str,
    ) -> str:
        raw = f"{camera_id}:{event_type}:{entity_id or ''}:{round(timestamp, 2)}:{summary}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def add_entry(
        cls,
        timeline: list[IncidentTimelineEntry],
        entry: IncidentTimelineEntry,
    ) -> bool:
        """Insert entry in chronological order if not already present. Returns True if inserted."""
        # 1. Deduplication by canonical event_id
        if entry.event_id is not None:
            for existing in timeline:
                if existing.event_id is not None and str(existing.event_id) == str(
                    entry.event_id
                ):
                    return False  # Duplicate event

        # 2. Deduplication by fingerprint
        entry_fp = cls._entry_fingerprint(
            entry.camera_id,
            entry.event_type,
            entry.entities[0] if entry.entities else None,
            entry.timestamp,
            entry.summary,
        )
        for existing in timeline:
            ex_fp = cls._entry_fingerprint(
                existing.camera_id,
                existing.event_type,
                existing.entities[0] if existing.entities else None,
                existing.timestamp,
                existing.summary,
            )
            if entry_fp == ex_fp:
                return False

        # 3. Insertion with chronological sorting
        timeline.append(entry)
        timeline.sort(key=lambda e: (e.timestamp, e.entry_id))
        return True
