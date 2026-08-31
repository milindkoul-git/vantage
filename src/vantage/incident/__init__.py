"""Incident Intelligence & Multi-Event Reasoning Subsystem.

Provides situational incident correlation, explainable multi-factor attribution,
negative continuity penalties, chronological timeline dossiers, and lifecycle state machines.
"""

from __future__ import annotations

from vantage.incident.config import IncidentCorrelatorConfig
from vantage.incident.correlator import IncidentCorrelator
from vantage.incident.models import (
    CanonicalIncident,
    IncidentCorrelationBreakdown,
    IncidentCorrelationDecision,
    IncidentSeverityBreakdown,
    IncidentState,
    IncidentTimelineEntry,
)
from vantage.incident.service import IncidentService
from vantage.incident.timeline import IncidentTimelineManager

__all__ = [
    "CanonicalIncident",
    "IncidentCorrelationBreakdown",
    "IncidentCorrelationDecision",
    "IncidentCorrelator",
    "IncidentCorrelatorConfig",
    "IncidentService",
    "IncidentSeverityBreakdown",
    "IncidentState",
    "IncidentTimelineEntry",
    "IncidentTimelineManager",
]
