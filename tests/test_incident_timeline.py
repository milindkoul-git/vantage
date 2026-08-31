"""Unit tests for IncidentTimelineManager deduplication and chronological ordering."""

from __future__ import annotations

from vantage.incident.models import IncidentTimelineEntry
from vantage.incident.timeline import IncidentTimelineManager


def test_timeline_chronological_ordering() -> None:
    timeline: list[IncidentTimelineEntry] = []

    e3 = IncidentTimelineEntry(
        "tle_3",
        130.0,
        "ev_3",
        "group_dispersion",
        "cam_01",
        ("p1", "p2"),
        (),
        None,
        "dispersion",
    )
    e1 = IncidentTimelineEntry(
        "tle_1", 100.0, "ev_1", "exclusion_breach", "cam_01", ("p1",), (), "vault", "breach"
    )
    e2 = IncidentTimelineEntry(
        "tle_2", 115.0, "ev_2", "following_pattern", "cam_01", ("p2",), (), None, "following"
    )

    # Insert out-of-order
    assert IncidentTimelineManager.add_entry(timeline, e3) is True
    assert IncidentTimelineManager.add_entry(timeline, e1) is True
    assert IncidentTimelineManager.add_entry(timeline, e2) is True

    # Check strict chronological ordering
    assert [t.entry_id for t in timeline] == ["tle_1", "tle_2", "tle_3"]
    assert [t.timestamp for t in timeline] == [100.0, 115.0, 130.0]


def test_timeline_event_id_deduplication() -> None:
    timeline: list[IncidentTimelineEntry] = []

    e1 = IncidentTimelineEntry(
        "tle_1", 100.0, "ev_canonical_42", "loitering", "cam_01", ("p1",), (), None, "loitering"
    )
    dup_e1 = IncidentTimelineEntry(
        "tle_dup",
        100.0,
        "ev_canonical_42",
        "loitering",
        "cam_01",
        ("p1",),
        (),
        None,
        "loitering",
    )

    assert IncidentTimelineManager.add_entry(timeline, e1) is True
    assert IncidentTimelineManager.add_entry(timeline, dup_e1) is False
    assert len(timeline) == 1
