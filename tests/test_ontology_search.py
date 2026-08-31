"""Unit Tests for Security Event Ontology Search Expansion."""

from __future__ import annotations

from vantage.search.semantic import (
    EventOntologyExpander,
    ParsedSearchIntent,
    StructuredQueryParser,
)


def test_event_ontology_expander() -> None:
    expander = EventOntologyExpander()

    tokens = {"fall", "fight", "bag"}
    expanded = expander.expand(tokens)

    # Fall concepts
    assert "sudden_collapse" in expanded
    assert "prone" in expanded

    # Fight concepts
    assert "erratic_high_energy_motion" in expanded
    assert "group_convergence" in expanded

    # Bag concepts
    assert "unattended_object_dwell" in expanded


def test_incident_search_with_ontology_parsing() -> None:
    parser = StructuredQueryParser()

    intent = parser.parse("Find critical collapse events near retail")
    assert isinstance(intent, ParsedSearchIntent)
    assert intent.camera_filter == "retail"
    assert intent.severity_filter == "alert"
    assert intent.rule_filter == "sudden_collapse"
    assert "sudden_collapse" in intent.expanded_concepts
    assert "floor" in intent.expanded_concepts
