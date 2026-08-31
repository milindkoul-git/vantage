"""Unit Tests for IncidentSearch Architecture (StructuredQueryParser, LexicalRetriever)."""

from __future__ import annotations

from vantage.search.semantic import (
    IncidentSearch,
    ParsedSearchIntent,
    SemanticEventSearch,
    StructuredQueryParser,
)


def test_structured_query_parser() -> None:
    parser = StructuredQueryParser()

    # 1. Parse complex natural language query
    intent = parser.parse(
        "Find critical tailgating events near corridor involving global_person_4"
    )
    assert isinstance(intent, ParsedSearchIntent)
    assert intent.camera_filter == "corridor"
    assert intent.severity_filter == "alert"
    assert intent.rule_filter == "tailgating"
    assert intent.entity_filter == "global_person_4"
    assert "critical" in intent.raw_tokens
    assert "tailgating" in intent.raw_tokens

    # 2. Verify backward compatibility alias
    assert SemanticEventSearch is IncidentSearch
