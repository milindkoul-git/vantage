"""Entity Intelligence: Canonical entity context, identity hierarchy, and immutable snapshots."""

from vantage.entity.context import EntityContext
from vantage.entity.contracts import (
    ActivityContext,
    EntitySnapshot,
    EventContext,
    IdentityEvidence,
    IdentityLevel,
    JourneyContext,
    RelationshipContext,
    SpatialContext,
    SpatialPresence,
    TemporalKinematics,
)
from vantage.entity.manager import EntityContextManager

__all__ = [
    "ActivityContext",
    "EntityContext",
    "EntityContextManager",
    "EntitySnapshot",
    "EventContext",
    "IdentityEvidence",
    "IdentityLevel",
    "JourneyContext",
    "RelationshipContext",
    "SpatialContext",
    "SpatialPresence",
    "TemporalKinematics",
]
