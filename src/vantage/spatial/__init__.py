"""Spatial and interaction understanding.

Where entities are, and how they stand to each other. Produces the scene graph
the event engine of the next phase consumes: nodes are entities with zone
membership, edges are relations between pairs.

No model and no weights - this is geometry over the tracks and poses the earlier
phases already produce.

What a single camera cannot tell you
------------------------------------
Everything here rests on entities sharing a ground plane, with the bottom edge
of a box as the point where they meet it. That assumption is load-bearing and it
fails in specific, predictable ways:

* **There is no depth.** Two people on opposite sides of a room can have boxes
  that overlap perfectly. Proximity is a good approximation for entities at
  similar depth and degrades as their depths diverge.
* **Distances are not metres.** They are entity heights, on the rough basis that
  a standing adult is about as tall as two paces are long. Thresholds should be
  tuned per camera rather than trusted as physical distances.
* **A floating object breaks the anchor.** The ground point of something held or
  mounted is not on the floor, so its distance to anything else is wrong by
  however far off the floor it is.

Interaction is the claim most exposed to all three, which is why it is reported
at two confidence levels: a wrist landmark inside the object's box is
reach-confirmed evidence, while sustained proximity alone is capped low and says
so in its evidence string.
"""

from vantage.spatial.contracts import (
    EntitySpatial,
    Relation,
    RelationObservation,
    SpatialResult,
    Zone,
    ZoneEvent,
    ZoneOccupancy,
    to_scene_record,
)

__all__ = [
    "EntitySpatial",
    "Relation",
    "RelationObservation",
    "SpatialAnalyzer",
    "SpatialEngine",
    "SpatialParams",
    "SpatialResult",
    "Zone",
    "ZoneEvent",
    "ZoneOccupancy",
    "build_spatial_engine",
    "to_scene_record",
]


def __getattr__(name: str):
    if name in ("SpatialEngine", "build_spatial_engine"):
        from vantage.spatial import engine

        return getattr(engine, name)
    if name in ("SpatialAnalyzer", "SpatialParams"):
        from vantage.spatial import analyzer

        return getattr(analyzer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
