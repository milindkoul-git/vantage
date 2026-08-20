"""Temporal activity recognition.

The first thing in the platform that cannot be answered from one frame:
``sitting_down`` is a change of posture, not a posture, and ``loitering`` is
indistinguishable from standing until you know how long it has lasted.

Costs nothing to run - it reads signals Phases 3 and 4 already produce and holds
a small bounded buffer per entity. No model, no weights.

On the fall rule
----------------
``falling`` is the one activity here with a consequence attached, so its limits
belong next to its name rather than buried:

* It is **not a certified fall detector** and must not be relied on where one is
  required. It reports that a body went from upright to horizontal quickly.
* It **needs legs.** Posture needs hips and knees, so a camera that sees people
  only from the waist up can never report a fall at all.
* It **inherits the posture rules' blind spot**: a steeply angled camera
  compresses vertical geometry and will eventually read upright bodies as
  horizontal ones.
* A person lowering themselves deliberately is reported as **nothing**, not as a
  low-confidence fall. A hedged alert is worse than none, because it teaches
  whoever reads it to ignore the real one.
"""

from vantage.activity.contracts import (
    Activity,
    ActivityObservation,
    ActivityResult,
    EntityActivity,
    to_observation_record,
)

__all__ = [
    "Activity",
    "ActivityEngine",
    "ActivityObservation",
    "ActivityParams",
    "ActivityResult",
    "EntityActivity",
    "Recognizer",
    "RuleRecognizer",
    "build_activity_engine",
    "to_observation_record",
]


def __getattr__(name: str):
    if name in ("ActivityEngine", "build_activity_engine"):
        from vantage.activity import engine

        return getattr(engine, name)
    if name in ("ActivityParams", "RuleRecognizer"):
        from vantage.activity import recognizer

        return getattr(recognizer, name)
    if name == "Recognizer":
        from vantage.activity.base import Recognizer

        return Recognizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
