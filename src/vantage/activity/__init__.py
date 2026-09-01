"""Temporal activity recognition.

The first thing in the platform that cannot be answered from one frame:
``sitting_down`` is a change of posture, not a posture, and ``loitering`` is
indistinguishable from standing until you know how long it has lasted.

Costs nothing to run - it reads signals Phases 3 and 4 already produce and holds
a small bounded buffer per entity. No model, no weights.

Who has activities
------------------
People, and only while something is actually being seen. Both gates are
``ActivityEngine``'s rather than the recogniser's, and both were put there after
running the pipeline over real street footage rather than over scenarios:

* ``activity.labels`` decides which detected classes are eligible, defaulting to
  ``("person",)``. These are verbs about people. Left open to every tracked
  class, 73% of everything this engine reported across five clips was about a
  car, a potted plant, a traffic light or a handbag - ``potted plant_2 is
  running`` reached the event log.
* An entity whose box was **predicted rather than detected** this frame is
  skipped. A coasting track keeps moving on the tracker's motion model, and that
  drift was being measured as speed: a further fifth to a quarter of everything
  reported. ``EntityState.observed`` had carried this distinction since Phase 4
  and no consumer had ever read it.

What speed cannot tell you
--------------------------
``walking`` and ``running`` are separated by speed in entity heights per second,
which is what makes the threshold survive a change of resolution and of distance.
It does not survive a change of **depth**. Someone walking toward the camera has
a box that grows while their feet cross the frame, and the ratio spikes; measured
on real footage, that is the whole of the remaining false-positive population.
Separating approach from pace needs a ground plane or an estimate of camera-
relative motion, neither of which exists here.

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
* The posture underneath it is cross-checked against the detector's own box. A
  skeleton reading horizontal inside a box taller than it is wide is reported as
  ``unknown``, because far-field joints are noisy and the torso vector flips.
  Measured on 2,652 frames containing no falls, that check removes all 77 false
  ``lying`` readings and with them all four false ALERTs.
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
