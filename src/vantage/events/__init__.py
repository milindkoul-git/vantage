"""The event engine: turning continuous observations into discrete events.

Everything before this phase says what is *true right now*. An event says what
*happened*, once, at a time. That reduction is the whole phase, and the hard
part is not choosing what is interesting - the rules are short - but ensuring a
condition true for forty-five consecutive frames produces one event rather than
forty-five.

No model, no weights: rules over the scene graph and the activity stream.
"""

from vantage.events.contracts import Event, EventResult, Severity
from vantage.events.rules import DEFAULT_RULES, RuleSpec, SceneContext

__all__ = [
    "DEFAULT_RULES",
    "Event",
    "EventEngine",
    "EventResult",
    "RuleSpec",
    "SceneContext",
    "Severity",
    "build_event_engine",
]


def __getattr__(name: str):
    if name in ("EventEngine", "build_event_engine"):
        from vantage.events import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
