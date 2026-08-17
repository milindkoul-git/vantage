"""Exception hierarchy.

Every failure the platform raises deliberately derives from :class:`VantageError`
so that a host application can distinguish "the platform said no" from "something
unexpected exploded". Nothing here is caught-and-ignored anywhere in the codebase.
"""

from __future__ import annotations


class VantageError(Exception):
    """Base class for all errors raised deliberately by the platform."""


class ConfigError(VantageError):
    """Configuration was missing, malformed, or semantically invalid."""


class SourceError(VantageError):
    """Base class for video-source failures."""


class SourceOpenError(SourceError):
    """A video source could not be opened or validated.

    Raised with an actionable message: what was tried, and what to check.
    """


class SourceReadError(SourceError):
    """A source was open but frame acquisition failed unrecoverably."""


class SourceExhausted(SourceError):
    """The source reached its natural end.

    This is control flow, not a fault: a file played to its last frame, or a
    synthetic source produced its configured frame count. Live sources never
    raise it on their own.
    """


class SourceStateError(SourceError):
    """A source method was called in the wrong lifecycle state."""
