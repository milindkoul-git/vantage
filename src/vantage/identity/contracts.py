"""Identity contracts, and the constraints they exist to enforce.

This is the only part of the platform that answers *who*, and the specification
fenced it more carefully than anything else in the project. Those fences are
implemented here as types rather than left as documentation.

What the design guarantees
--------------------------
**Tracking never depends on identity.** The pipeline runs identically with this
subsystem off; identity attaches to an existing anonymous ``entity_id`` and adds
a name, or does not. That was the seam every phase since 4 left open, and
nothing above it changes now that something fills it.

**Enrolment is an act, not an observation.** A face cannot become an identity by
being seen. It is added by a person at the machine running an explicit command
with an explicit consent flag, and there is no code path from the live pipeline
to the gallery.

**Only templates are kept, never images.** An enrolment stores a 128-dimensional
vector. That is still biometric data and is treated as such - it is not a
photograph, and it should not be described as anonymised either.

**Unknown is a real answer.** Below the similarity threshold the result is
:attr:`Identity.UNKNOWN`, never the closest guess. A system that always names
someone is wrong about strangers by construction.

**Everything is audited.** Enrolment, revocation and every identification are
recorded with a timestamp, so the question "when did this system decide it knew
who I was" has an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

UNKNOWN = "unknown"
"""The name used when no enrolled template matches well enough."""

EMBEDDING_DIM = 128
"""Length of an SFace template. Checked on load, because a vector of the wrong
length silently produces meaningless similarities rather than an error."""


class AuditAction(str, Enum):
    """What happened, for the audit trail."""

    ENROLLED = "enrolled"
    REVOKED = "revoked"
    IDENTIFIED = "identified"
    REJECTED = "rejected"
    """A face was compared and matched nobody well enough. Recorded because
    "the system looked at me and decided it did not know me" is a thing that
    happened, and a trail that only logs successes cannot show it."""


@dataclass(frozen=True, slots=True)
class Enrollment:
    """One enrolled person, as stored."""

    name: str
    template: tuple[float, ...]
    """The embedding. Not a photograph, and not anonymous either."""

    samples: int
    """How many face captures were averaged into this template."""

    enrolled_at: float
    note: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an enrolment needs a name")
        if len(self.template) != EMBEDDING_DIM:
            raise ValueError(
                f"template for {self.name!r} has {len(self.template)} dimensions, "
                f"expected {EMBEDDING_DIM}. A wrong-length vector produces "
                "meaningless similarities rather than an error, so it is refused here."
            )
        if self.samples < 1:
            raise ValueError("an enrolment needs at least one sample")

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.enrolled_at, tz=UTC)

    def describe(self) -> str:
        return (
            f"{self.name:20s} {self.samples:3d} samples  "
            f"enrolled {self.when.strftime('%Y-%m-%d %H:%M')}"
            + (f"  ({self.note})" if self.note else "")
        )


@dataclass(frozen=True, slots=True)
class IdentityMatch:
    """The outcome of comparing one face against the gallery."""

    name: str
    similarity: float
    """Cosine similarity to the best template, in roughly [-1, 1]."""

    runner_up: str | None = None
    runner_up_similarity: float = 0.0
    """The second-best match. Reported because a confident-looking score means
    much less when another template scored nearly the same - which is exactly
    the situation where a wrong name is most likely."""

    @property
    def known(self) -> bool:
        return self.name != UNKNOWN

    @property
    def margin(self) -> float:
        """How far ahead the winner is. Small means the answer is contested."""
        return (
            self.similarity - self.runner_up_similarity if self.runner_up else self.similarity
        )

    def describe(self) -> str:
        if not self.known:
            return f"unknown (best {self.similarity:.3f})"
        contested = (
            f", runner-up {self.runner_up} {self.runner_up_similarity:.3f}"
            if self.runner_up
            else ""
        )
        return f"{self.name} ({self.similarity:.3f}{contested})"


@dataclass(frozen=True, slots=True)
class EntityIdentity:
    """What the system currently believes about one tracked entity."""

    track_id: int
    entity_id: str
    name: str
    similarity: float
    votes: int
    resolved: bool
    """Whether enough agreeing observations have accumulated to commit.

    Until then the name is provisional and should not be shown as fact. One
    face crop at an awkward angle is not evidence of who someone is."""

    attempts: int = 0

    @property
    def known(self) -> bool:
        return self.resolved and self.name != UNKNOWN

    def describe(self) -> str:
        if not self.resolved:
            return f"{self.entity_id}: identifying ({self.votes}/{self.attempts})"
        if self.name == UNKNOWN:
            return f"{self.entity_id}: unknown"
        return f"{self.entity_id}: {self.name} ({self.similarity:.2f})"


@dataclass(frozen=True, slots=True)
class IdentityResult:
    """Identity state for one frame."""

    identities: tuple[EntityIdentity, ...]
    source_id: str
    frame_index: int
    capture_wall: float
    attempted: int = 0
    """Faces actually compared this frame. Usually zero: identification runs on
    an interval and stops once a track is resolved."""

    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.identities)

    def __iter__(self):
        return iter(self.identities)

    def by_track(self) -> dict[int, EntityIdentity]:
        return {item.track_id: item for item in self.identities}

    def named(self) -> tuple[EntityIdentity, ...]:
        return tuple(item for item in self.identities if item.known)

    def describe(self) -> str:
        if not self.identities:
            return "no entities"
        return "; ".join(item.describe() for item in self.identities)


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One line of the identity audit trail."""

    action: AuditAction
    name: str
    timestamp: float
    detail: str = ""
    entity_id: str | None = None
    similarity: float | None = None

    @property
    def when(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=UTC)

    def describe(self) -> str:
        stamp = self.when.strftime("%Y-%m-%d %H:%M:%S")
        who = f" {self.entity_id}" if self.entity_id else ""
        score = f" ({self.similarity:.3f})" if self.similarity is not None else ""
        return f"{stamp}  {self.action.value:11s} {self.name}{who}{score}  {self.detail}"

    def to_record(self) -> dict[str, Any]:
        return {
            "timestamp": self.when.isoformat(),
            "action": self.action.value,
            "name": self.name,
            "entity_id": self.entity_id,
            "similarity": self.similarity,
            "detail": self.detail,
        }
