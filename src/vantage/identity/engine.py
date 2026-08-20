"""Identifying tracks rather than frames.

A face costs about 41 ms here - 12 to detect, 29 to embed - so running it on
every person on every frame is not affordable and would not help if it were.
Identity does not change during a track. The engine therefore:

* attempts identification only on tracks it has not resolved,
* only every ``interval`` tracking steps,
* accumulates agreeing results until ``min_votes``, then commits,
* and re-checks a resolved track occasionally, because the *tracker* can be
  wrong even when the recogniser is not.

Why votes rather than one good look
-----------------------------------
One face crop is one angle, one blink, one moment of motion blur. Committing on
a single comparison means a person walking past at the wrong instant gets
someone else's name for the rest of their time on camera. Requiring several
agreeing observations costs a second or two of "identifying" and removes almost
all of that.

Why re-check after committing
-----------------------------
Phase 3 keeps identity through occlusion by association, and association can
swap two people who cross. If that happens, the name attached to the track is
now on the wrong person, and nothing downstream would ever notice. A periodic
re-check catches it: a resolved track whose face stops matching is returned to
unresolved rather than left confidently wrong.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from vantage.core.errors import ConfigError
from vantage.core.frame import Frame
from vantage.core.logging import get_logger
from vantage.identity.contracts import (
    UNKNOWN,
    AuditAction,
    AuditRecord,
    EntityIdentity,
    IdentityResult,
)
from vantage.identity.gallery import Gallery
from vantage.identity.recognizer import FaceRecognizer
from vantage.tracking.contracts import Track, TrackingResult

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IdentityParams:
    """How hard to try, and how sure to be."""

    interval: int = 10
    """Tracking steps between attempts on an unresolved track."""

    min_votes: int = 3
    """Agreeing observations before a name is committed."""

    max_attempts: int = 25
    """Give up after this many, and settle on unknown.

    Without a ceiling, a track of someone facing away is retried for as long as
    they are on camera - 41 ms of face work every interval, forever, for a
    question that cannot be answered from the back of a head."""

    reverify_interval: int = 150
    """Steps between re-checks of a resolved track. Zero disables."""

    min_face_fraction: float = 0.04
    """Smallest face, as a fraction of the person box height, worth trying.

    A distant person's face is a dozen pixels; embedding it produces a number
    with no information in it, and voting on those is how a gallery starts
    naming strangers."""

    def __post_init__(self) -> None:
        for name in ("interval", "min_votes", "max_attempts"):
            if getattr(self, name) < 1:
                raise ConfigError(f"identity.{name} must be >= 1")
        if self.reverify_interval < 0:
            raise ConfigError("identity.reverify_interval must be >= 0 (0 disables)")
        if not 0.0 <= self.min_face_fraction <= 1.0:
            raise ConfigError("identity.min_face_fraction must be in [0, 1]")


@dataclass(slots=True)
class _TrackState:
    votes: dict[str, int] = field(default_factory=dict)
    attempts: int = 0
    steps: int = 0
    resolved: bool = False
    name: str = UNKNOWN
    similarity: float = 0.0
    resolved_at_step: int = 0

    def leader(self) -> tuple[str, int]:
        if not self.votes:
            return UNKNOWN, 0
        name = max(self.votes, key=lambda key: self.votes[key])
        return name, self.votes[name]


class IdentityEngine:
    """Attaches names to tracks, or leaves them anonymous."""

    def __init__(
        self,
        recognizer: FaceRecognizer,
        gallery: Gallery,
        *,
        params: IdentityParams | None = None,
        store=None,
        person_labels: tuple[str, ...] = ("person",),
    ) -> None:
        self._recognizer = recognizer
        self._gallery = gallery
        self._params = params or IdentityParams()
        self._store = store
        self._labels = tuple(label.lower() for label in person_labels)
        self._states: dict[int, _TrackState] = {}
        self._attempts = 0
        self._identified = 0

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    @property
    def params(self) -> IdentityParams:
        return self._params

    @property
    def tracked(self) -> int:
        return len(self._states)

    def stats(self) -> dict[str, object]:
        return {
            "enrolled": len(self._gallery),
            "attempts": self._attempts,
            "identified": self._identified,
            "tracked": len(self._states),
        }

    def update(self, frame: Frame, tracking: TrackingResult) -> IdentityResult:
        """Advance identification for the people in this frame."""
        attempted = 0
        identities: list[EntityIdentity] = []
        live: set[int] = set()

        for track in tracking.tracks:
            if track.label.lower() not in self._labels:
                continue
            live.add(track.track_id)
            state = self._states.setdefault(track.track_id, _TrackState())
            state.steps += 1

            if self._should_attempt(state):
                attempted += int(self._attempt(frame, track, state))

            identities.append(
                EntityIdentity(
                    track_id=track.track_id,
                    entity_id=track.entity_id,
                    name=state.name,
                    similarity=state.similarity,
                    votes=state.leader()[1],
                    resolved=state.resolved,
                    attempts=state.attempts,
                )
            )

        # Entities the tracker retired take their votes with them. Keeping them
        # would also mean a recycled track id inherited a stranger's name.
        for track_id in self._states.keys() - live:
            del self._states[track_id]

        return IdentityResult(
            identities=tuple(identities),
            source_id=tracking.source_id,
            frame_index=tracking.frame_index,
            capture_wall=tracking.capture_wall,
            attempted=attempted,
            metadata={"enrolled": len(self._gallery)},
        )

    def _should_attempt(self, state: _TrackState) -> bool:
        params = self._params
        if not state.resolved:
            if state.attempts >= params.max_attempts:
                return False
            return state.steps % params.interval == 0
        if not params.reverify_interval:
            return False
        return (state.steps - state.resolved_at_step) % params.reverify_interval == 0

    def _attempt(self, frame: Frame, track: Track, state: _TrackState) -> bool:
        """One comparison. Returns whether a face was actually embedded."""
        crop = _crop(frame.image, track)
        if crop is None:
            return False
        if crop.shape[0] * self._params.min_face_fraction < 1.0:
            return False

        template = self._recognizer.template_for(crop)
        if template is None:
            return False

        self._attempts += 1
        state.attempts += 1
        match = self._gallery.match(template)
        state.votes[match.name] = state.votes.get(match.name, 0) + 1

        if state.resolved:
            self._reverify(track, state, match)
            return True

        leader, votes = state.leader()
        if votes >= self._params.min_votes:
            state.resolved = True
            state.name = leader
            state.similarity = match.similarity if match.name == leader else state.similarity
            state.resolved_at_step = state.steps
            self._commit(track, state, match)
        elif state.attempts >= self._params.max_attempts:
            # Out of attempts without agreement. Unknown is the honest outcome:
            # the evidence never converged.
            state.resolved = True
            state.name = UNKNOWN
            state.resolved_at_step = state.steps
            self._record(
                AuditAction.REJECTED, UNKNOWN, track, match.similarity, "attempts exhausted"
            )
        return True

    def _commit(self, track: Track, state: _TrackState, match) -> None:
        if state.name == UNKNOWN:
            self._record(
                AuditAction.REJECTED, UNKNOWN, track, match.similarity, "no template matched"
            )
            return
        self._identified += 1
        log.info(
            "identity resolved",
            extra={
                "vantage_fields": {
                    "entity": track.entity_id,
                    "name": state.name,
                    "similarity": round(state.similarity, 3),
                    "votes": state.votes.get(state.name, 0),
                }
            },
        )
        self._record(AuditAction.IDENTIFIED, state.name, track, state.similarity, "committed")

    def _reverify(self, track: Track, state: _TrackState, match) -> None:
        """Check a resolved track still looks like who it was named after."""
        if match.name == state.name:
            state.similarity = match.similarity
            return
        log.warning(
            "identity no longer matches; returning to unresolved",
            extra={
                "vantage_fields": {
                    "entity": track.entity_id,
                    "was": state.name,
                    "now": match.name,
                    "similarity": round(match.similarity, 3),
                    "likely_cause": "a track swap, or the first identification was wrong",
                }
            },
        )
        self._record(
            AuditAction.REJECTED, state.name, track, match.similarity, "re-check disagreed"
        )
        state.resolved = False
        state.name = UNKNOWN
        state.similarity = 0.0
        state.votes.clear()
        state.attempts = 0

    def _record(
        self, action: AuditAction, name: str, track: Track, similarity: float, detail: str
    ) -> None:
        if self._store is None:
            return
        self._store.audit(
            AuditRecord(
                action=action,
                name=name,
                timestamp=time.time(),
                detail=detail,
                entity_id=track.entity_id,
                similarity=similarity,
            )
        )

    def reset(self) -> None:
        self._states.clear()


def _crop(image: np.ndarray, track: Track) -> np.ndarray | None:
    """The upper portion of a person box, where a face is if there is one.

    The whole box would give the detector a body to search, which is slower and
    invites it to find a face on a T-shirt. The top 45% is generous enough to
    hold the head of a standing, sitting or leaning person.
    """
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1, x2, y2 = track.box.clipped(width, height).to_int()
    head_height = max(1, int((y2 - y1) * 0.45))
    y2 = min(y2, y1 + head_height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return image[y1:y2, x1:x2]
