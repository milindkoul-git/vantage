"""Stage isolation: one failing stage must not take the process with it.

The problem this exists for
---------------------------
Every analysis stage - detection, tracking, pose, state, activity, spatial - was
called directly inside the run loop, with a single ``try`` around the whole
thing. That is correct for a benchmark and wrong for a deployment: one malformed
frame, one transient driver fault, and a camera that was supposed to run for
weeks stops. This platform has already met exactly such a fault once, when the
iGPU returned ``CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST`` under a
too-large token budget.

So each stage runs behind a guard that converts a failure into a *skipped stage
for that frame* rather than a stopped process.

What this is not
----------------
It is **not** silent exception handling, which this project forbids for good
reason. Three things make the difference:

1. **Every failure is logged**, the first with a full traceback.
2. **Every failure is counted**, and the counts appear in the run summary and
   the HUD, so a stage quietly failing on half the frames is visible rather
   than merely invisible-and-slow.
3. **Persistent failure stops the stage loudly.** A stage that throws on
   :attr:`StageGuard.max_consecutive` frames in a row is not having a bad
   moment, it is broken. Continuing to call it burns latency on every frame and
   floods the log, so it is disabled for the rest of the run with an ERROR that
   names it.

The alternative - retrying forever - produces the worst outcome available: a
system that is neither working nor obviously broken.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

from vantage.core.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

_LOG_EVERY = 50
"""After the first, log one failure in this many.

A stage failing on every frame at 30 fps writes 1800 lines a minute, which
buries whatever else the log had to say. The count is always exact; only the
log lines are thinned.
"""


@dataclass(slots=True)
class StageStats:
    """What happened to one stage over a run."""

    name: str
    calls: int = 0
    failures: int = 0
    consecutive: int = 0
    worst_streak: int = 0
    disabled: bool = False
    last_error: str = ""
    error_types: dict[str, int] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    @property
    def healthy(self) -> bool:
        return not self.disabled and self.failures == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 4),
            "worst_streak": self.worst_streak,
            "disabled": self.disabled,
            "last_error": self.last_error,
            "error_types": dict(self.error_types),
        }

    def describe(self) -> str:
        if self.disabled:
            return f"{self.name}: DISABLED after {self.worst_streak} consecutive failures"
        if not self.failures:
            return f"{self.name}: ok"
        return (
            f"{self.name}: {self.failures}/{self.calls} failed "
            f"({self.failure_rate:.1%}), last: {self.last_error}"
        )


class StageGuard:
    """Runs one pipeline stage, surviving its failures up to a budget."""

    __slots__ = ("_max_consecutive", "_stats")

    def __init__(self, name: str, max_consecutive: int = 5) -> None:
        if max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")
        self._stats = StageStats(name=name)
        self._max_consecutive = max_consecutive

    @property
    def name(self) -> str:
        return self._stats.name

    @property
    def stats(self) -> StageStats:
        return self._stats

    @property
    def enabled(self) -> bool:
        return not self._stats.disabled

    @property
    def max_consecutive(self) -> int:
        return self._max_consecutive

    def run(
        self,
        work: Callable[..., T],
        /,
        *args: Any,
        default: T | None = None,
        **kwargs: Any,
    ) -> T | None:
        """Call ``work(*args, **kwargs)``, returning ``default`` if it fails.

        Arguments are passed through rather than captured in a closure at the
        call site. That is not style: a run loop is full of variables that
        change every frame, and ``lambda: detect(frame)`` is the exact shape of
        a late-binding bug. It happens to be safe here because this method calls
        ``work`` immediately, but nothing in the signature said so, and a future
        change that deferred or retried the call would turn eleven safe closures
        into eleven wrong ones at once. Passing the arguments makes the binding
        explicit and the whole class of mistake unavailable.

        ``work`` is positional-only and ``default`` keyword-only, so neither can
        collide with a keyword the wrapped callable wants.

        ``MemoryError`` is deliberately not caught. Skipping a frame does not
        give memory back, and a process that keeps running while out of memory
        produces corrupt results instead of a clear failure. Anything derived
        from ``BaseException`` - ``KeyboardInterrupt``, ``SystemExit`` - passes
        through untouched for the same reason: those are the operator asking to
        stop, not a stage misbehaving.
        """
        stats = self._stats
        if stats.disabled:
            return default

        stats.calls += 1
        try:
            result = work(*args, **kwargs)
        except MemoryError:
            raise
        except Exception as exc:
            self._record(exc)
            return default

        stats.consecutive = 0
        return result

    def _record(self, exc: Exception) -> None:
        stats = self._stats
        stats.failures += 1
        stats.consecutive += 1
        stats.worst_streak = max(stats.worst_streak, stats.consecutive)
        kind = type(exc).__name__
        stats.error_types[kind] = stats.error_types.get(kind, 0) + 1
        stats.last_error = f"{kind}: {exc}"

        fields = {
            "stage": stats.name,
            "error": stats.last_error,
            "consecutive": stats.consecutive,
            "failures": stats.failures,
            "calls": stats.calls,
        }
        if stats.failures == 1:
            # The first one carries the traceback. Later ones would repeat it.
            log.warning("stage failed", exc_info=True, extra={"vantage_fields": fields})
        elif stats.failures % _LOG_EVERY == 0:
            log.warning("stage still failing", extra={"vantage_fields": fields})

        if stats.consecutive >= self._max_consecutive:
            stats.disabled = True
            log.error(
                "stage disabled after repeated failures",
                extra={
                    "vantage_fields": {
                        **fields,
                        "reason": (
                            f"{stats.consecutive} consecutive failures reached the "
                            "budget; a stage failing every frame is broken rather "
                            "than unlucky, and continuing would cost latency on "
                            "every frame and flood the log"
                        ),
                    }
                },
            )

    def reset(self) -> None:
        """Clear counters and re-enable. For tests and for a supervised restart."""
        name = self._stats.name
        self._stats = StageStats(name=name)


class StageRegistry:
    """The guards for one run, so the summary can report all of them at once."""

    def __init__(self, max_consecutive: int = 5) -> None:
        self._guards: dict[str, StageGuard] = {}
        self._max_consecutive = max_consecutive

    def guard(self, name: str) -> StageGuard:
        guard = self._guards.get(name)
        if guard is None:
            guard = StageGuard(name, self._max_consecutive)
            self._guards[name] = guard
        return guard

    def __iter__(self):
        return iter(self._guards.values())

    def __len__(self) -> int:
        return len(self._guards)

    @property
    def healthy(self) -> bool:
        return all(guard.stats.healthy for guard in self._guards.values())

    @property
    def degraded(self) -> tuple[StageStats, ...]:
        """Stages that failed at least once, worst first."""
        return tuple(
            sorted(
                (g.stats for g in self._guards.values() if not g.stats.healthy),
                key=lambda s: (not s.disabled, -s.failures),
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {name: guard.stats.to_dict() for name, guard in self._guards.items()}

    def summary(self) -> str:
        degraded = self.degraded
        if not degraded:
            return ""
        return "; ".join(stats.describe() for stats in degraded)
