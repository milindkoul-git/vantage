"""Adaptive load shedding: analyse fewer frames well, rather than fall behind.

The failure this prevents
------------------------
A detector that takes 84 ms cannot run on every frame of a 30 fps camera. Left
alone, the pipeline does not slow down - it *drops* frames under backpressure,
so the analysis silently sees an arbitrary subset of reality while the frame
rate looks fine. Every later stage inherits that: a tracker fed irregular gaps
loses identity, and a dwell timer measured in footage time measures the wrong
thing.

The lever, and why it is this one
---------------------------------
Three were available:

* **Throttle capture** (``RatePacer.set_target``). Rejected: it discards frames
  before anything sees them, so the display stutters and nothing is gained that
  raising the interval does not give more cheaply.
* **Shrink the model.** Rejected here: swapping weights mid-run changes what the
  system can detect at all, which is a decision for an operator rather than a
  control loop.
* **Analyse every Nth frame.** Chosen. The display stays at full rate, the
  tracker's variable timestep already absorbs irregular gaps by design, and the
  thing given up - temporal resolution of the analysis - is the thing that
  degrades most gracefully.

Computed, not hunted for
------------------------
The required interval is arithmetic rather than a search. If analysis costs
``C`` milliseconds and frames arrive every ``B`` milliseconds, then analysing
one frame in ``N`` costs ``C/N`` per delivered frame, so the smallest workable
``N`` is ``ceil(C / (B * headroom))``. A controller that nudged the interval up
and down looking for equilibrium would take seconds to converge and oscillate
around it; this lands on the answer in one step and then only has to be stopped
from twitching.

Which is what the hysteresis is for: the same dead-band lesson as the motion
state machine in Phase 4, applied to a different signal. Raising the interval is
allowed promptly, because being over budget is a live problem. Lowering it
requires the headroom to have persisted, because a single cheap frame is not
evidence that the load has gone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GovernorParams:
    """Thresholds for the load governor."""

    headroom: float = 0.7
    """Fraction of the frame budget analysis may occupy.

    Not 1.0. Analysis is not the only work in a frame - decoding, colour
    conversion, overlay drawing and display all take time - and a target that
    consumed the entire budget would leave the pipeline permanently on the edge
    of dropping. Thirty percent kept back is the difference between "keeps up"
    and "keeps up unless something twitches".
    """

    max_interval: int = 8
    """Ceiling on how far the interval may be raised.

    Past this the analysis is so sparse that the tracker's association gaps grow
    beyond what its motion model can bridge, and the honest outcome is a slow
    system rather than a fast one producing nonsense. Reaching the ceiling is
    reported rather than absorbed.
    """

    raise_after_s: float = 1.0
    """How long over budget before the interval is raised."""

    lower_after_s: float = 6.0
    """How long under budget before it is lowered again.

    Deliberately much longer than :attr:`raise_after_s`. The costs are
    asymmetric: raising too eagerly loses a little temporal resolution, while
    lowering too eagerly puts the pipeline straight back into the overload it
    just escaped, and the two together produce a system that oscillates instead
    of settling.
    """

    def __post_init__(self) -> None:
        if not 0.0 < self.headroom <= 1.0:
            raise ConfigError(f"app.adaptive.headroom must be in (0, 1], got {self.headroom}")
        if self.max_interval < 1:
            raise ConfigError("app.adaptive.max_interval must be >= 1")
        for name in ("raise_after_s", "lower_after_s"):
            if getattr(self, name) < 0:
                raise ConfigError(f"app.adaptive.{name} must be >= 0")


@dataclass(slots=True)
class GovernorStats:
    """What the governor did over a run."""

    interval: int = 1
    base_interval: int = 1
    raises: int = 0
    lowers: int = 0
    at_ceiling_s: float = 0.0
    degraded_s: float = 0.0
    """Footage time spent analysing less often than configured."""

    peak_interval: int = 1
    last_cost_ms: float = 0.0
    last_budget_ms: float = 0.0

    @property
    def degraded(self) -> bool:
        return self.interval > self.base_interval

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "interval": self.interval,
            "base_interval": self.base_interval,
            "peak_interval": self.peak_interval,
            "raises": self.raises,
            "lowers": self.lowers,
            "degraded_s": round(self.degraded_s, 2),
            "at_ceiling_s": round(self.at_ceiling_s, 2),
            "last_cost_ms": round(self.last_cost_ms, 2),
            "last_budget_ms": round(self.last_budget_ms, 2),
        }

    def describe(self) -> str:
        if self.peak_interval == self.base_interval:
            return f"interval {self.interval} (never degraded)"
        return (
            f"interval {self.interval}, peaked at {self.peak_interval}, "
            f"{self.degraded_s:.0f}s degraded ({self.raises} raises, {self.lowers} lowers)"
        )


class LoadGovernor:
    """Chooses how often to analyse, from measured cost against the frame budget."""

    __slots__ = ("_over_s", "_params", "_stats", "_under_s")

    def __init__(self, base_interval: int = 1, params: GovernorParams | None = None) -> None:
        if base_interval < 1:
            raise ConfigError("detection.interval must be >= 1")
        self._params = params or GovernorParams()
        if self._params.max_interval < base_interval:
            raise ConfigError(
                f"app.adaptive.max_interval ({self._params.max_interval}) must be at "
                f"least detection.interval ({base_interval}), or the governor would "
                "be asked to shed load below the floor it was told to start at"
            )
        # peak_interval seeded to the base, not left at its default of 1. A
        # governor configured to analyse every 4th frame that reported a peak of
        # 1 would be claiming a peak below its own floor, and the run summary
        # decides whether to mention degradation by comparing the two.
        self._stats = GovernorStats(
            interval=base_interval, base_interval=base_interval, peak_interval=base_interval
        )
        self._over_s = 0.0
        self._under_s = 0.0

    @property
    def interval(self) -> int:
        return self._stats.interval

    @property
    def stats(self) -> GovernorStats:
        return self._stats

    @property
    def params(self) -> GovernorParams:
        return self._params

    def observe(self, cost_ms: float, budget_ms: float, elapsed_s: float) -> int:
        """Record one frame's analysis cost and return the interval to use.

        Args:
            cost_ms: What analysis actually took on the frames it ran on.
            budget_ms: Milliseconds between delivered frames.
            elapsed_s: Time since the previous observation.
        """
        stats = self._stats
        stats.last_cost_ms = cost_ms
        stats.last_budget_ms = budget_ms

        if stats.degraded:
            stats.degraded_s += elapsed_s
        if stats.interval >= self._params.max_interval:
            stats.at_ceiling_s += elapsed_s

        if budget_ms <= 0.0 or cost_ms <= 0.0:
            # No budget to measure against, or nothing measured yet.
            return stats.interval

        target = budget_ms * self._params.headroom
        # Cost is per analysed frame, so at interval N the per-delivered-frame
        # cost is cost/N. Solve for the smallest N that fits.
        required = max(1, math.ceil(cost_ms / target))

        if required > stats.interval:
            self._over_s += elapsed_s
            self._under_s = 0.0
            if self._over_s >= self._params.raise_after_s:
                self._set(min(required, self._params.max_interval), cost_ms, budget_ms)
                self._over_s = 0.0
        elif required < stats.interval and stats.interval > stats.base_interval:
            self._under_s += elapsed_s
            self._over_s = 0.0
            if self._under_s >= self._params.lower_after_s:
                # One step at a time on the way down. Dropping straight to the
                # required interval would hand the pipeline the full load again
                # in a single frame, which is how a controller ends up
                # oscillating between two states forever.
                self._set(
                    max(required, stats.base_interval, stats.interval - 1),
                    cost_ms,
                    budget_ms,
                )
                self._under_s = 0.0
        else:
            self._over_s = 0.0
            self._under_s = 0.0
        return stats.interval

    def _set(self, interval: int, cost_ms: float, budget_ms: float) -> None:
        stats = self._stats
        interval = max(stats.base_interval, min(interval, self._params.max_interval))
        if interval == stats.interval:
            return

        raising = interval > stats.interval
        stats.raises += raising
        stats.lowers += not raising
        previous, stats.interval = stats.interval, interval
        stats.peak_interval = max(stats.peak_interval, interval)

        # Always logged. A system that silently changes how much of reality it
        # looks at is exactly the kind of thing that makes later results
        # inexplicable.
        log.info(
            "analysis interval changed",
            extra={
                "vantage_fields": {
                    "from": previous,
                    "to": interval,
                    "reason": "over budget" if raising else "headroom recovered",
                    "cost_ms": round(cost_ms, 1),
                    "budget_ms": round(budget_ms, 1),
                    "headroom": self._params.headroom,
                }
            },
        )
        if raising and interval >= self._params.max_interval:
            log.warning(
                "analysis interval at its ceiling",
                extra={
                    "vantage_fields": {
                        "interval": interval,
                        "cost_ms": round(cost_ms, 1),
                        "budget_ms": round(budget_ms, 1),
                        "advice": (
                            "analysis cannot keep up even at the maximum interval; "
                            "a smaller model or fewer stages is the real fix"
                        ),
                    }
                },
            )

    def reset(self) -> None:
        base = self._stats.base_interval
        self._stats = GovernorStats(interval=base, base_interval=base, peak_interval=base)
        self._over_s = self._under_s = 0.0
