"""Lightweight instrumentation.

Phase 1 already needs FPS and latency to prove the pipeline behaves; the same
primitives serve the Phase 12 observability requirements (inference latency,
dropped frames, model confidence). Everything here is allocation-free on the hot
path and safe to call from the capture thread.

Deliberately not a Prometheus client: the exporter is a Phase 9+ concern, and
:meth:`MetricsRegistry.snapshot` produces a plain dict that any exporter can
consume.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


class Counter:
    """Monotonically increasing integer counter."""

    __slots__ = ("_lock", "_value")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def inc(self, amount: int = 1) -> None:
        with self._lock:
            self._value += amount

    @property
    def value(self) -> int:
        return self._value

    def snapshot(self) -> int:
        return self._value


class RateMeter:
    """Measures events per second.

    Reports two numbers because they answer different questions:

    ``rate``
        Exponentially weighted moving average - what the pipeline is doing
        *now*, responsive enough to show a stall within a second.
    ``mean_rate``
        Total events divided by total elapsed time - what the pipeline did
        *overall*, the honest number for a run summary.

    The smoothing is applied to the *interval* between events and then
    inverted, not to the instantaneous rate. Averaging rates over-weights short
    intervals: bursty USB webcam delivery (a burst of 10 ms gaps, then a 60 ms
    stall) reported ~100 fps against a true 27 fps until this was corrected.
    Averaging intervals gives the arrival rate an observer would actually measure.
    """

    __slots__ = ("_alpha", "_count", "_first", "_interval", "_last", "_lock")

    def __init__(self, alpha: float = 0.15) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = alpha
        self._lock = threading.Lock()
        self._count = 0
        self._interval: float | None = None
        self._first: float | None = None
        self._last: float | None = None

    def tick(self, now: float, amount: int = 1) -> None:
        """Record ``amount`` events observed at monotonic time ``now``."""
        with self._lock:
            self._count += amount
            if self._first is None:
                self._first = now
                self._last = now
                return
            dt = now - self._last  # type: ignore[operator]
            self._last = now
            if dt <= 0:
                return
            per_event = dt / max(1, amount)
            self._interval = (
                per_event
                if self._interval is None
                else (self._alpha * per_event + (1.0 - self._alpha) * self._interval)
            )

    @property
    def count(self) -> int:
        return self._count

    @property
    def rate(self) -> float:
        """Smoothed events per second; ``0.0`` before two events are seen."""
        return 1.0 / self._interval if self._interval else 0.0

    @property
    def elapsed(self) -> float:
        if self._first is None or self._last is None:
            return 0.0
        return self._last - self._first

    @property
    def mean_rate(self) -> float:
        elapsed = self.elapsed
        if elapsed <= 0 or self._count < 2:
            return 0.0
        # The first event starts the clock rather than occupying an interval.
        return (self._count - 1) / elapsed

    def snapshot(self) -> dict[str, float | int]:
        return {
            "count": self._count,
            "rate": round(self.rate, 3),
            "mean_rate": round(self.mean_rate, 3),
            "elapsed_s": round(self.elapsed, 3),
        }


class LatencyTracker:
    """Rolling latency distribution over a bounded window of samples.

    A ring buffer rather than a histogram: at video frame rates the window is
    small, exact percentiles are cheap, and there are no bucket boundaries to
    get wrong.
    """

    __slots__ = ("_lock", "_max_ever", "_samples", "_total", "_total_count")

    def __init__(self, window: int = 240) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._lock = threading.Lock()
        self._samples: deque[float] = deque(maxlen=window)
        self._max_ever = 0.0
        self._total = 0.0
        self._total_count = 0

    def observe(self, value_ms: float) -> None:
        with self._lock:
            self._samples.append(value_ms)
            self._total += value_ms
            self._total_count += 1
            if value_ms > self._max_ever:
                self._max_ever = value_ms

    def percentile(self, p: float) -> float:
        """Nearest-rank percentile over the current window; ``p`` in [0, 100]."""
        if not 0.0 <= p <= 100.0:
            raise ValueError("percentile must be in [0, 100]")
        with self._lock:
            if not self._samples:
                return 0.0
            ordered = sorted(self._samples)
        rank = max(1, int(round(p / 100.0 * len(ordered))))
        return ordered[min(rank, len(ordered)) - 1]

    @property
    def last(self) -> float:
        return self._samples[-1] if self._samples else 0.0

    @property
    def mean(self) -> float:
        return self._total / self._total_count if self._total_count else 0.0

    @property
    def max(self) -> float:
        return self._max_ever

    def snapshot(self) -> dict[str, float | int]:
        return {
            "samples": self._total_count,
            "last_ms": round(self.last, 3),
            "mean_ms": round(self.mean, 3),
            "p50_ms": round(self.percentile(50), 3),
            "p95_ms": round(self.percentile(95), 3),
            "max_ms": round(self.max, 3),
        }


@dataclass(slots=True)
class MetricsRegistry:
    """Named collection of metrics belonging to one subsystem instance.

    Registries nest (a future ``IngestionManager`` holds one per camera), and
    :meth:`snapshot` renders the whole tree as JSON-compatible primitives.
    """

    name: str
    _counters: dict[str, Counter] = field(default_factory=dict, init=False, repr=False)
    _rates: dict[str, RateMeter] = field(default_factory=dict, init=False, repr=False)
    _latencies: dict[str, LatencyTracker] = field(default_factory=dict, init=False, repr=False)
    _children: dict[str, MetricsRegistry] = field(default_factory=dict, init=False, repr=False)

    def counter(self, name: str) -> Counter:
        return self._counters.setdefault(name, Counter())

    def rate(self, name: str, alpha: float = 0.15) -> RateMeter:
        if name not in self._rates:
            self._rates[name] = RateMeter(alpha=alpha)
        return self._rates[name]

    def latency(self, name: str, window: int = 240) -> LatencyTracker:
        if name not in self._latencies:
            self._latencies[name] = LatencyTracker(window=window)
        return self._latencies[name]

    def child(self, name: str) -> MetricsRegistry:
        if name not in self._children:
            self._children[name] = MetricsRegistry(name=name)
        return self._children[name]

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name}
        if self._counters:
            out["counters"] = {k: v.snapshot() for k, v in self._counters.items()}
        if self._rates:
            out["rates"] = {k: v.snapshot() for k, v in self._rates.items()}
        if self._latencies:
            out["latencies"] = {k: v.snapshot() for k, v in self._latencies.items()}
        if self._children:
            out["children"] = {k: v.snapshot() for k, v in self._children.items()}
        return out
