"""Rate control: how many frames to take, and how fast.

Two independent mechanisms, because they answer different questions.

:class:`StrideFilter` - *which* frames
    Keep every Nth frame. Deterministic and resolution-independent: the same
    input file yields the same frames on any machine, which is what offline
    evaluation requires.

:class:`RatePacer` - *when* frames
    Hold acquisition to a wall-clock rate. Time-based rather than count-based,
    so it adapts to whatever the source actually delivers.

Both take a :class:`~vantage.core.clock.Clock`, so their behaviour is verified
in tests without any real waiting.

Why this matters beyond Phase 1: on CPU-only hardware, the cheapest way to fit a
detector is to run it on fewer frames rather than to run a worse detector.
Processing 10 frames per second well beats processing 30 badly and falling
progressively further behind. Phase 12's adaptive sampling will drive
:meth:`RatePacer.set_target` from measured inference latency; the control point
already exists.
"""

from __future__ import annotations

from vantage.core.clock import SYSTEM_CLOCK, Clock


class StrideFilter:
    """Keeps one frame in every ``stride``."""

    __slots__ = ("_skipped", "_stride")

    def __init__(self, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self._stride = stride
        self._skipped = 0

    @property
    def stride(self) -> int:
        return self._stride

    @property
    def skipped(self) -> int:
        """Frames rejected so far - reported separately from queue drops, since
        these are an intentional sampling decision rather than lost work."""
        return self._skipped

    def keep(self, index: int) -> bool:
        """Whether the frame with source index ``index`` should be delivered.

        Keyed on the source index rather than an internal counter, so the
        selected set is identical regardless of where a run started or whether
        frames were dropped elsewhere.
        """
        if self._stride == 1:
            return True
        if index % self._stride == 0:
            return True
        self._skipped += 1
        return False


class RatePacer:
    """Limits throughput to at most ``target_fps`` frames per second.

    Deadlines advance on a fixed grid rather than "now + interval", so transient
    jitter does not accumulate into permanent drift. A pacer that falls far
    behind (a long stall) resynchronises to the present instead of sprinting to
    catch up, which would defeat the point of rate limiting.
    """

    __slots__ = ("_clock", "_interval", "_next_deadline", "_waited_s")

    def __init__(self, target_fps: float | None, clock: Clock = SYSTEM_CLOCK) -> None:
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive or None")
        self._clock = clock
        self._interval = 1.0 / target_fps if target_fps else 0.0
        self._next_deadline: float | None = None
        self._waited_s = 0.0

    @property
    def target_fps(self) -> float | None:
        return 1.0 / self._interval if self._interval else None

    @property
    def waited_s(self) -> float:
        """Total time spent sleeping - how much headroom the pipeline has."""
        return self._waited_s

    def set_target(self, target_fps: float | None) -> None:
        """Change the rate at runtime. The control point for adaptive sampling."""
        if target_fps is not None and target_fps <= 0:
            raise ValueError("target_fps must be positive or None")
        self._interval = 1.0 / target_fps if target_fps else 0.0
        self._next_deadline = None

    def wait(self) -> float:
        """Block until the next frame is due. Returns the seconds waited."""
        if not self._interval:
            return 0.0

        now = self._clock.monotonic()
        if self._next_deadline is None:
            self._next_deadline = now + self._interval
            return 0.0

        delay = self._next_deadline - now
        if delay > 0:
            self._clock.sleep(delay)
            self._waited_s += delay
            self._next_deadline += self._interval
            return delay

        # Behind schedule. Re-anchor the grid on the present rather than
        # emitting a burst to "catch up", which would defeat rate limiting.
        # Re-anchoring (rather than adding N missed intervals) also avoids the
        # floating-point case where the recomputed deadline lands exactly on
        # `now` and the next call reports another miss.
        self._next_deadline = now + self._interval
        return 0.0


class MediaClockPacer:
    """Paces a recorded source to its own timeline, like a live camera.

    Anchors on the first frame, then sleeps until each frame's presentation
    timestamp is due. Used for previewing files at natural speed; batch
    processing leaves it off and runs as fast as the machine allows.
    """

    __slots__ = ("_anchor_mono", "_anchor_pts", "_clock", "_fallback_interval", "_waited_s")

    def __init__(self, fallback_fps: float | None = None, clock: Clock = SYSTEM_CLOCK) -> None:
        self._clock = clock
        self._anchor_mono: float | None = None
        self._anchor_pts: float | None = None
        self._fallback_interval = 1.0 / fallback_fps if fallback_fps else 0.0
        self._waited_s = 0.0

    @property
    def waited_s(self) -> float:
        return self._waited_s

    def wait_for(self, media_pts: float | None) -> float:
        """Block until ``media_pts`` is due relative to the first frame seen."""
        now = self._clock.monotonic()

        if media_pts is None:
            # Container reported no timestamp; fall back to a fixed interval.
            if not self._fallback_interval:
                return 0.0
            self._clock.sleep(self._fallback_interval)
            self._waited_s += self._fallback_interval
            return self._fallback_interval

        if self._anchor_mono is None or self._anchor_pts is None:
            self._anchor_mono, self._anchor_pts = now, media_pts
            return 0.0

        target = self._anchor_mono + (media_pts - self._anchor_pts)
        delay = target - now
        if delay <= 0:
            return 0.0
        self._clock.sleep(delay)
        self._waited_s += delay
        return delay
