"""Background writer: storage that cannot stall the frame loop.

Writing to disk on the analysis thread would put an unpredictable, occasionally
tens-of-milliseconds pause in the middle of a frame budget measured in tens of
milliseconds. So the run loop only ever *enqueues*, and a background thread
batches and commits.

Two queues, two policies, and the difference is the point
---------------------------------------------------------
**Observations** are continuous - roughly one row per entity per analysed frame.
Losing one costs almost nothing, because the next frame says nearly the same
thing. Their queue is bounded and drops on overflow.

**Events** are discrete and rare. Each is the output of a rule that already
decided it was worth someone's attention. Their queue is separate, so a flood of
observations can never crowd one out, and a dropped event is logged as an ERROR
rather than counted quietly.

A single shared queue was the first design and is wrong for exactly that reason:
under load the thing you lose is whichever arrived when it was full, and
observations arrive a hundred times more often than events.

On dropping rather than blocking
--------------------------------
Blocking would apply backpressure to the analysis thread, which is the one thing
this module exists to avoid: a slow disk would become dropped *frames*, and a
frame lost at the camera is lost to every stage, not just to storage. Dropping
here costs one row. The counts are reported, and
``storage.observation_interval`` is the lever for controlling volume
deliberately rather than by overflow.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from typing import Any

from vantage.core.logging import get_logger
from vantage.storage.contracts import Store, WriteStats

log = get_logger(__name__)

_SENTINEL = object()
"""Pushed on close to wake the thread immediately rather than at the next poll."""


class StoreWriter:
    """Batches records onto a background thread and commits them."""

    def __init__(
        self,
        store: Store,
        *,
        batch_size: int = 200,
        flush_interval_s: float = 2.0,
        observation_queue: int = 5000,
        event_queue: int = 1000,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if flush_interval_s <= 0:
            raise ValueError("flush_interval_s must be positive")

        self._store = store
        self._batch_size = batch_size
        self._flush_interval_s = flush_interval_s
        self._observations: queue.Queue[Any] = queue.Queue(maxsize=observation_queue)
        self._events: queue.Queue[Any] = queue.Queue(maxsize=event_queue)
        self._stats = WriteStats()
        self._lock = threading.Lock()
        self._closing = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="vantage-store-writer", daemon=True
        )
        self._thread.start()

    @property
    def stats(self) -> WriteStats:
        return self._stats

    @property
    def pending(self) -> int:
        return self._observations.qsize() + self._events.qsize()

    # -- enqueue (called from the analysis thread) -----------------------

    def add_event(self, record: dict[str, Any]) -> bool:
        """Queue one event. Returns whether it was accepted."""
        try:
            self._events.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._stats.events_dropped += 1
                dropped = self._stats.events_dropped
            # ERROR, not a counter increment. An event is the output of a rule
            # that already decided it mattered; losing one is the worst thing
            # this subsystem can do, and it should look like it.
            log.error(
                "event dropped: store queue full",
                extra={
                    "vantage_fields": {
                        "rule": record.get("rule"),
                        "summary": record.get("summary"),
                        "dropped_total": dropped,
                        "advice": (
                            "the writer is not keeping up; check disk latency or "
                            "raise storage.event_queue"
                        ),
                    }
                },
            )
            return False
        with self._lock:
            self._stats.events_queued += 1
        return True

    def add_observation(self, record: dict[str, Any]) -> bool:
        """Queue one observation. Returns whether it was accepted."""
        try:
            self._observations.put_nowait(record)
        except queue.Full:
            with self._lock:
                self._stats.observations_dropped += 1
                total = self._stats.observations_dropped
            # Logged at intervals rather than per row: under sustained overload
            # this fires hundreds of times a second, and the count is exact
            # regardless of how often it is mentioned.
            if total == 1 or total % 1000 == 0:
                log.warning(
                    "observations dropped: store queue full",
                    extra={
                        "vantage_fields": {
                            "dropped_total": total,
                            "advice": (
                                "raise storage.observation_interval to sample "
                                "deliberately instead of losing rows to overflow"
                            ),
                        }
                    },
                )
            return False
        with self._lock:
            self._stats.observations_queued += 1
        return True

    # -- background thread ------------------------------------------------

    def _run(self) -> None:
        last_flush = time.monotonic()
        events: list[dict[str, Any]] = []
        observations: list[dict[str, Any]] = []

        while True:
            stop = self._drain(events, observations)
            now = time.monotonic()
            full = len(events) + len(observations) >= self._batch_size
            due = (now - last_flush) >= self._flush_interval_s

            if (events or observations) and (full or due or stop):
                self._flush(events, observations)
                last_flush = now
            if stop:
                # One more pass: records may have arrived between the sentinel
                # being queued and the queues being drained.
                self._drain(events, observations)
                self._flush(events, observations)
                return
            if not full:
                time.sleep(0.01)

    def _drain(self, events: list, observations: list) -> bool:
        """Move everything currently queued into the batches. True to stop."""
        stop = False
        while True:
            try:
                item = self._events.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                stop = True
                continue
            events.append(item)
        while True:
            try:
                item = self._observations.get_nowait()
            except queue.Empty:
                break
            if item is _SENTINEL:
                stop = True
                continue
            observations.append(item)
        return stop or self._closing.is_set()

    def _flush(self, events: list, observations: list) -> None:
        if not events and not observations:
            return
        try:
            written_events = self._store.write_events(events) if events else 0
            written_observations = (
                self._store.write_observations(observations) if observations else 0
            )
        except Exception as exc:
            with self._lock:
                self._stats.write_errors += 1
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
                errors = self._stats.write_errors
            # The batch is discarded rather than retried forever. Retrying a
            # batch that fails deterministically - a full disk, a corrupt file -
            # would stall the writer permanently and grow the queues until they
            # dropped everything anyway, which converts one loud failure into a
            # slow silent one.
            log.error(
                "store write failed; batch discarded",
                exc_info=errors == 1,
                extra={
                    "vantage_fields": {
                        "events": len(events),
                        "observations": len(observations),
                        "errors_total": errors,
                    }
                },
            )
        else:
            with self._lock:
                self._stats.events_written += written_events
                self._stats.observations_written += written_observations
                self._stats.batches += 1
        finally:
            events.clear()
            observations.clear()

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Block until the queues are empty. For tests and for shutdown."""
        deadline = time.monotonic() + timeout_s
        while self.pending and time.monotonic() < deadline:
            time.sleep(0.01)
        return self.pending == 0

    def close(self, timeout_s: float = 5.0) -> WriteStats:
        """Stop the thread after writing what is queued."""
        if self._closing.is_set():
            return self._stats
        self._closing.set()
        # Wake it now rather than at the next poll.
        for target in (self._events, self._observations):
            with contextlib.suppress(queue.Full):
                # A full queue is already about to be drained, so the thread
                # will see _closing on its next pass regardless.
                target.put_nowait(_SENTINEL)
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            log.warning(
                "store writer did not stop in time; queued records may be lost",
                extra={"vantage_fields": {"pending": self.pending}},
            )
        return self._stats
