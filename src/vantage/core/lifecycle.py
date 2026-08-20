"""Graceful shutdown coordination.

A video pipeline holds real operating-system resources - camera handles,
decoder threads, GUI windows - that Windows does not reliably reclaim if the
process is killed mid-frame. Ctrl+C must therefore mean "finish the current
frame, release everything, exit 0", not "raise KeyboardInterrupt from whatever
line happened to be executing".

:class:`ShutdownController` converts signals into a thread-safe
:class:`threading.Event` that every loop in the platform polls. A second signal
escalates to the default handler so an operator is never trapped in a hung
process.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable
from types import FrameType

from vantage.core.logging import get_logger

log = get_logger(__name__)

_SIGNALS = ("SIGINT", "SIGTERM", "SIGBREAK")  # SIGBREAK is Windows-only


class ShutdownController:
    """Turns termination signals into a cooperative stop flag.

    Usable as a context manager; handlers are restored on exit so that
    embedding the platform in a larger application does not hijack its signals.
    """

    __slots__ = ("_event", "_installed", "_on_shutdown", "_previous")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._previous: dict[int, object] = {}
        self._installed = False
        self._on_shutdown: list[Callable[[], None]] = []

    @property
    def event(self) -> threading.Event:
        """The flag itself, for passing to worker loops."""
        return self._event

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def request(self, reason: str = "requested") -> None:
        """Signal shutdown from application code (e.g. the viewer's quit key)."""
        if not self._event.is_set():
            log.info("shutdown requested", extra={"vantage_fields": {"reason": reason}})
            self._event.set()
            for callback in self._on_shutdown:
                callback()

    def on_shutdown(self, callback: Callable[[], None]) -> None:
        """Register a callback fired once when shutdown is first requested."""
        self._on_shutdown.append(callback)

    def install(self) -> ShutdownController:
        """Install signal handlers. Only valid on the main thread."""
        if self._installed:
            return self
        if threading.current_thread() is not threading.main_thread():
            # Not an error: a library embedding us may run off-main-thread and
            # simply drive `request()` itself.
            log.debug("signal handlers not installed (not on main thread)")
            return self
        for name in _SIGNALS:
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
            except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
                log.debug("could not install handler for %s: %s", name, exc)
        self._installed = True
        return self

    def restore(self) -> None:
        """Restore the handlers that were in place before :meth:`install`."""
        for signum, handler in self._previous.items():
            try:  # noqa: SIM105 - the comment below is the point
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass
        self._previous.clear()
        self._installed = False

    def _handle(self, signum: int, _frame: FrameType | None) -> None:
        if self._event.is_set():
            # Second interrupt: the operator is insisting. Hand control back to
            # the default disposition rather than swallowing it.
            log.warning("second signal received, escalating to default handler")
            self.restore()
            signal.raise_signal(signum)
            return
        self.request(reason=signal.Signals(signum).name)

    def __enter__(self) -> ShutdownController:
        return self.install()

    def __exit__(self, *_exc: object) -> None:
        self.restore()
