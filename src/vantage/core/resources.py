"""Process CPU and memory, without a dependency.

The spec asks for CPU and memory utilisation under observability, and for a
process that is meant to run for weeks on a camera feed it is the most important
number on the list: a slow leak is the characteristic failure of long-running
video pipelines, and it is invisible in a frame rate until the machine starts
swapping.

Why not psutil
--------------
It is the obvious answer and it would work. But it is a dependency added for one
number, and everything needed is already in the standard library or twenty lines
of :mod:`ctypes` - the same trade this project already made against SciPy for
the assignment solver and against a geometry library for point-in-polygon.

* **CPU** is fully portable: :func:`time.process_time` returns CPU seconds
  consumed by this process, so utilisation is its delta over the wall-clock
  delta. A value of 1.0 means one core saturated; on an eight-core machine the
  ceiling is 8.0, which is the honest way to report it rather than a percentage
  that silently means different things on different hardware.
* **Memory** needs one platform call each. Windows has ``GetProcessMemoryInfo``,
  Linux has ``/proc/self/statm``, macOS has ``getrusage``.

Where a platform is not covered, memory is reported as ``None`` - explicitly
unavailable rather than silently zero, which would read as "no memory used" and
make a leak look like perfect health.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One reading of what the process is consuming."""

    cpu_cores: float
    """Cores in use since the previous sample. 1.0 is one core saturated."""

    rss_bytes: int | None
    """Resident set size, or ``None`` where this platform is not covered."""

    peak_rss_bytes: int | None
    growth_bytes: int | None
    """RSS minus the first sample taken. The number that reveals a leak."""

    elapsed_s: float

    @property
    def rss_mb(self) -> float | None:
        return None if self.rss_bytes is None else self.rss_bytes / 1e6

    @property
    def growth_mb(self) -> float | None:
        return None if self.growth_bytes is None else self.growth_bytes / 1e6

    def describe(self) -> str:
        cpu = f"{self.cpu_cores:.2f} cores"
        if self.rss_bytes is None:
            return f"{cpu}, memory unavailable on this platform"
        growth = ""
        if self.growth_bytes is not None and abs(self.growth_bytes) > 1e6:
            growth = f" ({self.growth_mb:+.1f} MB since start)"
        return f"{cpu}, {self.rss_mb:.0f} MB RSS{growth}"

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "cpu_cores": round(self.cpu_cores, 3),
            "rss_bytes": self.rss_bytes,
            "rss_mb": None if self.rss_mb is None else round(self.rss_mb, 1),
            "peak_rss_mb": (
                None if self.peak_rss_bytes is None else round(self.peak_rss_bytes / 1e6, 1)
            ),
            "growth_mb": None if self.growth_mb is None else round(self.growth_mb, 1),
        }


class ResourceSampler:
    """Samples process CPU and memory, cheaply enough to call every second."""

    __slots__ = (
        "_first_rss",
        "_last_cpu",
        "_last_wall",
        "_peak_rss",
        "_start_cpu",
        "_started",
    )

    def __init__(self) -> None:
        self._last_cpu = time.process_time()
        self._start_cpu = self._last_cpu
        self._last_wall = time.monotonic()
        self._started = self._last_wall
        # Baseline taken here rather than on the first sample(), so "growth
        # since start" means since this object existed. Deferring it to the
        # first sample made the baseline whatever memory happened to be in use
        # after the models loaded, and a run that then released them reported a
        # *negative* growth of 130 MB - true of that baseline, and useless.
        self._first_rss: int | None = read_rss()
        self._peak_rss: int | None = self._first_rss

    def sample(self) -> ResourceSample:
        """Read usage since the previous call."""
        now_cpu = time.process_time()
        now_wall = time.monotonic()
        window = now_wall - self._last_wall
        # A zero or negative window means two samples in the same tick; report
        # no load rather than dividing by it.
        cores = (now_cpu - self._last_cpu) / window if window > 1e-6 else 0.0
        self._last_cpu, self._last_wall = now_cpu, now_wall

        rss = read_rss()
        if rss is not None:
            if self._first_rss is None:
                self._first_rss = rss
            self._peak_rss = rss if self._peak_rss is None else max(self._peak_rss, rss)

        return self._build(cores, rss, now_wall - self._started)

    def total(self) -> ResourceSample:
        """Usage over the whole life of the sampler, not since the last sample.

        The right figure for a run summary. Calling :meth:`sample` there instead
        reports CPU over whatever fraction of a second has passed since the last
        periodic reading, which is noise dressed as a measurement.
        """
        now_wall = time.monotonic()
        span = now_wall - self._started
        cores = (time.process_time() - self._start_cpu) / span if span > 1e-6 else 0.0
        rss = read_rss()
        if rss is not None:
            self._peak_rss = rss if self._peak_rss is None else max(self._peak_rss, rss)
        return self._build(cores, rss, span)

    def _build(self, cores: float, rss: int | None, elapsed: float) -> ResourceSample:
        growth = None if rss is None or self._first_rss is None else rss - self._first_rss
        return ResourceSample(
            cpu_cores=max(0.0, cores),
            rss_bytes=rss,
            peak_rss_bytes=self._peak_rss,
            growth_bytes=growth,
            elapsed_s=elapsed,
        )


def read_rss() -> int | None:
    """Resident set size in bytes, or ``None`` if this platform is not covered."""
    if sys.platform == "win32":
        return _rss_windows()
    if sys.platform.startswith("linux"):
        return _rss_linux()
    if sys.platform == "darwin":
        return _rss_macos()
    return None


def _rss_windows() -> int | None:
    """``GetProcessMemoryInfo`` from psapi.

    ``WorkingSetSize`` is the resident set - the pages actually in physical
    memory - which is the figure comparable to RSS elsewhere. ``PagefileUsage``
    (what Task Manager used to label "VM Size") would be a different and
    generally larger number.
    """
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)

        # Declaring these is not decoration. GetCurrentProcess returns a HANDLE,
        # which is 64-bit here, and ctypes defaults an undeclared integer
        # argument to a 32-bit C int - truncating the pseudo-handle
        # (0xFFFFFFFFFFFFFFFF) into something the call rejects. Written without
        # the declarations, this returned failure every time and memory read as
        # "unavailable on this platform" on the platform it was written for.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetCurrentProcess.argtypes = []

        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.restype = wintypes.BOOL
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), wintypes.DWORD]

        if not get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except (ImportError, OSError, AttributeError):
        # A locked-down or unusual Windows build. Unavailable, not zero.
        return None


def _rss_linux() -> int | None:
    """Field two of ``/proc/self/statm`` is resident pages."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            resident_pages = int(handle.read().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return None


def _rss_macos() -> int | None:
    """``ru_maxrss`` is a peak rather than a current value, and is in bytes here.

    Reported anyway, because a peak that keeps climbing is still the signal a
    leak produces, and it is the only figure available without a dependency.
    """
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError):
        return None
