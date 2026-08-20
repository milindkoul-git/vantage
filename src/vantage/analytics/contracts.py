"""Analytics contracts: buckets, series, baselines and the anomalies they imply.

Every phase before this one answered questions about *now*. This one answers
questions about *usually* - which is a different kind of claim and fails in a
different way, so the types here carry the things that make such a claim
checkable rather than just plausible.

Three of those are load bearing.

**Coverage travels with every series.** A week-long query against a database
holding four hours of footage returns a perfectly well-formed answer that means
almost nothing. :attr:`Series.coverage` is the fraction of buckets that had any
data at all, and it is a field rather than a derived nicety because every
consumer needs to decide what to do when it is low.

**Sample counts travel with every baseline slot.** "Tuesday 3am is normally
quiet" is a statement about however many Tuesdays were observed. One Tuesday is
not a baseline, and a type that cannot express "I have seen this slot twice"
invites the caller to treat two samples and two hundred identically.

**Spread is reported alongside the centre.** An expected value with no expected
*range* cannot support the sentence "this is unusual", which is the only
sentence anomaly detection exists to produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

SECONDS_PER_HOUR = 3600.0
HOURS_PER_WEEK = 168


class Metric(str, Enum):
    """What a bucket counts.

    Deliberately a closed set. Each member maps to one aggregate expression over
    an indexed column; an interface that accepted arbitrary expressions could
    not promise that, and analytics that quietly table-scans a ten-million-row
    table is a different feature from analytics that returns.
    """

    ENTITIES = "entities"
    """Distinct entities seen in the bucket. The count a person means by "how
    many people were there" - observations would count the same person once per
    sample and report hundreds."""

    OBSERVATIONS = "observations"
    """Raw observation rows. Useful for judging sampling density, not traffic."""

    EVENTS = "events"
    MEAN_SPEED = "mean_speed"
    MOVING_FRACTION = "moving_fraction"
    """Share of observations in the bucket whose motion state was moving. A
    corridor and a waiting room can produce identical entity counts and are
    told apart by this."""

    @property
    def label(self) -> str:
        return {
            Metric.ENTITIES: "distinct entities",
            Metric.OBSERVATIONS: "observations",
            Metric.EVENTS: "events",
            Metric.MEAN_SPEED: "mean speed (heights/s)",
            Metric.MOVING_FRACTION: "fraction moving",
        }[self]

    @property
    def is_rate(self) -> bool:
        """Whether the value is an average rather than a count.

        Counts scale with bucket width and averages do not, so a caller
        re-bucketing a series must sum the first and cannot sum the second.
        """
        return self in (Metric.MEAN_SPEED, Metric.MOVING_FRACTION)


class Direction(str, Enum):
    """Which way an anomaly went. Both matter, and not equally.

    A camera seeing far more people than usual is a busy day. A camera seeing
    far *fewer* is often a camera that stopped working, which is the more
    urgent of the two and would be invisible to a detector that only looked for
    peaks.
    """

    ABOVE = "above"
    BELOW = "below"


@dataclass(frozen=True, slots=True)
class Bucket:
    """One time bucket and its value."""

    start: float
    """Unix time of the bucket's left edge."""

    width_s: float
    value: float
    samples: int = 0
    """Rows that went into the value. Zero means no rows were stored for this
    interval, which is ambiguous on its own - see :attr:`known_zero`."""

    known_zero: bool = False
    """Whether an absence of rows is known to mean "nobody was there".

    No rows is two different facts wearing one appearance: the scene was empty,
    or the pipeline was not running. They call for opposite responses - the first
    is data worth learning from, the second is a gap that must not be learned as
    normal - and a bucket cannot tell them apart by itself.

    :func:`vantage.analytics.coverage.mark_observed_zeros` sets this by looking
    at the neighbours, which can: an empty hour with recorded activity either
    side of it happened while the system was demonstrably running.

    This matters more than it sounds. Without it, an hour that is *always* empty
    - 3am in an office - never accumulates a single training sample, so its slot
    is never learned, so somebody walking through at 3am can never be flagged.
    The most valuable thing overnight analytics could tell you was structurally
    unreachable.
    """

    @property
    def end(self) -> float:
        return self.start + self.width_s

    @property
    def when(self) -> datetime:
        """The left edge as a local-time datetime.

        Local rather than UTC on purpose: every question this module exists to
        answer - "what happens overnight", "is Tuesday afternoon busy" - is
        asked in the timezone the camera is standing in.
        """
        return datetime.fromtimestamp(self.start, tz=UTC).astimezone()

    @property
    def empty(self) -> bool:
        """No usable reading. A known zero is a reading, so it is not empty."""
        return self.samples == 0 and not self.known_zero

    def describe(self) -> str:
        return f"{self.when.strftime('%Y-%m-%d %H:%M')}  {self.value:8.2f}  ({self.samples})"


@dataclass(frozen=True, slots=True)
class Series:
    """An ordered run of buckets over one metric.

    Always contiguous and always complete: an interval with no rows appears as
    an empty bucket rather than being left out. A series that silently omitted
    quiet hours would make every gap look like the camera was busy right up to
    the moment it wasn't.
    """

    metric: Metric
    buckets: tuple[Bucket, ...]
    interval_s: float
    since: float
    until: float
    camera_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.buckets)

    def __iter__(self):
        return iter(self.buckets)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(b.value for b in self.buckets)

    @property
    def occupied(self) -> tuple[Bucket, ...]:
        return tuple(b for b in self.buckets if not b.empty)

    @property
    def coverage(self) -> float:
        """Fraction of buckets that saw any data at all.

        The single number that says whether the rest of this series deserves to
        be believed. Reported everywhere it is used, and never silently.
        """
        if not self.buckets:
            return 0.0
        return len(self.occupied) / len(self.buckets)

    @property
    def total(self) -> float:
        """Sum of the values. Meaningless for rate metrics, so it refuses."""
        if self.metric.is_rate:
            raise ValueError(
                f"{self.metric.value} is an average, not a count: summing it across "
                "buckets produces a number with no interpretation. Use peak() or "
                "mean() instead."
            )
        return float(sum(self.values))

    def mean(self) -> float:
        """Mean over *occupied* buckets only.

        Empty buckets are excluded because they usually mean the system was not
        running. Including them would drag every average toward zero in
        proportion to how much downtime the database happens to contain, which
        would make the metric a measure of uptime wearing the name of traffic.
        """
        occupied = self.occupied
        if not occupied:
            return 0.0
        return sum(b.value for b in occupied) / len(occupied)

    def peak(self) -> Bucket | None:
        return max(self.buckets, key=lambda b: b.value) if self.buckets else None

    def quietest(self) -> Bucket | None:
        occupied = self.occupied
        return min(occupied, key=lambda b: b.value) if occupied else None

    def describe(self) -> str:
        if not self.buckets:
            return f"{self.metric.label}: no buckets in range"
        span_h = (self.until - self.since) / SECONDS_PER_HOUR
        return (
            f"{self.metric.label}: {len(self.buckets)} buckets over {span_h:.1f}h, "
            f"{self.coverage:.0%} covered"
        )


@dataclass(frozen=True, slots=True)
class Slot:
    """One learned slot of a baseline: a recurring position in the week."""

    index: int
    """Hour of week, 0 = Monday 00:00 local, or hour of day when the baseline
    was built with a daily period."""

    centre: float
    """Median, not mean. See :mod:`vantage.analytics.baseline`."""

    spread: float
    """Median absolute deviation, scaled to be comparable with a standard
    deviation. Zero is a real and common answer - see the note on constant
    slots in the baseline module."""

    samples: int
    """How many observed buckets fell in this slot. The number that decides
    whether the two above mean anything."""

    low: float = 0.0
    high: float = 0.0
    """The band a value must leave to be called anomalous."""

    @property
    def trustworthy(self) -> bool:
        return self.samples >= MIN_SLOT_SAMPLES

    def describe(self) -> str:
        day = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][self.index // 24]
        mark = "" if self.trustworthy else "  (too few samples)"
        return (
            f"{day} {self.index % 24:02d}:00  centre {self.centre:7.2f}  "
            f"band {self.low:.2f}-{self.high:.2f}  n={self.samples}{mark}"
        )


MIN_SLOT_SAMPLES = 3
"""Observed buckets a slot needs before it is allowed to declare anything normal.

Three is not a statistically comfortable number and is not claimed to be. It is
the point below which the answer is obviously meaningless - one sample has no
spread at all, two have a spread that is just the gap between them - and the
sample count is carried on every slot so a caller can demand more.
"""


@dataclass(frozen=True, slots=True)
class Baseline:
    """What normal looks like, per recurring slot."""

    metric: Metric
    slots: dict[int, Slot]
    period_hours: int
    """168 for a weekly baseline, 24 for a daily one."""

    interval_s: float
    """Bucket width the baseline was learned from. Comparing a value bucketed
    differently against it would be comparing an hour's traffic to a day's."""

    trained_from: float = 0.0
    trained_until: float = 0.0
    sensitivity: float = 3.5

    @property
    def trustworthy_slots(self) -> int:
        return sum(1 for slot in self.slots.values() if slot.trustworthy)

    @property
    def coverage(self) -> float:
        """Fraction of the period that has a usable slot."""
        if not self.period_hours:
            return 0.0
        return self.trustworthy_slots / self.period_hours

    def slot_for(self, when: datetime) -> Slot | None:
        return self.slots.get(slot_index(when, self.period_hours))

    def describe(self) -> str:
        return (
            f"{self.metric.label}: {self.trustworthy_slots}/{self.period_hours} slots "
            f"learned ({self.coverage:.0%} of the period)"
        )


@dataclass(frozen=True, slots=True)
class Anomaly:
    """One bucket that did not look like its slot usually does."""

    bucket: Bucket
    metric: Metric
    direction: Direction
    observed: float
    expected: float
    band_low: float
    band_high: float
    score: float
    """Robust z-score: deviations of MAD from the median. Not a probability, and
    deliberately not dressed up as one - the underlying distribution is unknown
    and assuming normality would put a confident percentage on a guess."""

    samples: int
    """Baseline samples behind the slot this was judged against."""

    @property
    def severity(self) -> str:
        if self.score >= 8.0:
            return "high"
        return "medium" if self.score >= 5.0 else "low"

    def describe(self) -> str:
        # ASCII rather than arrows. The Windows console encodes cp1252 by
        # default and raises UnicodeEncodeError on U+2191, which crashed the
        # summary command on the platform this project is developed on.
        arrow = "above" if self.direction is Direction.ABOVE else "below"
        return (
            f"{self.bucket.when.strftime('%a %d %b %H:%M')}  {arrow} "
            f"{self.observed:.2f} vs {self.expected:.2f} expected "
            f"({self.band_low:.2f}-{self.band_high:.2f}), "
            f"score {self.score:.1f}, n={self.samples}"
        )


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Everything one analytics run produced, including what it refused to say."""

    series: Series
    anomalies: tuple[Anomaly, ...] = ()
    baseline: Baseline | None = None
    skipped_untrained: int = 0
    """Buckets that could not be judged because their slot had too few samples.

    Reported rather than dropped. A run that examined 168 buckets, judged 12 and
    found no anomalies is not the same as a clean week, and the difference is
    invisible without this number.
    """

    def __len__(self) -> int:
        return len(self.anomalies)

    @property
    def judged(self) -> int:
        return len(self.series) - self.skipped_untrained

    def describe(self) -> str:
        lines = [self.series.describe()]
        if self.baseline is not None:
            lines.append(self.baseline.describe())
        lines.append(
            f"{len(self.anomalies)} anomalies from {self.judged} judged buckets"
            + (f", {self.skipped_untrained} unjudged" if self.skipped_untrained else "")
        )
        return "\n".join(lines)


def slot_index(when: datetime, period_hours: int) -> int:
    """Position of a moment within the recurring period, in local time.

    Local, and computed with a real timezone-aware datetime rather than by
    arithmetic on the epoch, because the two disagree twice a year. An hour-of-
    week derived by dividing a Unix timestamp is off by one hour for half the
    year in any zone that observes daylight saving, which would smear every
    learned slot into its neighbour for six months and then smear it back.
    """
    local = when.astimezone()
    if period_hours == HOURS_PER_WEEK:
        return local.weekday() * 24 + local.hour
    return local.hour
