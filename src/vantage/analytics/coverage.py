"""Telling "nobody was there" apart from "nothing was recorded".

An interval with no stored rows is ambiguous, and the two readings pull in
opposite directions. If the scene was empty, that is a measurement, and a
baseline that never learns it can never notice when the scene stops being empty.
If the pipeline was down, it is a gap, and a baseline that learns it as normal
will happily accept a dead camera forever.

Neither the storage layer nor the bucket knows which it is. The neighbours do.

The rule
--------
An empty bucket is a *known zero* if there is recorded activity within ``reach``
buckets on **both** sides. An hour with people either side of it happened while
the system was demonstrably running, so the absence in the middle is a fact
about the scene rather than about the recorder.

Requiring both sides is the whole point. A one-sided rule would mark the
beginning of an outage as an observed zero - there is data before it, and the
absence after is exactly what is being asked about - and the detector would
learn the first hours of every outage as normal quiet.

What this cannot do
-------------------
It cannot recover the edges. An outage running to the end of the window has no
right-hand neighbour, so those buckets stay ambiguous and unjudged, which is the
correct answer rather than a limitation to work around. Nor can it help a
database whose gaps are longer than ``reach`` in both directions - a camera off
for a week leaves nothing to bracket the middle of it.

A recorder heartbeat would settle all of this exactly, and is the right long-term
answer. This is the inference available from the rows that exist today, and it
is stated as an inference wherever its output is used.
"""

from __future__ import annotations

from dataclasses import replace

from vantage.analytics.contracts import Series

DEFAULT_REACH = 2
"""Buckets to look either side. At the shipped one-hour interval this treats a
gap of up to two hours as ordinary quiet and anything longer as an outage."""


def mark_observed_zeros(series: Series, *, reach: int = DEFAULT_REACH) -> Series:
    """Return a copy where bracketed empty buckets are marked as known zeros."""
    if reach < 1:
        raise ValueError("reach must be at least 1 bucket")

    buckets = list(series.buckets)
    has_data = [bucket.samples > 0 for bucket in buckets]

    updated = []
    for index, bucket in enumerate(buckets):
        if bucket.samples > 0 or bucket.known_zero:
            updated.append(bucket)
            continue
        before = any(has_data[max(0, index - reach) : index])
        after = any(has_data[index + 1 : index + 1 + reach])
        updated.append(replace(bucket, known_zero=before and after))

    return replace(series, buckets=tuple(updated))


def gap_report(series: Series) -> dict[str, int]:
    """Count what the marking decided, for reporting rather than for logic."""
    recorded = sum(1 for b in series if b.samples > 0)
    known_zero = sum(1 for b in series if b.samples == 0 and b.known_zero)
    unknown = sum(1 for b in series if b.samples == 0 and not b.known_zero)
    return {
        "recorded": recorded,
        "known_zero": known_zero,
        "unknown": unknown,
        "total": len(series),
    }


def mark_from_heartbeats(series: Series, heartbeats: list[float]) -> Series:
    """Mark empty buckets as known zeros wherever the recorder said it was alive.

    This is the answer the neighbour rule was approximating. A heartbeat inside
    a bucket is direct evidence that the pipeline completed a frame during that
    interval, so an absence of observations in it is a fact about the scene.
    No bracketing, no reach, and no failure on long quiet periods: a nine-hour
    overnight gap is fully resolved if the recorder was running through it, and
    fully unresolved if it was not.

    Buckets with no heartbeat are left alone rather than marked as outages. A
    database written before the heartbeat table existed has none at all, and
    treating that as "the camera was never running" would be a confident wrong
    answer where the honest one is that this database cannot say.

    Partial coverage is deliberately not modelled. A bucket containing a single
    heartbeat counts as covered even if the recorder was alive for only ten
    minutes of the hour, and its lower count then reads as reduced traffic
    rather than as reduced observation. This is a real limitation with a real
    consequence - an outage that starts mid-hour produces one below-baseline
    anomaly at its edge - and it is left visible rather than smoothed away,
    because the alternative is weighting every count by a coverage fraction and
    thereby reporting numbers no row in the database actually supports. The
    summary says plainly that a below-baseline anomaly may be a camera fault
    rather than a change in the scene, which is the honest handling of a
    genuinely ambiguous observation.
    """
    if not heartbeats:
        return series
    buckets = list(series.buckets)
    if not buckets:
        return series

    width = series.interval_s
    origin = buckets[0].start
    alive: set[int] = set()
    for beat in heartbeats:
        index = int((beat - origin) // width)
        if 0 <= index < len(buckets):
            alive.add(index)

    updated = [
        (
            bucket
            if bucket.samples > 0 or bucket.known_zero or index not in alive
            else replace(bucket, known_zero=True)
        )
        for index, bucket in enumerate(buckets)
    ]
    return replace(series, buckets=tuple(updated))
