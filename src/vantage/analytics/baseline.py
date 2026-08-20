"""Learning what normal looks like, robustly.

Why median and MAD rather than mean and standard deviation
----------------------------------------------------------
The textbook answer is mean plus k standard deviations, and on this data it
fails in the specific way that makes anomaly detection useless: it is destroyed
by the very events it is supposed to find.

Consider a corridor that normally sees 3 people an hour, and one afternoon a
group of 40 walks through. That single bucket pulls the mean up and inflates the
standard deviation enormously - and the band is *wider* than the anomaly it was
supposed to catch. Having seen one unusual event, the detector has taught itself
that unusual events are normal, and it will never fire again.

The median and the median absolute deviation do not move. Both tolerate up to
half the samples being contaminated before they shift at all, so the same
40-person bucket leaves the learned normal at 3 and is flagged. That property is
the whole reason for the choice: a detector that quietly stops detecting is
worse than no detector, because it looks like good news.

MAD is scaled by 1.4826 so that on normally distributed data it matches the
standard deviation, which keeps the familiar "3 sigma" intuition roughly valid
for anyone reading a threshold.

The constant-slot problem
-------------------------
A slot where every observed value was identical has a MAD of exactly zero, and
this is *common* rather than pathological - an office corridor at 3am is 0
people every single night. With a zero spread, any deviation divides by zero and
every non-zero reading becomes infinitely anomalous, so one person walking past
at 3am on one night produces a maximum-severity alert.

That is not obviously wrong - it genuinely is unusual - but it is not worth an
infinite score, and it makes the detector hypersensitive exactly where data is
sparsest. So a floor is applied, expressed relative to the slot's own centre
with an absolute minimum for centres at or near zero. The floor is a stated
assumption rather than a hidden guard, and it is exposed for tuning.

Small samples, and why a slot is not judged on its own spread alone
-------------------------------------------------------------------
A weekly baseline over four weeks gives each slot exactly four samples. The
median of four numbers is a serviceable centre; the *median absolute deviation*
of four numbers is close to meaningless, and it is wrong in the dangerous
direction - four samples that happen to land near each other produce a tiny
spread, which turns the next ordinary reading into a high-severity anomaly.

The accuracy harness found this rather than predicted it: a week of ordinary
variation produced nine confident false positives, every one of them at a quiet
slot whose four samples had clustered and whose spread had collapsed to the
floor.

The fix is to stop pretending four samples can measure spread. Each slot's own
MAD is shrunk toward a **pooled** estimate taken from the residuals of the whole
series - hundreds of points rather than four - with the weight given to the
slot's own estimate rising as its sample count does. A slot with 4 samples is
mostly judged by how much this camera varies in general; a slot with 40 is
mostly judged by itself. This is ordinary shrinkage, and it is the difference
between a detector that works after four weeks and one that needs a year.

The pooled estimate is scaled to each slot rather than applied flat, because
count data does not have constant variance: a corridor averaging 40 people an
hour varies by more people than one averaging 2, and a single pooled number
would be far too wide for the quiet slot and far too narrow for the busy one.
"""

from __future__ import annotations

from vantage.analytics.contracts import (
    MIN_SLOT_SAMPLES,
    Baseline,
    Series,
    Slot,
    slot_index,
)
from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger

log = get_logger(__name__)

MAD_TO_SIGMA = 1.4826
"""Scales the median absolute deviation onto the standard-deviation scale for
normally distributed data, so a sensitivity of 3.5 means roughly what a reader
expects "3.5 sigma" to mean."""

RELATIVE_SPREAD_FLOOR = 0.02
"""Minimum spread as a fraction of the slot's centre.

Guards one degenerate case and nothing else: a busy slot whose observed values
were all *identical*, where the measured spread is exactly zero and any
departure would otherwise score infinitely.

This was 0.15 until it was measured. At that value it was not a guard at all -
it exceeded the real spread of every well-behaved slot and became the binding
constraint on all of them, pinning every band to +/-52% of centre. The
consequence was that accumulating history bought nothing: four weeks and
fifty-two weeks of training produced identical bands, because the floor
overrode the estimate in both cases. A floor that dominates the measurement it
is protecting is not a floor, it is the answer.

At 2% it does what it was meant to do. Slots with real variation are governed by
their own shrunk MAD, which tightens as samples accumulate; only the genuinely
constant slot meets the floor.
"""

ABSOLUTE_SPREAD_FLOOR = 0.75
"""Minimum spread for slots whose centre is at or near zero.

The relative floor collapses to nothing when the centre is 0, which is exactly
the always-empty overnight slot. Below one whole unit of whatever is being
counted, the distinction being drawn is finer than the measurement itself.
"""

DEFAULT_SENSITIVITY = 3.5
"""Robust z-score at which a bucket is called anomalous."""

SHRINKAGE_SAMPLES = 8.0
"""Sample count at which a slot's own spread carries half the weight.

With the shipped four-week training span every slot has 4 samples and is
therefore weighted 1/3 toward its own MAD and 2/3 toward the pooled estimate.
A slot with 24 samples - six months of weekly history - is weighted 3/4 toward
its own, which is the intended behaviour: more evidence, more autonomy.
"""


def median(values: list[float]) -> float:
    """Plain median. Written out rather than imported from statistics because
    this module is called with lists that are frequently empty or of length one,
    and returning 0.0 for empty is more useful here than raising."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


_SMALL_SAMPLE_CORRECTION = {
    2: 2.20,
    3: 1.85,
    4: 1.49,
    5: 1.34,
    6: 1.26,
    8: 1.17,
    12: 1.12,
    16: 1.10,
    26: 1.05,
}
"""How much the MAD of *n* samples underestimates the true spread.

Measured rather than cited: 4000 draws of n standard-normal values, taking the
median of the resulting MADs, for each n. Reproduce with::

    xs = [gauss(0, 1) for _ in range(n)]
    ratios.append(mad(xs, median(xs)))
    correction = 1 / median(ratios)

The bias is real and large at the sizes this module actually sees. A weekly
baseline over four weeks gives four samples per slot, where the MAD lands at
0.67 of the true spread - so an uncorrected 3.5-sigma band is really a
2.3-sigma one, and fires roughly ten times a week on data with nothing wrong
in it. That is not a subtle effect; it is the difference between a usable
detector and an alarm that everyone learns to ignore.

Two sources compound here, and both are corrected by this factor: the MAD of a
small sample is biased low, and the centre it is measured against was fitted to
those same points, which shrinks every residual.
"""


def small_sample_correction(n: int) -> float:
    """Multiplier that removes the small-sample bias in :func:`mad`."""
    if n < 2:
        return _SMALL_SAMPLE_CORRECTION[2]
    if n in _SMALL_SAMPLE_CORRECTION:
        return _SMALL_SAMPLE_CORRECTION[n]
    known = sorted(_SMALL_SAMPLE_CORRECTION)
    if n > known[-1]:
        # Beyond the table the bias is small and decays as 1/n; the closed form
        # is used rather than extending the table with values nobody will read.
        return 1.0 + 1.3 / n
    lower = max(k for k in known if k < n)
    upper = min(k for k in known if k > n)
    span = upper - lower
    weight = (n - lower) / span
    return (
        _SMALL_SAMPLE_CORRECTION[lower] * (1 - weight)
        + _SMALL_SAMPLE_CORRECTION[upper] * weight
    )


def mad(values: list[float], centre: float | None = None) -> float:
    """Median absolute deviation, scaled to the standard-deviation scale.

    Includes the small-sample correction, because every caller here works with
    slots of four to a few dozen samples, and an uncorrected MAD at those sizes
    is wrong by a factor that dominates every other decision in this module.
    """
    if not values:
        return 0.0
    mid = median(values) if centre is None else centre
    raw = median([abs(value - mid) for value in values]) * MAD_TO_SIGMA
    return raw * small_sample_correction(len(values))


def floor_spread(
    spread: float,
    centre: float,
    *,
    relative: float = RELATIVE_SPREAD_FLOOR,
    absolute: float = ABSOLUTE_SPREAD_FLOOR,
) -> float:
    """Apply the spread floor described in the module docstring."""
    return max(spread, abs(centre) * relative, absolute)


def _scale_for(centre: float) -> float:
    """How much a slot at this level is expected to vary, relatively.

    The square root of the level, which is what count data does: independent
    arrivals give a variance equal to the mean, so the standard deviation grows
    as the square root of it. A corridor averaging 49 people an hour varies by
    roughly seven; one averaging 4 varies by roughly two. Applying one flat
    pooled number to both would make the quiet slot untouchable and the busy one
    hair-triggered.

    The ``+ 1`` keeps a zero-centred slot from scaling to nothing, and matches
    the variance-stabilising transform used for low counts.
    """
    return (abs(centre) + 1.0) ** 0.5


def pooled_dispersion(grouped: dict[int, list[float]], centres: dict[int, float]) -> float:
    """One dispersion figure for the whole series, in units of _scale_for.

    Every observation contributes its residual from its own slot's centre,
    divided by that slot's expected scale. The median of those normalised
    residuals says how variable this camera is per unit of traffic, estimated
    from every bucket rather than from the four in any one slot.

    Returns 0.0 for an empty or single-slot input rather than raising: a caller
    with one slot has no pooled information, and the floors in
    :func:`floor_spread` still apply.
    """
    normalised: list[float] = []
    contributing = 0
    for index, values in grouped.items():
        # Slots that never varied are skipped. An office corridor has nine hours
        # a day where the count is zero every single time, and those residuals
        # are all exactly zero - not a sample of how much this camera varies,
        # but a structural constant. Including them put 37% zeros into the
        # median and halved the pooled estimate for every busy slot, which is
        # the opposite of what pooling is for.
        if len(set(values)) <= 1:
            continue
        contributing += 1
        scale = _scale_for(centres[index])
        normalised.extend(abs(value - centres[index]) / scale for value in values)
    if not normalised:
        return 0.0
    per_slot = max(1, len(normalised) // max(1, contributing))
    return median(normalised) * MAD_TO_SIGMA * small_sample_correction(per_slot)


def learn(
    series: Series,
    *,
    period_hours: int = 168,
    sensitivity: float = DEFAULT_SENSITIVITY,
    include_empty: bool = False,
) -> Baseline:
    """Learn a per-slot baseline from an observed series.

    ``include_empty`` decides what an empty bucket means, and there is no
    correct default for both cases - which is why it is a parameter rather than
    an assumption. An empty bucket is either "nobody was there", which is data
    and should train the baseline, or "the system was not running", which is not
    and would teach it that silence is normal. The pipeline cannot tell these
    apart from the rows alone, so the caller states which it has. Excluding them
    is the safer default: a baseline that has learned zeros it never actually
    observed will not flag a camera that has gone dark.
    """
    if period_hours not in (24, 168):
        raise ConfigError(
            f"baseline period must be 24 (daily) or 168 (weekly) hours, got {period_hours}"
        )
    if sensitivity <= 0:
        raise ConfigError(f"sensitivity must be positive, got {sensitivity}")

    grouped: dict[int, list[float]] = {}
    for bucket in series:
        if bucket.empty and not include_empty:
            continue
        grouped.setdefault(slot_index(bucket.when, period_hours), []).append(bucket.value)

    centres = {index: median(values) for index, values in grouped.items()}
    dispersion = pooled_dispersion(grouped, centres)

    slots: dict[int, Slot] = {}
    for index, values in grouped.items():
        centre = centres[index]
        own = mad(values, centre)
        # Shrink the slot's own spread toward what this camera does in general,
        # weighted by how much evidence the slot actually has. See the module
        # docstring: four samples cannot measure a spread, and pretending they
        # can is what produced nine confident false positives on ordinary data.
        pooled = dispersion * _scale_for(centre)
        weight = len(values) / (len(values) + SHRINKAGE_SAMPLES)
        spread = floor_spread(weight * own + (1.0 - weight) * pooled, centre)
        slots[index] = Slot(
            index=index,
            centre=centre,
            spread=spread,
            samples=len(values),
            low=centre - sensitivity * spread,
            high=centre + sensitivity * spread,
        )

    baseline = Baseline(
        metric=series.metric,
        slots=slots,
        period_hours=period_hours,
        interval_s=series.interval_s,
        trained_from=series.since,
        trained_until=series.until,
        sensitivity=sensitivity,
    )
    log.info(
        "baseline learned",
        extra={
            "vantage_fields": {
                "metric": series.metric.value,
                "slots": len(slots),
                "usable": baseline.trustworthy_slots,
                "period_hours": period_hours,
                "min_samples": MIN_SLOT_SAMPLES,
            }
        },
    )
    return baseline
