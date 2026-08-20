"""Scoring the analytics against histories whose answers are known.

Every phase since tracking ships a harness that builds ground truth, runs the
real code over it, and scores the result. This one generates synthetic histories
with anomalies planted at known positions, then asks whether the detector found
those and nothing else.

Why synthetic history rather than recorded footage
--------------------------------------------------
The question here is not "does the camera see people" - four earlier phases
already answer that. It is "given a month of counts, does this correctly
identify the three unusual hours". Recorded footage cannot answer that without a
month of recording and a human labelling every hour of it, and the labels would
be the very judgement under test.

Generated histories make the answer checkable: the scenario knows it put 40
people in Tuesday's 14:00 bucket, so a detector that misses it has failed and a
detector that also flags Wednesday 09:00 has failed differently. Both are
counted separately, because a detector tuned until it finds everything by
flagging everything is the standard way this goes wrong.

The scenarios include cases designed to be *rejected*: a history too short to
learn from, a slot with a single sample, and a week of pure noise with nothing
planted. A harness that only measured detection would score a
flag-everything detector perfectly on all three.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from vantage.analytics.baseline import DEFAULT_SENSITIVITY, learn
from vantage.analytics.contracts import Bucket, Direction, Metric, Series
from vantage.analytics.detector import detect

HOUR = 3600.0


@dataclass(frozen=True, slots=True)
class Planted:
    """One deliberately unusual bucket."""

    index: int
    value: float
    direction: Direction


@dataclass(frozen=True, slots=True)
class Scenario:
    """A generated history with a known answer."""

    name: str
    description: str
    days: int
    weekday_profile: tuple[float, ...]
    """Expected value for each hour of a weekday, 24 entries."""

    weekend_profile: tuple[float, ...] | None = None
    planted: tuple[Planted, ...] = ()
    noise: float = 0.0
    """Deterministic jitter amplitude, at the busiest hour of the profile.

    Applied as a fixed pattern rather than a random one, so a failing run can be
    reproduced exactly rather than re-run until it passes.

    Scaled by the square root of each hour's own level, not applied flat. Count
    data behaves this way - a corridor averaging 49 people an hour varies by
    roughly seven, one averaging zero varies by zero - and the first version of
    this generator did apply it flat, which put a six-person swing on 3am slots
    whose true value was zero. That is not "heavy jitter on a busy profile",
    which is what this scenario is named for; it is a different and impossible
    distribution at every quiet hour, and no estimator could have characterised
    it from four samples. The false positives it produced were the generator's,
    not the detector's.
    """

    gap_hours: tuple[int, ...] = ()
    """Bucket indices left empty, standing in for recorder downtime."""

    expect_anomalies: bool = True
    metric: Metric = Metric.ENTITIES

    def build(self, origin: float) -> tuple[Series, set[int]]:
        """Materialise the history and the indices that should be flagged."""
        weekend = self.weekend_profile or self.weekday_profile
        buckets: list[Bucket] = []
        planted_at = {p.index: p for p in self.planted}
        gaps = set(self.gap_hours)

        for index in range(self.days * 24):
            start = origin + index * HOUR
            when = datetime.fromtimestamp(start, tz=UTC).astimezone()
            profile = weekend if when.weekday() >= 5 else self.weekday_profile
            value = profile[when.hour]

            if self.noise:
                # A fixed, repeatable wobble, scaled to this hour's own level so
                # that an empty hour stays empty. Deliberately not random: a
                # harness that fails one run in twenty on unlucky noise teaches
                # people to re-run it until it passes.
                peak = max(profile) or 1.0
                level = (value / peak) ** 0.5
                value += self.noise * level * math.sin(index * 1.7)

            plant = planted_at.get(index)
            if plant is not None:
                value = plant.value

            if index in gaps:
                buckets.append(Bucket(start=start, width_s=HOUR, value=0.0, samples=0))
                continue

            buckets.append(
                Bucket(
                    start=start,
                    width_s=HOUR,
                    value=max(0.0, value),
                    samples=max(1, int(max(0.0, value))),
                )
            )

        series = Series(
            metric=self.metric,
            buckets=tuple(buckets),
            interval_s=HOUR,
            since=origin,
            until=origin + self.days * 24 * HOUR,
        )
        return series, set(planted_at)


@dataclass(slots=True)
class ScenarioScore:
    name: str
    found: int = 0
    missed: int = 0
    false_positives: int = 0
    judged: int = 0
    unjudged: int = 0
    notes: str = ""

    @property
    def planted(self) -> int:
        return self.found + self.missed

    @property
    def recall(self) -> float:
        return self.found / self.planted if self.planted else 1.0

    @property
    def passed(self) -> bool:
        return self.missed == 0 and self.false_positives == 0


@dataclass(slots=True)
class EvaluationReport:
    scores: list[ScenarioScore] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(score.passed for score in self.scores)

    @property
    def total_found(self) -> int:
        return sum(s.found for s in self.scores)

    @property
    def total_planted(self) -> int:
        return sum(s.planted for s in self.scores)

    @property
    def total_false(self) -> int:
        return sum(s.false_positives for s in self.scores)

    def describe(self) -> str:
        header = f"{'scenario':<30} {'found':>10} {'missed':>7} {'false':>6} {'judged':>7}"
        lines = [header, "-" * len(header)]
        for score in self.scores:
            lines.append(
                f"{score.name:<30} {score.found:>4}/{score.planted:<5} "
                f"{score.missed:>7} {score.false_positives:>6} {score.judged:>7}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'POOLED':<30} {self.total_found:>4}/{self.total_planted:<5} "
            f"{self.total_planted - self.total_found:>7} {self.total_false:>6}"
        )
        notes = [f"  {s.name}: {s.notes}" for s in self.scores if s.notes]
        if notes:
            lines.append("")
            lines.extend(notes)
        return "\n".join(lines)


def _origin() -> float:
    """A fixed Monday 00:00 local, so weekday profiles land where intended.

    Anchored to a real date rather than to "now" because a harness whose result
    depends on the day it is run is not a regression test.
    """
    local = datetime(2026, 1, 5, 0, 0, 0).astimezone()  # a Monday
    return local.timestamp()


# Day indices into the generated history, which starts on Monday 5 January 2026.
# Anomalies are planted on *weekdays* on purpose. An earlier version of this file
# used bare numbers that happened to land on Saturday and Sunday, where the
# weekend profile is a flat zero - so three of the five planted anomalies were
# "more than zero people in a slot that is always zero", which any detector
# finds. The harness passed, and it was measuring almost nothing. Naming the days
# makes that class of mistake visible in the scenario itself.
MON, TUE, WED, THU, FRI, SAT, SUN = 21, 22, 23, 24, 25, 26, 27

QUIET_OFFICE = (0, 0, 0, 0, 0, 0, 1, 3, 8, 12, 11, 10, 14, 13, 12, 11, 9, 6, 2, 1, 0, 0, 0, 0)
QUIET_WEEKEND = (0,) * 24
BUSY_RETAIL = (
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    6,
    18,
    30,
    38,
    42,
    47,
    45,
    40,
    38,
    35,
    30,
    20,
    10,
    3,
    0,
    0,
    0,
)


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="crowd_spike",
        description="A single hour with far more people than that slot ever sees.",
        days=28,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        planted=(Planted(index=THU * 24 + 14, value=60.0, direction=Direction.ABOVE),),
        noise=0.6,
    ),
    Scenario(
        name="camera_went_dark",
        description="A busy afternoon that suddenly reports almost nobody.",
        days=28,
        weekday_profile=BUSY_RETAIL,
        planted=(Planted(index=THU * 24 + 12, value=1.0, direction=Direction.BELOW),),
        noise=1.2,
    ),
    Scenario(
        name="after_hours_presence",
        description="Somebody in the building at 03:00, where the baseline is a flat zero.",
        days=28,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        planted=(Planted(index=THU * 24 + 3, value=4.0, direction=Direction.ABOVE),),
    ),
    Scenario(
        name="two_spikes_one_week",
        description="Two unusual hours on different days; both must be found.",
        days=28,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        planted=(
            Planted(index=TUE * 24 + 10, value=55.0, direction=Direction.ABOVE),
            Planted(index=THU * 24 + 16, value=48.0, direction=Direction.ABOVE),
        ),
        noise=0.6,
    ),
    Scenario(
        name="ordinary_week",
        description="Nothing planted. Any flag here is a false positive.",
        days=28,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        noise=1.5,
        expect_anomalies=False,
    ),
    Scenario(
        name="noisy_but_normal",
        description="Heavy jitter on a busy profile. Variation is not anomaly.",
        days=28,
        weekday_profile=BUSY_RETAIL,
        noise=6.0,
        expect_anomalies=False,
    ),
    Scenario(
        name="recorder_downtime",
        description="A day of gaps. Downtime must not be reported as a change in traffic.",
        days=28,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        gap_hours=tuple(range(WED * 24, WED * 24 + 24)),
        noise=0.6,
        expect_anomalies=False,
    ),
    Scenario(
        name="too_little_history",
        description="Three days. No slot has enough samples, so nothing may be judged.",
        days=3,
        weekday_profile=QUIET_OFFICE,
        weekend_profile=QUIET_WEEKEND,
        planted=(Planted(index=2 * 24 + 14, value=60.0, direction=Direction.ABOVE),),
        expect_anomalies=False,
    ),
)


def evaluate(
    scenarios: tuple[Scenario, ...] = SCENARIOS,
    *,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> EvaluationReport:
    """Run every scenario and score what the detector made of it."""
    report = EvaluationReport()
    origin = _origin()

    for scenario in scenarios:
        series, planted = scenario.build(origin)
        baseline = learn(series, period_hours=168, sensitivity=sensitivity)
        result = detect(series, baseline)

        flagged = {int((a.bucket.start - origin) / HOUR) for a in result.anomalies}
        found = flagged & planted
        score = ScenarioScore(
            name=scenario.name,
            found=len(found),
            missed=len(planted - flagged),
            false_positives=len(flagged - planted),
            judged=result.judged,
            unjudged=result.skipped_untrained,
        )

        if scenario.name == "too_little_history":
            # This scenario passes by refusing, not by detecting. A planted
            # spike it cannot judge is the correct outcome, so a miss here is
            # not a failure - but a *flag* would be, because it would mean the
            # detector judged a slot it had no history for.
            score.missed = 0
            score.notes = (
                f"judged {result.judged} of {len(series)} buckets - "
                "correctly declined for want of history"
            )
        elif not scenario.expect_anomalies and score.false_positives:
            score.notes = f"expected silence, flagged {score.false_positives}"

        report.scores.append(score)

    return report


# ---------------------------------------------------------------------------
# Characterisation on random data
# ---------------------------------------------------------------------------
#
# The scenarios above use deterministic jitter so that a failure is reproducible.
# That is the right property for a regression test and the wrong one for judging
# how the detector behaves in the field: bounded, repeating variation has no
# tails, and a detector can pass every scenario above while firing constantly on
# real footage.
#
# It did, in fact. The scenario suite passed with zero false positives at a point
# when the detector was producing about ten false alarms a week on random data of
# the same shape, because its spread estimate was low by a factor of 2.7. Nothing
# in the pass/fail suite could have shown that.
#
# So this second harness trades reproducibility for realism deliberately. It
# draws counts from a distribution with real tails, plants nothing, and counts
# what gets flagged anyway; then plants spikes of known size and counts how often
# they are caught. Both numbers are seeded, so a given run repeats exactly, but
# the point is that they are measured across many seeds rather than one.


@dataclass(frozen=True, slots=True)
class Characterisation:
    """False-alarm rate and detection power, measured on random data."""

    false_alarms_per_week: float
    worst_week: float
    detection: tuple[tuple[float, float], ...]
    """(relative size of spike, fraction of trials detected), ascending."""

    trials: int
    training_weeks: int

    def detection_at(self, fraction: float) -> float | None:
        """Smallest spike detected in at least ``fraction`` of trials."""
        for size, hit_rate in self.detection:
            if hit_rate >= fraction:
                return size
        return None

    def describe(self) -> str:
        lines = [
            f"Measured over {self.trials} seeds, {self.training_weeks} weeks of history:",
            "",
            f"  False alarms on clean data   {self.false_alarms_per_week:.2f} per week "
            f"(worst week {self.worst_week:.2f})",
            "",
            "  Detection power:",
        ]
        for size, rate in self.detection:
            bar = "#" * int(rate * 20)
            lines.append(f"    +{size:>4.0%}   {rate:>5.0%}  {bar}")
        half = self.detection_at(0.5)
        near = self.detection_at(0.95)
        lines.append("")
        if half is not None:
            lines.append(f"  Caught half the time at +{half:.0%} above normal")
        if near is not None:
            lines.append(f"  Caught almost always at +{near:.0%} above normal")
        else:
            lines.append("  Never reached 95% detection within the sizes tried")
        return "\n".join(lines)


def _random_series(
    days: int,
    profile: tuple[float, ...],
    seed: int,
    *,
    spike: tuple[int, float] | None = None,
    dispersion: float = 1.0,
) -> Series:
    """Counts drawn with variance proportional to the mean, as arrivals behave."""
    import random

    rng = random.Random(seed)
    origin = _origin()
    buckets: list[Bucket] = []
    for index in range(days * 24):
        start = origin + index * HOUR
        when = datetime.fromtimestamp(start, tz=UTC).astimezone()
        mean = profile[when.hour]
        value = max(0.0, rng.gauss(mean, dispersion * (mean**0.5)))
        if spike is not None and index == spike[0]:
            value = spike[1]
        buckets.append(
            Bucket(start=start, width_s=HOUR, value=value, samples=max(1, int(value)))
        )
    return Series(
        metric=Metric.ENTITIES,
        buckets=tuple(buckets),
        interval_s=HOUR,
        since=origin,
        until=origin + days * 24 * HOUR,
    )


def characterise(
    *,
    trials: int = 20,
    training_weeks: int = 4,
    sensitivity: float = DEFAULT_SENSITIVITY,
    profile: tuple[float, ...] = BUSY_RETAIL,
) -> Characterisation:
    """Measure false-alarm rate and detection power on random data."""
    days = training_weeks * 7
    target = (days - 4) * 24 + 12  # a Thursday at noon, where the profile peaks
    origin = _origin()
    normal = profile[12]

    clean = []
    for seed in range(trials):
        series = _random_series(days, profile, seed)
        baseline = learn(series, period_hours=168, sensitivity=sensitivity)
        clean.append(len(detect(series, baseline).anomalies))

    detection: list[tuple[float, float]] = []
    for multiple in (1.15, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5):
        value = normal * multiple
        hits = 0
        for seed in range(trials):
            spiked = _random_series(days, profile, seed, spike=(target, value))
            baseline = learn(
                _random_series(days, profile, seed),
                period_hours=168,
                sensitivity=sensitivity,
            )
            flagged = {
                int((a.bucket.start - origin) / HOUR)
                for a in detect(spiked, baseline).anomalies
            }
            hits += target in flagged
        detection.append((multiple - 1.0, hits / trials))

    return Characterisation(
        false_alarms_per_week=sum(clean) / len(clean) / training_weeks,
        worst_week=max(clean) / training_weeks,
        detection=tuple(detection),
        trials=trials,
        training_weeks=training_weeks,
    )
