"""Judging buckets against a baseline.

The detector is deliberately thin. Everything that makes the answer defensible -
robust centre, floored spread, sample counts - was decided when the baseline was
learned; this module's only jobs are to refuse to judge what it cannot, and to
describe what it does judge in terms a reader can check.

What it refuses
---------------
A bucket whose slot has fewer than :data:`MIN_SLOT_SAMPLES` observations is not
judged at all, and is counted in ``skipped_untrained``. It is not judged
leniently, and it is not judged with a wider band: both would produce a verdict,
and a verdict from two samples is a guess wearing the same clothes as a
measurement.

Empty buckets are also not judged by default, for the reason given in the
baseline module - the pipeline cannot distinguish "nobody was there" from "the
system was off", and flagging downtime as an anomaly in human traffic would fill
the report with events about the recorder rather than about the scene. A caller
who knows the system ran continuously can pass ``judge_empty=True`` and get
exactly the opposite, useful behaviour: a camera that goes dark starts producing
below-baseline anomalies immediately.
"""

from __future__ import annotations

from vantage.analytics.contracts import (
    AnalysisResult,
    Anomaly,
    Baseline,
    Direction,
    Series,
)
from vantage.core.errors import VantageError
from vantage.core.logging import get_logger

log = get_logger(__name__)


def detect(
    series: Series,
    baseline: Baseline,
    *,
    judge_empty: bool = False,
) -> AnalysisResult:
    """Score every bucket in ``series`` against ``baseline``."""
    if series.metric is not baseline.metric:
        raise VantageError(
            f"cannot judge a {series.metric.value} series against a "
            f"{baseline.metric.value} baseline"
        )
    if abs(series.interval_s - baseline.interval_s) > 1e-6:
        # An hourly value against a baseline of daily totals would be flagged as
        # catastrophically low every single hour, and the report would look
        # exactly like a real finding.
        raise VantageError(
            f"bucket width mismatch: the series is bucketed at {series.interval_s:g}s "
            f"but the baseline was learned at {baseline.interval_s:g}s. Re-learn the "
            "baseline at the interval you want to judge, or re-bucket the series."
        )

    anomalies: list[Anomaly] = []
    skipped = 0

    for bucket in series:
        if bucket.empty and not judge_empty:
            skipped += 1
            continue
        slot = baseline.slot_for(bucket.when)
        if slot is None or not slot.trustworthy:
            skipped += 1
            continue

        deviation = bucket.value - slot.centre
        score = abs(deviation) / slot.spread if slot.spread > 0 else 0.0
        if score < baseline.sensitivity:
            continue

        anomalies.append(
            Anomaly(
                bucket=bucket,
                metric=series.metric,
                direction=Direction.ABOVE if deviation > 0 else Direction.BELOW,
                observed=bucket.value,
                expected=slot.centre,
                band_low=slot.low,
                band_high=slot.high,
                score=score,
                samples=slot.samples,
            )
        )

    result = AnalysisResult(
        series=series,
        anomalies=tuple(anomalies),
        baseline=baseline,
        skipped_untrained=skipped,
    )
    if anomalies:
        log.info(
            "anomalies detected",
            extra={
                "vantage_fields": {
                    "metric": series.metric.value,
                    "found": len(anomalies),
                    "judged": result.judged,
                    "unjudged": skipped,
                }
            },
        )
    return result
