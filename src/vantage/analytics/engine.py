"""The analytics entry point: a store in, an answer out.

This is the only module here that knows about storage. Everything below it -
aggregation, baselines, detection, prose - works on plain series and can be
tested without a database, which is why the accuracy harness needs no SQLite
file and no recorded footage.

Where the training window comes from
------------------------------------
Judging a period against a baseline learned from that same period is circular:
an anomaly large enough to matter also moves the thing it is being compared
against. Robust statistics blunt this - a median barely notices one outlier -
but they do not remove it, and with a short window a single bad day can be a
large share of the samples.

So the default is to train on history *before* the window under examination, and
the training window is reported in the result rather than left implicit. Callers
who genuinely want the self-comparison - looking for the odd hour within one
week, with no earlier history to draw on - ask for it explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.analytics.aggregate import bucket_series
from vantage.analytics.baseline import DEFAULT_SENSITIVITY, learn
from vantage.analytics.contracts import AnalysisResult, Baseline, Metric, Series
from vantage.analytics.coverage import (
    DEFAULT_REACH,
    mark_from_heartbeats,
    mark_observed_zeros,
)
from vantage.analytics.detector import detect
from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger

log = get_logger(__name__)

HOUR = 3600.0
DAY = 86_400.0
WEEK = 7 * DAY


@dataclass(frozen=True, slots=True)
class AnalyticsParams:
    """How to bucket, how far back to train, and how sure to be."""

    interval_s: float = HOUR
    period_hours: int = 168
    sensitivity: float = DEFAULT_SENSITIVITY
    training_span_s: float = 4 * WEEK
    """How much history before the window to learn from. Four weeks gives a
    weekly baseline four samples per slot - one above the minimum, which is the
    least that can be called a pattern rather than a coincidence."""

    judge_empty: bool = False
    include_empty_in_training: bool = False
    """Whether buckets that are *still ambiguous* after coverage inference train
    the baseline.

    False, and it should stay false. Once :attr:`infer_zeros` has run, an empty
    bucket bracketed by activity is no longer empty - it is a known zero, and it
    trains the baseline like any other reading. What remains empty is what could
    not be established either way: outages, and the edges of the window.

    Setting this true would teach the baseline that every unexplained silence is
    normal quiet, which is precisely how a detector learns to accept a dead
    camera without complaint.
    """

    infer_zeros: bool = True
    """Mark empty buckets bracketed by activity as observed zeros.

    Without this, an hour that is always empty never accumulates a training
    sample and can never be judged - so nobody walking through at 3am is
    detectable, which is most of the value of overnight analytics."""

    zero_reach: int = DEFAULT_REACH

    def __post_init__(self) -> None:
        if self.interval_s <= 0:
            raise ConfigError("analytics.interval_s must be positive")
        if self.training_span_s < 0:
            raise ConfigError("analytics.training_span_s must be >= 0")
        if self.period_hours not in (24, 168):
            raise ConfigError("analytics.period_hours must be 24 or 168")
        if self.zero_reach < 1:
            raise ConfigError("analytics.zero_reach must be at least 1 bucket")
        if self.include_empty_in_training and not self.infer_zeros:
            raise ConfigError(
                "include_empty_in_training without infer_zeros trains the baseline "
                "on every silent bucket, including recorder downtime, as though it "
                "were a quiet scene - so a camera that stopped working would never "
                "be flagged. Enable infer_zeros, which establishes which silences "
                "are real, or stop training on empty buckets."
            )


class AnalyticsEngine:
    """Aggregates, learns and judges against a store."""

    def __init__(self, store, *, params: AnalyticsParams | None = None) -> None:
        self._store = store
        self._params = params or AnalyticsParams()

    @property
    def params(self) -> AnalyticsParams:
        return self._params

    def _connection(self):
        # Reaches for the store's connection rather than adding an aggregate
        # method to the Store protocol. Analytics is a reader; making every
        # implementation of that protocol provide GROUP BY support would push
        # this phase's requirements onto a contract that four other subsystems
        # already satisfy.
        connection = getattr(self._store, "_require", None)
        if connection is None:
            raise ConfigError(
                f"{type(self._store).__name__} does not expose a SQL connection; "
                "analytics needs one to aggregate in the database rather than in memory"
            )
        return connection()

    def _heartbeats(self, since: float, until: float) -> list[float]:
        """Liveness markers in a window, or nothing if the store predates them."""
        reader = getattr(self._store, "heartbeats", None)
        if reader is None:
            return []
        try:
            return reader(since, until)
        except Exception as exc:  # pragma: no cover - a store without the table
            log.debug(
                "no heartbeat data available",
                extra={"vantage_fields": {"error": f"{type(exc).__name__}: {exc}"}},
            )
            return []

    def series(
        self,
        metric: Metric,
        *,
        since: float,
        until: float,
        interval_s: float | None = None,
        camera_id: str | None = None,
        entity_type: str | None = None,
        zone: str | None = None,
    ) -> Series:
        """One metric, bucketed over a window."""
        series = bucket_series(
            self._connection(),
            metric,
            since=since,
            until=until,
            interval_s=interval_s or self._params.interval_s,
            camera_id=camera_id,
            entity_type=entity_type,
            zone=zone,
        )
        if self._params.infer_zeros:
            # Heartbeats first, because they are evidence rather than inference.
            # The neighbour rule runs afterwards on whatever they left ambiguous,
            # which for a database recorded before the heartbeat table existed is
            # everything - so older stores keep working exactly as before.
            beats = self._heartbeats(series.since, series.until)
            if beats:
                series = mark_from_heartbeats(series, beats)
            series = mark_observed_zeros(series, reach=self._params.zero_reach)
        return series

    def baseline(
        self,
        metric: Metric,
        *,
        until: float,
        span_s: float | None = None,
        interval_s: float | None = None,
        camera_id: str | None = None,
        zone: str | None = None,
    ) -> Baseline:
        """Learn a baseline from the history ending at ``until``."""
        params = self._params
        span = params.training_span_s if span_s is None else span_s
        if span <= 0:
            raise ConfigError("a baseline needs a training span greater than zero")
        training = self.series(
            metric,
            since=until - span,
            until=until,
            interval_s=interval_s,
            camera_id=camera_id,
            zone=zone,
        )
        return learn(
            training,
            period_hours=params.period_hours,
            sensitivity=params.sensitivity,
            include_empty=params.include_empty_in_training,
        )

    def analyse(
        self,
        metric: Metric,
        *,
        since: float,
        until: float,
        interval_s: float | None = None,
        camera_id: str | None = None,
        entity_type: str | None = None,
        zone: str | None = None,
        train_on_window: bool = False,
    ) -> AnalysisResult:
        """Bucket a window, learn what is normal, and judge the window."""
        params = self._params
        width = interval_s or params.interval_s
        observed = self.series(
            metric,
            since=since,
            until=until,
            interval_s=width,
            camera_id=camera_id,
            entity_type=entity_type,
            zone=zone,
        )

        if train_on_window:
            baseline = learn(
                observed,
                period_hours=params.period_hours,
                sensitivity=params.sensitivity,
                include_empty=params.include_empty_in_training,
            )
        else:
            baseline = self.baseline(
                metric,
                until=since,
                interval_s=width,
                camera_id=camera_id,
                zone=zone,
            )
            if baseline.trustworthy_slots == 0:
                # Falling back silently would present a self-compared result as
                # though it had been judged against history, which is a
                # materially weaker claim wearing a stronger one's clothes.
                log.warning(
                    "no usable history before the window; nothing can be judged",
                    extra={
                        "vantage_fields": {
                            "metric": metric.value,
                            "trained_from": baseline.trained_from,
                            "trained_until": baseline.trained_until,
                            "hint": "pass train_on_window=True to compare the window "
                            "against itself, accepting that a large anomaly also "
                            "shifts what it is compared against",
                        }
                    },
                )

        return detect(observed, baseline, judge_empty=params.judge_empty)

    def zone_breakdown(
        self,
        zones: list[str],
        *,
        since: float,
        until: float,
        camera_id: str | None = None,
    ) -> dict[str, int]:
        """Distinct entities seen in each named zone over the window.

        One query per zone rather than a single grouped one, because the zones
        column is a denormalised list and a row can belong to several. Grouping
        by it would produce a bucket per *combination* of zones, and a person
        who stood on a boundary would appear under neither of the two zones they
        were actually in.
        """
        totals: dict[str, int] = {}
        for zone in zones:
            series = self.series(
                Metric.ENTITIES,
                since=since,
                until=until,
                interval_s=max(1.0, until - since),
                camera_id=camera_id,
                zone=zone,
            )
            totals[zone] = int(sum(series.values))
        return totals
