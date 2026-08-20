"""Tests for Phase 11 analytics.

What these are actually guarding
--------------------------------
Two things went wrong while building this phase, both of them invisible to an
ordinary "does the function return the right shape" test, and both are pinned
here so they cannot come back.

The first is **a spread estimate that was wrong by a factor of 2.7**. Every
scenario in the accuracy harness passed while the detector was producing about
ten false alarms a week on realistic data, because the harness used bounded
deterministic noise and real counts have tails. The tests below check the
estimator against a known distribution rather than against a fixture.

The second is **the empty-bucket ambiguity**. An hour with no rows means either
"nobody was there" or "nothing was recording", the two call for opposite
handling, and for a while the code silently chose one. The tests pin the whole
chain: what a bare bucket means, what a heartbeat changes, and - most
importantly - that an outage never trains the baseline as though it were quiet.

What is deliberately not tested
-------------------------------
Whether the detector is *useful* on real footage. That needs months of recorded
history from a real camera, and no fixture can stand in for it. What is measured
here is that the statistics are unbiased, the ambiguity is handled explicitly,
and the aggregation matches what a hand count of the same rows gives.
"""

from __future__ import annotations

import random
import statistics
import time
from datetime import UTC, datetime, timedelta

import pytest

from vantage.analytics.aggregate import bucket_series, local_midnight, next_local_midnight
from vantage.analytics.baseline import (
    ABSOLUTE_SPREAD_FLOOR,
    RELATIVE_SPREAD_FLOOR,
    learn,
    mad,
    median,
    pooled_dispersion,
    small_sample_correction,
)
from vantage.analytics.contracts import (
    MIN_SLOT_SAMPLES,
    Bucket,
    Direction,
    Metric,
    Series,
    slot_index,
)
from vantage.analytics.coverage import mark_from_heartbeats, mark_observed_zeros
from vantage.analytics.detector import detect
from vantage.analytics.engine import AnalyticsEngine, AnalyticsParams
from vantage.analytics.evaluation import evaluate
from vantage.analytics.summary import summarise_analysis, summarise_series
from vantage.core.errors import ConfigError, VantageError
from vantage.storage.sqlite_store import SqliteStore

HOUR = 3600.0


def make_series(values, *, start=None, metric=Metric.ENTITIES, samples=None):
    """A series from bare numbers. ``None`` marks a bucket with no rows."""
    origin = local_midnight(time.time() - 40 * 86400) if start is None else start
    buckets = []
    for index, value in enumerate(values):
        if value is None:
            buckets.append(Bucket(start=origin + index * HOUR, width_s=HOUR, value=0.0))
            continue
        buckets.append(
            Bucket(
                start=origin + index * HOUR,
                width_s=HOUR,
                value=float(value),
                samples=samples or max(1, int(value)) or 1,
            )
        )
    return Series(
        metric=metric,
        buckets=tuple(buckets),
        interval_s=HOUR,
        since=origin,
        until=origin + len(values) * HOUR,
    )


class TestRobustStatistics:
    def test_median_of_empty_is_zero_rather_than_an_exception(self) -> None:
        assert median([]) == 0.0

    def test_mad_survives_a_gross_outlier(self) -> None:
        """The property the whole module is built on.

        A mean-and-stddev baseline would widen so far after one 40-person bucket
        that it could never flag another.
        """
        ordinary = [1.0, 3.0, 5.0, 2.0, 6.0, 3.0, 4.0, 2.0]
        contaminated = [*ordinary, 400.0]
        assert median(contaminated) == pytest.approx(median(ordinary), abs=1.0)
        # A mean/stddev pair would move by orders of magnitude here.
        assert mad(ordinary) > 0, "the fixture must have real spread to be a test"
        assert mad(contaminated) < 2 * mad(ordinary)
        assert statistics.stdev(contaminated) > 20 * statistics.stdev(ordinary)

    def test_small_sample_correction_shrinks_toward_one(self) -> None:
        assert small_sample_correction(3) > small_sample_correction(8)
        assert small_sample_correction(8) > small_sample_correction(26)
        assert small_sample_correction(1000) == pytest.approx(1.0, abs=0.01)

    def test_small_sample_correction_interpolates_between_table_entries(self) -> None:
        between = small_sample_correction(7)
        assert small_sample_correction(8) < between < small_sample_correction(6)

    def test_mad_is_unbiased_on_known_normal_data(self) -> None:
        """The regression that matters most.

        Before the correction existed, the MAD of four samples came out at 0.67
        of the true spread. A nominal 3.5-sigma band was really 2.3 sigma, and
        the detector fired about ten times a week on data with nothing wrong in
        it. This asserts the estimator is honest at the sample sizes actually
        used, which is four.
        """
        for n in (4, 8, 26):
            rng = random.Random(n * 31)
            estimates = [mad([rng.gauss(0.0, 1.0) for _ in range(n)]) for _ in range(2000)]
            assert statistics.median(estimates) == pytest.approx(1.0, abs=0.12), (
                f"MAD of {n} samples is biased; a 3.5-sigma band would not be one"
            )

    def test_pooled_dispersion_ignores_slots_that_never_varied(self) -> None:
        """Structural zeros are not evidence about variability.

        An office is empty for nine hours a day, every day. Those residuals are
        all exactly zero, and including them put 37% zeros into the pooled
        median and halved the estimate for every busy slot.
        """
        varying = {0: [10.0, 12.0, 8.0, 11.0]}
        centres = {0: 10.5, 1: 0.0}
        with_constant = {**varying, 1: [0.0, 0.0, 0.0, 0.0]}
        assert pooled_dispersion(with_constant, centres) == pytest.approx(
            pooled_dispersion(varying, {0: 10.5})
        )

    def test_pooled_dispersion_of_nothing_is_zero_not_an_error(self) -> None:
        assert pooled_dispersion({}, {}) == 0.0


class TestSpreadFloors:
    def test_a_constant_slot_does_not_get_a_zero_spread(self) -> None:
        """Otherwise one person at 3am scores infinitely."""
        series = make_series([0.0] * (24 * 28), samples=1)
        baseline = learn(series, period_hours=24)
        assert all(slot.spread >= ABSOLUTE_SPREAD_FLOOR for slot in baseline.slots.values())

    def test_the_relative_floor_does_not_override_a_real_estimate(self) -> None:
        """The bug that made history worthless.

        At 0.15 the relative floor exceeded the real spread of every well-behaved
        slot, so it - not the data - set every band. Four weeks and fifty-two
        weeks of training produced identical results, and collecting more history
        bought nothing at all.
        """
        assert RELATIVE_SPREAD_FLOOR <= 0.05

        rng = random.Random(4)
        values = [max(0.0, rng.gauss(50.0, 7.0)) for _ in range(24 * 28)]
        baseline = learn(make_series(values), period_hours=24)
        slot = next(iter(baseline.slots.values()))
        assert slot.spread > slot.centre * RELATIVE_SPREAD_FLOOR


class TestBaseline:
    def test_a_slot_with_too_few_samples_is_not_trustworthy(self) -> None:
        series = make_series([5.0, 6.0])
        baseline = learn(series, period_hours=24)
        assert all(not slot.trustworthy for slot in baseline.slots.values())
        assert baseline.trustworthy_slots == 0

    def test_min_slot_samples_is_enforced_not_merely_declared(self) -> None:
        series = make_series([5.0] * (MIN_SLOT_SAMPLES - 1) * 24, samples=3)
        baseline = learn(series, period_hours=24)
        assert baseline.trustworthy_slots == 0

    def test_period_must_be_daily_or_weekly(self) -> None:
        with pytest.raises(ConfigError, match=r"24 .*or 168"):
            learn(make_series([1.0] * 48), period_hours=72)

    def test_negative_sensitivity_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="sensitivity"):
            learn(make_series([1.0] * 48), sensitivity=-1.0)


class TestDetector:
    def test_a_clear_spike_is_flagged(self) -> None:
        values = [10.0] * (24 * 28)
        values[24 * 27 + 5] = 90.0
        series = make_series(values)
        result = detect(series, learn(series, period_hours=24))
        assert len(result.anomalies) == 1
        assert result.anomalies[0].direction is Direction.ABOVE

    def test_a_collapse_is_flagged_and_marked_below(self) -> None:
        """A camera that stopped seeing people is the more urgent direction."""
        values = [40.0 + (index % 5) for index in range(24 * 28)]
        values[24 * 27 + 5] = 0.0
        series = make_series(values, samples=40)
        result = detect(series, learn(series, period_hours=24))
        assert any(a.direction is Direction.BELOW for a in result.anomalies)

    def test_a_metric_mismatch_is_refused(self) -> None:
        entities = make_series([1.0] * 48)
        events = make_series([1.0] * 48, metric=Metric.EVENTS)
        with pytest.raises(VantageError, match="cannot judge"):
            detect(entities, learn(events, period_hours=24))

    def test_an_interval_mismatch_is_refused(self) -> None:
        """Judging hourly values against daily totals would flag every hour.

        And it would look exactly like a real finding, which is why this raises
        rather than rescaling something the caller did not ask to be rescaled.
        """
        hourly = make_series([5.0] * 48)
        daily = Series(
            metric=Metric.ENTITIES,
            buckets=hourly.buckets,
            interval_s=86400.0,
            since=hourly.since,
            until=hourly.until,
        )
        with pytest.raises(VantageError, match="bucket width mismatch"):
            detect(hourly, learn(daily, period_hours=24))

    def test_untrained_slots_are_counted_not_silently_skipped(self) -> None:
        """'No anomalies' and 'nothing was looked at' must be distinguishable."""
        series = make_series([7.0, 7.0])
        result = detect(series, learn(series, period_hours=24))
        assert result.anomalies == ()
        assert result.skipped_untrained == len(series)
        assert result.judged == 0


class TestCoverage:
    def test_a_bare_empty_bucket_is_not_a_reading(self) -> None:
        series = make_series([5.0, None, 5.0])
        assert series.buckets[1].empty
        assert series.coverage == pytest.approx(2 / 3)

    def test_neighbours_can_establish_that_a_gap_was_a_quiet_hour(self) -> None:
        marked = mark_observed_zeros(make_series([5.0, None, 5.0]), reach=1)
        assert not marked.buckets[1].empty
        assert marked.buckets[1].known_zero
        assert marked.coverage == 1.0

    def test_a_one_sided_gap_stays_ambiguous(self) -> None:
        """The start of an outage has data before it and nothing after.

        A one-sided rule would learn the first hours of every outage as normal
        quiet, which is how a detector comes to accept a dead camera.
        """
        marked = mark_observed_zeros(make_series([5.0, None, None, None]), reach=1)
        assert all(b.empty for b in marked.buckets[1:])

    def test_heartbeats_resolve_a_gap_longer_than_any_reach(self) -> None:
        """The case neighbour inference structurally cannot handle.

        An office is empty for nine hours overnight. No reach distinguishes that
        from a nine-hour outage, so before heartbeats existed those slots never
        accumulated a training sample and nobody walking through at 3am could
        ever be flagged.
        """
        series = make_series([5.0] + [None] * 9 + [5.0])
        beats = [b.start + 60.0 for b in series.buckets]
        marked = mark_from_heartbeats(series, beats)
        assert all(not b.empty for b in marked.buckets)
        assert marked.coverage == 1.0

    def test_a_gap_with_no_heartbeat_stays_ambiguous(self) -> None:
        series = make_series([5.0] + [None] * 9 + [5.0])
        alive = [series.buckets[0].start + 60.0, series.buckets[-1].start + 60.0]
        marked = mark_from_heartbeats(series, alive)
        assert all(b.empty for b in marked.buckets[1:10])

    def test_a_store_with_no_heartbeats_is_unchanged_rather_than_blanked(self) -> None:
        """Databases written before schema v2 must keep working."""
        series = make_series([5.0, None, 5.0])
        assert mark_from_heartbeats(series, []) == series

    def test_an_outage_never_trains_the_baseline(self) -> None:
        """The single most consequential guarantee in this phase.

        If downtime trains the baseline, the system learns that silence is
        normal and a dead camera is never reported.
        """
        values = [20.0] * (24 * 27) + [None] * 24
        series = mark_observed_zeros(make_series(values), reach=2)
        baseline = learn(series, period_hours=24, include_empty=False)
        assert all(slot.centre > 0 for slot in baseline.slots.values())


class TestAggregation:
    def test_bucket_index_is_correct_left_of_the_origin(self) -> None:
        """Truncation toward zero would put every pre-origin timestamp one
        bucket too high, silently shifting half of any window that straddles
        local midnight."""
        from vantage.analytics.aggregate import _index_of

        assert _index_of(-1.0, 0.0, 10.0) == -1
        assert _index_of(-11.0, 0.0, 10.0) == -2
        assert _index_of(9.0, 0.0, 10.0) == 0

    def test_local_midnight_is_a_real_midnight(self) -> None:
        midnight = local_midnight(time.time())
        local = datetime.fromtimestamp(midnight, tz=UTC).astimezone()
        assert (local.hour, local.minute, local.second) == (0, 0, 0)

    def test_next_local_midnight_advances_one_day_not_86400_seconds(self) -> None:
        """The two differ by an hour on a daylight-saving transition day."""
        start = local_midnight(time.time())
        following = next_local_midnight(start)
        a = datetime.fromtimestamp(start, tz=UTC).astimezone()
        b = datetime.fromtimestamp(following, tz=UTC).astimezone()
        assert (b - a).days == 1
        assert (b.hour, b.minute) == (0, 0)

    def test_slot_index_uses_local_weekday_not_epoch_arithmetic(self) -> None:
        when = datetime.now().astimezone().replace(hour=9, minute=0, second=0, microsecond=0)
        assert slot_index(when, 168) == when.weekday() * 24 + 9
        assert slot_index(when, 24) == 9

    def test_slot_index_is_stable_across_a_week(self) -> None:
        when = datetime.now().astimezone().replace(hour=14, minute=30)
        assert slot_index(when, 168) == slot_index(when + timedelta(days=7), 168)


class TestSeriesContract:
    def test_summing_a_rate_metric_is_refused(self) -> None:
        """A mean speed summed across buckets has no interpretation."""
        series = make_series([0.3, 0.4], metric=Metric.MEAN_SPEED)
        with pytest.raises(ValueError, match="average, not a count"):
            _ = series.total

    def test_mean_excludes_empty_buckets(self) -> None:
        """Otherwise the metric measures uptime while wearing the name of traffic."""
        assert make_series([10.0, None, 10.0]).mean() == pytest.approx(10.0)

    def test_coverage_of_an_empty_series_is_zero_not_an_error(self) -> None:
        assert make_series([]).coverage == 0.0


class TestAgainstARealStore:
    """The aggregation runs as SQL, so it has to be tested against real SQL."""

    def build(self, tmp_path, counts_by_hour):
        store = SqliteStore(tmp_path / "a.db")
        origin = local_midnight(time.time() - 3 * 86400)
        rows = []
        entity = 0
        for index, count in enumerate(counts_by_hour):
            for k in range(count):
                entity += 1
                rows.append(
                    {
                        "timestamp": origin + index * HOUR + k,
                        "camera_id": "cam0",
                        "entity_id": f"person_{entity}",
                        "identity": None,
                        "entity_type": "person",
                        "motion": "moving" if k % 2 else "stationary",
                        "speed": 0.4,
                        "posture": None,
                        "zones": None,
                        "activities": None,
                        "frame_index": k,
                        "elapsed_s": 0.0,
                    }
                )
        if rows:
            store.write_observations(rows)
        return store, origin

    def test_distinct_entities_matches_a_hand_count(self, tmp_path) -> None:
        store, origin = self.build(tmp_path, [3, 0, 5])
        try:
            series = bucket_series(
                store._require(),
                Metric.ENTITIES,
                since=origin,
                until=origin + 3 * HOUR,
                interval_s=HOUR,
            )
            assert [b.value for b in series.buckets] == [3.0, 0.0, 5.0]
            assert series.buckets[1].empty
        finally:
            store.close()

    def test_moving_fraction_counts_rows_with_no_motion_in_the_denominator(
        self, tmp_path
    ) -> None:
        store, origin = self.build(tmp_path, [4])
        try:
            series = bucket_series(
                store._require(),
                Metric.MOVING_FRACTION,
                since=origin,
                until=origin + HOUR,
                interval_s=HOUR,
            )
            assert series.buckets[0].value == pytest.approx(0.5)
        finally:
            store.close()

    def test_an_absurd_bucket_count_is_refused_rather_than_attempted(self, tmp_path) -> None:
        store, origin = self.build(tmp_path, [1])
        try:
            with pytest.raises(VantageError, match="ceiling"):
                bucket_series(
                    store._require(),
                    Metric.ENTITIES,
                    since=origin,
                    until=origin + 365 * 86400,
                    interval_s=1.0,
                )
        finally:
            store.close()

    def test_an_inverted_window_is_refused(self, tmp_path) -> None:
        store, origin = self.build(tmp_path, [1])
        try:
            with pytest.raises(VantageError, match="empty or inverted"):
                bucket_series(
                    store._require(),
                    Metric.ENTITIES,
                    since=origin + HOUR,
                    until=origin,
                    interval_s=HOUR,
                )
        finally:
            store.close()

    def test_heartbeats_round_trip_through_the_store(self, tmp_path) -> None:
        store = SqliteStore(tmp_path / "hb.db")
        try:
            now = time.time()
            store.write_heartbeats(
                [{"camera_id": "cam0", "timestamp": now + i} for i in range(5)]
            )
            assert len(store.heartbeats(now - 1, now + 10)) == 5
            assert store.heartbeats(now + 100, now + 200) == []
        finally:
            store.close()

    def test_pruning_covers_heartbeats(self, tmp_path) -> None:
        """A table only the retention policy forgot is the one that fills a disk."""
        store = SqliteStore(tmp_path / "p.db")
        try:
            old = time.time() - 10 * 86400
            store.write_heartbeats([{"camera_id": "cam0", "timestamp": old}])
            removed = store.prune(time.time())
            assert removed["heartbeat"] == 1
        finally:
            store.close()

    def test_the_engine_refuses_a_store_with_no_sql_connection(self) -> None:
        engine = AnalyticsEngine(object())
        with pytest.raises(ConfigError, match="SQL connection"):
            engine.series(Metric.ENTITIES, since=0.0, until=HOUR)


class TestParams:
    def test_training_on_ambiguous_buckets_without_inference_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="infer_zeros"):
            AnalyticsParams(include_empty_in_training=True, infer_zeros=False)

    def test_defaults_are_internally_consistent(self) -> None:
        params = AnalyticsParams()
        assert params.infer_zeros
        assert not params.include_empty_in_training


class TestSummary:
    def test_a_low_coverage_window_says_so(self) -> None:
        report = summarise_series(make_series([5.0] + [None] * 30))
        assert report.warnings
        assert any("of the window has any data" in w for w in report.warnings)

    def test_nothing_recorded_is_stated_rather_than_dressed_up(self) -> None:
        report = summarise_series(make_series([None] * 10))
        assert "nothing recorded" in report.headline

    def test_judging_nothing_is_not_reported_as_a_clean_result(self) -> None:
        """'No anomalies from 0 judged buckets' must not read as good news."""
        series = make_series([4.0, 4.0])
        result = detect(series, learn(series, period_hours=24))
        report = summarise_analysis(result)
        assert any("absent" in w or "not compared" in w for w in report.warnings)

    def test_below_baseline_anomalies_carry_the_camera_caveat(self) -> None:
        values = [30.0 + (i % 4) for i in range(24 * 28)]
        values[24 * 27 + 8] = 0.0
        series = make_series(values, samples=30)
        result = detect(series, learn(series, period_hours=24))
        text = summarise_analysis(result).describe()
        assert "stopped seeing people" in text

    def test_the_report_is_ascii(self) -> None:
        """The Windows console encodes cp1252 and raises on anything else.

        Arrow glyphs in the anomaly description crashed the summary command on
        the platform this project is developed on.
        """
        values = [10.0] * (24 * 28)
        values[24 * 27 + 5] = 90.0
        series = make_series(values)
        result = detect(series, learn(series, period_hours=24))
        summarise_analysis(result).describe().encode("cp1252")


class TestEvaluationHarness:
    def test_every_scenario_passes(self) -> None:
        report = evaluate()
        assert report.passed, report.describe()

    def test_the_harness_would_fail_a_detector_that_flags_everything(self) -> None:
        """Guards the guard.

        A harness that only counted detections would score a flag-everything
        detector perfectly. False positives are counted separately and are a
        failure, which is what makes the silent scenarios worth running.
        """
        report = evaluate()
        silent = [s for s in report.scores if s.planted == 0]
        assert silent, "no scenario expects silence, so precision is untested"
        assert all(s.false_positives == 0 for s in silent)

    def test_a_history_too_short_to_learn_from_judges_nothing(self) -> None:
        report = evaluate()
        short = next(s for s in report.scores if s.name == "too_little_history")
        assert short.judged == 0
        assert short.false_positives == 0


class TestConfigValidationIsWiredUp:
    """Guards a failure mode that silently disabled every storage check.

    While adding the heartbeat setting, a programmatic edit dedented
    ``StorageConfig.__post_init__`` out of its class and into module scope. The
    file still imported, ruff still formatted it, mypy still passed, and every
    validation rule in it stopped running - a config with an empty path, a zero
    batch size or an inverted retention order was accepted in silence.

    One unrelated test happened to catch it. These make that deliberate.
    """

    def test_no_dataclass_has_an_orphaned_post_init(self) -> None:
        import inspect
        from dataclasses import is_dataclass

        import vantage.config.schema as schema

        orphaned = []
        for name, obj in vars(schema).items():
            if not (inspect.isclass(obj) and is_dataclass(obj)):
                continue
            source = inspect.getsource(obj)
            if "ConfigError" in source and "def __post_init__" not in source:
                orphaned.append(name)
        assert not orphaned, (
            f"{orphaned} raise ConfigError but define no __post_init__, so their "
            "validation never runs. This is what a dedented method looks like."
        )

    def test_storage_config_still_rejects_what_it_documents(self) -> None:
        from vantage.config.schema import StorageConfig

        for kwargs in (
            {"path": "   "},
            {"batch_size": 0},
            {"observation_interval": 0},
            {"flush_interval_s": 0},
            {"heartbeat_interval_s": 0},
            {"retention_days": 30, "event_retention_days": 7},
        ):
            with pytest.raises(ConfigError):
                StorageConfig(**kwargs)

    def test_analytics_config_still_rejects_what_it_documents(self) -> None:
        from vantage.config.schema import AnalyticsConfig

        for kwargs in (
            {"interval_s": 0},
            {"period_hours": 72},
            {"sensitivity": 0},
            {"training_span_s": -1},
            {"zero_reach": 0},
        ):
            with pytest.raises(ConfigError):
                AnalyticsConfig(**kwargs)
