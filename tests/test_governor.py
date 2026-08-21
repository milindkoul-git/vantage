"""Adaptive load shedding.

The governor decides how often analysis runs, from measured cost against the
real frame budget. Its correctness is arithmetic plus a state machine, so all of
it is checkable without a camera or a model.
"""

from __future__ import annotations

import dataclasses

import pytest

from vantage.core.errors import ConfigError
from vantage.core.governor import GovernorParams, LoadGovernor


def settle(governor: LoadGovernor, cost_ms: float, budget_ms: float, seconds: float) -> int:
    """Hold a load for a while, one observation per notional frame."""
    step = budget_ms / 1000.0 or 0.03
    for _ in range(max(1, int(seconds / step))):
        governor.observe(cost_ms, budget_ms, step)
    return governor.interval


class TestArithmetic:
    def test_within_budget_stays_at_the_base_interval(self) -> None:
        governor = LoadGovernor(base_interval=1)
        assert settle(governor, cost_ms=10.0, budget_ms=33.3, seconds=5.0) == 1

    def test_over_budget_raises_to_the_computed_interval(self) -> None:
        """ceil(cost / (budget * headroom)): 60 / (33.3 * 0.7) = 2.57 -> 3."""
        governor = LoadGovernor(base_interval=1, params=GovernorParams(raise_after_s=0.2))
        assert settle(governor, cost_ms=60.0, budget_ms=33.3, seconds=3.0) == 3

    def test_the_interval_lands_in_one_step(self) -> None:
        """A hunting controller would take seconds and oscillate around it."""
        governor = LoadGovernor(base_interval=1, params=GovernorParams(raise_after_s=0.2))
        before = governor.stats.raises
        settle(governor, cost_ms=200.0, budget_ms=33.3, seconds=1.0)
        assert governor.stats.raises == before + 1

    def test_headroom_is_respected(self) -> None:
        """A stricter headroom demands a longer interval for the same cost."""
        loose = LoadGovernor(1, GovernorParams(headroom=1.0, raise_after_s=0.2))
        tight = LoadGovernor(1, GovernorParams(headroom=0.5, raise_after_s=0.2))
        assert settle(loose, 60.0, 33.3, 3.0) < settle(tight, 60.0, 33.3, 3.0)

    def test_the_ceiling_is_respected(self) -> None:
        governor = LoadGovernor(1, GovernorParams(max_interval=3, raise_after_s=0.2))
        assert settle(governor, cost_ms=5000.0, budget_ms=33.3, seconds=3.0) == 3

    def test_reaching_the_ceiling_is_reported(self, caplog) -> None:
        """Absorbing it silently would hide that analysis still cannot keep up."""
        import logging

        caplog.set_level(logging.WARNING, logger="vantage")
        governor = LoadGovernor(1, GovernorParams(max_interval=2, raise_after_s=0.2))
        settle(governor, cost_ms=5000.0, budget_ms=33.3, seconds=3.0)
        assert any("ceiling" in record.message for record in caplog.records)

    def test_never_goes_below_the_configured_interval(self) -> None:
        """The operator asked for every 4th frame; free capacity is not licence
        to override that."""
        governor = LoadGovernor(base_interval=4, params=GovernorParams(lower_after_s=0.2))
        assert settle(governor, cost_ms=1.0, budget_ms=100.0, seconds=5.0) == 4

    def test_no_budget_means_no_decision(self) -> None:
        """The first frame has no gap to measure against."""
        governor = LoadGovernor(1)
        assert governor.observe(cost_ms=100.0, budget_ms=0.0, elapsed_s=0.03) == 1

    def test_no_cost_yet_means_no_decision(self) -> None:
        governor = LoadGovernor(1)
        assert governor.observe(cost_ms=0.0, budget_ms=33.3, elapsed_s=0.03) == 1


class TestHysteresis:
    def test_a_brief_spike_does_not_raise(self) -> None:
        """One slow frame is not a load."""
        governor = LoadGovernor(1, GovernorParams(raise_after_s=1.0))
        for _ in range(3):
            governor.observe(500.0, 33.3, 0.033)
        assert governor.interval == 1

    def test_sustained_load_does_raise(self) -> None:
        governor = LoadGovernor(1, GovernorParams(raise_after_s=0.5))
        assert settle(governor, 60.0, 33.3, seconds=2.0) > 1

    def test_recovery_is_slower_than_degradation(self) -> None:
        """Lowering early puts the pipeline back into the overload it escaped."""
        params = GovernorParams(raise_after_s=0.3, lower_after_s=5.0)
        governor = LoadGovernor(1, params)
        settle(governor, 60.0, 33.3, seconds=2.0)
        raised = governor.interval
        assert raised > 1

        settle(governor, 5.0, 33.3, seconds=1.0)  # briefly cheap
        assert governor.interval == raised

    def test_sustained_headroom_lowers_the_interval(self) -> None:
        params = GovernorParams(raise_after_s=0.3, lower_after_s=1.0)
        governor = LoadGovernor(1, params)
        settle(governor, 60.0, 33.3, seconds=2.0)
        raised = governor.interval
        assert settle(governor, 2.0, 33.3, seconds=10.0) < raised

    def test_recovery_steps_down_rather_than_jumping(self) -> None:
        """Returning the full load in one frame is how a controller oscillates."""
        params = GovernorParams(raise_after_s=0.2, lower_after_s=0.5, max_interval=8)
        governor = LoadGovernor(1, params)
        settle(governor, 400.0, 33.3, seconds=2.0)
        high = governor.interval
        assert high >= 4

        step = 0.0333
        for _ in range(int(0.6 / step)):
            governor.observe(1.0, 33.3, step)
        assert governor.interval == high - 1

    def test_alternating_load_does_not_thrash(self) -> None:
        params = GovernorParams(raise_after_s=1.0, lower_after_s=6.0)
        governor = LoadGovernor(1, params)
        for index in range(600):
            governor.observe(60.0 if index % 2 else 5.0, 33.3, 0.0333)
        # Twenty seconds of alternating load; a controller without dead bands
        # would have changed state dozens of times.
        assert governor.stats.raises + governor.stats.lowers <= 4


class TestReporting:
    def test_degraded_time_is_accumulated(self) -> None:
        governor = LoadGovernor(1, GovernorParams(raise_after_s=0.2))
        settle(governor, 60.0, 33.3, seconds=3.0)
        assert governor.stats.degraded_s > 1.0

    def test_peak_is_remembered_after_recovery(self) -> None:
        """A run that coped only by degrading must not look like one that never did."""
        params = GovernorParams(raise_after_s=0.2, lower_after_s=0.3)
        governor = LoadGovernor(1, params)
        settle(governor, 300.0, 33.3, seconds=2.0)
        peak = governor.stats.peak_interval
        settle(governor, 1.0, 33.3, seconds=20.0)
        assert governor.interval == 1
        assert governor.stats.peak_interval == peak > 1

    def test_stats_are_serialisable(self) -> None:
        import json

        json.dumps(LoadGovernor(1).stats.to_dict())

    def test_changes_are_logged(self, caplog) -> None:
        """Silently changing how much of reality is analysed makes later
        results inexplicable."""
        import logging

        caplog.set_level(logging.INFO, logger="vantage")
        governor = LoadGovernor(1, GovernorParams(raise_after_s=0.2))
        settle(governor, 60.0, 33.3, seconds=2.0)
        assert any("interval changed" in record.message for record in caplog.records)

    def test_reset_returns_to_base(self) -> None:
        governor = LoadGovernor(2, GovernorParams(raise_after_s=0.2))
        settle(governor, 300.0, 33.3, seconds=2.0)
        governor.reset()
        assert governor.interval == 2 and governor.stats.peak_interval == 2

    def test_peak_is_never_below_the_base_interval(self) -> None:
        """Regression: peak defaulted to 1, so a governor told to analyse every
        4th frame reported a peak below its own floor."""
        assert LoadGovernor(4).stats.peak_interval == 4


class TestParams:
    def test_headroom_must_be_a_fraction(self) -> None:
        with pytest.raises(ConfigError, match="headroom"):
            GovernorParams(headroom=1.5)

    def test_ceiling_below_base_is_refused(self) -> None:
        with pytest.raises(ConfigError, match="max_interval"):
            LoadGovernor(base_interval=5, params=GovernorParams(max_interval=2))

    def test_base_interval_must_be_positive(self) -> None:
        with pytest.raises(ConfigError):
            LoadGovernor(base_interval=0)

    def test_config_guard_catches_the_same_inversion(self) -> None:
        from vantage.config.schema import (
            AdaptiveConfig,
            AppConfig,
            DetectionConfig,
            VantageConfig,
        )

        with pytest.raises(ConfigError, match="max_interval"):
            VantageConfig(
                app=AppConfig(adaptive=AdaptiveConfig(max_interval=2)),
                detection=DetectionConfig(interval=5),
            )


class TestPipelineIntegration:
    """The behaviour that matters: a live source keeps up by analysing less."""

    def build(self, *, adaptive: bool, cost_ms: float):
        import time

        from tests.fakes import make_engine
        from vantage.config.schema import (
            AdaptiveConfig,
            AppConfig,
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        engine, _ = make_engine()

        class Slow:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def detect(self, frame):
                time.sleep(cost_ms / 1000.0)
                return self._inner.detect(frame)

        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&live=true"),
            ingest=IngestConfig(max_frames=60),
            app=AppConfig(
                adaptive=AdaptiveConfig(enabled=adaptive, raise_after_s=0.2),
                resource_interval_s=0,
            ),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        return config, Slow(engine)

    def test_a_slow_detector_raises_the_interval(self) -> None:
        from vantage.app import run_ingestion

        config, engine = self.build(adaptive=True, cost_ms=40.0)
        result = run_ingestion(config, engine=engine)
        assert result.adaptive["peak_interval"] > 1
        assert result.frames == 60

    def test_a_fast_detector_leaves_it_alone(self) -> None:
        """A detector that costs nothing must not drive sustained escalation.

        Asserting ``peak_interval == 1`` was wrong, and flaky about one run in
        three. The governor measures real wall-clock analysis cost, so a busy
        machine can push one frame over budget even when the detector itself is
        free - and raising the interval in response is the governor working, not
        failing. The test was asserting that the machine running it was idle.

        What is actually guaranteed is that a free detector produces no
        *sustained* pressure: at most a single step in response to transient
        load, and the interval comes back down.
        """
        from vantage.app import run_ingestion

        config, engine = self.build(adaptive=True, cost_ms=0.0)
        result = run_ingestion(config, engine=engine)
        adaptive = result.adaptive

        assert adaptive["peak_interval"] <= 2, (
            f"a zero-cost detector escalated to {adaptive['peak_interval']}, which "
            "is more than transient machine load can explain"
        )
        assert adaptive["interval"] == 1, (
            "the interval did not return to 1 after the load passed, which is a "
            "governor that ratchets rather than adapts"
        )

    def test_disabled_means_no_governor_at_all(self) -> None:
        from vantage.app import run_ingestion

        config, engine = self.build(adaptive=False, cost_ms=40.0)
        result = run_ingestion(config, engine=engine)
        assert result.adaptive == {}

    def test_a_recorded_source_is_never_governed(self) -> None:
        """No deadline to miss: shedding load would discard data for nothing."""
        from vantage.app import run_ingestion

        config, engine = self.build(adaptive=True, cost_ms=40.0)
        config = dataclasses.replace(
            config,
            source=dataclasses.replace(
                config.source, uri="synthetic://?width=160&height=120&fps=30"
            ),
        )
        result = run_ingestion(config, engine=engine)
        assert result.adaptive.get("peak_interval", 1) == 1

    def test_degradation_appears_in_the_summary(self) -> None:
        from vantage.app import run_ingestion

        config, engine = self.build(adaptive=True, cost_ms=40.0)
        result = run_ingestion(config, engine=engine)
        assert "adaptive:" in result.summary()
