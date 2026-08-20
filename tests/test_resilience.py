"""Stage isolation and resource accounting - the production-readiness pass.

Two things a deployment needs that a benchmark does not: a failing stage must
not take the process down, and a slow leak must be visible before it becomes a
crash. Both are checkable without a camera, a model or a runtime.
"""

from __future__ import annotations

import pytest

from vantage.core.errors import ConfigError
from vantage.core.resilience import StageGuard, StageRegistry
from vantage.core.resources import ResourceSampler, read_rss


def boom() -> None:
    raise RuntimeError("simulated fault")


class TestStageGuard:
    def test_success_passes_the_value_through(self) -> None:
        guard = StageGuard("detection")
        assert guard.run(lambda: 42) == 42
        assert guard.stats.calls == 1
        assert guard.stats.healthy

    def test_failure_returns_the_default_rather_than_raising(self) -> None:
        guard = StageGuard("detection")
        assert guard.run(boom, default="fallback") == "fallback"
        assert guard.stats.failures == 1

    def test_failure_is_counted_and_described(self) -> None:
        """Not silent: the count and the error text both survive."""
        guard = StageGuard("pose")
        guard.run(boom)
        assert guard.stats.last_error.startswith("RuntimeError: simulated fault")
        assert guard.stats.error_types == {"RuntimeError": 1}

    def test_failure_is_logged_with_a_traceback(self, caplog) -> None:
        import logging

        caplog.set_level(logging.WARNING, logger="vantage")
        StageGuard("pose").run(boom)
        assert any("stage failed" in record.message for record in caplog.records)
        assert any(record.exc_info for record in caplog.records)

    def test_a_recovered_stage_resets_its_streak(self) -> None:
        """A bad frame among good ones must not accumulate toward the budget."""
        guard = StageGuard("detection", max_consecutive=3)
        guard.run(boom)
        guard.run(lambda: 1)
        guard.run(boom)
        assert guard.enabled
        assert guard.stats.consecutive == 1
        assert guard.stats.failures == 2

    def test_consecutive_failures_disable_the_stage(self) -> None:
        guard = StageGuard("detection", max_consecutive=3)
        for _ in range(3):
            guard.run(boom)
        assert not guard.enabled
        assert guard.stats.disabled

    def test_a_disabled_stage_is_not_called_again(self) -> None:
        """Continuing to call it would cost latency on every frame."""
        calls = []

        def counted():
            calls.append(1)
            raise RuntimeError("x")

        guard = StageGuard("detection", max_consecutive=2)
        for _ in range(10):
            guard.run(counted)
        assert len(calls) == 2

    def test_disabling_is_logged_as_an_error(self, caplog) -> None:
        import logging

        caplog.set_level(logging.ERROR, logger="vantage")
        guard = StageGuard("detection", max_consecutive=2)
        guard.run(boom)
        guard.run(boom)
        assert any("stage disabled" in record.message for record in caplog.records)

    def test_memory_errors_are_never_swallowed(self) -> None:
        """Skipping a frame does not give memory back; carrying on corrupts results."""

        def out_of_memory():
            raise MemoryError("out of memory")

        with pytest.raises(MemoryError):
            StageGuard("detection").run(out_of_memory)

    def test_keyboard_interrupt_passes_through(self) -> None:
        """That is the operator asking to stop, not a stage misbehaving."""

        def interrupted():
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            StageGuard("detection").run(interrupted)

    def test_failure_rate_is_reported(self) -> None:
        guard = StageGuard("detection", max_consecutive=99)
        for index in range(10):
            guard.run(boom if index % 2 else (lambda: 1))
        assert guard.stats.failure_rate == pytest.approx(0.5)

    def test_reset_re_enables(self) -> None:
        guard = StageGuard("detection", max_consecutive=1)
        guard.run(boom)
        assert not guard.enabled
        guard.reset()
        assert guard.enabled and guard.stats.failures == 0

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            StageGuard("x", max_consecutive=0)


class TestStageRegistry:
    def test_guards_are_reused_by_name(self) -> None:
        registry = StageRegistry()
        assert registry.guard("pose") is registry.guard("pose")
        assert len(registry) == 1

    def test_healthy_until_something_fails(self) -> None:
        registry = StageRegistry()
        registry.guard("pose").run(lambda: 1)
        assert registry.healthy
        registry.guard("pose").run(boom)
        assert not registry.healthy

    def test_disabled_stages_sort_first(self) -> None:
        """The worst news goes at the top of the report."""
        registry = StageRegistry(max_consecutive=2)
        registry.guard("flaky").run(boom)
        for _ in range(2):
            registry.guard("broken").run(boom)
        assert registry.degraded[0].name == "broken"

    def test_summary_is_empty_when_healthy(self) -> None:
        """The absence of the line is the healthy signal."""
        registry = StageRegistry()
        registry.guard("pose").run(lambda: 1)
        assert registry.summary() == ""

    def test_snapshot_is_serialisable(self) -> None:
        import json

        registry = StageRegistry()
        registry.guard("pose").run(boom)
        json.dumps(registry.to_dict())


class TestResourceSampler:
    def test_cpu_is_reported_in_cores(self) -> None:
        """A fraction of a core, not a percentage that means different things
        on different hardware."""
        sampler = ResourceSampler()
        # Burn a measurable slice of CPU so the reading is not all zero.
        total = sum(index * index for index in range(200_000))
        assert total > 0
        sample = sampler.sample()
        assert 0.0 <= sample.cpu_cores <= 64.0

    def test_memory_is_a_number_or_explicitly_none(self) -> None:
        """Never a silent zero, which would make a leak look like health."""
        rss = read_rss()
        assert rss is None or rss > 0

    def test_growth_is_measured_from_construction(self) -> None:
        """Not from the first sample: with the models already loaded, that
        baseline made a run that released them report negative growth."""
        sampler = ResourceSampler()
        held = [bytearray(2_000_000) for _ in range(20)]
        sample = sampler.sample()
        if sample.rss_bytes is None:
            pytest.skip("memory reporting unavailable on this platform")
        assert sample.growth_bytes is not None
        assert sample.growth_bytes > 10_000_000
        del held

    def test_total_covers_the_whole_life(self) -> None:
        sampler = ResourceSampler()
        sampler.sample()
        total = sampler.total()
        assert total.elapsed_s >= 0.0

    def test_sample_is_serialisable(self) -> None:
        import json

        json.dumps(ResourceSampler().sample().to_dict())

    def test_describe_says_when_memory_is_unavailable(self) -> None:
        from vantage.core.resources import ResourceSample

        sample = ResourceSample(1.0, None, None, None, 1.0)
        assert "unavailable" in sample.describe()


class TestConfig:
    def test_failure_budget_must_allow_one_bad_frame(self) -> None:
        from vantage.config.schema import AppConfig

        with pytest.raises(ConfigError, match=r"stage_failure_budget"):
            AppConfig(stage_failure_budget=0)

    def test_resource_interval_may_be_disabled(self) -> None:
        from vantage.config.schema import AppConfig

        assert AppConfig(resource_interval_s=0).resource_interval_s == 0

    def test_negative_resource_interval_is_refused(self) -> None:
        from vantage.config.schema import AppConfig

        with pytest.raises(ConfigError):
            AppConfig(resource_interval_s=-1.0)


class TestRunLoopIsolation:
    """The behaviour that matters: a failing stage does not stop the run."""

    def config(self):
        from vantage.config.schema import (
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        return VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=30&frames=40"),
            ingest=IngestConfig(max_frames=20),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )

    def flaky_engine(self, mode: str):
        from tests.fakes import make_engine

        engine, _ = make_engine()

        class Flaky:
            def __init__(self, inner):
                self._inner = inner
                self.calls = 0

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def detect(self, frame):
                self.calls += 1
                if mode == "always" or self.calls % 3 == 0:
                    raise RuntimeError("simulated inference fault")
                return self._inner.detect(frame)

        return Flaky(engine)

    def test_intermittent_failure_does_not_stop_the_run(self) -> None:
        from vantage.app import run_ingestion

        result = run_ingestion(self.config(), engine=self.flaky_engine("intermittent"))
        assert result.frames == 20
        health = result.stage_health["detection"]
        assert health["failures"] > 0
        assert not health["disabled"]
        assert result.detections_run > 0

    def test_persistent_failure_disables_the_stage_but_finishes(self) -> None:
        from vantage.app import run_ingestion

        result = run_ingestion(self.config(), engine=self.flaky_engine("always"))
        assert result.frames == 20
        health = result.stage_health["detection"]
        assert health["disabled"]
        assert result.detections_run == 0

    def test_degradation_is_in_the_summary(self) -> None:
        """A degraded run must not read as a healthy one."""
        from vantage.app import run_ingestion

        result = run_ingestion(self.config(), engine=self.flaky_engine("always"))
        assert "DEGRADED" in result.summary()

    def test_a_healthy_run_says_nothing_about_stages(self) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion

        engine, _ = make_engine()
        result = run_ingestion(self.config(), engine=engine)
        assert "DEGRADED" not in result.summary()
        assert all(not s["failures"] for s in result.stage_health.values())


class TestSoak:
    """Memory over a long run. The failure mode of a 24/7 vision process."""

    @pytest.mark.slow
    def test_a_long_run_does_not_grow_without_bound(self) -> None:
        from tests.fakes import make_engine
        from vantage.app import run_ingestion
        from vantage.config.schema import (
            DetectionConfig,
            DisplayConfig,
            IngestConfig,
            SourceConfig,
            TrackingConfig,
            VantageConfig,
        )

        if read_rss() is None:
            pytest.skip("memory reporting unavailable on this platform")

        engine, _ = make_engine()
        config = VantageConfig(
            source=SourceConfig(uri="synthetic://?width=160&height=120&fps=60&frames=4000"),
            ingest=IngestConfig(max_frames=2000),
            detection=DetectionConfig(enabled=True),
            tracking=TrackingConfig(enabled=True),
            display=DisplayConfig(enabled=False),
        )
        result = run_ingestion(config, engine=engine)

        assert result.frames == 2000
        growth_mb = result.resources.get("growth_mb")
        assert growth_mb is not None
        # Generous, because interpreters allocate arenas and the first frames
        # populate caches. What this catches is per-frame retention: at 2000
        # frames a leak of even 20 kB a frame would be 40 MB.
        assert growth_mb < 60.0, f"grew {growth_mb} MB over 2000 frames"
