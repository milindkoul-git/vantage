"""Tests for BaselineFeed and dynamic adaptive event rules."""

from __future__ import annotations

from vantage.analytics.baseline_feed import BaselineFeed
from vantage.analytics.contracts import Baseline, Metric, Slot
from vantage.events.contracts import Severity
from vantage.events.rules import RuleSpec, SceneContext, evaluate


class FakeSpatialResult:
    def __init__(self, occupancy_map: dict[str, int]) -> None:
        self._occupancy = occupancy_map

    def occupancy(self) -> dict[str, int]:
        return self._occupancy


def test_baseline_feed_expectation() -> None:
    feed = BaselineFeed(store=None)
    slots = {i: Slot(index=i, centre=3.0, spread=1.0, samples=5) for i in range(168)}
    baseline = Baseline(
        metric=Metric.ENTITIES,
        slots=slots,
        period_hours=168,
        interval_s=3600.0,
        trained_from=0.0,
        trained_until=1000.0,
        sensitivity=3.5,
    )
    feed._baselines[(Metric.ENTITIES, "*")] = baseline

    exp = feed.get_expectation(Metric.ENTITIES, timestamp=1700000000.0)
    assert exp.is_known is True
    assert exp.center == 3.0
    assert exp.spread == 1.0

    # Test normal count vs anomalous count
    is_anom, score, _details = feed.check_anomaly(
        Metric.ENTITIES, value=4.0, timestamp=1700000000.0, multiplier=2.0
    )
    assert is_anom is False  # 4 <= 3 + 2*1

    is_anom, score, _details = feed.check_anomaly(
        Metric.ENTITIES, value=12.0, timestamp=1700000000.0, multiplier=2.0
    )
    assert is_anom is True  # 12 > 5.0
    assert score > 0


def test_adaptive_occupancy_rule() -> None:
    feed = BaselineFeed(store=None)
    slots = {i: Slot(index=i, centre=2.0, spread=0.5, samples=6) for i in range(168)}
    feed._baselines[(Metric.ENTITIES, "*")] = Baseline(
        metric=Metric.ENTITIES,
        slots=slots,
        period_hours=168,
        interval_s=3600.0,
        trained_from=0.0,
        trained_until=1000.0,
        sensitivity=3.5,
    )

    spec = RuleSpec(
        type="adaptive_occupancy",
        zones=("lobby",),
        severity=Severity.ALERT,
        factor=2.0,
        min_count=10,
    )

    # 1. Normal occupancy in lobby (2 entities) -> No event
    ctx_normal = SceneContext(
        tracking=None,
        state=None,
        activity=None,
        spatial=FakeSpatialResult({"lobby": 2}),  # type: ignore
        elapsed_s=10.0,
        frame_index=100,
        capture_wall=1700000000.0,
        source_id="cam0",
        baseline_feed=feed,
    )
    events = evaluate(spec, ctx_normal)
    assert len(events) == 0

    # 2. Anomalous occupancy in lobby (8 entities) -> Fires adaptive event
    ctx_anom = SceneContext(
        tracking=None,
        state=None,
        activity=None,
        spatial=FakeSpatialResult({"lobby": 8}),  # type: ignore
        elapsed_s=15.0,
        frame_index=150,
        capture_wall=1700000000.0,
        source_id="cam0",
        baseline_feed=feed,
    )
    events = evaluate(spec, ctx_anom)
    assert len(events) == 1
    assert "Anomalous occupancy" in events[0].summary
    assert events[0].evidence["count"] == 8

    # 3. Fallback when feed is None
    ctx_nofeed = SceneContext(
        tracking=None,
        state=None,
        activity=None,
        spatial=FakeSpatialResult({"lobby": 12}),  # type: ignore
        elapsed_s=20.0,
        frame_index=200,
        capture_wall=1700000000.0,
        source_id="cam0",
        baseline_feed=None,
    )
    events = evaluate(spec, ctx_nofeed)
    assert len(events) == 1
    assert "static fallback" in events[0].summary
