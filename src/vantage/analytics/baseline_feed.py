"""Live baseline-to-rule feedback feed.

Periodically computes and caches current slot baselines (Median & MAD)
from stored history, and provides expected values and anomaly scores to
the live event rule engine.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from vantage.analytics.contracts import Baseline, Metric, slot_index
from vantage.analytics.engine import AnalyticsEngine, AnalyticsParams
from vantage.core.logging import get_logger

if TYPE_CHECKING:
    from vantage.storage.contracts import Store

log = get_logger(__name__)


@dataclass(slots=True)
class BaselineExpectation:
    """Expected baseline values for a specific timestamp slot."""

    center: float
    spread: float
    samples: int
    is_known: bool

    @property
    def upper_bound(self) -> float:
        return self.center + self.spread


class BaselineFeed:
    """Provides cached historical baselines to live rule evaluators."""

    def __init__(
        self,
        store: Store | None = None,
        *,
        params: AnalyticsParams | None = None,
        refresh_interval_s: float = 1800.0,  # 30 mins
    ) -> None:
        self._store = store
        self._params = params or AnalyticsParams()
        self._refresh_interval_s = refresh_interval_s
        self._lock = threading.Lock()
        self._baselines: dict[tuple[Metric, str], Baseline] = {}
        self._last_refresh = 0.0

    def refresh(self, now: float | None = None) -> None:
        """Query storage and update all cached baselines."""
        if self._store is None:
            return

        current_time = now if now is not None else time.time()
        engine = AnalyticsEngine(self._store, params=self._params)

        with self._lock:
            try:
                # 1. Global entity baseline
                self._baselines[(Metric.ENTITIES, "*")] = engine.baseline(
                    Metric.ENTITIES,
                    until=current_time,
                )
                self._last_refresh = current_time
                log.debug(
                    "baseline feed refreshed", extra={"vantage_fields": {"now": current_time}}
                )
            except Exception as e:
                log.warning(
                    "baseline feed refresh failed", extra={"vantage_fields": {"error": str(e)}}
                )

    def get_expectation(
        self,
        metric: Metric,
        timestamp: float,
        zone: str | None = None,
    ) -> BaselineExpectation:
        """Lookup expected center and spread for a timestamp."""
        key = (metric, zone or "*")
        with self._lock:
            baseline = self._baselines.get(key)
            if baseline is None:
                # Fallback to wildcard
                baseline = self._baselines.get((metric, "*"))

        if baseline is None:
            return BaselineExpectation(center=0.0, spread=0.0, samples=0, is_known=False)

        dt = datetime.fromtimestamp(timestamp, UTC)
        slot = slot_index(dt, baseline.period_hours)
        slot_data = baseline.slots.get(slot)

        if slot_data is None or slot_data.samples < 3:
            samples = slot_data.samples if slot_data is not None else 0
            center = slot_data.centre if slot_data is not None else 0.0
            spread = slot_data.spread if slot_data is not None else 0.0
            return BaselineExpectation(
                center=center,
                spread=spread,
                samples=samples,
                is_known=False,
            )

        return BaselineExpectation(
            center=slot_data.centre,
            spread=slot_data.spread,
            samples=slot_data.samples,
            is_known=True,
        )

    def check_anomaly(
        self,
        metric: Metric,
        value: float,
        timestamp: float,
        zone: str | None = None,
        multiplier: float = 2.0,
    ) -> tuple[bool, float, dict[str, Any]]:
        """Check if a measured value exceeds historical expectation."""
        exp = self.get_expectation(metric, timestamp, zone)
        if not exp.is_known:
            return False, 0.0, {"known": False, "samples": exp.samples}

        threshold = exp.center + multiplier * max(exp.spread, 1.0)
        is_anom = value > threshold
        score = (value - exp.center) / max(exp.spread, 1.0) if exp.spread > 0 else 0.0

        details = {
            "known": True,
            "measured": value,
            "expected_center": round(exp.center, 2),
            "expected_spread": round(exp.spread, 2),
            "samples": exp.samples,
            "threshold": round(threshold, 2),
            "score": round(score, 2),
        }
        return is_anom, score, details
