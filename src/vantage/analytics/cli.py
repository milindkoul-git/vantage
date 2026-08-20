"""The ``vantage analytics`` command.

Five actions, and the split between them is the useful part:

``summary``    what happened in a window, in sentences
``series``     the buckets themselves, for a chart or a spreadsheet
``anomalies``  what did not look like the history behind it
``eval``       score the detector against histories with known answers
``characterise``  measure false-alarm rate and detection power on random data

The last two exist because the numbers in the README have to come from
somewhere a reader can re-run. ``eval`` is the pass/fail regression suite;
``characterise`` is the honest performance picture, and they disagree in a way
that matters - see the note at the bottom of :mod:`vantage.analytics.evaluation`.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from vantage.analytics.contracts import Metric
from vantage.analytics.engine import AnalyticsEngine, AnalyticsParams
from vantage.analytics.summary import summarise_analysis, summarise_series
from vantage.core.errors import VantageError
from vantage.core.logging import get_logger

log = get_logger(__name__)


def parse_window(text: str) -> float:
    """Parse ``24h``, ``7d``, ``90m`` into seconds.

    Shared with the history command's spelling on purpose: a person who learned
    ``--since 24h`` there should not discover a different vocabulary here.
    """
    text = text.strip().lower()
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}
    if text and text[-1] in units:
        try:
            return float(text[:-1]) * units[text[-1]]
        except ValueError as exc:
            raise VantageError(f"could not read a duration from {text!r}") from exc
    try:
        return float(text)
    except ValueError as exc:
        raise VantageError(
            f"could not read a duration from {text!r}. Use forms like 90m, 24h, 7d, 4w."
        ) from exc


def run(args: argparse.Namespace) -> int:
    if args.action == "eval":
        return _eval(args)
    if args.action == "characterise":
        return _characterise(args)
    return _query(args)


def _eval(args: argparse.Namespace) -> int:
    from vantage.analytics.evaluation import SCENARIOS, evaluate

    scenarios = SCENARIOS
    if getattr(args, "scenarios", None):
        wanted = {name.strip() for name in args.scenarios.split(",")}
        scenarios = tuple(s for s in SCENARIOS if s.name in wanted)
        if not scenarios:
            print(f"No scenario matched {args.scenarios!r}.")
            print("Available: " + ", ".join(s.name for s in SCENARIOS))
            return 1

    report = evaluate(scenarios)
    if args.json:
        print(
            json.dumps(
                {
                    "passed": report.passed,
                    "found": report.total_found,
                    "planted": report.total_planted,
                    "false_positives": report.total_false,
                    "scenarios": [
                        {
                            "name": s.name,
                            "found": s.found,
                            "missed": s.missed,
                            "false_positives": s.false_positives,
                            "judged": s.judged,
                        }
                        for s in report.scores
                    ],
                },
                indent=2,
            )
        )
    else:
        print(report.describe())
        print()
        print("PASSED" if report.passed else "FAILED")
    return 0 if report.passed else 1


def _characterise(args: argparse.Namespace) -> int:
    from vantage.analytics.evaluation import characterise

    result = characterise(trials=args.trials, training_weeks=args.weeks)
    if args.json:
        print(
            json.dumps(
                {
                    "false_alarms_per_week": result.false_alarms_per_week,
                    "worst_week": result.worst_week,
                    "trials": result.trials,
                    "training_weeks": result.training_weeks,
                    "detection": [
                        {"relative_size": size, "detected": rate}
                        for size, rate in result.detection
                    ],
                },
                indent=2,
            )
        )
    else:
        print(result.describe())
    return 0


def _open_store(args: argparse.Namespace):
    from vantage.storage.sqlite_store import SqliteStore

    return SqliteStore(args.db, read_only=True)


def _query(args: argparse.Namespace) -> int:
    try:
        metric = Metric(args.metric)
    except ValueError:
        print(
            f"Unknown metric {args.metric!r}. Available: " + ", ".join(m.value for m in Metric)
        )
        return 1

    span = parse_window(args.since)
    interval = parse_window(args.interval)
    until = time.time()
    since = until - span

    store = _open_store(args)
    try:
        engine = AnalyticsEngine(
            store,
            params=AnalyticsParams(
                interval_s=interval,
                period_hours=args.period,
                sensitivity=args.sensitivity,
                training_span_s=parse_window(args.train),
                judge_empty=args.judge_empty,
            ),
        )

        if args.action == "series":
            series = engine.series(
                metric,
                since=since,
                until=until,
                camera_id=args.camera,
                zone=args.zone,
            )
            if args.json:
                print(
                    json.dumps(
                        {
                            "metric": metric.value,
                            "interval_s": series.interval_s,
                            "coverage": series.coverage,
                            "buckets": [
                                {
                                    "start": b.start,
                                    "value": b.value,
                                    "samples": b.samples,
                                }
                                for b in series
                            ],
                        },
                        indent=2,
                    )
                )
            else:
                print(series.describe())
                print()
                for bucket in series:
                    print(f"  {bucket.describe()}")
            return 0

        result = engine.analyse(
            metric,
            since=since,
            until=until,
            camera_id=args.camera,
            zone=args.zone,
            train_on_window=args.self_compare,
        )

        if args.json:
            print(json.dumps(_as_json(result), indent=2))
            return 0

        if args.action == "anomalies":
            if not result.anomalies:
                print(result.describe())
                return 0
            print(f"{len(result.anomalies)} anomalies:\n")
            for anomaly in sorted(result.anomalies, key=lambda a: -a.score):
                print(f"  {anomaly.describe()}")
            return 0

        print(summarise_analysis(result).describe())
        return 0
    finally:
        store.close()


def _as_json(result) -> dict[str, Any]:
    return {
        "metric": result.series.metric.value,
        "coverage": result.series.coverage,
        "buckets": len(result.series),
        "judged": result.judged,
        "unjudged": result.skipped_untrained,
        "baseline": (
            {
                "slots_learned": result.baseline.trustworthy_slots,
                "period_hours": result.baseline.period_hours,
                "coverage": result.baseline.coverage,
            }
            if result.baseline
            else None
        ),
        "anomalies": [
            {
                "start": a.bucket.start,
                "when": a.bucket.when.isoformat(),
                "direction": a.direction.value,
                "observed": a.observed,
                "expected": a.expected,
                "band": [a.band_low, a.band_high],
                "score": a.score,
                "severity": a.severity,
                "baseline_samples": a.samples,
            }
            for a in sorted(result.anomalies, key=lambda a: -a.score)
        ],
    }


__all__ = ["parse_window", "run", "summarise_series"]
