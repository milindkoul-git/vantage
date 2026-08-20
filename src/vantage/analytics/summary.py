"""Turning series into sentences.

The numbers are already correct by the time they reach this module. What it adds
is the thing a person actually asked for - "what happened yesterday" - in a form
that can be read in one pass.

Two rules govern everything here.

**Never state a figure without the evidence behind it.** "Busiest at 14:00" is a
different claim depending on whether the day was 95% covered or 12% covered, so
coverage appears in the report rather than in a footnote.

**Say when there is nothing to say.** A summariser that produces confident prose
from four buckets of data is worse than one that says it has four buckets. Every
path here has a low-data branch, and those branches state the shortfall instead
of hedging the prose.
"""

from __future__ import annotations

from dataclasses import dataclass

from vantage.analytics.contracts import (
    SECONDS_PER_HOUR,
    AnalysisResult,
    Direction,
    Metric,
    Series,
)

MIN_BUCKETS_FOR_PROSE = 6
"""Below this many occupied buckets, the report states the shortfall and stops.

Six is not a magic number; it is roughly the point below which "busiest hour"
and "quietest hour" are the same observation said twice.
"""


@dataclass(frozen=True, slots=True)
class Report:
    """A readable account of one analysis."""

    headline: str
    lines: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def describe(self) -> str:
        parts = [self.headline, ""]
        parts.extend(f"  {line}" for line in self.lines)
        if self.warnings:
            parts.append("")
            parts.extend(f"  ! {warning}" for warning in self.warnings)
        return "\n".join(parts)


def _fmt(value: float, metric: Metric) -> str:
    if metric is Metric.MOVING_FRACTION:
        return f"{value:.0%}"
    if metric is Metric.MEAN_SPEED:
        return f"{value:.2f} h/s"
    return f"{value:,.0f}"


def _coverage_warnings(series: Series) -> list[str]:
    warnings: list[str] = []
    coverage = series.coverage
    if coverage < 0.25:
        warnings.append(
            f"Only {coverage:.0%} of the window has any data "
            f"({len(series.occupied)} of {len(series)} buckets). "
            "Treat everything above as a description of those buckets, not of the period."
        )
    elif coverage < 0.7:
        warnings.append(
            f"{coverage:.0%} coverage - the system was not recording for much of this "
            "window, so totals understate reality by an unknown amount."
        )
    return warnings


def summarise_series(series: Series) -> Report:
    """Describe one series without reference to a baseline."""
    span_h = (series.until - series.since) / SECONDS_PER_HOUR
    occupied = series.occupied

    if not occupied:
        return Report(
            headline=f"{series.metric.label}: nothing recorded",
            lines=(
                f"Window covers {span_h:.1f}h in {len(series)} buckets, all empty.",
                "Either the pipeline was not running, or storage was disabled.",
            ),
        )

    lines: list[str] = []
    if not series.metric.is_rate:
        lines.append(f"Total: {_fmt(series.total, series.metric)} over {span_h:.1f}h")
    lines.append(
        f"Typical bucket: {_fmt(series.mean(), series.metric)} "
        f"(across {len(occupied)} buckets with data)"
    )

    if len(occupied) < MIN_BUCKETS_FOR_PROSE:
        lines.append(
            f"Too few buckets ({len(occupied)}) to describe a pattern - "
            "peaks and troughs would just be restating the same points."
        )
        return Report(
            headline=f"{series.metric.label}: {len(occupied)} buckets with data",
            lines=tuple(lines),
            warnings=tuple(_coverage_warnings(series)),
        )

    peak = series.peak()
    quiet = series.quietest()
    if peak is not None:
        lines.append(
            f"Busiest: {peak.when.strftime('%a %d %b %H:%M')} "
            f"at {_fmt(peak.value, series.metric)}"
        )
    if quiet is not None and peak is not None and quiet.start != peak.start:
        lines.append(
            f"Quietest (with data): {quiet.when.strftime('%a %d %b %H:%M')} "
            f"at {_fmt(quiet.value, series.metric)}"
        )

    return Report(
        headline=(
            f"{series.metric.label}: {len(series)} buckets of "
            f"{series.interval_s / SECONDS_PER_HOUR:.2g}h, {series.coverage:.0%} covered"
        ),
        lines=tuple(lines),
        warnings=tuple(_coverage_warnings(series)),
    )


def summarise_analysis(result: AnalysisResult) -> Report:
    """Describe a series together with what the baseline made of it."""
    base = summarise_series(result.series)
    lines = list(base.lines)
    warnings = list(base.warnings)

    baseline = result.baseline
    if baseline is None:
        return Report(headline=base.headline, lines=tuple(lines), warnings=tuple(warnings))

    lines.append("")
    lines.append(
        f"Baseline: {baseline.trustworthy_slots} of {baseline.period_hours} slots learned "
        f"({baseline.coverage:.0%} of the {'week' if baseline.period_hours == 168 else 'day'})"
    )

    if result.judged <= 0:
        lines.append("No bucket could be judged: no slot had enough history behind it.")
        warnings.append(
            "Nothing was compared against anything. This is not a clean result - it is "
            "an absent one. Collect more history, or widen the bucket interval so each "
            "slot accumulates faster."
        )
        return Report(headline=base.headline, lines=tuple(lines), warnings=tuple(warnings))

    if not result.anomalies:
        lines.append(f"No anomalies across {result.judged} judged buckets.")
    else:
        above = sum(1 for a in result.anomalies if a.direction is Direction.ABOVE)
        below = len(result.anomalies) - above
        lines.append(
            f"{len(result.anomalies)} anomalies from {result.judged} judged buckets "
            f"({above} above baseline, {below} below):"
        )
        for anomaly in sorted(result.anomalies, key=lambda a: -a.score)[:8]:
            lines.append(f"  {anomaly.describe()}")
        if len(result.anomalies) > 8:
            lines.append(f"  ... and {len(result.anomalies) - 8} more")

        if below:
            lines.append("")
            lines.append(
                f"{below} of these are *below* baseline. A camera that stopped seeing "
                "people looks exactly like this, so rule that out before reading them "
                "as a change in the scene."
            )

    if result.skipped_untrained:
        share = result.skipped_untrained / max(1, len(result.series))
        message = (
            f"{result.skipped_untrained} of {len(result.series)} buckets were not judged "
            "(empty, or their slot had too little history)."
        )
        if share > 0.5:
            warnings.append(
                message + " That is most of the window, so 'no anomalies' here means "
                "'mostly not looked at' rather than 'mostly fine'."
            )
        else:
            lines.append(message)

    return Report(headline=base.headline, lines=tuple(lines), warnings=tuple(warnings))
