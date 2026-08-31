/**
 * The bucketed history chart.
 *
 * Three things it has to keep distinct, because the analytics engine goes to
 * real trouble to distinguish them and a chart that flattens them is worse than
 * no chart:
 *
 * - a bucket with readings -> a bar of its value
 * - a bucket that was measured and was zero (`known_zero`) -> a bar at the floor
 * - a bucket with no reading at all -> a hatched gap, never a zero
 *
 * The last one is the whole reason the heartbeat table exists. "Nobody was
 * there" and "nothing was recording" are different facts, and drawing both as a
 * flat line at zero is exactly the lie the coverage machinery was built to
 * prevent.
 *
 * Anomalies are drawn as a cap and a dot on top of the existing bar rather than
 * by recolouring it, so the bar still reads as its value first and as a flagged
 * value second.
 */

import React, { useMemo, useState } from 'react';
import type { AnalyticsAnomaly, AnalyticsBucket } from '../../contracts/types';

const NO_DATA = '#6B5545';
const BAR = '#B08D57';
const BAR_ZERO = 'rgba(176,141,87,0.35)';
const ANOMALY = '#B33A2E';

interface Props {
  buckets: AnalyticsBucket[];
  anomalies: AnalyticsAnomaly[];
  intervalSeconds: number;
  unitLabel: string;
  height?: number;
}

const bucketLabel = (start: number, intervalSeconds: number): string => {
  const date = new Date(start * 1000);
  if (intervalSeconds >= 86400) {
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  }
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
};

export const TrendChart: React.FC<Props> = ({
  buckets,
  anomalies,
  intervalSeconds,
  unitLabel,
  height = 200,
}) => {
  const [hover, setHover] = useState<number | null>(null);

  const anomalyByStart = useMemo(() => {
    const map = new Map<number, AnalyticsAnomaly>();
    for (const anomaly of anomalies) map.set(anomaly.start, anomaly);
    return map;
  }, [anomalies]);

  const peak = useMemo(
    () => Math.max(1e-9, ...buckets.map((b) => (b.samples > 0 || b.known_zero ? b.value : 0))),
    [buckets],
  );

  if (buckets.length === 0) return null;

  // Two ends of a 24-hour window read as the same clock time without a date.
  const spansMoreThanADay =
    buckets[buckets.length - 1].start - buckets[0].start >= 86_400 - intervalSeconds;

  const width = 1000;
  const padTop = 12;
  const padBottom = 22;
  const plot = height - padTop - padBottom;
  const slot = width / buckets.length;
  const barWidth = Math.max(1.5, Math.min(slot - 1.5, 26));

  // Four gridlines is enough to read a magnitude off and few enough not to
  // compete with the bars for attention.
  const gridValues = [0.25, 0.5, 0.75, 1].map((fraction) => peak * fraction);

  const hovered = hover === null ? null : buckets[hover];
  const hoveredAnomaly = hovered ? anomalyByStart.get(hovered.start) : undefined;

  return (
    <figure className="flex min-h-0 flex-col gap-1">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width="100%"
        height={height}
        preserveAspectRatio="none"
        role="img"
        aria-label={`${buckets.length} buckets of ${unitLabel}, peak ${peak.toFixed(2)}`}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          {/* Hatching, not a tint: a gap has to look like a different kind of
              thing from a low bar, not like a darker one. */}
          <pattern
            id="no-data"
            width="6"
            height="6"
            patternTransform="rotate(45)"
            patternUnits="userSpaceOnUse"
          >
            <rect width="6" height="6" fill="rgba(107,85,69,0.10)" />
            <line x1="0" y1="0" x2="0" y2="6" stroke={NO_DATA} strokeWidth="1.5" opacity="0.5" />
          </pattern>
        </defs>
        {gridValues.map((value) => {
          const y = padTop + plot - (value / peak) * plot;
          return (
            <g key={value}>
              <line
                x1={0}
                x2={width}
                y1={y}
                y2={y}
                stroke="rgba(176,141,87,0.12)"
                strokeWidth={1}
                vectorEffect="non-scaling-stroke"
              />
              <text x={4} y={y - 3} fill="#6B5545" fontSize={9} fontFamily="ui-monospace, monospace">
                {value >= 10 ? value.toFixed(0) : value.toFixed(2)}
              </text>
            </g>
          );
        })}

        {buckets.map((bucket, index) => {
          const missing = bucket.samples === 0 && !bucket.known_zero;
          const x = index * slot + (slot - barWidth) / 2;
          const anomaly = anomalyByStart.get(bucket.start);

          if (missing) {
            return (
              <rect
                key={bucket.start}
                x={x}
                y={padTop}
                width={barWidth}
                height={plot}
                fill="url(#no-data)"
                opacity={hover === index ? 1 : 0.75}
                onMouseEnter={() => setHover(index)}
              >
                <title>{`${bucketLabel(bucket.start, intervalSeconds)} — no data recorded`}</title>
              </rect>
            );
          }

          const barHeight = Math.max(bucket.value > 0 ? 1.5 : 1, (bucket.value / peak) * plot);
          const y = padTop + plot - barHeight;
          return (
            <g key={bucket.start} onMouseEnter={() => setHover(index)}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={bucket.value === 0 ? BAR_ZERO : BAR}
                opacity={hover === null || hover === index ? 1 : 0.55}
                rx={1}
              >
                <title>
                  {`${bucketLabel(bucket.start, intervalSeconds)} — ${bucket.value} ${unitLabel}`}
                </title>
              </rect>
              {anomaly && (
                <>
                  <rect x={x} y={y - 2.5} width={barWidth} height={2.5} fill={ANOMALY} />
                  <circle cx={x + barWidth / 2} cy={y - 8} r={2.6} fill={ANOMALY} />
                </>
              )}
            </g>
          );
        })}

        <line
          x1={0}
          x2={width}
          y1={padTop + plot}
          y2={padTop + plot}
          stroke="rgba(176,141,87,0.35)"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
        />
      </svg>

      <figcaption className="flex items-center justify-between gap-3 px-1">
        <span className="stamp tabular-nums text-ink-faint">
          {spansMoreThanADay
            ? new Date(buckets[0].start * 1000).toLocaleString([], {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
              })
            : bucketLabel(buckets[0].start, intervalSeconds)}
        </span>
        <span className="text-tiny text-warm-white" aria-live="polite">
          {hovered ? (
            <>
              <span className="stamp text-brass">
                {bucketLabel(hovered.start, intervalSeconds)}
              </span>
              {'  '}
              {hovered.samples === 0 && !hovered.known_zero ? (
                <span className="text-ink-faint">no data recorded</span>
              ) : (
                <>
                  <span className="font-mono tabular-nums">{hovered.value}</span>{' '}
                  <span className="text-ink-faint">{unitLabel}</span>
                  {hoveredAnomaly && (
                    <span style={{ color: ANOMALY }}>
                      {'  '}
                      {hoveredAnomaly.direction} expected {hoveredAnomaly.expected} (z{' '}
                      {hoveredAnomaly.score})
                    </span>
                  )}
                </>
              )}
            </>
          ) : (
            <span className="text-ink-faint">Hover a bucket for its reading</span>
          )}
        </span>
        <span className="stamp tabular-nums text-ink-faint">
          {bucketLabel(buckets[buckets.length - 1].start, intervalSeconds)}
        </span>
      </figcaption>
    </figure>
  );
};
