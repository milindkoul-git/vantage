/**
 * Historical analytics -- baselines, coverage and anomalies.
 *
 * This exists because the feature did, and the page had lost it. The analytics
 * engine (hourly buckets, a median/MAD baseline shrunk toward a pooled estimate,
 * anomalies scored against the matching hour of the week) was reachable only from
 * `vantage analytics` on the command line; the browser had two `<select>`
 * elements with the right ids sitting in a `display:none` block so that a test
 * asserting the controls existed would keep passing. This is that panel, wired
 * to `/api/analytics` and visible.
 *
 * The ids `an-metric` and `an-since` are the ones `tests/test_dashboard.py`
 * checks. They are on the real controls.
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, TrendingDown, TrendingUp } from 'lucide-react';
import { api } from '../../data/source';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import {
  ANALYTICS_WINDOWS,
  METRICS,
  METRIC_LABELS,
  RATE_METRICS,
} from '../../contracts/vocabulary';
import type { AnalyticsWindow, Metric } from '../../contracts/vocabulary';
import { Empty, Panel, Resolved, Stat } from '../../components/common/Panels';
import { dayTime } from '../../lib/format';
import { TrendChart } from './TrendChart';

const SELECT_CLASS =
  'stamp cursor-pointer rounded-sm border border-brass/25 bg-board px-2 py-1 text-warm-white ' +
  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brass';

export const AnalyticsWorkspace: React.FC = () => {
  const { analyticsMetric, analyticsWindow, setAnalyticsMetric, setAnalyticsWindow } =
    useInvestigationStore();

  const query = useQuery({
    queryKey: ['analytics', analyticsMetric, analyticsWindow],
    queryFn: ({ signal }) =>
      api.analytics({ metric: analyticsMetric, since: analyticsWindow }, signal),
    refetchInterval: 60_000,
    retry: false,
  });

  const isRate = RATE_METRICS.has(analyticsMetric);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto p-4 custom-scrollbar">
      <div className="flex flex-wrap items-end gap-4">
        <label className="flex flex-col gap-1">
          <span className="stamp text-ink-faint">Metric</span>
          <select
            id="an-metric"
            aria-label="Metric"
            className={SELECT_CLASS}
            value={analyticsMetric}
            onChange={(event) => setAnalyticsMetric(event.target.value as Metric)}
          >
            {METRICS.map((metric) => (
              <option key={metric} value={metric}>
                {METRIC_LABELS[metric]}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="stamp text-ink-faint">Window</span>
          <select
            id="an-since"
            aria-label="Window"
            className={SELECT_CLASS}
            value={analyticsWindow}
            onChange={(event) => setAnalyticsWindow(event.target.value as AnalyticsWindow)}
          >
            {ANALYTICS_WINDOWS.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>

        <p className="ml-auto max-w-sm pb-1.5 text-right text-micro leading-relaxed text-ink-faint">
          Buckets come from recorded history, not the live feed. Hatched columns are
          hours with no recording — not quiet hours.
        </p>
      </div>

      <Panel
        title={`${METRIC_LABELS[analyticsMetric]} over ${
          ANALYTICS_WINDOWS.find((w) => w.value === analyticsWindow)?.label ?? analyticsWindow
        }`}
        bodyClassName="p-3"
      >
        <Resolved
          query={query}
          what="analytics"
          emptyWhen={(data) => (data.buckets?.length ?? 0) === 0}
          emptyLabel="No buckets in this window"
          emptyHint="Widen the window, or let the pipeline run for longer with --store."
          unavailableHint={
            <>
              Start the pipeline with <code className="text-brass">--store</code> to record the
              history this reads.
            </>
          }
        >
          {(data) => {
            const buckets = data.buckets ?? [];
            const anomalies = data.anomalies ?? [];
            const occupied = buckets.filter((b) => b.samples > 0 || b.known_zero);
            const total = occupied.reduce((sum, b) => sum + b.value, 0);
            const mean = occupied.length > 0 ? total / occupied.length : null;
            const coverage = data.coverage ?? 0;

            return (
              <div className="flex flex-col gap-4">
                <TrendChart
                  buckets={buckets}
                  anomalies={anomalies}
                  intervalSeconds={data.interval_s ?? 3600}
                  unitLabel={data.label ?? METRIC_LABELS[analyticsMetric]}
                />

                <div className="grid grid-cols-2 gap-4 border-t border-brass/15 pt-3 sm:grid-cols-4">
                  <Stat
                    label="Coverage"
                    value={`${(coverage * 100).toFixed(0)}%`}
                    tone={coverage < 0.5 ? 'alert' : 'normal'}
                    hint="Fraction of buckets in this window that hold a reading."
                  />
                  <Stat
                    label="Mean per bucket"
                    value={mean === null ? null : mean.toFixed(2)}
                    hint="Averaged over buckets that hold a reading, not over the whole window."
                  />
                  {/* Series.total refuses rate metrics on the Python side; summing
                      a mean speed across buckets would be meaningless here too. */}
                  <Stat
                    label={isRate ? 'Total' : 'Total in window'}
                    value={isRate ? null : total.toFixed(0)}
                    hint={isRate ? 'Not meaningful for a rate metric.' : undefined}
                  />
                  <Stat
                    label="Buckets compared"
                    value={
                      data.anomalies_available
                        ? `${data.judged ?? 0} of ${(data.judged ?? 0) + (data.unjudged ?? 0)}`
                        : null
                    }
                    hint="A bucket can only be judged once its slot has enough history behind it."
                  />
                </div>

                <div
                  className="h-1 w-full overflow-hidden rounded-sm bg-board"
                  role="meter"
                  aria-valuenow={Math.round(coverage * 100)}
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-label="Coverage of the selected window"
                >
                  <div
                    className="h-full"
                    style={{
                      width: `${Math.max(0, Math.min(1, coverage)) * 100}%`,
                      backgroundColor: coverage < 0.5 ? '#B33A2E' : '#B08D57',
                    }}
                  />
                </div>
              </div>
            );
          }}
        </Resolved>
      </Panel>

      <Panel title="Anomalies" bodyClassName="p-2">
        <Resolved query={query} what="anomalies">
          {(data) => {
            if (!data.anomalies_available) {
              return (
                <Empty
                  what="Nothing has been compared yet"
                  hint={
                    data.anomalies_reason ??
                    'A baseline needs several weeks of the same hour before it can call anything unusual.'
                  }
                />
              );
            }
            const anomalies = data.anomalies ?? [];
            if (anomalies.length === 0) {
              return (
                <Empty
                  what="Nothing unusual"
                  hint={`${data.judged ?? 0} buckets were compared against their baseline and all sat inside it.`}
                />
              );
            }
            return (
              <ul className="flex flex-col divide-y divide-brass/10">
                {anomalies.map((anomaly) => (
                  <li
                    key={anomaly.start}
                    className="flex items-center gap-3 px-2 py-2 text-tiny text-warm-white"
                  >
                    {anomaly.direction === 'above' ? (
                      <TrendingUp className="h-3.5 w-3.5 flex-none text-string-red" aria-hidden />
                    ) : (
                      <TrendingDown className="h-3.5 w-3.5 flex-none text-brass" aria-hidden />
                    )}
                    <span className="stamp w-32 flex-none tabular-nums text-ink-faint">
                      {dayTime(anomaly.start)}
                    </span>
                    <span className="flex-1">
                      <span className="font-mono tabular-nums">{anomaly.observed}</span>{' '}
                      <span className="text-ink-faint">
                        {anomaly.direction} an expected{' '}
                        <span className="font-mono tabular-nums">{anomaly.expected}</span>
                      </span>
                    </span>
                    <span className="stamp flex-none tabular-nums text-brass" title="Robust z-score">
                      z {anomaly.score}
                    </span>
                  </li>
                ))}
              </ul>
            );
          }}
        </Resolved>
      </Panel>

      <p className="flex items-center gap-2 px-1 text-micro text-ink-faint">
        <Activity className="h-3 w-3" aria-hidden />
        The same numbers <code className="text-brass">vantage analytics</code> prints, over the
        same store.
      </p>
    </div>
  );
};
