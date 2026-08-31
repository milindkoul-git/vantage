/**
 * The words the backend actually uses.
 *
 * Everything in this file mirrors a Python enum or a published dict key, and
 * `tests/test_dashboard.py` reads this file to check it still does. That check
 * exists because the drift is silent in both directions: the page is TypeScript,
 * the contracts are Python enums, and nothing connects them. Two such drifts
 * shipped already -- a health panel branching on `stage.broken` (the real field
 * is `disabled`, so a stage whose circuit had opened rendered as "ok"), and an
 * event list styling `warning` / `critical` (the real values are `notice` /
 * `alert`, so two of three severities rendered as unknown and the filter offered
 * two options no row could ever match).
 *
 * Add a value here only when the backend emits it. A severity invented on this
 * side is a filter option that matches nothing.
 */

/** `vantage.events.contracts.Severity`. */
export const SEVERITIES = ['info', 'notice', 'alert'] as const;
export type Severity = (typeof SEVERITIES)[number];

export const SEVERITY_LABELS: Record<Severity, string> = {
  info: 'Info',
  notice: 'Notice',
  alert: 'Alert',
};

/**
 * The colour each severity is drawn in, everywhere.
 *
 * Here rather than in a component so that the dot, the tag, the row rule, the
 * incident card's left border and the header's threat chip cannot disagree --
 * and so that adding a severity on the Python side breaks the build in one
 * place instead of rendering as five different unstyled defaults.
 */
export const SEVERITY_COLOR: Record<Severity, string> = {
  info: '#6B8F6B',
  notice: '#B08D57',
  alert: '#B33A2E',
};

/** Ascending, so `SEVERITY_RANK[a] > SEVERITY_RANK[b]` reads as "worse than". */
export const SEVERITY_RANK: Record<Severity, number> = { info: 0, notice: 1, alert: 2 };

export function isSeverity(value: unknown): value is Severity {
  return typeof value === 'string' && (SEVERITIES as readonly string[]).includes(value);
}

/** `vantage.analytics.contracts.Metric`. */
export const METRICS = [
  'entities',
  'observations',
  'events',
  'mean_speed',
  'moving_fraction',
] as const;
export type Metric = (typeof METRICS)[number];

export const METRIC_LABELS: Record<Metric, string> = {
  entities: 'Distinct entities',
  observations: 'Observations',
  events: 'Events',
  mean_speed: 'Mean speed',
  moving_fraction: 'Fraction moving',
};

/**
 * Metrics that are rates rather than counts. `Series.total` refuses these on the
 * Python side, so the chart must not offer a total for them either.
 */
export const RATE_METRICS: ReadonlySet<Metric> = new Set<Metric>(['mean_speed', 'moving_fraction']);

/** `vantage.incident.models.IncidentState`, lowercase as the API emits it. */
export const INCIDENT_STATES = ['active', 'quiescent', 'resolved', 'expired'] as const;
export type IncidentState = (typeof INCIDENT_STATES)[number];

export const INCIDENT_STATE_LABELS: Record<IncidentState, string> = {
  active: 'Active',
  quiescent: 'Quiet',
  resolved: 'Resolved',
  expired: 'Archived',
};

/**
 * The fields `vantage.core.resilience.StageRegistry` publishes per stage.
 *
 * The health panel must branch on these and no others.
 */
export const STAGE_FIELDS = ['calls', 'failures', 'disabled', 'last_error'] as const;
export type StageField = (typeof STAGE_FIELDS)[number];

/** Analytics windows the window selector offers, as `parse_window` accepts them. */
export const ANALYTICS_WINDOWS = [
  { value: '6h', label: '6 hours' },
  { value: '12h', label: '12 hours' },
  { value: '24h', label: '24 hours' },
  { value: '3d', label: '3 days' },
  { value: '7d', label: '7 days' },
  { value: '30d', label: '30 days' },
] as const;

export type AnalyticsWindow = (typeof ANALYTICS_WINDOWS)[number]['value'];
