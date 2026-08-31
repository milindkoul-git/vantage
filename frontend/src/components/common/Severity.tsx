/**
 * Severity rendering, in one place, driven by the vocabulary file.
 *
 * Mapping severities by hand in each component is how the page came to style
 * `info` / `warning` / `critical` when the backend has only ever raised `info` /
 * `notice` / `alert`: two of three rendered as an unrecognised value and the
 * filter offered options no row could match. Everything here is keyed by the
 * `Severity` union, so a new severity on the Python side is a compile error
 * rather than a silent blank.
 */

import React from 'react';
import type { Severity } from '../../contracts/vocabulary';
import { SEVERITY_COLOR, SEVERITY_LABELS } from '../../contracts/vocabulary';

export const SeverityDot: React.FC<{ severity: Severity; className?: string }> = ({
  severity,
  className = '',
}) => (
  <span
    className={`inline-block h-1.5 w-1.5 flex-none rounded-full ${className}`}
    style={{ backgroundColor: SEVERITY_COLOR[severity] }}
    aria-label={SEVERITY_LABELS[severity]}
    role="img"
  />
);

export const SeverityTag: React.FC<{ severity: Severity }> = ({ severity }) => {
  const color = SEVERITY_COLOR[severity];
  return (
    <span
      className="stamp flex-none rounded-sm px-1.5 py-0.5"
      style={{ color, border: `1px solid ${color}59`, backgroundColor: `${color}1A` }}
    >
      {SEVERITY_LABELS[severity]}
    </span>
  );
};

/**
 * A row in the event list.
 *
 * The `ev` / `ev-<severity>` class names are load-bearing: `index.css` styles the
 * left rule per severity and `tests/test_dashboard.py` checks that every value of
 * the Python `Severity` enum has styling, so an unstyled severity fails the build
 * rather than rendering as an anonymous grey line.
 */
export const EventRow: React.FC<{
  severity: Severity;
  when: string;
  summary: string;
  rule: string;
  meta?: React.ReactNode;
  onClick?: () => void;
  selected?: boolean;
}> = ({ severity, when, summary, rule, meta, onClick, selected }) => {
  const body = (
    <>
      <span className="bar" style={{ backgroundColor: SEVERITY_COLOR[severity] }} aria-hidden />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-tiny text-warm-white">{summary}</span>
          <span className="stamp flex-none tabular-nums text-ink-faint">{when}</span>
        </div>
        <div className="mt-0.5 flex items-center gap-2">
          <span className="stamp text-ink-faint">{rule}</span>
          {meta}
        </div>
      </div>
    </>
  );

  const className = `ev ev-${severity} flex w-full items-stretch gap-2 rounded-sm px-2 py-1.5 text-left transition-colors ${
    selected ? 'bg-string-red/10' : 'hover:bg-brass/5'
  }`;

  return onClick ? (
    <button type="button" onClick={onClick} className={className}>
      {body}
    </button>
  ) : (
    <div className={className}>{body}</div>
  );
};
