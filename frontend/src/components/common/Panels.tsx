/**
 * The pieces every workspace uses to say what it knows and what it does not.
 *
 * The reason these exist as components rather than as inline JSX is that the
 * three states they cover kept collapsing into one another. A panel with no data
 * rendered the same whether the subsystem was switched off, the request had
 * failed, or the facility was simply quiet -- and the previous version papered
 * over all three with fallback numbers (`fps || 24.0`, `count || 3`), so a dead
 * pipeline read as a healthy one. Each state now looks different and says which
 * it is.
 */

import React from 'react';
import { AlertTriangle, Loader2, PowerOff, Search } from 'lucide-react';
import type { Availability } from '../../contracts/types';
import { AnimatedNumber } from './AnimatedNumber';

// ---------------------------------------------------------------------------
// surfaces
// ---------------------------------------------------------------------------

export const Panel: React.FC<{
  title?: React.ReactNode;
  aside?: React.ReactNode;
  className?: string;
  bodyClassName?: string;
  children: React.ReactNode;
}> = ({ title, aside, className = '', bodyClassName = '', children }) => (
  <section
    className={`flex min-h-0 flex-col overflow-hidden rounded-sm border border-brass/20 bg-board-surface/95 shadow-paper ${className}`}
  >
    {(title || aside) && (
      <header className="flex flex-none items-center justify-between gap-3 border-b border-brass/15 px-3 py-2">
        <h2 className="stamp truncate text-brass">{title}</h2>
        {aside}
      </header>
    )}
    <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
  </section>
);

/** A label/value pair. `value` of `null`/`undefined` renders an em dash.
 *
 * A numeric value counts to its new reading rather than snapping; anything else
 * is set outright. The distinction matters: a count that moved is worth drawing
 * the eye to, and a label that changed is not the same kind of event.
 */
export const Stat: React.FC<{
  label: string;
  value: React.ReactNode;
  hint?: string;
  tone?: 'normal' | 'alert' | 'muted';
  decimals?: number;
}> = ({ label, value, hint, tone = 'normal', decimals = 0 }) => {
  const empty = value === null || value === undefined || value === '';
  const color = empty || tone === 'muted' ? '#6B5545' : tone === 'alert' ? '#B33A2E' : '#E8E2D4';
  return (
    <div className="flex flex-col gap-0.5" title={hint}>
      <span className="stamp text-ink-faint">{label}</span>
      <span className="font-mono text-sm tabular-nums" style={{ color }}>
        {typeof value === 'number' ? (
          <AnimatedNumber value={value} decimals={decimals} />
        ) : (
          (empty ? '—' : value)
        )}
      </span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// the three not-showing-data states
// ---------------------------------------------------------------------------

/**
 * The frame the three not-showing-data states share.
 *
 * They are a third of what this console displays - a single-camera run has four
 * workspaces that legitimately have nothing in them - so they get composed
 * rather than left as centred grey text. The shape is the same each time so the
 * three read as one family: a mark, a stamped headline, a sentence, and where
 * there is one, the command that would change the situation.
 */
const State: React.FC<{
  mark: React.ReactNode;
  rule: string;
  headline: React.ReactNode;
  headlineColor: string;
  body?: React.ReactNode;
  hint?: React.ReactNode;
  action?: React.ReactNode;
}> = ({ mark, rule, headline, headlineColor, body, hint, action }) => (
  <div className="flex h-full min-h-[140px] flex-col items-center justify-center px-6 py-8 text-center">
    <div
      className="mb-3 flex h-9 w-9 items-center justify-center rounded-sm border"
      style={{ borderColor: `${rule}33`, backgroundColor: `${rule}0F`, color: rule }}
      aria-hidden
    >
      {mark}
    </div>
    <p className="stamp" style={{ color: headlineColor }}>
      {headline}
    </p>
    {/* A hairline the width of the headline, which is what stops the block
        reading as a paragraph that happens to be centred. */}
    <span className="my-2 block h-px w-10" style={{ backgroundColor: `${rule}40` }} aria-hidden />
    {body && <p className="max-w-md text-tiny leading-relaxed text-ink-faint">{body}</p>}
    {hint && (
      <div className="mt-2 max-w-md text-micro leading-relaxed text-ink-faint/75">{hint}</div>
    )}
    {action && <div className="mt-3">{action}</div>}
  </div>
);

export const Loading: React.FC<{ what?: string }> = ({ what = 'data' }) => (
  <div className="flex h-full min-h-[140px] items-center justify-center gap-2 text-ink-faint">
    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
    <span className="stamp">Reading {what}…</span>
  </div>
);

/**
 * The subsystem is not running. Renders the server's own `reason`, which is
 * written for an operator and usually names the flag that would turn it on.
 */
export const Unavailable: React.FC<{ what: string; reason?: string; hint?: React.ReactNode }> = ({
  what,
  reason,
  hint,
}) => (
  <State
    mark={<PowerOff className="h-4 w-4" />}
    rule="#B08D57"
    headlineColor="#B08D57"
    headline={`${what} unavailable`}
    body={reason}
    hint={hint}
  />
);

/** Running, connected, and genuinely nothing to show. */
export const Empty: React.FC<{ what: string; hint?: React.ReactNode }> = ({ what, hint }) => (
  <State
    mark={<Search className="h-4 w-4" />}
    rule="#6B5545"
    headlineColor="#6B5545"
    headline={what}
    hint={hint}
  />
);

/** The request itself failed. Distinct from the subsystem being off. */
export const Failed: React.FC<{ what: string; error: unknown; onRetry?: () => void }> = ({
  what,
  error,
  onRetry,
}) => (
  <State
    mark={<AlertTriangle className="h-4 w-4" />}
    rule="#B33A2E"
    headlineColor="#B33A2E"
    headline={`Could not load ${what}`}
    body={
      <span className="break-words font-mono text-micro">
        {error instanceof Error ? error.message : String(error)}
      </span>
    }
    action={
      onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="stamp rounded-sm border border-brass/30 px-3 py-1.5 text-brass transition-colors hover:bg-brass/10"
        >
          Retry
        </button>
      )
    }
  />
);

// ---------------------------------------------------------------------------
// the gate
// ---------------------------------------------------------------------------

export interface QueryLike<T> {
  data: T | undefined;
  isPending: boolean;
  isError: boolean;
  error: unknown;
  refetch?: () => void;
}

/**
 * Render `children` only once there is a real, available payload.
 *
 * Takes the query rather than the data so that pending, failed, unavailable and
 * present stay four distinct branches at every call site. `emptyWhen` is how a
 * caller says "available, but there is nothing in it" without every panel
 * reinventing the check.
 */
export function Resolved<T extends Availability>({
  query,
  what,
  emptyWhen,
  emptyLabel,
  emptyHint,
  unavailableHint,
  children,
}: {
  query: QueryLike<T>;
  what: string;
  emptyWhen?: (data: T) => boolean;
  emptyLabel?: string;
  emptyHint?: React.ReactNode;
  unavailableHint?: React.ReactNode;
  children: (data: T) => React.ReactNode;
}): React.ReactElement {
  if (query.isError) {
    return <Failed what={what} error={query.error} onRetry={query.refetch} />;
  }
  if (query.isPending || query.data === undefined) {
    return <Loading what={what} />;
  }
  if (!query.data.available) {
    return <Unavailable what={what} reason={query.data.reason} hint={unavailableHint} />;
  }
  if (emptyWhen?.(query.data)) {
    return <Empty what={emptyLabel ?? `No ${what} yet`} hint={emptyHint} />;
  }
  return <>{children(query.data)}</>;
}
