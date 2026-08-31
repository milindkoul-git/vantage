/**
 * Pipeline telemetry: which stages are healthy, which have been switched off by
 * their own circuit breaker, and what the store holds.
 *
 * The stage table is the panel whose entire purpose is to say when something has
 * stopped, and it once branched on `stage.broken` and `stage.circuit_open` --
 * neither of which `StageRegistry` publishes -- so a stage whose circuit had
 * opened rendered as "ok". The fields it reads are now taken from
 * `STAGE_FIELDS`, which is the list `tests/test_dashboard.py` checks against the
 * registry itself.
 */

import React from 'react';
import { X } from 'lucide-react';
import type { StageHealth, SystemStatsResponse } from '../../contracts/types';
import { STAGE_FIELDS } from '../../contracts/vocabulary';
import { Stat } from '../common/Panels';
import { bytes, duration } from '../../lib/format';
import { useInvestigationStore } from '../../store/useInvestigationStore';

const StageRow: React.FC<{ stage: StageHealth }> = ({ stage }) => {
  // Read exactly the fields StageRegistry publishes: `disabled` is the circuit
  // breaker, `failures` and `calls` give the rate, `last_error` is the reason.
  const disabled = stage.disabled;
  const failures = stage.failures;
  const calls = stage.calls;
  const lastError = stage.last_error;

  const state = disabled ? 'disabled' : failures > 0 ? 'degraded' : 'ok';
  const color = disabled ? '#B33A2E' : failures > 0 ? '#B08D57' : '#6B8F6B';

  return (
    <li className="flex flex-col gap-0.5 border-b border-brass/10 px-3 py-2 last:border-b-0">
      <div className="flex items-baseline justify-between gap-3">
        <span className="flex items-center gap-2">
          <span
            className="h-1.5 w-1.5 flex-none rounded-full"
            style={{ backgroundColor: color }}
            aria-hidden
          />
          <span className="font-mono text-tiny text-warm-white">{stage.name}</span>
        </span>
        <span className="stamp flex-none" style={{ color }}>
          {state}
        </span>
      </div>
      <div className="flex items-baseline justify-between gap-3 pl-3.5">
        <span className="stamp tabular-nums text-ink-faint">
          {failures} / {calls} failed
        </span>
        {stage.worst_streak > 1 && (
          <span className="stamp tabular-nums text-ink-faint">
            worst streak {stage.worst_streak}
          </span>
        )}
      </div>
      {lastError && (
        <p className="break-words pl-3.5 text-micro leading-relaxed text-string-red/80">
          {lastError}
        </p>
      )}
    </li>
  );
};

export const OperationsDrawer: React.FC<{
  health: Record<string, StageHealth> | undefined;
  stats: SystemStatsResponse | undefined;
}> = ({ health, stats }) => {
  const { isOperationsDrawerOpen, setOperationsDrawerOpen } = useInvestigationStore();
  if (!isOperationsDrawerOpen) return null;

  const stages = Object.values(health ?? {});
  const store = stats?.store ?? null;

  return (
    <aside
      className="absolute bottom-0 right-0 top-0 z-30 flex w-full max-w-[360px] flex-col border-l border-brass/25 bg-board-surface shadow-paper-lift"
      role="complementary"
      aria-label="Pipeline telemetry"
    >
      <header className="flex flex-none items-center justify-between border-b border-brass/20 px-4 py-3">
        <h2 className="stamp text-brass">Pipeline telemetry</h2>
        <button
          type="button"
          onClick={() => setOperationsDrawerOpen(false)}
          aria-label="Close pipeline telemetry"
          className="rounded-sm p-1 text-ink-faint transition-colors hover:bg-brass/10 hover:text-warm-white"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
        <section className="grid grid-cols-2 gap-3 px-4 py-3">
          <Stat label="Camera" value={stats?.camera_id ?? null} />
          <Stat
            label="Uptime"
            value={stats?.uptime_s === undefined ? null : duration(stats.uptime_s)}
          />
          <Stat label="Live feed" value={stats?.live === undefined ? null : stats.live ? 'yes' : 'no'} />
          <Stat label="Schema" value={stats?.schema_version ?? null} />
          <Stat label="Events stored" value={store ? store.events : null} />
          <Stat label="Observations" value={store ? store.observations : null} />
          <Stat label="Store size" value={store ? bytes(store.bytes) : null} />
        </section>

        <section className="border-t border-brass/15">
          <h3 className="stamp px-4 py-2 text-brass">Stages</h3>
          {stages.length === 0 ? (
            <p className="px-4 pb-3 text-tiny text-ink-faint">
              No stage telemetry — nothing is running in this process.
            </p>
          ) : (
            <ul className="flex flex-col">
              {stages.map((stage) => (
                <StageRow key={stage.name} stage={stage} />
              ))}
            </ul>
          )}
          <p className="px-4 py-2 text-micro leading-relaxed text-ink-faint/70">
            A stage that fails repeatedly is disabled rather than retried, so one
            broken subsystem cannot take the run down with it. Reported per stage as{' '}
            {STAGE_FIELDS.join(', ')}.
          </p>
        </section>
      </div>
    </aside>
  );
};
