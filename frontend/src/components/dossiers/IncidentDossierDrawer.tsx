/**
 * Everything recorded about one incident: its timeline, how its severity was
 * arrived at, and -- when the event was attached rather than opening a new
 * incident -- the correlation factors that decided it.
 *
 * The correlation panel is the one that used to be fabricated. It now renders
 * `IncidentCorrelationBreakdown`, which the correlator computes and the service
 * stores on attach, and it is absent on an incident that was opened rather than
 * joined, because in that case nothing was correlated and there is nothing to
 * show.
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { api } from '../../data/source';
import { buildIncidentEvidenceViewModel } from '../../lib/transforms/viewModels';
import { SeverityTag } from '../common/Severity';
import { Empty, Resolved } from '../common/Panels';
import { clockTime, duration, shortEntity } from '../../lib/format';
import { SEVERITY_COLOR } from '../../contracts/vocabulary';

const Bar: React.FC<{ fraction: number; color?: string }> = ({ fraction, color = '#B08D57' }) => (
  <div className="h-1.5 w-full overflow-hidden rounded-sm bg-board">
    <div
      className="h-full transition-[width] duration-300"
      style={{ width: `${Math.max(0, Math.min(1, fraction)) * 100}%`, backgroundColor: color }}
    />
  </div>
);

const Section: React.FC<{ title: string; children: React.ReactNode; note?: string }> = ({
  title,
  children,
  note,
}) => (
  <section className="border-t border-brass/15 px-4 py-3">
    <h3 className="stamp mb-2 text-brass">{title}</h3>
    {children}
    {note && <p className="mt-2 text-micro leading-relaxed text-ink-faint/80">{note}</p>}
  </section>
);

export const IncidentDossierDrawer: React.FC<{ incidentId: string; onClose: () => void }> = ({
  incidentId,
  onClose,
}) => {
  const query = useQuery({
    queryKey: ['incident', incidentId],
    queryFn: ({ signal }) => api.incident(incidentId, signal),
    refetchInterval: 5_000,
    retry: false,
  });

  return (
    <aside
      className="folder-pull absolute bottom-0 right-0 top-0 z-40 flex w-full max-w-[520px] flex-col border-l border-brass/25 bg-board-surface shadow-paper-lift"
      role="dialog"
      aria-label="Incident dossier"
    >
      <header className="flex flex-none items-center justify-between gap-3 border-b border-brass/20 px-4 py-3">
        <div className="min-w-0">
          <p className="stamp text-ink-faint">Incident</p>
          <p className="truncate font-mono text-tiny text-brass">{incidentId}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close incident dossier"
          className="rounded-sm p-1 text-ink-faint transition-colors hover:bg-brass/10 hover:text-warm-white"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
        <Resolved
          query={query}
          what="this incident"
          emptyWhen={(data) => data.found === false}
          emptyLabel="That incident is no longer held"
          emptyHint="Incidents are evicted from memory once resolved; only those written to the store survive a restart."
        >
          {(data) => {
            if (!data.incident) {
              return <Empty what="That incident is no longer held" />;
            }
            const model = buildIncidentEvidenceViewModel(data.incident);
            const incident = data.incident;

            return (
              <>
                <div className="px-4 py-3">
                  <h2 className="font-serif text-base leading-snug text-warm-white">
                    {model.title}
                  </h2>
                  <div className="mt-2 flex flex-wrap items-center gap-2">
                    <SeverityTag severity={model.severity} />
                    <span className="stamp text-ink-faint">{model.state}</span>
                    <span className="stamp tabular-nums text-ink-faint">
                      {duration(model.durationSeconds)} · {model.eventCount} events
                    </span>
                  </div>
                  <dl className="mt-3 grid grid-cols-2 gap-3 text-micro">
                    <div>
                      <dt className="stamp text-ink-faint">Entities</dt>
                      <dd className="mt-0.5 font-mono text-tiny text-warm-white">
                        {model.involvedEntities.length > 0
                          ? model.involvedEntities.map(shortEntity).join(', ')
                          : '—'}
                      </dd>
                    </div>
                    <div>
                      <dt className="stamp text-ink-faint">Zones</dt>
                      <dd className="mt-0.5 text-tiny text-warm-white">
                        {model.zones.length > 0 ? model.zones.join(', ') : '—'}
                      </dd>
                    </div>
                  </dl>
                </div>

                <Section
                  title="Timeline"
                  note="Each entry links back to the event that produced it."
                >
                  {incident.timeline.length === 0 ? (
                    <p className="text-tiny text-ink-faint">Nothing recorded.</p>
                  ) : (
                    <ol className="flex flex-col gap-2">
                      {incident.timeline.map((entry) => (
                        <li key={entry.entry_id} className="flex gap-2">
                          <span
                            className="mt-1 h-1.5 w-1.5 flex-none rounded-full"
                            style={{
                              backgroundColor:
                                SEVERITY_COLOR[entry.evidence_ref?.severity ?? 'info'] ?? '#6B5545',
                            }}
                            aria-hidden
                          />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="truncate text-tiny text-warm-white">
                                {entry.summary}
                              </span>
                              <span className="stamp flex-none tabular-nums text-ink-faint">
                                {clockTime(entry.timestamp)}
                              </span>
                            </div>
                            <p className="text-micro text-ink-faint">
                              {entry.event_type}
                              {entry.zone ? ` · ${entry.zone}` : ''}
                              {entry.camera_id ? ` · ${entry.camera_id}` : ''}
                            </p>
                          </div>
                        </li>
                      ))}
                    </ol>
                  )}
                </Section>

                <Section
                  title="How this severity was reached"
                  note={`Severity score ${model.severityScore.toFixed(2)}.`}
                >
                  <dl className="flex flex-col gap-1.5">
                    {model.severityFactors.map((factor) => (
                      <div key={factor.name} className="flex items-baseline justify-between gap-3">
                        <dt className="text-tiny text-ink-faint">{factor.name}</dt>
                        <dd className="font-mono text-tiny tabular-nums text-warm-white">
                          {factor.value}
                        </dd>
                      </div>
                    ))}
                  </dl>
                </Section>

                <Section
                  title="Why the last event joined this incident"
                  note={
                    model.correlation
                      ? `Decision: ${model.correlation.decision.replace(/_/g, ' ')}. ` +
                        `Positive ${model.correlation.positive.toFixed(2)} less a continuity ` +
                        `penalty of ${model.correlation.penalty.toFixed(2)}.`
                      : undefined
                  }
                >
                  {model.correlation === null ? (
                    <p className="text-tiny leading-relaxed text-ink-faint">
                      This incident was opened by its first event rather than joined to an existing
                      one, so there was no correlation to score.
                    </p>
                  ) : (
                    <>
                      <ul className="flex flex-col gap-2">
                        {model.correlation.factors.map((factor) => (
                          <li key={factor.name}>
                            <div className="flex items-baseline justify-between gap-2">
                              <span className="text-tiny text-warm-white">{factor.name}</span>
                              <span
                                className="stamp tabular-nums text-ink-faint"
                                title={`weight ${factor.weight} from ${factor.why}`}
                              >
                                {factor.score.toFixed(2)} × {factor.weight} ={' '}
                                <span className="text-brass">{factor.contribution.toFixed(3)}</span>
                              </span>
                            </div>
                            <div className="mt-1">
                              <Bar fraction={factor.score} />
                            </div>
                          </li>
                        ))}
                      </ul>
                      <div className="mt-3 flex items-baseline justify-between gap-2 border-t border-brass/10 pt-2">
                        <span className="stamp text-brass">Total</span>
                        <span className="font-mono text-sm tabular-nums text-warm-white">
                          {model.correlation.total.toFixed(3)}
                        </span>
                      </div>
                      {model.correlation.explanation && (
                        <p className="mt-2 text-micro leading-relaxed text-ink-faint">
                          {model.correlation.explanation}
                        </p>
                      )}
                    </>
                  )}
                </Section>

                {incident.correlation_candidates.length > 0 && (
                  <Section
                    title="Ambiguous links"
                    note="Scored between the candidate and attach thresholds: recorded on both incidents rather than merged."
                  >
                    <ul className="flex flex-col gap-2">
                      {incident.correlation_candidates.map((candidate, index) => (
                        <li key={`${candidate.related_incident_id}-${index}`}>
                          <div className="flex items-baseline justify-between gap-2">
                            <span className="truncate font-mono text-tiny text-brass">
                              {candidate.related_incident_id}
                            </span>
                            <span className="stamp tabular-nums text-ink-faint">
                              {candidate.score.toFixed(2)}
                            </span>
                          </div>
                          <p className="text-micro text-ink-faint">{candidate.explanation}</p>
                        </li>
                      ))}
                    </ul>
                  </Section>
                )}
              </>
            );
          }}
        </Resolved>
      </div>
    </aside>
  );
};
