/**
 * One entity's recorded history: what it did, where, and who it kept appearing
 * with.
 *
 * Built from `/api/entity_timeline`, which projects the stored observations and
 * events into contiguous state intervals. That works on a single camera as well
 * as a facility, which the previous version did not -- it read the multi-camera
 * entity snapshot only, and filled the gaps with `?? 0.94` for identity
 * confidence and `?? 14` for the number of appearance prototypes, so an entity
 * the system knew nothing about still displayed a 94% confident identity.
 *
 * The identifiers here are the tracker's own anonymous ones. A name appears only
 * where the identity subsystem resolved one against a consented enrolment.
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { Users, X } from 'lucide-react';
import { api } from '../../data/source';
import { isSeverity } from '../../contracts/vocabulary';
import { EventRow } from '../common/Severity';
import { Resolved } from '../common/Panels';
import { clockTime, dayTime, duration, shortEntity } from '../../lib/format';
import { useInvestigationStore } from '../../store/useInvestigationStore';

const Section: React.FC<{ title: string; children: React.ReactNode; aside?: React.ReactNode }> = ({
  title,
  children,
  aside,
}) => (
  <section className="border-t border-brass/15 px-4 py-3">
    <div className="mb-2 flex items-baseline justify-between gap-2">
      <h3 className="stamp text-brass">{title}</h3>
      {aside}
    </div>
    {children}
  </section>
);

export const EntityDossierDrawer: React.FC<{ entityId: string; onClose: () => void }> = ({
  entityId,
  onClose,
}) => {
  const selectRelationship = useInvestigationStore((state) => state.selectRelationship);

  const timeline = useQuery({
    queryKey: ['entity-timeline', entityId],
    queryFn: ({ signal }) => api.entityTimeline(entityId, signal),
    refetchInterval: 5_000,
    retry: false,
  });

  const associates = useQuery({
    queryKey: ['relationships', entityId],
    queryFn: ({ signal }) => api.relationships(entityId, signal),
    refetchInterval: 10_000,
    retry: false,
  });

  return (
    <aside
      className="folder-pull absolute bottom-0 right-0 top-0 z-40 flex w-full max-w-[480px] flex-col border-l border-brass/25 bg-board-surface shadow-paper-lift"
      role="dialog"
      aria-label="Entity dossier"
    >
      <header className="flex flex-none items-center justify-between gap-3 border-b border-brass/20 px-4 py-3">
        <div className="min-w-0">
          <p className="stamp text-ink-faint">Entity</p>
          <p className="truncate font-mono text-sm text-brass">{shortEntity(entityId)}</p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close entity dossier"
          className="rounded-sm p-1 text-ink-faint transition-colors hover:bg-brass/10 hover:text-warm-white"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto custom-scrollbar">
        <Resolved
          query={timeline}
          what="this entity"
          emptyWhen={(data) => data.found === false}
          emptyLabel="Nothing recorded for this entity"
          emptyHint="Observations are only written when the run has --store."
        >
          {(data) => {
            const segments = data.segments ?? [];
            const events = (data.events ?? []).filter((event) => isSeverity(event.severity));
            const zones = [...new Set(segments.flatMap((segment) => segment.zones))];
            const activities = [...new Set(segments.flatMap((segment) => segment.activities))];

            return (
              <>
                <div className="px-4 py-3">
                  {data.identity ? (
                    <p className="font-serif text-base text-warm-white">{data.identity}</p>
                  ) : (
                    <p className="text-tiny text-ink-faint">
                      Anonymous — identity resolution is off, or nobody was recognised.
                    </p>
                  )}
                  <dl className="mt-3 grid grid-cols-2 gap-3 text-micro">
                    <div>
                      <dt className="stamp text-ink-faint">Seen for</dt>
                      <dd className="mt-0.5 font-mono text-tiny tabular-nums text-warm-white">
                        {duration(data.total_duration_s ?? 0)}
                      </dd>
                    </div>
                    <div>
                      <dt className="stamp text-ink-faint">Camera</dt>
                      <dd className="mt-0.5 text-tiny text-warm-white">{data.camera_id ?? '—'}</dd>
                    </div>
                    <div>
                      <dt className="stamp text-ink-faint">First seen</dt>
                      <dd className="mt-0.5 text-tiny tabular-nums text-warm-white">
                        {data.first_seen ? dayTime(data.first_seen) : '—'}
                      </dd>
                    </div>
                    <div>
                      <dt className="stamp text-ink-faint">Last seen</dt>
                      <dd className="mt-0.5 text-tiny tabular-nums text-warm-white">
                        {data.last_seen ? dayTime(data.last_seen) : '—'}
                      </dd>
                    </div>
                  </dl>
                  {(zones.length > 0 || activities.length > 0) && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                      {zones.map((zone) => (
                        <span
                          key={`zone-${zone}`}
                          className="stamp rounded-sm border border-brass/25 px-1.5 py-0.5 text-brass"
                        >
                          {zone}
                        </span>
                      ))}
                      {activities.map((activity) => (
                        <span
                          key={`act-${activity}`}
                          className="stamp rounded-sm border border-string-red/30 px-1.5 py-0.5 text-string-red"
                        >
                          {activity}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                <Section
                  title="State intervals"
                  aside={<span className="stamp tabular-nums text-ink-faint">{segments.length}</span>}
                >
                  {segments.length === 0 ? (
                    <p className="text-tiny text-ink-faint">No observations recorded.</p>
                  ) : (
                    <ol className="flex flex-col gap-1.5">
                      {segments.map((segment) => (
                        <li
                          key={segment.start_time}
                          className="flex items-baseline justify-between gap-3 text-tiny"
                        >
                          <span className="stamp w-20 flex-none tabular-nums text-ink-faint">
                            {clockTime(segment.start_time)}
                          </span>
                          <span className="min-w-0 flex-1 truncate text-warm-white">
                            {segment.motion ?? 'unknown'}
                            {segment.posture ? ` · ${segment.posture}` : ''}
                            {segment.zones.length > 0 ? ` · ${segment.zones.join(', ')}` : ''}
                            {segment.activities.length > 0 ? ` · ${segment.activities.join(', ')}` : ''}
                          </span>
                          <span className="stamp flex-none tabular-nums text-brass">
                            {duration(segment.duration_s)}
                          </span>
                        </li>
                      ))}
                    </ol>
                  )}
                </Section>

                <Section
                  title="Events involving this entity"
                  aside={<span className="stamp tabular-nums text-ink-faint">{events.length}</span>}
                >
                  {events.length === 0 ? (
                    <p className="text-tiny text-ink-faint">No rule fired on this entity.</p>
                  ) : (
                    <ul className="flex flex-col gap-0.5">
                      {events.map((event, index) => (
                        <li key={`${event.timestamp}-${index}`}>
                          <EventRow
                            severity={event.severity}
                            when={clockTime(event.timestamp)}
                            summary={event.summary}
                            rule={event.rule}
                            meta={
                              event.zone ? <span className="stamp text-brass">{event.zone}</span> : null
                            }
                          />
                        </li>
                      ))}
                    </ul>
                  )}
                </Section>
              </>
            );
          }}
        </Resolved>

        <Section
          title="Seen with"
          aside={<Users className="h-3.5 w-3.5 text-ink-faint" aria-hidden />}
        >
          <Resolved
            query={associates}
            what="associations"
            emptyWhen={(data) => (data.relationships?.length ?? 0) === 0}
            emptyLabel="Nobody, so far"
            emptyHint="An association needs two entities in frame together over several observations."
            unavailableHint={
              <>
                Relationship tracking is off. Turn it on with{' '}
                <code className="text-brass">--set relationships.enabled=true</code>.
              </>
            }
          >
            {(data) => (
              <ul className="flex flex-col gap-1">
                {(data.relationships ?? []).map((relationship) => {
                  const other =
                    relationship.entity_a === entityId
                      ? relationship.entity_b
                      : relationship.entity_a;
                  return (
                    <li key={`${relationship.entity_a}-${relationship.entity_b}`}>
                      <button
                        type="button"
                        onClick={() => selectRelationship(entityId, other)}
                        className="flex w-full items-baseline justify-between gap-2 rounded-sm px-2 py-1.5 text-left transition-colors hover:bg-brass/5"
                      >
                        <span className="font-mono text-tiny text-warm-white">
                          {shortEntity(other)}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-micro text-ink-faint">
                          {relationship.primary_derived_pattern?.replace(/_/g, ' ') ??
                            relationship.evidence_summary}
                        </span>
                        <span className="stamp flex-none tabular-nums text-brass">
                          {relationship.active_strength.toFixed(2)}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </Resolved>
        </Section>
      </div>
    </aside>
  );
};
