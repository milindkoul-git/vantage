/**
 * Incidents: groups of raised events that the correlator decided belong
 * together, newest activity first.
 *
 * Every number on this screen comes off the incident record. The state chips,
 * the severity, the event count and the timeline are all as the server sent
 * them, and an incident whose correlation was never recorded says so instead of
 * displaying a table of representative-looking factors.
 */

import React, { useMemo } from 'react';
import { Clock, Layers, MapPin, Users } from 'lucide-react';
import type { CanonicalIncident, IncidentsResponse } from '../../contracts/types';
import type { IncidentState } from '../../contracts/vocabulary';
import { INCIDENT_STATE_LABELS, SEVERITY_RANK } from '../../contracts/vocabulary';
import {SeverityTag } from '../../components/common/Severity';
import { SEVERITY_COLOR } from '../../contracts/vocabulary';
import { Empty, Panel, Resolved } from '../../components/common/Panels';
import { clockTime, duration, shortEntity } from '../../lib/format';
import type { QueryLike } from '../../components/common/Panels';
import { useInvestigationStore } from '../../store/useInvestigationStore';

const STATE_STYLE: Record<IncidentState, { color: string; border: string }> = {
  active: { color: '#B33A2E', border: 'rgba(179,58,46,0.4)' },
  quiescent: { color: '#B08D57', border: 'rgba(176,141,87,0.4)' },
  resolved: { color: '#6B8F6B', border: 'rgba(107,143,107,0.4)' },
  expired: { color: '#6B5545', border: 'rgba(107,85,69,0.4)' },
};

const StateChip: React.FC<{ state: IncidentState }> = ({ state }) => {
  const style = STATE_STYLE[state] ?? STATE_STYLE.expired;
  return (
    <span
      className="stamp flex-none rounded-sm px-1.5 py-0.5"
      style={{ color: style.color, border: `1px solid ${style.border}` }}
    >
      {INCIDENT_STATE_LABELS[state] ?? state}
    </span>
  );
};

const IncidentCard: React.FC<{
  incident: CanonicalIncident;
  selected: boolean;
  onSelect: () => void;
}> = ({ incident, selected, onSelect }) => (
  <button
    type="button"
    onClick={onSelect}
    aria-pressed={selected}
    className="w-full rounded-sm border bg-board-surface/90 p-3 text-left shadow-paper transition-all hover:border-brass/40"
    style={{
      borderColor: selected ? 'rgba(179,58,46,0.5)' : 'rgba(176,141,87,0.18)',
      borderLeftWidth: '3px',
      borderLeftColor: SEVERITY_COLOR[incident.severity],
    }}
  >
    <div className="flex items-start justify-between gap-3">
      <h3 className="min-w-0 flex-1 font-serif text-sm leading-snug text-warm-white">
        {incident.title}
      </h3>
      <div className="flex flex-none items-center gap-1.5">
        <SeverityTag severity={incident.severity} />
        <StateChip state={incident.state} />
      </div>
    </div>

    <dl className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-micro text-ink-faint">
      <div className="flex items-center gap-1">
        <Clock className="h-3 w-3" aria-hidden />
        <dt className="sr-only">Last activity</dt>
        <dd className="tabular-nums">
          {clockTime(incident.last_seen)} · {duration(incident.duration_s)}
        </dd>
      </div>
      <div className="flex items-center gap-1">
        <Layers className="h-3 w-3" aria-hidden />
        <dt className="sr-only">Events</dt>
        <dd className="tabular-nums">
          {incident.event_count} event{incident.event_count === 1 ? '' : 's'}
        </dd>
      </div>
      {incident.involved_entities.length > 0 && (
        <div className="flex items-center gap-1">
          <Users className="h-3 w-3" aria-hidden />
          <dt className="sr-only">Entities</dt>
          <dd className="font-mono">{incident.involved_entities.map(shortEntity).join(', ')}</dd>
        </div>
      )}
      {incident.zones.length > 0 && (
        <div className="flex items-center gap-1">
          <MapPin className="h-3 w-3" aria-hidden />
          <dt className="sr-only">Zones</dt>
          <dd>{incident.zones.join(', ')}</dd>
        </div>
      )}
    </dl>

    {incident.timeline.length > 0 && (
      <p className="mt-2 truncate border-t border-brass/10 pt-2 text-tiny text-ink-faint">
        Latest: {incident.timeline[incident.timeline.length - 1].summary}
      </p>
    )}
  </button>
);

export const IncidentsWorkspace: React.FC<{ query: QueryLike<IncidentsResponse> }> = ({ query }) => {
  const { selectedIncidentId, selectIncident } = useInvestigationStore();

  const sorted = useMemo(() => {
    const incidents = query.data?.incidents ?? [];
    // Worst first, then most recent. An operator scanning this list is looking
    // for what is wrong now, not for what happened in order.
    return [...incidents].sort((a, b) => {
      const bySeverity = (SEVERITY_RANK[b.severity] ?? 0) - (SEVERITY_RANK[a.severity] ?? 0);
      return bySeverity !== 0 ? bySeverity : b.last_seen - a.last_seen;
    });
  }, [query.data]);

  const active = sorted.filter((incident) => incident.state === 'active').length;

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-3 p-4">
      <Panel
        title="Incidents"
        aside={
          <span className="stamp tabular-nums text-ink-faint">
            {sorted.length} total{active > 0 ? ` · ${active} active` : ''}
          </span>
        }
        bodyClassName="overflow-y-auto custom-scrollbar p-3"
      >
        <Resolved
          query={query}
          what="incidents"
          emptyWhen={(data) => (data.incidents?.length ?? 0) === 0}
          emptyLabel="No incidents"
          emptyHint="An incident opens when a rule raises its first event. A quiet scene raises none."
          unavailableHint={
            <>
              Incident correlation groups raised events. It needs{' '}
              <code className="text-brass">events.enabled</code>, and{' '}
              <code className="text-brass">--store</code> to survive a restart.
            </>
          }
        >
          {() =>
            sorted.length === 0 ? (
              <Empty what="No incidents" />
            ) : (
              <ul className="flex flex-col gap-2">
                {sorted.map((incident) => (
                  <li key={incident.incident_id}>
                    <IncidentCard
                      incident={incident}
                      selected={selectedIncidentId === incident.incident_id}
                      onSelect={() =>
                        selectIncident(
                          selectedIncidentId === incident.incident_id ? null : incident.incident_id,
                        )
                      }
                    />
                  </li>
                ))}
              </ul>
            )
          }
        </Resolved>
      </Panel>
    </div>
  );
};
