/**
 * The recorded event log, filterable, plus ontology-expanded search over it.
 *
 * The severity filter (`id="ev-sev"`) offers exactly the values the Python
 * `Severity` enum defines, taken from the vocabulary file rather than typed out
 * here, because the last time they were typed out the list read info / warning /
 * critical and two of the three options could never match a row.
 */

import React, { useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Search } from 'lucide-react';
import { api } from '../../data/source';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import { SEVERITIES, SEVERITY_LABELS, isSeverity } from '../../contracts/vocabulary';
import type { Severity } from '../../contracts/vocabulary';
import type { EventsResponse } from '../../contracts/types';
import type { QueryLike } from '../../components/common/Panels';
import { Empty, Panel, Resolved } from '../../components/common/Panels';
import { dayTime, shortEntity } from '../../lib/format';
import { EventRow, SeverityTag } from '../../components/common/Severity';

const CONTROL_CLASS =
  'stamp rounded-sm border border-brass/25 bg-board px-2 py-1 text-warm-white ' +
  'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brass';

const SearchPanel: React.FC = () => {
  const [draft, setDraft] = useState('');
  const [submitted, setSubmitted] = useState('');

  const query = useQuery({
    queryKey: ['search', submitted],
    queryFn: ({ signal }) => api.search(submitted, signal),
    enabled: submitted.trim().length > 0,
    retry: false,
  });

  return (
    <Panel title="Search the log" bodyClassName="flex flex-col min-h-0">
      <form
        className="flex flex-none gap-2 p-3"
        onSubmit={(event) => {
          event.preventDefault();
          setSubmitted(draft);
        }}
      >
        <input
          type="search"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder="e.g. someone collapsed near the lobby"
          aria-label="Search recorded events"
          className="min-w-0 flex-1 rounded-sm border border-brass/25 bg-board px-2 py-1.5 text-tiny text-warm-white placeholder:text-ink-faint focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-brass"
        />
        <button
          type="submit"
          className="flex flex-none items-center gap-1.5 rounded-sm border border-brass/30 px-3 py-1.5 text-brass transition-colors hover:bg-brass/10"
        >
          <Search className="h-3.5 w-3.5" aria-hidden />
          <span className="stamp">Search</span>
        </button>
      </form>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2 custom-scrollbar">
        {submitted.trim().length === 0 ? (
          <Empty
            what="Nothing searched yet"
            hint="Plain words are expanded into related concepts before matching — “collapse” also finds falls and prone postures."
          />
        ) : (
          <Resolved
            query={query}
            what="search results"
            emptyWhen={(data) => (data.results?.length ?? 0) === 0}
            emptyLabel="Nothing matched"
            emptyHint="Try a broader phrase, or widen the recorded window."
          >
            {(data) => (
              <>
                {data.parsed_intent && data.parsed_intent.expanded_concepts.length > 0 && (
                  <p className="px-1 pb-2 text-micro leading-relaxed text-ink-faint">
                    Expanded to: {data.parsed_intent.expanded_concepts.slice(0, 8).join(', ')}
                  </p>
                )}
                <ul className="flex flex-col gap-0.5">
                  {(data.results ?? []).filter((r) => isSeverity(r.severity)).map((result) => (
                    <li key={`${result.id}-${result.timestamp}`}>
                      <EventRow
                        severity={result.severity}
                        when={dayTime(result.timestamp)}
                        summary={result.summary}
                        rule={result.rule}
                        meta={
                          <span className="stamp text-brass">
                            score {result.score.toFixed(2)}
                            {result.zone ? ` · ${result.zone}` : ''}
                          </span>
                        }
                      />
                    </li>
                  ))}
                </ul>
              </>
            )}
          </Resolved>
        )}
      </div>
    </Panel>
  );
};

export const InvestigateWorkspace: React.FC<{ eventsQuery: QueryLike<EventsResponse> }> = ({
  eventsQuery,
}) => {
  const {
    severityFilter,
    setSeverityFilter,
    eventQuery,
    setEventQuery,
    selectEntity,
    selectedEntityId,
  } = useInvestigationStore();

  const filtered = useMemo(() => {
    const events = eventsQuery.data?.events ?? [];
    const needle = eventQuery.trim().toLowerCase();
    return events.filter((event) => {
      if (severityFilter && event.severity !== severityFilter) return false;
      if (!needle) return true;
      return (
        event.summary.toLowerCase().includes(needle) ||
        event.rule.toLowerCase().includes(needle) ||
        (event.entity_id ?? '').toLowerCase().includes(needle) ||
        (event.zone ?? '').toLowerCase().includes(needle)
      );
    });
  }, [eventsQuery.data, severityFilter, eventQuery]);

  const tally = useMemo(() => {
    const counts: Record<Severity, number> = { info: 0, notice: 0, alert: 0 };
    for (const event of eventsQuery.data?.events ?? []) {
      if (isSeverity(event.severity)) counts[event.severity] += 1;
    }
    return counts;
  }, [eventsQuery.data]);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-4 lg:grid-cols-[3fr_2fr]">
      <Panel
        title="Recorded events"
        aside={
          <div className="flex items-center gap-2">
            {SEVERITIES.map((severity) => (
              <span key={severity} className="flex items-center gap-1">
                <SeverityTag severity={severity} />
                <span className="stamp tabular-nums text-ink-faint">{tally[severity]}</span>
              </span>
            ))}
          </div>
        }
        bodyClassName="flex flex-col min-h-0"
      >
        <div className="flex flex-none flex-wrap items-end gap-3 border-b border-brass/12 p-3">
          <label className="flex flex-col gap-1">
            <span className="stamp text-ink-faint">Severity</span>
            <select
              id="ev-sev"
              aria-label="Severity"
              className={CONTROL_CLASS}
              value={severityFilter}
              onChange={(event) => setSeverityFilter(event.target.value as Severity | '')}
            >
              <option value="">All severities</option>
              {SEVERITIES.map((severity) => (
                <option key={severity} value={severity}>
                  {SEVERITY_LABELS[severity]}
                </option>
              ))}
            </select>
          </label>

          <label className="flex min-w-[180px] flex-1 flex-col gap-1">
            <span className="stamp text-ink-faint">Contains</span>
            <input
              type="search"
              value={eventQuery}
              onChange={(event) => setEventQuery(event.target.value)}
              placeholder="rule, zone or entity"
              className={`${CONTROL_CLASS} placeholder:text-ink-faint`}
            />
          </label>

          <span className="stamp pb-1.5 tabular-nums text-ink-faint">
            {filtered.length} shown
          </span>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-2 custom-scrollbar">
          <Resolved
            query={eventsQuery}
            what="the event log"
            emptyWhen={(data) => (data.events?.length ?? 0) === 0}
            emptyLabel="No events recorded"
            emptyHint="Rules raise events only when something happens. A quiet scene produces none."
            unavailableHint={
              <>
                Start the pipeline with <code className="text-brass">--store</code> to keep a log.
              </>
            }
          >
            {() =>
              filtered.length === 0 ? (
                <Empty what="Nothing matches this filter" />
              ) : (
                <ul className="flex flex-col gap-0.5">
                  {filtered.filter((event) => isSeverity(event.severity)).map((event) => (
                    <li key={event.id}>
                      <EventRow
                        severity={event.severity}
                        when={dayTime(event.timestamp)}
                        summary={event.summary}
                        rule={event.rule}
                        selected={Boolean(event.entity_id) && event.entity_id === selectedEntityId}
                        onClick={
                          event.entity_id
                            ? () =>
                                selectEntity(
                                  selectedEntityId === event.entity_id ? null : event.entity_id,
                                )
                            : undefined
                        }
                        meta={
                          <>
                            {event.entity_id && (
                              <span className="stamp font-mono text-brass">
                                {event.identity ?? shortEntity(event.entity_id)}
                              </span>
                            )}
                            {event.zone && <span className="stamp text-ink-faint">{event.zone}</span>}
                          </>
                        }
                      />
                    </li>
                  ))}
                </ul>
              )
            }
          </Resolved>
        </div>
      </Panel>

      <SearchPanel />
    </div>
  );
};
