/**
 * A single strip of counts under the header.
 *
 * Each cell is `number | null`, and null renders as an em dash rather than as a
 * plausible figure. That is the whole change from the version this replaces,
 * which was fed `entityCount || 3`, `relationshipCount || 1`, `incidentCount ||
 * 2` and a `sceneEdgeCount` of a literal `4` that read from nothing at all -- so
 * a dashboard that could not reach its own API displayed a facility with three
 * people, one association and two incidents in it.
 */

import React from 'react';
import { AlertTriangle, Link2, Radio, Users } from 'lucide-react';

interface Cell {
  label: string;
  value: number | null;
  icon: React.ReactNode;
  hint: string;
}

export interface RibbonProps {
  tracked: number | null;
  incidents: number | null;
  associations: number | null;
  eventsStored: number | null;
}

export const IntelligenceRibbon: React.FC<RibbonProps> = ({
  tracked,
  incidents,
  associations,
  eventsStored,
}) => {
  const cells: Cell[] = [
    {
      label: 'In frame',
      value: tracked,
      icon: <Users className="h-3 w-3" />,
      hint: 'Entities the tracker is holding right now',
    },
    {
      label: 'Incidents',
      value: incidents,
      icon: <AlertTriangle className="h-3 w-3" />,
      hint: 'Correlated groups of raised events',
    },
    {
      label: 'Associations',
      value: associations,
      icon: <Link2 className="h-3 w-3" />,
      hint: 'Pairs of entities seen together often enough to score an edge',
    },
    {
      label: 'Events stored',
      value: eventsStored,
      icon: <Radio className="h-3 w-3" />,
      hint: 'Rows in the events table of the attached store',
    },
  ];

  return (
    <div className="flex flex-none items-center gap-6 border-b border-brass/12 bg-board/60 px-4 py-1.5">
      {cells.map((cell) => (
        <div key={cell.label} className="flex items-center gap-1.5" title={cell.hint}>
          <span className="text-ink-faint" aria-hidden>
            {cell.icon}
          </span>
          <span className="stamp text-ink-faint">{cell.label}</span>
          <span
            className="font-mono text-tiny tabular-nums"
            style={{ color: cell.value === null ? '#6B5545' : '#E8E2D4' }}
          >
            {cell.value === null ? '—' : cell.value}
          </span>
        </div>
      ))}
    </div>
  );
};
