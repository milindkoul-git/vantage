/**
 * The top bar: where you are, whether the pipeline is alive, and the way into
 * the telemetry drawer.
 *
 * The readouts here have no defaults. The previous version declared
 * `fps = 24.0`, `threatLevel = 'ALERT'` and `activeIncidentsCount = 2` as default
 * props and painted the status dot green unconditionally, so a header rendered
 * with no data at all looked like a healthy pipeline watching a facility in
 * alarm. Every value is now either measured or an em dash.
 */

import React from 'react';
import {
  Activity,
  AlertTriangle,
  Layers,
  LineChart,
  Network,
  Search,
  Settings,
  Video,
} from 'lucide-react';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import type { WorkspaceId } from '../../store/useInvestigationStore';
import type { Severity } from '../../contracts/vocabulary';
import { SEVERITY_COLOR } from '../../contracts/vocabulary';

export interface HeaderProps {
  /** Frames per second the pipeline is delivering, or null when not live. */
  fps: number | null;
  /** True once the live feed has published at least one frame. */
  streaming: boolean;
  /** Worst severity among currently active incidents, or null when there are none. */
  worstActive: Severity | null;
  activeIncidents: number | null;
}

const WORKSPACES: Array<{ id: WorkspaceId; label: string; icon: React.ReactNode }> = [
  { id: 'live', label: 'Live', icon: <Video className="h-3.5 w-3.5" /> },
  { id: 'incidents', label: 'Incidents', icon: <AlertTriangle className="h-3.5 w-3.5" /> },
  { id: 'trends', label: 'Trends', icon: <LineChart className="h-3.5 w-3.5" /> },
  { id: 'intelligence', label: 'Intelligence', icon: <Network className="h-3.5 w-3.5" /> },
  { id: 'investigate', label: 'Investigate', icon: <Search className="h-3.5 w-3.5" /> },
  { id: 'twin', label: 'Twin', icon: <Layers className="h-3.5 w-3.5" /> },
];

export const Header: React.FC<HeaderProps> = ({ fps, streaming, worstActive, activeIncidents }) => {
  const {
    activeWorkspace,
    setActiveWorkspace,
    isOperationsDrawerOpen,
    setOperationsDrawerOpen,
  } = useInvestigationStore();

  return (
    <header className="relative z-50 flex h-[52px] flex-none items-center justify-between gap-4 border-b border-brass/20 bg-board-surface px-4">
      <div className="flex flex-none items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-brass" aria-hidden />
        <span className="font-serif text-[17px] font-bold tracking-wide text-warm-white">
          Vantage
        </span>
      </div>

      <nav aria-label="Workspaces" className="flex min-w-0 flex-1 items-center justify-center">
        {WORKSPACES.map((workspace) => {
          const isActive = activeWorkspace === workspace.id;
          const badge = workspace.id === 'incidents' ? activeIncidents : null;
          return (
            <button
              key={workspace.id}
              type="button"
              onClick={() => setActiveWorkspace(workspace.id)}
              aria-current={isActive ? 'page' : undefined}
              className="relative flex h-[52px] items-center gap-1.5 px-3 text-tiny transition-colors"
              style={{
                color: isActive ? '#E8E2D4' : '#6B5545',
                fontWeight: isActive ? 500 : 400,
              }}
            >
              {isActive && (
                <span className="absolute bottom-0 left-2 right-2 h-[1.5px] bg-brass" aria-hidden />
              )}
              <span style={{ opacity: isActive ? 1 : 0.7 }}>{workspace.icon}</span>
              <span className="hidden md:inline">{workspace.label}</span>
              {badge !== null && badge > 0 && (
                <span className="ml-0.5 flex h-4 min-w-4 items-center justify-center rounded-full bg-string-red px-1 font-mono text-[9px] font-bold text-warm-white">
                  {badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="flex flex-none items-center gap-3">
        <div
          className="flex items-center gap-1.5 rounded-sm border border-brass/15 bg-board/80 px-2 py-1"
          title={streaming ? 'Frames are arriving' : 'No frames are arriving'}
        >
          <span
            className="h-1.5 w-1.5 rounded-full"
            style={{ backgroundColor: streaming ? '#6B8F6B' : '#6B5545' }}
            aria-hidden
          />
          <span className="stamp tabular-nums text-warm-white">
            {fps === null ? '—' : fps.toFixed(1)}
          </span>
          <span className="stamp text-ink-faint">fps</span>
        </div>

        <div
          className="rounded-sm px-2 py-0.5"
          style={{
            border: `1px solid ${worstActive ? `${SEVERITY_COLOR[worstActive]}66` : 'rgba(107,85,69,0.4)'}`,
            backgroundColor: worstActive ? `${SEVERITY_COLOR[worstActive]}1F` : 'transparent',
          }}
          title="Worst severity among incidents that are currently active"
        >
          <span
            className="stamp"
            style={{ color: worstActive ? SEVERITY_COLOR[worstActive] : '#6B5545' }}
          >
            {activeIncidents === null ? 'No data' : worstActive ? worstActive : 'Quiet'}
          </span>
        </div>

        <span className="h-4 w-px bg-brass/20" aria-hidden />

        <button
          type="button"
          onClick={() => setOperationsDrawerOpen(!isOperationsDrawerOpen)}
          aria-label="Toggle pipeline telemetry"
          aria-expanded={isOperationsDrawerOpen}
          className="rounded-sm p-1.5 transition-colors"
          style={{
            color: isOperationsDrawerOpen ? '#B08D57' : '#6B5545',
            backgroundColor: isOperationsDrawerOpen ? 'rgba(176,141,87,0.10)' : 'transparent',
          }}
        >
          {isOperationsDrawerOpen ? (
            <Activity className="h-4 w-4" />
          ) : (
            <Settings className="h-4 w-4" />
          )}
        </button>
      </div>
    </header>
  );
};
