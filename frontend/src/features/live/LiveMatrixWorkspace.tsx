/**
 * The live view: the annotated stream, what is being tracked in it, and the
 * events the rules have just raised.
 *
 * Two things were removed here rather than restyled. The stream had an
 * `onError` handler that swapped in `assets/hero.png` -- a stock photograph --
 * so a camera that failed to open still filled the panel with something that
 * looked like footage. And a card in the corner read `ALERT` and `40m x 24m COV`
 * as fixed text, regardless of any measurement. A dead stream now says it is
 * dead, and nothing on this screen is a constant pretending to be a reading.
 */

import React, { useState } from 'react';
import { Crosshair, Layers, VideoOff } from 'lucide-react';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import type { LiveResponse } from '../../contracts/types';
import { isSeverity } from '../../contracts/vocabulary';
import { EventRow } from '../../components/common/Severity';
import { Empty, Stat, Unavailable } from '../../components/common/Panels';
import { clockTime, shortEntity } from '../../lib/format';

const OVERLAY_KEYS = [
  { key: 'boxes' as const, label: 'Boxes' },
  { key: 'ids' as const, label: 'IDs' },
  { key: 'pose' as const, label: 'Pose' },
  { key: 'behavior' as const, label: 'Behaviour' },
  { key: 'zones' as const, label: 'Zones' },
];

const OverlayToggles: React.FC = () => {
  const { overlayOptions, toggleOverlay } = useInvestigationStore();
  return (
    <div className="flex items-center gap-2 rounded-sm border border-brass/20 bg-board-surface/95 px-2 py-1.5 shadow-paper">
      <div className="flex items-center gap-1.5 border-r border-brass/20 pr-2">
        <Layers className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
        <span className="stamp text-ink-faint">Layers</span>
      </div>
      <div className="flex items-center gap-1">
        {OVERLAY_KEYS.map(({ key, label }) => {
          const active = overlayOptions[key];
          return (
            <button
              key={key}
              type="button"
              onClick={() => toggleOverlay(key)}
              aria-pressed={active}
              className="stamp rounded-sm border px-2.5 py-1 transition-colors"
              style={{
                backgroundColor: active ? 'rgba(179,58,46,0.15)' : 'transparent',
                borderColor: active ? 'rgba(179,58,46,0.35)' : 'rgba(176,141,87,0.12)',
                color: active ? '#B33A2E' : '#6B5545',
              }}
            >
              {label}
            </button>
          );
        })}
      </div>
      <span className="pl-1 text-micro text-ink-faint/70">drawn server-side</span>
    </div>
  );
};

const Stream: React.FC<{ live: boolean }> = ({ live }) => {
  const [broken, setBroken] = useState(false);

  if (!live || broken) {
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-board">
        <VideoOff className="h-7 w-7 text-ink-faint" aria-hidden />
        <p className="stamp text-brass">No video</p>
        <p className="max-w-md px-6 text-center text-tiny leading-relaxed text-ink-faint">
          {broken
            ? 'The MJPEG stream stopped responding. The pipeline may have exited, or another program may have taken the camera.'
            : 'This dashboard is reading recorded history; no pipeline is publishing frames into it.'}
        </p>
        <p className="max-w-md px-6 text-center text-micro leading-relaxed text-ink-faint/70">
          <code className="text-brass">vantage run --source webcam:0 --track --pose --dashboard</code>
        </p>
      </div>
    );
  }

  return (
    <img
      src="/stream.mjpg"
      alt="Annotated live camera stream"
      className="pointer-events-none absolute inset-0 h-full w-full object-contain"
      onError={() => setBroken(true)}
    />
  );
};

export const LiveMatrixWorkspace: React.FC<{ live: LiveResponse | undefined; pending: boolean }> = ({
  live,
  pending,
}) => {
  const { selectedEntityId, selectEntity } = useInvestigationStore();

  if (!pending && live && !live.available) {
    return (
      <div className="flex-1">
        <Unavailable
          what="Live view"
          reason={live.reason}
          hint={
            <>
              Recorded history is still readable from the other workspaces. To watch a camera, start
              the pipeline with <code className="text-brass">--dashboard</code>.
            </>
          }
        />
      </div>
    );
  }

  const entities = live?.entities ?? [];
  const events = (live?.events ?? []).filter((event) => isSeverity(event.severity));
  const stats = live?.stats;
  const streaming = Boolean(live?.available && live?.has_frame);

  return (
    <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
      <Stream live={streaming} />

      <div
        className="pointer-events-none absolute inset-0 z-0"
        style={{
          background:
            'radial-gradient(ellipse at center, transparent 55%, rgba(20,17,13,0.7) 100%)',
        }}
        aria-hidden
      />

      {/* stream telemetry */}
      <div className="absolute left-5 top-5 z-10 flex items-center gap-4 rounded-sm border border-brass/25 bg-board-surface/95 px-3 py-2 shadow-paper">
        <span
          className="h-2 w-2 flex-none rounded-full"
          style={{ backgroundColor: streaming ? '#6B8F6B' : '#6B5545' }}
          aria-hidden
        />
        <Stat label="Feed" value={stats?.source ?? null} />
        <Stat label="fps" value={stats ? stats.fps.toFixed(1) : null} />
        <Stat
          label="Dropped"
          value={stats?.dropped ?? null}
          tone={stats && stats.dropped > 0 ? 'alert' : 'normal'}
        />
        <Stat label="Frame age" value={live?.age_s === undefined ? null : `${live.age_s}s`} />
      </div>

      <div className="absolute bottom-5 left-5 z-10">
        <OverlayToggles />
      </div>

      {/* tracked entities + recent events */}
      <div className="absolute bottom-5 right-5 top-5 z-10 flex w-[260px] flex-col gap-3">
        <section className="flex min-h-0 flex-[3] flex-col overflow-hidden rounded-sm border border-brass/25 bg-board-surface/95 shadow-paper">
          <header className="flex flex-none items-center justify-between border-b border-brass/20 px-3 py-2">
            <span className="flex items-center gap-2">
              <Crosshair className="h-3.5 w-3.5 text-ink-faint" aria-hidden />
              <span className="stamp text-brass">Tracked</span>
            </span>
            <span className="stamp rounded-sm bg-brass/10 px-1.5 py-0.5 tabular-nums text-ink-faint">
              {entities.length}
            </span>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto p-1.5 custom-scrollbar">
            {entities.length === 0 ? (
              <Empty what={streaming ? 'Nobody in frame' : 'Not tracking'} />
            ) : (
              <ul className="flex flex-col gap-1">
                {entities.map((entity) => {
                  const selected = selectedEntityId === entity.entity_id;
                  return (
                    <li key={entity.entity_id}>
                      <button
                        type="button"
                        onClick={() => selectEntity(selected ? null : entity.entity_id)}
                        aria-pressed={selected}
                        className="w-full rounded-sm px-2.5 py-2 text-left transition-colors"
                        style={{
                          backgroundColor: selected ? 'rgba(179,58,46,0.12)' : 'transparent',
                          borderLeft: `2.5px solid ${selected ? '#B33A2E' : 'transparent'}`,
                        }}
                      >
                        <div className="flex items-baseline justify-between gap-2">
                          <span
                            className="font-mono text-tiny font-semibold"
                            style={{ color: selected ? '#B33A2E' : '#C4B898' }}
                          >
                            {entity.identity ?? shortEntity(entity.entity_id)}
                          </span>
                          <span className="stamp tabular-nums text-ink-faint">
                            {entity.speed.toFixed(2)} h/s
                          </span>
                        </div>
                        <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-micro text-ink-faint">
                          <span>{entity.motion}</span>
                          {entity.posture && <span>· {entity.posture}</span>}
                          {entity.activities.length > 0 && (
                            <span className="text-brass">· {entity.activities.join(', ')}</span>
                          )}
                          {entity.zones.length > 0 && <span>· {entity.zones.join(', ')}</span>}
                        </div>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        </section>

        <section className="flex min-h-0 flex-[2] flex-col overflow-hidden rounded-sm border border-brass/25 bg-board-surface/95 shadow-paper">
          <header className="flex-none border-b border-brass/20 px-3 py-2">
            <span className="stamp text-brass">Just raised</span>
          </header>
          <div className="min-h-0 flex-1 overflow-y-auto p-1.5 custom-scrollbar">
            {events.length === 0 ? (
              <Empty what="No events" />
            ) : (
              <ul className="flex flex-col gap-0.5">
                {events
                  .slice()
                  .reverse()
                  .map((event, index) => (
                    <li key={`${event.timestamp}-${index}`}>
                      <EventRow
                        severity={event.severity}
                        when={clockTime(event.timestamp)}
                        summary={event.summary}
                        rule={event.rule}
                        meta={event.zone ? <span className="stamp text-brass">{event.zone}</span> : null}
                      />
                    </li>
                  ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
