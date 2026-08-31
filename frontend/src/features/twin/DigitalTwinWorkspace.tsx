/**
 * The facility seen from outside: a 3D reconstruction with a top-down plot
 * beside it.
 *
 * Both need a calibrated facility model and cross-camera positions, which only
 * the multi-camera pipeline produces. On a single-camera run this workspace says
 * so and names the command that would give it something to draw, rather than
 * rendering an invented building.
 */

import React, { Suspense, lazy } from 'react';
import { FloorplanRadar2D } from '../../components/visualizations/FloorplanRadar2D';
import type { RadarResponse, TwinResponse } from '../../contracts/types';
import type { QueryLike } from '../../components/common/Panels';
import { Loading, Panel, Resolved, Stat } from '../../components/common/Panels';
import { useInvestigationStore } from '../../store/useInvestigationStore';

// three.js is around three quarters of the JavaScript this app ships and is
// used by this one panel. Splitting it out keeps it off the critical path for
// every other workspace.
const SpatialTwin3D = lazy(() =>
  import('../../components/visualizations/SpatialTwin3D').then((module) => ({
    default: module.SpatialTwin3D,
  })),
);

const NEEDS_FACILITY = (
  <>
    Both views are built from cross-camera positions. Start the facility pipeline —{' '}
    <code className="text-brass">vantage facility --cameras front=webcam:0 yard=rtsp://…</code>
  </>
);

export const DigitalTwinWorkspace: React.FC<{
  twinQuery: QueryLike<TwinResponse>;
  radarQuery: QueryLike<RadarResponse>;
}> = ({ twinQuery, radarQuery }) => {
  const { selectedEntityId } = useInvestigationStore();

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-4 lg:grid-cols-[2fr_1fr]">
      <Panel
        title="Facility reconstruction"
        aside={
          <span className="stamp tabular-nums text-ink-faint">
            {twinQuery.data?.available
              ? `${twinQuery.data.cameras?.length ?? 0} cameras · ${
                  twinQuery.data.entities?.length ?? 0
                } tracked`
              : ''}
          </span>
        }
        bodyClassName="bg-board"
      >
        <Resolved
          query={twinQuery}
          what="the digital twin"
          unavailableHint={NEEDS_FACILITY}
          emptyWhen={(data) => !data.facility}
          emptyLabel="No facility model"
          emptyHint="The twin is attached but has no rooms defined yet."
        >
          {(data) => (
            <Suspense fallback={<Loading what="the 3D renderer" />}>
              <SpatialTwin3D twin={data} selectedEntityId={selectedEntityId} />
            </Suspense>
          )}
        </Resolved>
      </Panel>

      <div className="flex min-h-0 flex-col gap-3">
        <Panel title="Top-down" bodyClassName="bg-board">
          <Resolved query={radarQuery} what="the floor plot" unavailableHint={NEEDS_FACILITY}>
            {(data) => <FloorplanRadar2D radar={data} />}
          </Resolved>
        </Panel>

        <Panel title="On the floor" bodyClassName="p-3">
          <Resolved query={radarQuery} what="floor occupancy" unavailableHint={NEEDS_FACILITY}>
            {(data) => (
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Tracked now" value={data.active_count ?? 0} />
                <Stat label="Camera footprints" value={data.zones?.length ?? 0} />
                <Stat
                  label="Moving"
                  value={(data.entities ?? []).filter((e) => e.motion !== 'stationary').length}
                />
                <Stat
                  label="Stationary"
                  value={(data.entities ?? []).filter((e) => e.motion === 'stationary').length}
                />
              </div>
            )}
          </Resolved>
        </Panel>
      </div>
    </div>
  );
};
