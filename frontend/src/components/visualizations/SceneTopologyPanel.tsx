/**
 * What the transient scene graph is seeing in each camera right now:
 * interaction edges between people, collective behaviours, and objects whose
 * owner has walked away from them.
 *
 * The shapes here are the ones `SceneGraphSnapshot.to_dict` emits. The version
 * this replaces read `entities`, `interaction_edges`, `clusters`,
 * `density_level` and `group_motion` -- none of which the snapshot has ever had
 * -- and defaulted the cohesion state to the string `converging_convoy` when it
 * found nothing, so an empty scene reported a convoy.
 */

import React from 'react';
import { Package, Users } from 'lucide-react';
import type { SceneResponse } from '../../contracts/types';
import { duration, shortEntity } from '../../lib/format';

export const SceneTopologyPanel: React.FC<{ sceneData: SceneResponse }> = ({ sceneData }) => {
  const cameras = Object.entries(sceneData.cameras ?? {});

  return (
    <div className="flex flex-col gap-2 p-2">
      {cameras.map(([cameraId, camera]) => (
        <article
          key={cameraId}
          className="rounded-sm border border-brass/20 bg-board p-3 text-tiny"
        >
          <header className="flex items-baseline justify-between gap-2">
            <span className="font-mono text-brass">{cameraId}</span>
            <span className="stamp tabular-nums text-ink-faint">
              {camera.entity_count} in frame · {camera.active_edges.length} interactions
            </span>
          </header>

          {camera.active_edges.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1">
              {camera.active_edges.slice(0, 6).map((edge) => (
                <li
                  key={`${edge.source}-${edge.target}-${edge.relation}`}
                  className="flex items-baseline gap-2"
                >
                  <span className="flex-none font-mono text-micro text-warm-white">
                    {shortEntity(edge.source)} ↔ {shortEntity(edge.target)}
                  </span>
                  <span className="min-w-0 flex-1 truncate text-micro text-ink-faint">
                    {edge.relation.replace(/_/g, ' ')} · {edge.evidence}
                  </span>
                  <span className="stamp flex-none tabular-nums text-brass">
                    {edge.confidence.toFixed(2)}
                  </span>
                </li>
              ))}
            </ul>
          )}

          {camera.collective_behaviors.length > 0 && (
            <ul className="mt-2 flex flex-col gap-1 border-t border-brass/10 pt-2">
              {camera.collective_behaviors.map((behavior) => (
                <li key={`${behavior.type}-${behavior.entities.join()}`} className="flex gap-2">
                  <Users className="mt-0.5 h-3 w-3 flex-none text-brass" aria-hidden />
                  <span className="min-w-0 flex-1">
                    <span className="text-warm-white">{behavior.type.replace(/_/g, ' ')}</span>{' '}
                    <span className="text-micro text-ink-faint">
                      ({behavior.entities.length} entities) — {behavior.evidence}
                    </span>
                  </span>
                </li>
              ))}
            </ul>
          )}

          {camera.unattended_objects.map((object) => (
            <div
              key={object.object_id}
              className="mt-2 rounded-sm border border-string-red/35 bg-string-red/10 p-2"
            >
              <div className="flex items-baseline justify-between gap-2">
                <span className="flex items-center gap-1.5 font-semibold text-string-red">
                  <Package className="h-3.5 w-3.5" aria-hidden />
                  Unattended {object.label}
                </span>
                <span className="stamp tabular-nums text-string-red">
                  {duration(object.unattended_dwell_s)}
                </span>
              </div>
              <p className="mt-1 text-micro text-ink-faint">
                Owner {object.owner_id ? shortEntity(object.owner_id) : 'unknown'} ·{' '}
                {object.owner_distance_norm.toFixed(2)} of a frame away
              </p>
            </div>
          ))}

          {camera.active_edges.length === 0 &&
            camera.collective_behaviors.length === 0 &&
            camera.unattended_objects.length === 0 && (
              <p className="mt-2 text-micro text-ink-faint">Nothing interacting.</p>
            )}
        </article>
      ))}
    </div>
  );
};
