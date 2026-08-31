/**
 * Who keeps appearing with whom, and what is happening between people in one
 * scene right now.
 *
 * The graph is the persistent relationship layer, which a single camera can
 * build. The scene topology beside it is transient per-camera structure and only
 * exists in the facility pipeline; it says so rather than rendering an empty
 * board.
 */

import React from 'react';
import type { RelationshipGraphResponse, SceneResponse } from '../../contracts/types';
import type { QueryLike } from '../../components/common/Panels';
import { Panel, Resolved } from '../../components/common/Panels';
import { shortEntity } from '../../lib/format';
import { ForceDirectedGraph } from '../../components/visualizations/ForceDirectedGraph';
import { SceneTopologyPanel } from '../../components/visualizations/SceneTopologyPanel';
import { useInvestigationStore } from '../../store/useInvestigationStore';

export const IntelligenceWorkspace: React.FC<{
  graphQuery: QueryLike<RelationshipGraphResponse>;
  sceneQuery: QueryLike<SceneResponse>;
}> = ({ graphQuery, sceneQuery }) => {
  const selectRelationship = useInvestigationStore((state) => state.selectRelationship);

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 p-4 lg:grid-cols-[2fr_1fr]">
      <Panel
        title="Association graph"
        aside={
          graphQuery.data?.graph ? (
            <span className="stamp tabular-nums text-ink-faint">
              {graphQuery.data.graph.total_nodes} entities · {graphQuery.data.graph.total_edges} edges
            </span>
          ) : null
        }
        bodyClassName="relative bg-board"
      >
        <Resolved
          query={graphQuery}
          what="the association graph"
          emptyWhen={(data) => (data.graph?.nodes.length ?? 0) === 0}
          emptyLabel="No associations yet"
          emptyHint="An edge needs two entities in frame together across several observations."
          unavailableHint={
            <>
              Relationship tracking is off by default because it accumulates state about pairs across
              a whole session. Turn it on with{' '}
              <code className="text-brass">--set relationships.enabled=true</code>.
            </>
          }
        >
          {(data) => <ForceDirectedGraph graphData={data.graph} />}
        </Resolved>
      </Panel>

      <div className="flex min-h-0 flex-col gap-3">
        <Panel title="Strongest pairs" bodyClassName="overflow-y-auto custom-scrollbar p-2">
          <Resolved
            query={graphQuery}
            what="associations"
            emptyWhen={(data) => (data.graph?.edges.length ?? 0) === 0}
            emptyLabel="No pairs scored yet"
          >
            {(data) => (
              <ul className="flex flex-col gap-1">
                {[...(data.graph?.edges ?? [])]
                  .sort((a, b) => b.active_strength - a.active_strength)
                  .slice(0, 25)
                  .map((edge) => (
                    <li key={`${edge.source}-${edge.target}`}>
                      <button
                        type="button"
                        onClick={() => selectRelationship(edge.source, edge.target)}
                        className="flex w-full items-baseline gap-2 rounded-sm px-2 py-1.5 text-left transition-colors hover:bg-brass/5"
                      >
                        <span className="flex-none font-mono text-tiny text-warm-white">
                          {shortEntity(edge.source)} ↔ {shortEntity(edge.target)}
                        </span>
                        <span className="min-w-0 flex-1 truncate text-micro text-ink-faint">
                          {edge.pattern?.replace(/_/g, ' ') ?? 'co-occurrence'}
                        </span>
                        <span className="stamp flex-none tabular-nums text-brass">
                          {edge.active_strength.toFixed(2)}
                        </span>
                      </button>
                    </li>
                  ))}
              </ul>
            )}
          </Resolved>
        </Panel>

        <Panel title="Scene topology" bodyClassName="overflow-y-auto custom-scrollbar">
          <Resolved
            query={sceneQuery}
            what="scene topology"
            emptyWhen={(data) => Object.keys(data.cameras ?? {}).length === 0}
            emptyLabel="No scene graph"
            unavailableHint={
              <>
                Transient per-camera structure — clusters, interactions, unattended objects — is
                built by the facility pipeline. Start it with{' '}
                <code className="text-brass">vantage facility</code>.
              </>
            }
          >
            {(data) => <SceneTopologyPanel sceneData={data} />}
          </Resolved>
        </Panel>
      </div>
    </div>
  );
};
