/**
 * The shell: one place that owns the polling, and six workspaces that render
 * whatever it got.
 *
 * Queries are declared here rather than inside each workspace so that the header
 * and the ribbon read the same responses the panels do -- a header showing a
 * frame rate the live panel disagrees with is worse than a header showing
 * nothing. Every count handed downward is `number | null`; nothing on this
 * screen substitutes a plausible figure for a missing one.
 *
 * Polling intervals differ by how fast the thing underneath actually changes.
 * The live snapshot moves every frame; the stored event log does not, and asking
 * for it thirty times a second would be work the store has to do for no new
 * information.
 */

import React, { useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from './data/source';
import { useInvestigationStore, WORKSPACE_IDS } from './store/useInvestigationStore';
import type { WorkspaceId } from './store/useInvestigationStore';
import { SEVERITY_RANK, isSeverity } from './contracts/vocabulary';
import type { Severity } from './contracts/vocabulary';
import { ErrorBoundary } from './ErrorBoundary';
import { Header } from './components/shell/Header';
import { IntelligenceRibbon } from './components/shell/IntelligenceRibbon';
import { OperationsDrawer } from './components/shell/OperationsDrawer';
import { EntityDossierDrawer } from './components/dossiers/EntityDossierDrawer';
import { IncidentDossierDrawer } from './components/dossiers/IncidentDossierDrawer';
import { RelationshipInspectorDrawer } from './components/dossiers/RelationshipInspectorDrawer';
import { VideoClipPlayerModal } from './components/visualizations/VideoClipPlayerModal';
import { LiveMatrixWorkspace } from './features/live/LiveMatrixWorkspace';
import { IncidentsWorkspace } from './features/incidents/IncidentsWorkspace';
import { AnalyticsWorkspace } from './features/analytics/AnalyticsWorkspace';
import { IntelligenceWorkspace } from './features/intelligence/IntelligenceWorkspace';
import { InvestigateWorkspace } from './features/investigate/InvestigateWorkspace';
import { DigitalTwinWorkspace } from './features/twin/DigitalTwinWorkspace';

export const App: React.FC = () => (
  <ErrorBoundary>
    <AppContent />
  </ErrorBoundary>
);

/**
 * Keep the workspace in step with the address bar.
 *
 * The store seeds itself from `location.hash` once at startup and writes to it
 * on every change, which made the URL shareable but one-way: the browser's back
 * button, a bookmark opened in an already-loaded tab, and a hand-edited hash all
 * changed the address and left the page where it was.
 */
const useHashRouting = () => {
  const setActiveWorkspace = useInvestigationStore((state) => state.setActiveWorkspace);

  useEffect(() => {
    const onHashChange = () => {
      const hash = window.location.hash.replace('#', '');
      if ((WORKSPACE_IDS as string[]).includes(hash)) {
        setActiveWorkspace(hash as WorkspaceId);
      }
    };
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, [setActiveWorkspace]);
};

/** Number keys jump between workspaces; Escape closes whatever is open. */
const useShortcuts = () => {
  const {
    setActiveWorkspace,
    selectEntity,
    selectIncident,
    clearRelationship,
    setOperationsDrawerOpen,
  } = useInvestigationStore();

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      // Never steal a keystroke from something the operator is typing into.
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;

      if (event.key === 'Escape') {
        selectEntity(null);
        selectIncident(null);
        clearRelationship();
        setOperationsDrawerOpen(false);
        return;
      }
      const index = Number.parseInt(event.key, 10) - 1;
      if (Number.isInteger(index) && index >= 0 && index < WORKSPACE_IDS.length) {
        setActiveWorkspace(WORKSPACE_IDS[index]);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [setActiveWorkspace, selectEntity, selectIncident, clearRelationship, setOperationsDrawerOpen]);
};

const AppContent: React.FC = () => {
  const {
    activeWorkspace,
    selectedEntityId,
    selectEntity,
    selectedIncidentId,
    selectIncident,
    selectedRelationship,
    clearRelationship,
  } = useInvestigationStore();

  useHashRouting();
  useShortcuts();

  const liveQuery = useQuery({
    queryKey: ['live'],
    queryFn: ({ signal }) => api.live(signal),
    refetchInterval: 1_000,
    retry: false,
  });
  const statsQuery = useQuery({
    queryKey: ['stats'],
    queryFn: ({ signal }) => api.stats(signal),
    refetchInterval: 10_000,
    retry: false,
  });
  const incidentsQuery = useQuery({
    queryKey: ['incidents'],
    queryFn: ({ signal }) => api.incidents(50, signal),
    refetchInterval: 5_000,
    retry: false,
  });
  const eventsQuery = useQuery({
    queryKey: ['events'],
    queryFn: ({ signal }) => api.events(200, signal),
    refetchInterval: 10_000,
    retry: false,
  });
  const graphQuery = useQuery({
    queryKey: ['relationship-graph'],
    queryFn: ({ signal }) => api.relationshipGraph(signal),
    refetchInterval: 15_000,
    retry: false,
    // Always polled, unlike the scene and twin queries below: the ribbon reports
    // the association count on every workspace, and a count that only appears
    // after you visit one particular page is worse than no count.
  });
  const sceneQuery = useQuery({
    queryKey: ['scene'],
    queryFn: ({ signal }) => api.scene(signal),
    refetchInterval: 5_000,
    retry: false,
    enabled: activeWorkspace === 'intelligence',
  });
  const twinQuery = useQuery({
    queryKey: ['twin'],
    queryFn: ({ signal }) => api.twin(signal),
    refetchInterval: 3_000,
    retry: false,
    enabled: activeWorkspace === 'twin',
  });
  const radarQuery = useQuery({
    queryKey: ['radar'],
    queryFn: ({ signal }) => api.radar(signal),
    refetchInterval: 2_000,
    retry: false,
    enabled: activeWorkspace === 'twin',
  });

  const live = liveQuery.data;
  const incidents = incidentsQuery.data?.incidents;
  const activeIncidents = incidents?.filter((incident) => incident.state === 'active');

  const worstActive: Severity | null =
    activeIncidents && activeIncidents.length > 0
      ? activeIncidents
          .map((incident) => incident.severity)
          .filter(isSeverity)
          .reduce<Severity>(
            (worst, severity) => (SEVERITY_RANK[severity] > SEVERITY_RANK[worst] ? severity : worst),
            'info',
          )
      : null;

  return (
    <div className="bg-cork flex h-screen w-screen select-none flex-col overflow-hidden font-sans text-warm-white">
      <Header
        fps={live?.available ? (live.stats?.fps ?? null) : null}
        streaming={Boolean(live?.available && live.has_frame)}
        worstActive={worstActive}
        activeIncidents={incidentsQuery.data?.available ? (activeIncidents?.length ?? 0) : null}
      />

      <IntelligenceRibbon
        tracked={live?.available ? (live.entities?.length ?? 0) : null}
        incidents={incidentsQuery.data?.available ? (incidents?.length ?? 0) : null}
        associations={
          graphQuery.data?.available ? (graphQuery.data.graph?.total_edges ?? 0) : null
        }
        eventsStored={statsQuery.data?.store ? statsQuery.data.store.events : null}
      />

      <main className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        {activeWorkspace === 'live' && (
          <LiveMatrixWorkspace live={live} pending={liveQuery.isPending} />
        )}
        {activeWorkspace === 'incidents' && <IncidentsWorkspace query={incidentsQuery} />}
        {activeWorkspace === 'trends' && <AnalyticsWorkspace />}
        {activeWorkspace === 'intelligence' && (
          <IntelligenceWorkspace graphQuery={graphQuery} sceneQuery={sceneQuery} />
        )}
        {activeWorkspace === 'investigate' && <InvestigateWorkspace eventsQuery={eventsQuery} />}
        {activeWorkspace === 'twin' && (
          <DigitalTwinWorkspace twinQuery={twinQuery} radarQuery={radarQuery} />
        )}

        {selectedEntityId && (
          <EntityDossierDrawer entityId={selectedEntityId} onClose={() => selectEntity(null)} />
        )}
        {selectedIncidentId && (
          <IncidentDossierDrawer
            incidentId={selectedIncidentId}
            onClose={() => selectIncident(null)}
          />
        )}
        {selectedRelationship && (
          <RelationshipInspectorDrawer
            entityA={selectedRelationship.entityAId}
            entityB={selectedRelationship.entityBId}
            onClose={clearRelationship}
          />
        )}

        <OperationsDrawer health={live?.health} stats={statsQuery.data} />
      </main>

      <VideoClipPlayerModal />
    </div>
  );
};
