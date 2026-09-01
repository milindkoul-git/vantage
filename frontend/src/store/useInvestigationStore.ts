import { create } from 'zustand';
import { prefersReducedMotion } from '../lib/motion';
import type { RendererStats } from '../components/visualizations/SpatialTwin3D';
import type { AnalyticsWindow, Metric, Severity } from '../contracts/vocabulary';

export type WorkspaceId = 'live' | 'incidents' | 'trends' | 'intelligence' | 'investigate' | 'twin';

export const WORKSPACE_IDS: WorkspaceId[] = [
  'live',
  'incidents',
  'trends',
  'intelligence',
  'investigate',
  'twin',
];

export interface OverlayOptions {
  boxes: boolean;
  ids: boolean;
  pose: boolean;
  behavior: boolean;
  zones: boolean;
  trails: boolean;
}

export interface InvestigationStore {
  activeWorkspace: WorkspaceId;
  selectedEntityId: string | null;
  selectedIncidentId: string | null;
  selectedRelationship: { entityAId: string; entityBId: string } | null;
  activeEvidenceClip: { url: string; title: string } | null;
  isOperationsDrawerOpen: boolean;
  overlayOptions: OverlayOptions;

  /** Event list filter. Empty string means "every severity". */
  severityFilter: Severity | '';
  /** Free-text filter over the event list, applied client-side. */
  eventQuery: string;

  analyticsMetric: Metric;
  analyticsWindow: AnalyticsWindow;

  /**
   * The last frame's `renderer.info` from the twin, or null when it is not open.
   *
   * Here rather than local to the twin because the operations drawer is where
   * this console reports its own health, and the browser is part of that. A
   * dashboard meant to run for a shift needs the same leak check the pipeline
   * gets: if geometries or textures climb while nothing is happening, something
   * is not being disposed.
   */
  rendererStats: RendererStats | null;

  setActiveWorkspace: (ws: WorkspaceId) => void;
  selectEntity: (id: string | null) => void;
  selectIncident: (id: string | null) => void;
  selectRelationship: (entityA: string, entityB: string) => void;
  clearRelationship: () => void;
  playEvidenceClip: (url: string, title: string) => void;
  closeEvidenceClip: () => void;
  setOperationsDrawerOpen: (open: boolean) => void;
  toggleOverlay: (key: keyof OverlayOptions) => void;
  setSeverityFilter: (severity: Severity | '') => void;
  setEventQuery: (query: string) => void;
  setAnalyticsMetric: (metric: Metric) => void;
  setAnalyticsWindow: (window: AnalyticsWindow) => void;
  setRendererStats: (stats: RendererStats | null) => void;
}

const isWorkspace = (value: string): value is WorkspaceId =>
  (WORKSPACE_IDS as string[]).includes(value);

const initialWorkspace = (): WorkspaceId => {
  if (typeof window === 'undefined') return 'live';
  const hash = window.location.hash.replace('#', '');
  return isWorkspace(hash) ? hash : 'live';
};

const queryParam = (name: string): string | null => {
  if (typeof window === 'undefined') return null;
  return new URLSearchParams(window.location.search).get(name);
};

/**
 * There is deliberately no demo mode here.
 *
 * This store used to carry `isDemoMode: true` -- commented "so evaluator
 * immediately sees rich intelligence" -- which meant the app opened onto a
 * hand-written fixture set rather than the camera. Everything the page shows now
 * came from the pipeline, or the page says it has nothing.
 */
export const useInvestigationStore = create<InvestigationStore>((set) => ({
  activeWorkspace: initialWorkspace(),
  selectedEntityId: queryParam('entity'),
  selectedIncidentId: queryParam('incident'),
  selectedRelationship: null,
  activeEvidenceClip: null,
  isOperationsDrawerOpen: queryParam('operations') === 'true',
  overlayOptions: {
    boxes: true,
    ids: true,
    pose: true,
    behavior: true,
    zones: true,
    trails: false,
  },
  severityFilter: '',
  eventQuery: '',
  analyticsMetric: 'entities',
  analyticsWindow: '24h',
  rendererStats: null,

  setActiveWorkspace: (ws) => {
    if (typeof window !== 'undefined') window.location.hash = ws;
    // View Transitions where the browser has them: the six workspaces are
    // different views of one facility, and a hard cut between them makes the
    // operator re-find themselves each time. Same-document transitions are
    // supported everywhere current; where they are not, or where the viewer has
    // asked for less movement, this is a plain state change and the page cuts.
    const start = (document as Document & {
      startViewTransition?: (cb: () => void) => unknown;
    }).startViewTransition;
    if (start && !prefersReducedMotion()) {
      start.call(document, () => set({ activeWorkspace: ws }));
      return;
    }
    set({ activeWorkspace: ws });
  },
  selectEntity: (id) => set({ selectedEntityId: id }),
  selectIncident: (id) => set({ selectedIncidentId: id }),
  selectRelationship: (entityAId, entityBId) => set({ selectedRelationship: { entityAId, entityBId } }),
  clearRelationship: () => set({ selectedRelationship: null }),
  playEvidenceClip: (url, title) => set({ activeEvidenceClip: { url, title } }),
  closeEvidenceClip: () => set({ activeEvidenceClip: null }),
  setOperationsDrawerOpen: (open) => set({ isOperationsDrawerOpen: open }),
  toggleOverlay: (key) =>
    set((state) => ({
      overlayOptions: { ...state.overlayOptions, [key]: !state.overlayOptions[key] },
    })),
  setSeverityFilter: (severityFilter) => set({ severityFilter }),
  setEventQuery: (eventQuery) => set({ eventQuery }),
  setAnalyticsMetric: (analyticsMetric) => set({ analyticsMetric }),
  setAnalyticsWindow: (analyticsWindow) => set({ analyticsWindow }),
  setRendererStats: (rendererStats) => set({ rendererStats }),
}));
