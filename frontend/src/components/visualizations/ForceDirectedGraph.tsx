
import React, { useEffect, useRef, useState, useCallback } from 'react';
import { useInvestigationStore } from '../../store/useInvestigationStore';
import type { RelationshipGraphResponse } from '../../contracts/types';
import { DURATION, EASE, duration } from '../../lib/motion';

interface GraphProps {
  graphData?: RelationshipGraphResponse['graph'];
  /**
   * How many entities to lay out at once.
   *
   * A facility that has been running for an afternoon produces hundreds of
   * pairs, and no layout makes three hundred of these cards readable: each is
   * 120x150, so two dozen already claim half the panel whatever the algorithm
   * does. The graph shows the strongest sub-graph and says so in its legend;
   * the full ranking is in the list beside it, which stays readable at any size.
   */
  maxNodes?: number;
}

interface NodeState {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  degree: number;
  maxStrength: number;
  rot: number;
}

// Stable hash-based rotation — does not re-randomize on re-render
function seededRot(seed: string): number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(31, h) + seed.charCodeAt(i) | 0;
  }
  return ((h >>> 0) / 0xFFFFFFFF) * 5 - 2.5;
}

/**
 * The type roles on a card, as three sizes rather than six inline ones.
 *
 * This component arrived with `fontSize` written inline twelve times at 7px,
 * 8px, 9px, 11px and 0.6rem - five sizes doing three jobs, none of them from
 * the project's scale, and three of them below the 10px floor the rest of the
 * interface treats as the smallest thing worth asking anyone to read. The names
 * below are the jobs; the sizes match `tailwind.config.js`.
 */
const TYPE = {
  /** The entity id on a card. The one thing on it that must be readable. */
  cardName: { fontSize: '0.625rem', letterSpacing: '0.06em' },
  /** Its link count, and the edge labels on the strings between cards. */
  cardMeta: { fontSize: '0.625rem', letterSpacing: '0.04em' },
  /** The board watermark and the legend. Decorative, deliberately quiet. */
  board: { fontSize: '0.6875rem', letterSpacing: '0.15em' },
} as const;

const CARD_W = 120;
const CARD_H = 150;

/**
 * A Fruchterman-Reingold relaxation, run once when the graph changes.
 *
 * This component was called a force-directed graph and did no force layout: it
 * placed every node on one circle and left it there, so anything past a handful
 * of entities rendered as a ring of overlapping cards. The pass below is the
 * layout the name promised.
 *
 * Run to completion synchronously rather than animated per frame. The graph is
 * repolled every fifteen seconds, and cards drifting into place each time would
 * make a panel an operator is trying to read move under them; a settled layout
 * that changes when the data does is the more useful behaviour. A few hundred
 * iterations over a few dozen nodes is well under a frame's budget.
 */
function relax(
  nodes: NodeState[],
  edges: Array<{ source: string; target: string; active_strength: number }>,
  width: number,
  height: number,
): void {
  const count = nodes.length;
  if (count < 2) return;

  const index = new Map(nodes.map((node, i) => [node.id, i]));
  // The ideal separation for this many nodes in this much room, which is what
  // makes both forces scale-free: k is the distance at which they balance.
  // Scaled down from the full area because the nodes are cards rather than
  // points - at the textbook value they push each other flat against the panel
  // edges and leave the middle empty.
  const k = Math.sqrt((width * height * 0.45) / count);
  const centreX = width / 2 - CARD_W / 2;
  const centreY = height / 2 - CARD_H / 2;
  const ITERATIONS = 300;

  for (let step = 0; step < ITERATIONS; step += 1) {
    // Cooling: large corrections first, then settling, so the layout converges
    // instead of oscillating between two equally bad arrangements.
    const temperature = k * 0.25 * (1 - step / ITERATIONS);
    const dx = new Float64Array(count);
    const dy = new Float64Array(count);

    for (let i = 0; i < count; i += 1) {
      for (let j = i + 1; j < count; j += 1) {
        let vx = nodes[i].x - nodes[j].x;
        let vy = nodes[i].y - nodes[j].y;
        let distance = Math.hypot(vx, vy);
        if (distance < 1e-3) {
          // Two nodes exactly on top of each other have no direction to
          // separate along; nudge them deterministically by index so the layout
          // stays reproducible rather than depending on a random seed.
          vx = ((i % 7) - 3) * 0.5 + 0.1;
          vy = ((j % 5) - 2) * 0.5 + 0.1;
          distance = Math.hypot(vx, vy);
        }
        const repulsion = (k * k) / distance;
        const ux = (vx / distance) * repulsion;
        const uy = (vy / distance) * repulsion;
        dx[i] += ux;
        dy[i] += uy;
        dx[j] -= ux;
        dy[j] -= uy;
      }
    }

    for (const edge of edges) {
      const a = index.get(edge.source);
      const b = index.get(edge.target);
      if (a === undefined || b === undefined) continue;
      const vx = nodes[a].x - nodes[b].x;
      const vy = nodes[a].y - nodes[b].y;
      const distance = Math.max(1e-3, Math.hypot(vx, vy));
      // Stronger associations pull harder, so the graph's shape carries the
      // same information the edge widths do.
      const attraction = ((distance * distance) / k) * (0.5 + edge.active_strength);
      const ux = (vx / distance) * attraction;
      const uy = (vy / distance) * attraction;
      dx[a] -= ux;
      dy[a] -= uy;
      dx[b] += ux;
      dy[b] += uy;
    }

    // Gravity toward the middle. Without it the components of a sparse graph -
    // and a relationship graph is mostly sparse - have nothing holding them
    // together and drift until the bounds clamp stops them.
    for (let i = 0; i < count; i += 1) {
      dx[i] += (centreX - nodes[i].x) * 0.08;
      dy[i] += (centreY - nodes[i].y) * 0.08;
    }

    for (let i = 0; i < count; i += 1) {
      const magnitude = Math.max(1e-6, Math.hypot(dx[i], dy[i]));
      const limit = Math.min(magnitude, temperature);
      nodes[i].x += (dx[i] / magnitude) * limit;
      nodes[i].y += (dy[i] / magnitude) * limit;
      // Keep whole cards on screen: a node placed at the panel edge would put
      // most of its card outside it.
      nodes[i].x = Math.max(8, Math.min(width - CARD_W - 8, nodes[i].x));
      nodes[i].y = Math.max(8, Math.min(height - CARD_H - 8, nodes[i].y));
    }
  }

  separate(nodes, width, height);
}

/**
 * Push apart any two cards that still overlap after relaxation.
 *
 * The force pass treats nodes as points, so a tightly connected pair is pulled
 * closer than two 120x150 cards can sit without covering each other - and a card
 * you cannot read is worse than one slightly out of position. Separation is on
 * the axis of least penetration, so a pair that overlaps only at the corner moves
 * the short way rather than being flung across the panel.
 */
function separate(nodes: NodeState[], width: number, height: number): void {
  const PADDING = 10;
  for (let pass = 0; pass < 24; pass += 1) {
    let moved = false;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = nodes[i];
        const b = nodes[j];
        const overlapX = CARD_W + PADDING - Math.abs(a.x - b.x);
        const overlapY = CARD_H + PADDING - Math.abs(a.y - b.y);
        if (overlapX <= 0 || overlapY <= 0) continue;

        moved = true;
        if (overlapX < overlapY) {
          const shift = (overlapX / 2) * (a.x <= b.x ? -1 : 1);
          a.x += shift;
          b.x -= shift;
        } else {
          const shift = (overlapY / 2) * (a.y <= b.y ? -1 : 1);
          a.y += shift;
          b.y -= shift;
        }
        a.x = Math.max(8, Math.min(width - CARD_W - 8, a.x));
        a.y = Math.max(8, Math.min(height - CARD_H - 8, a.y));
        b.x = Math.max(8, Math.min(width - CARD_W - 8, b.x));
        b.y = Math.max(8, Math.min(height - CARD_H - 8, b.y));
      }
    }
    if (!moved) return;
  }
}

function cardCenter(node: NodeState) {
  return { x: node.x + CARD_W / 2, y: node.y + CARD_H / 2 };
}

const RedactedSilhouette: React.FC = () => (
  <svg viewBox="0 0 100 125" className="w-full h-full" preserveAspectRatio="xMidYMid slice">
    <rect width="100" height="125" fill="#C4A882" />
    <circle cx="50" cy="42" r="16" fill="#2A1F14" opacity="0.65" />
    <path d="M 15 125 Q 15 78 50 72 Q 85 78 85 125 Z" fill="#2A1F14" opacity="0.65" />
    <line x1="0" y1="20" x2="80" y2="125" stroke="#B33A2E" strokeWidth="1.5" opacity="0.4" />
    <line x1="15" y1="0" x2="100" y2="110" stroke="#B33A2E" strokeWidth="1.5" opacity="0.4" />
    <line x1="0" y1="55" x2="45" y2="125" stroke="#B33A2E" strokeWidth="1.5" opacity="0.4" />
    <line x1="50" y1="0" x2="100" y2="70" stroke="#B33A2E" strokeWidth="1.5" opacity="0.4" />
    <circle cx="50" cy="62.5" r="48" fill="none" stroke="#A89070" strokeWidth="1.5" opacity="0.35" />
  </svg>
);

const BrassPin: React.FC<{ hasIncident: boolean }> = ({ hasIncident }) => (
  <div
    className="absolute left-1/2 -translate-x-1/2 z-10 flex flex-col items-center"
    style={{ top: '-12px', filter: 'drop-shadow(0 3px 4px rgba(20,17,13,0.55))' }}
  >
    <div
      style={{
        width: '18px',
        height: '18px',
        borderRadius: '50%',
        backgroundColor: hasIncident ? '#B33A2E' : '#B08D57',
        border: '2px solid #C9A96E',
        boxShadow: '0 2px 6px rgba(20,17,13,0.45)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <div
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: 'rgba(255,255,255,0.5)',
        }}
      />
    </div>
    <div
      style={{
        width: '1px',
        height: '8px',
        backgroundColor: 'rgba(176,141,87,0.5)',
      }}
    />
  </div>
);

export const ForceDirectedGraph: React.FC<GraphProps> = ({ graphData, maxNodes = 18 }) => {
  // Take the strongest edges, then the entities they touch: picking nodes first
  // would leave most of them with nothing drawn between them.
  const view = React.useMemo(() => {
    if (!graphData) return undefined;
    if (graphData.nodes.length <= maxNodes) return { graph: graphData, truncated: false };
    const ranked = [...graphData.edges].sort((a, b) => b.active_strength - a.active_strength);
    const keep = new Set<string>();
    const edges: typeof graphData.edges = [];
    for (const edge of ranked) {
      const additions = [edge.source, edge.target].filter((id) => !keep.has(id));
      if (keep.size + additions.length > maxNodes) continue;
      for (const id of additions) keep.add(id);
      edges.push(edge);
    }
    return {
      graph: {
        ...graphData,
        nodes: graphData.nodes.filter((node) => keep.has(node.id)),
        edges,
      },
      truncated: true,
    };
  }, [graphData, maxNodes]);

  const { selectedEntityId, selectEntity } = useInvestigationStore();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [nodes, setNodes] = useState<NodeState[]>([]);

  const nodesRef = useRef<NodeState[]>([]);
  const dragNodeIdRef = useRef<string | null>(null);
  const dragOffsetRef = useRef({ x: 0, y: 0 });
  const isDraggedRef = useRef<boolean>(false);
  const animFrameRef = useRef<number | null>(null);
  const reducedMotionRef = useRef<boolean>(false);

  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);

  useEffect(() => {
    reducedMotionRef.current =
      window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  const trimmed = view?.graph;
  const nodeCount = trimmed?.nodes?.length ?? 0;

  useEffect(() => {
    if (!containerRef.current || nodeCount === 0) {
      nodesRef.current = [];
      setNodes([]);
      return;
    }
    const w = containerRef.current.clientWidth || 900;
    const h = containerRef.current.clientHeight || 600;
    const count = trimmed!.nodes.length;

    // Seed on a ring wide enough that the cards do not already overlap, then
    // relax. Starting them all inside a fixed 0.3-of-the-panel circle put a
    // dozen 120px cards on 30px of arc each, and since nothing then moved them,
    // that is exactly how they stayed: a solid disc with the graph inside it.
    const cx = w / 2 - CARD_W / 2;
    const cy = h / 2 - CARD_H / 2;
    const seedRadius = Math.max(
      Math.min(w, h) * 0.28,
      (count * CARD_W * 0.85) / (2 * Math.PI),
    );

    const initialized: NodeState[] = trimmed!.nodes.map((n, i) => {
      const angle = (i / Math.max(1, count)) * Math.PI * 2;
      return {
        id: n.id,
        x: cx + Math.cos(angle) * seedRadius,
        y: cy + Math.sin(angle) * seedRadius,
        vx: 0,
        vy: 0,
        degree: n.degree,
        maxStrength: n.max_strength,
        rot: seededRot(n.id),
      };
    });

    relax(initialized, trimmed!.edges, w, h);
    nodesRef.current = initialized;
    setNodes([...initialized]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nodeCount]);

  const runSpringSettle = useCallback((nodeId: string) => {
    if (reducedMotionRef.current) return;

    const DAMPING = 0.8;
    const DURATION = 150;
    const start = performance.now();

    const tick = (now: number) => {
      const elapsed = now - start;
      const n = nodesRef.current.find(n => n.id === nodeId);
      if (elapsed >= DURATION) {
        if (n) { n.vx = 0; n.vy = 0; }
        setNodes([...nodesRef.current]);
        return;
      }
      if (n) {
        n.vx *= DAMPING;
        n.vy *= DAMPING;
        n.x += n.vx;
        n.y += n.vy;
        setNodes([...nodesRef.current]);
      }
      animFrameRef.current = requestAnimationFrame(tick);
    };

    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    animFrameRef.current = requestAnimationFrame(tick);
  }, []);

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragNodeIdRef.current) return;
      isDraggedRef.current = true;
      const n = nodesRef.current.find(n => n.id === dragNodeIdRef.current);
      if (!n) return;
      const prevX = n.x;
      const prevY = n.y;
      n.x = e.clientX - dragOffsetRef.current.x;
      n.y = e.clientY - dragOffsetRef.current.y;
      n.vx = n.x - prevX;
      n.vy = n.y - prevY;
      setNodes([...nodesRef.current]);
    };

    const onMouseUp = () => {
      if (!dragNodeIdRef.current) return;
      const nodeId = dragNodeIdRef.current;
      dragNodeIdRef.current = null;
      setDraggingId(null);
      runSpringSettle(nodeId);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    return () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [runSpringSettle]);

  useEffect(() => {
    return () => {
      if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    };
  }, []);

  const handleNodeMouseDown = useCallback(
    (e: React.MouseEvent, node: NodeState) => {
      e.preventDefault();
      isDraggedRef.current = false;
      dragNodeIdRef.current = node.id;
      dragOffsetRef.current = {
        x: e.clientX - node.x,
        y: e.clientY - node.y,
      };
      setDraggingId(node.id);
    },
    []
  );

  const handleNodeClick = useCallback(
    (node: NodeState) => {
      if (!isDraggedRef.current) {
        selectEntity(node.id);
      }
    },
    [selectEntity]
  );

  // Zero under prefers-reduced-motion, which turns the transition off entirely.
  const settle = duration(DURATION.view);
  const edges = trimmed?.edges ?? [];
  const totalEdges = graphData?.total_edges ?? 0;

  const getCursor = () => {
    if (draggingId) return 'grabbing';
    if (hoveredId) return 'zoom-in';
    return 'crosshair';
  };

  if (!trimmed || nodeCount === 0) {
    const placeholders = [
      { id: 'p1', x: '25%', y: '30%', rot: -2 },
      { id: 'p2', x: '50%', y: '55%', rot: 3 },
      { id: 'p3', x: '75%', y: '35%', rot: -1 },
    ];

    return (
      <div
        className="relative w-full h-full overflow-hidden select-none bg-cork"
        style={{
          background: `
            repeating-linear-gradient(18deg, rgba(160,120,60,0.028) 0px, transparent 1px, transparent 22px),
            repeating-linear-gradient(-18deg, rgba(150,110,50,0.02) 0px, transparent 1px, transparent 28px),
            radial-gradient(ellipse at 60% 35%, #2A1E0F 0%, #1C1409 45%, #14110D 70%, #0D0B08 100%)
          `,
          backgroundColor: '#14110D',
        }}
      >
        {/* Dashed "pending connection" string — sits behind cards */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 1 }}
          viewBox="0 0 1000 600"
          preserveAspectRatio="xMidYMid meet"
        >
          {/* String from card 1 (25%, 30%) to card 2 (50%, 55%) in viewBox coords */}
          <path
            d="M 250,180 Q 350,280 500,330"
            fill="none"
            stroke="rgba(179,58,46,0.30)"
            strokeWidth="2"
            strokeDasharray="6 4"
            strokeLinecap="round"
          />
        </svg>

        {placeholders.map((p) => (
          <div
            key={p.id}
            style={{
              position: 'absolute',
              left: p.x,
              top: p.y,
              width: CARD_W,
              transform: `translate(-50%, -50%) rotate(${p.rot}deg)`,
              zIndex: 2,
            }}
          >
            <BrassPin hasIncident={false} />
            <div
              style={{
                marginTop: '8px',
                backgroundColor: '#E8DCC0',
                color: '#1A1512',
                boxShadow: '0 4px 20px rgba(20,17,13,0.35), 2px 2px 0 rgba(160,130,80,0.12)',
                borderRadius: '2px',
                border: '1px solid #C9B896',
                overflow: 'hidden',
              }}
            >
              <div style={{ width: '100%', aspectRatio: '4/5', backgroundColor: '#C4A882', position: 'relative', overflow: 'hidden' }}>
                <RedactedSilhouette />
                <div
                  style={{
                    position: 'absolute',
                    top: '6px',
                    right: '5px',
                    transform: 'rotate(-3deg)',
                    fontFamily: "'IBM Plex Mono', monospace",
                    ...TYPE.cardName,
                    fontWeight: 700,
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase' as const,
                    color: '#B33A2E',
                    border: '1.5px solid #B33A2E',
                    padding: '1px 5px',
                    display: 'inline-block',
                    lineHeight: 1.3,
                    backgroundColor: 'rgba(240,232,208,0.65)',
                  }}
                >
                  PENDING
                </div>
              </div>
              <div style={{ padding: '5px 6px', borderTop: '1px solid #C9B896', backgroundColor: '#EDE3C8', textAlign: 'center' }}>
                <div
                  style={{
                    fontFamily: "'IBM Plex Mono', monospace",
                    ...TYPE.cardName,
                    fontWeight: 700,
                    letterSpacing: '0.10em',
                    textTransform: 'uppercase' as const,
                    color: '#1A1512',
                  }}
                >
                  UNIDENTIFIED
                </div>
              </div>
            </div>
          </div>
        ))}

        <div style={{ position: 'absolute', bottom: 20, right: 24, zIndex: 5, fontFamily: '"Source Serif 4", serif', ...TYPE.board, color: 'rgba(176,141,87,0.18)', letterSpacing: '0.15em', textTransform: 'uppercase' as const, userSelect: 'none', pointerEvents: 'none' }}>
          Case Board · Vantage Intelligence
        </div>
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden select-none"
      style={{
        background: `
          repeating-linear-gradient(18deg, rgba(160,120,60,0.028) 0px, transparent 1px, transparent 22px),
          repeating-linear-gradient(-18deg, rgba(150,110,50,0.02) 0px, transparent 1px, transparent 28px),
          radial-gradient(ellipse at 60% 35%, #2A1E0F 0%, #1C1409 45%, #14110D 70%, #0D0B08 100%)
        `,
        backgroundColor: '#14110D',
        cursor: getCursor(),
      }}
    >
      {/* Cork grain texture */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            'radial-gradient(circle, rgba(176,141,87,0.06) 1px, transparent 1px)',
          backgroundSize: '18px 18px',
          zIndex: 0,
        }}
      />

      {/* SVG string layer */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none"
        style={{ zIndex: 1 }}
      >
        {edges.map((edge, i) => {
          // The rendered positions come from state, not from the ref the
          // simulation mutates. They hold the same values -- every write to the
          // ref is followed by a setNodes -- but reading the ref during render
          // is reading a value React has not been told about, so a frame where
          // only the ref moved would draw strings to where the cards no longer
          // are.
          const srcNode = nodes.find(n => n.id === edge.source);
          const tgtNode = nodes.find(n => n.id === edge.target);
          if (!srcNode || !tgtNode) return null;

          const sc = cardCenter(srcNode);
          const tc = cardCenter(tgtNode);
          const sx = sc.x;
          const sy = sc.y;
          const tx = tc.x;
          const ty = tc.y;
          const dist = Math.hypot(tx - sx, ty - sy);
          const cpx = (sx + tx) / 2;
          const cpy = (sy + ty) / 2 + dist * 0.22;

          const isFollowing = edge.pattern?.toUpperCase().includes('FOLLOWING');
          const strokeColor = isFollowing ? '#B33A2E' : '#8A6040';
          const strokeWidth = isFollowing ? 2 : 1.5;
          const dashArray = isFollowing ? '6 4' : undefined;

          const midT = 0.5;
          const midX =
            (1 - midT) * (1 - midT) * sx +
            2 * (1 - midT) * midT * cpx +
            midT * midT * tx;
          const midY =
            (1 - midT) * (1 - midT) * sy +
            2 * (1 - midT) * midT * cpy +
            midT * midT * ty;

          const pathD = `M ${sx},${sy} Q ${cpx},${cpy} ${tx},${ty}`;
          const shadowD = `M ${sx},${sy + 3} Q ${cpx},${cpy + 3} ${tx},${ty + 3}`;
          const rawLabel = `${edge.pattern ?? 'linked'} · ${edge.active_strength.toFixed(2)}`;
          const labelText = rawLabel.length > 14 ? rawLabel.slice(0, 14) : rawLabel;

          return (
            <g key={`edge-${i}`}>
              <path d={shadowD} fill="none" stroke="rgba(20,17,13,0.30)" strokeWidth={3} />
              <path
                d={pathD}
                fill="none"
                stroke={strokeColor}
                strokeWidth={strokeWidth}
                strokeDasharray={dashArray}
                strokeLinecap="round"
              />
              <circle cx={sx} cy={sy} r={4} fill="#B08D57" />
              <circle cx={tx} cy={ty} r={4} fill="#B08D57" />
              <g transform={`translate(${midX - 28}, ${midY - 7})`}>
                <rect width={56} height={14} rx={1.5} fill="#E8DCC0" stroke="#C9B896" strokeWidth={0.5} />
                <text
                  x={28}
                  y={9.5}
                  textAnchor="middle"
                  fontFamily='"IBM Plex Mono", monospace'
                  fontSize={7}
                  fill="#1A1512"
                >
                  {labelText}
                </text>
              </g>
            </g>
          );
        })}
      </svg>

      {/* Entity photograph cards */}
      {nodes.map(node => {
        const isSelected = node.id === selectedEntityId;
        const strong = node.maxStrength >= 0.5;
        const isHovered = hoveredId === node.id;
        const isDragging = draggingId === node.id;
        const displayId = node.id.length > 12 ? node.id.slice(0, 12) + '…' : node.id;

        return (
          <div
            key={node.id}
            style={{
              // Positioned by transform rather than by left/top so a relayout
              // is a compositor job rather than a layout one, and so it can be
              // transitioned. The graph repolls every fifteen seconds and the
              // relaxation can move a node a long way; without this the card
              // teleports and the operator cannot tell which one moved.
              position: 'absolute',
              left: 0,
              top: 0,
              width: CARD_W,
              zIndex: isSelected ? 10 : isDragging ? 8 : isHovered ? 6 : 2,
              transform: `translate3d(${node.x}px, ${node.y}px, 0) rotate(${node.rot}deg)`,
              // Never while dragging: the card must sit under the cursor, not
              // lag a fifth of a second behind it.
              transition: isDragging ? 'none' : `transform ${settle}ms ${EASE.inOut}`,
              cursor: isDragging ? 'grabbing' : 'zoom-in',
              userSelect: 'none',
              willChange: 'transform',
            }}
            onMouseEnter={() => setHoveredId(node.id)}
            onMouseLeave={() => setHoveredId(null)}
            onMouseDown={e => handleNodeMouseDown(e, node)}
            onClick={() => handleNodeClick(node)}
          >
            <BrassPin hasIncident={strong} />

            <div
              style={{
                backgroundColor: '#F0E8D0',
                borderRadius: '2px',
                overflow: 'hidden',
                boxShadow: isSelected
                  ? '0 0 0 2px #B33A2E, 0 8px 32px rgba(20,17,13,0.55)'
                  : isDragging
                  ? '0 12px 40px rgba(20,17,13,0.8), 2px 2px 0 rgba(140,110,70,0.3)'
                  : isHovered
                  ? '0 8px 28px rgba(20,17,13,0.65), 2px 2px 0 rgba(140,110,70,0.3)'
                  : '0 4px 18px rgba(20,17,13,0.55), 2px 2px 0 rgba(140,110,70,0.25)',
                border: '1px solid #C9B896',
                marginTop: '8px',
              }}
            >
              {/* Photograph area */}
              <div
                style={{
                  width: '100%',
                  aspectRatio: '4 / 5',
                  backgroundColor: '#C4A882',
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <RedactedSilhouette />

                {strong && (
                  <div
                    style={{
                      position: 'absolute',
                      top: '6px',
                      right: '5px',
                      transform: 'rotate(-3deg)',
                      border: '1.5px solid #B33A2E',
                      borderRadius: '1px',
                      padding: '1px 4px',
                      fontFamily: '"IBM Plex Mono", monospace',
                      ...TYPE.cardMeta,
                      fontWeight: 700,
                      color: '#B33A2E',
                      letterSpacing: '0.1em',
                      lineHeight: 1.4,
                      opacity: 0.9,
                      backgroundColor: 'rgba(240,232,208,0.55)',
                    }}
                  >
                    ALERT
                  </div>
                )}
              </div>

              {/* Label strip */}
              <div
                style={{
                  padding: '5px 6px 4px',
                  borderTop: '1px solid #C9B896',
                  backgroundColor: '#EDE3C8',
                }}
              >
                <div
                  style={{
                    fontFamily: '"IBM Plex Mono", monospace',
                    ...TYPE.cardName,
                    fontWeight: 700,
                    color: '#1A1512',
                    textTransform: 'uppercase' as const,
                    letterSpacing: '0.05em',
                    whiteSpace: 'nowrap',
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                  }}
                >
                  {displayId}
                </div>
                <div
                  style={{
                    fontFamily: '"IBM Plex Mono", monospace',
                    ...TYPE.cardMeta,
                    color: '#7A6545',
                    marginTop: '1px',
                  }}
                >
                  {node.degree} link{node.degree !== 1 ? 's' : ''}
                  {strong && ` · ${node.maxStrength.toFixed(2)} peak`}
                </div>

                {isSelected && (
                  <div
                    style={{
                      marginTop: '4px',
                      border: '1.5px solid #B33A2E',
                      borderRadius: '1px',
                      padding: '1px 4px',
                      fontFamily: '"IBM Plex Mono", monospace',
                      ...TYPE.cardMeta,
                      fontWeight: 700,
                      color: '#B33A2E',
                      letterSpacing: '0.12em',
                      textAlign: 'center' as const,
                    }}
                  >
                    ACTIVE
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}

      {/* Legend card */}
      <div
        style={{
          position: 'absolute',
          bottom: '16px',
          right: '16px',
          zIndex: 20,
          transform: 'rotate(1deg)',
          backgroundColor: '#E8DCC0',
          border: '1px solid #C9B896',
          borderRadius: '2px',
          padding: '10px 12px',
          boxShadow:
            '0 4px 18px rgba(20,17,13,0.5), 2px 2px 0 rgba(140,110,70,0.25)',
          minWidth: '148px',
        }}
      >
        <div
          style={{
            fontFamily: '"Source Serif 4", "Georgia", serif',
            ...TYPE.board,
            fontWeight: 600,
            color: '#1A1512',
            marginBottom: '8px',
            borderBottom: '1px solid #C9B896',
            paddingBottom: '5px',
          }}
        >
          Relationship Key
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '5px' }}>
          <svg width="28" height="8" style={{ flexShrink: 0 }}>
            <line x1="0" y1="4" x2="28" y2="4" stroke="#B33A2E" strokeWidth="2" strokeDasharray="5 3" />
          </svg>
          <span style={{ fontFamily: '"IBM Plex Mono", monospace', ...TYPE.cardMeta, color: '#1A1512' }}>
            FOLLOWING
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '7px', marginBottom: '8px' }}>
          <svg width="28" height="8" style={{ flexShrink: 0 }}>
            <line x1="0" y1="4" x2="28" y2="4" stroke="#8A6040" strokeWidth="1.5" />
          </svg>
          <span style={{ fontFamily: '"IBM Plex Mono", monospace', ...TYPE.cardMeta, color: '#1A1512' }}>
            PROXIMITY / OTHER
          </span>
        </div>
        <div
          style={{
            borderTop: '1px solid #C9B896',
            paddingTop: '5px',
            fontFamily: '"IBM Plex Mono", monospace',
            ...TYPE.cardMeta,
            color: '#7A6545',
            display: 'flex',
            justifyContent: 'space-between',
            gap: '10px',
          }}
        >
          <span>
            {view?.truncated
              ? `strongest ${nodeCount} of ${graphData?.total_nodes ?? nodeCount}`
              : `${nodeCount} entities`}
          </span>
          <span>
            {view?.truncated ? `${edges.length} of ${totalEdges} edges` : `${totalEdges} edges`}
          </span>
        </div>
      </div>
      
      <div style={{ position: 'absolute', bottom: 20, right: 24, zIndex: 5, fontFamily: '"Source Serif 4", serif', ...TYPE.board, color: 'rgba(176,141,87,0.14)', letterSpacing: '0.15em', textTransform: 'uppercase', userSelect: 'none', pointerEvents: 'none' }}>
        Case Board · Vantage Intelligence
      </div>
    </div>
  );
};
