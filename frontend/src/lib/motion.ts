/**
 * The motion budget, in one place.
 *
 * A console is watched for a whole shift by someone who has seen it a thousand
 * times and is looking for the one thing that changed. Every animation is a
 * claim that something happened, so the useful ones are short, and the
 * decorative ones are absent: a panel that flourishes on each poll teaches its
 * reader to ignore movement, which is exactly the signal that has to stay in
 * reserve for the frame where somebody falls over.
 *
 * The bands are Material's - roughly 100-200ms for a micro-interaction and
 * 200-500ms for a larger transition - taken at the fast end of both, because
 * this is an instrument rather than a narrative.
 *
 * Nothing here animates layout, colour or size to *carry* information. Motion is
 * allowed to draw the eye to a number that changed; it is not allowed to be the
 * only way to notice it. Everything below degrades to an instant change under
 * `prefers-reduced-motion`, and every consumer routes through `duration()` so
 * that is one decision rather than thirty.
 */

import { useEffect, useState } from 'react';

export const DURATION = {
  /** A value ticking over, a toggle flipping. Barely perceptible, on purpose. */
  micro: 120,
  /** A panel resolving, a chart drawing itself in. */
  panel: 220,
  /** The largest move allowed: a workspace change, the first paint. */
  view: 300,
} as const;

export const EASE = {
  /** Decelerate into rest. The default for anything arriving. */
  out: 'cubic-bezier(0.22, 1, 0.36, 1)',
  /** Symmetric, for something moving between two places it belongs. */
  inOut: 'cubic-bezier(0.65, 0, 0.35, 1)',
} as const;

/**
 * How long a stagger step should be, given how many things are staggering.
 *
 * A fixed per-item delay reads as snappy across eight items and as a slow wipe
 * across eighty. The total is what wants to be constant, so the step shrinks as
 * the count grows - the same rule the GSAP SplitText demos use to keep a
 * per-character reveal from dragging when a per-line one does not.
 */
export function staggerStep(count: number, totalMs = DURATION.panel): number {
  if (count <= 1) return 0;
  return Math.min(24, totalMs / count);
}

let reducedMotion: MediaQueryList | null = null;

function query(): MediaQueryList | null {
  if (reducedMotion) return reducedMotion;
  if (typeof window === 'undefined' || !window.matchMedia) return null;
  reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  return reducedMotion;
}

/** Whether the viewer has asked for less movement. Safe to call outside React. */
export function prefersReducedMotion(): boolean {
  return query()?.matches ?? false;
}

/**
 * A duration, or zero when the viewer has asked for less movement.
 *
 * Zero rather than "shorter". Reduced motion is a request to remove the
 * animation, not to hurry it, and a 40ms version of a transition is still a
 * transition to someone it makes ill.
 */
export function duration(ms: number): number {
  return prefersReducedMotion() ? 0 : ms;
}

/** The same, as a React hook that re-renders if the preference changes mid-session. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(prefersReducedMotion);

  useEffect(() => {
    const media = query();
    if (!media) return;
    const onChange = (event: MediaQueryListEvent) => setReduced(event.matches);
    media.addEventListener('change', onChange);
    return () => media.removeEventListener('change', onChange);
  }, []);

  return reduced;
}

/**
 * Run `onFrame(progress)` from 0 to 1 over `ms`, then once more at exactly 1.
 *
 * Returns a cancel function. Under reduced motion it calls `onFrame(1)` once and
 * returns a no-op, so a caller never has to branch.
 *
 * requestAnimationFrame rather than a CSS transition because the things this
 * drives - a counting number, a bar height in an SVG - are values React owns
 * rather than styles the browser can interpolate.
 */
export function tween(ms: number, onFrame: (progress: number) => void): () => void {
  if (duration(ms) === 0) {
    onFrame(1);
    return () => {};
  }

  let frame = 0;
  const started = performance.now();

  const step = (now: number) => {
    const elapsed = now - started;
    if (elapsed >= ms) {
      onFrame(1);
      return;
    }
    // Cubic ease-out: fast départ, settling finish. Matches EASE.out closely
    // enough that a tweened number and a CSS-transitioned panel beside it read
    // as one movement.
    const t = elapsed / ms;
    onFrame(1 - (1 - t) ** 3);
    frame = requestAnimationFrame(step);
  };

  frame = requestAnimationFrame(step);
  return () => cancelAnimationFrame(frame);
}
