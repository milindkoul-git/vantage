/**
 * A number that counts to its new value instead of jumping to it.
 *
 * This is the one place in the console where motion carries information rather
 * than decorating it. Counts here update on a poll - every one, five or ten
 * seconds - and a figure that snaps between two renders is a change the eye has
 * no reason to catch. A short count draws attention to the digit that moved,
 * and then gets out of the way.
 *
 * Three things it deliberately does not do:
 *
 * - It never animates the *first* value. Counting up from zero on load would be
 *   a small lie about a measurement that was already whatever it was.
 * - It never animates a change to or from "no reading". An em dash is a
 *   different kind of statement from a number and must not be tweened through.
 * - It is not the only way to see the value. Under `prefers-reduced-motion` it
 *   sets the number outright, and screen readers are given the settled figure
 *   rather than the intermediate ones.
 */

import React, { useEffect, useRef, useState } from 'react';
import { DURATION, tween } from '../../lib/motion';

export const AnimatedNumber: React.FC<{
  value: number | null | undefined;
  /** Decimal places. Counting through fractional values of an integer is noise. */
  decimals?: number;
  /** Rendered when there is no reading. */
  placeholder?: string;
  className?: string;
  ms?: number;
}> = ({ value, decimals = 0, placeholder = '—', className, ms = DURATION.panel }) => {
  const [shown, setShown] = useState(value ?? 0);
  const previous = useRef<number | null>(null);

  useEffect(() => {
    if (value === null || value === undefined) {
      previous.current = null;
      return;
    }
    const from = previous.current;
    previous.current = value;

    // First reading, or arriving from nothing: show it, do not count to it.
    if (from === null || from === value) {
      setShown(value);
      return;
    }
    return tween(ms, (progress) => setShown(from + (value - from) * progress));
  }, [value, ms]);

  if (value === null || value === undefined) {
    return <span className={className}>{placeholder}</span>;
  }

  return (
    <span className={className}>
      {/* The settled value for assistive tech; the counting one is visual only. */}
      <span className="sr-only">{value.toFixed(decimals)}</span>
      <span aria-hidden>{shown.toFixed(decimals)}</span>
    </span>
  );
};
