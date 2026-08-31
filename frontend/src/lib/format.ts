/**
 * Formatting helpers shared across the workspaces.
 *
 * Separate from the components that use them because a module exporting both
 * components and plain functions cannot be hot-reloaded as a unit -- and,
 * separately, because a number's presentation is the kind of thing worth
 * getting right once. `duration` returning an em dash for a negative or
 * non-finite input is not defensive dressing: `last_seen - first_seen` on a
 * partially written incident really can be either.
 */

export const clockTime = (unixSeconds: number): string =>
  new Date(unixSeconds * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

export const dayTime = (unixSeconds: number): string =>
  new Date(unixSeconds * 1000).toLocaleString([], {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

export const duration = (seconds: number): string => {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${Math.round(seconds % 60)}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
};

export const bytes = (value: number): string => {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  if (value < 1024 ** 3) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${(value / 1024 ** 3).toFixed(2)} GB`;
};

/**
 * `person_17` -> `P-17`, so a dense list stays readable.
 *
 * Shortens the prefix only. The number is the tracker's anonymous identifier and
 * is the whole content of the label; truncating it would make two entities look
 * like one.
 */
export const shortEntity = (entityId: string): string =>
  entityId.replace(/^global_person_/i, 'P-').replace(/^person_/i, 'P-');
