/**
 * One pair of entities, and what the score between them is made of.
 *
 * The four contributions are read off `score_breakdown`, and they are shown as
 * shares of the raw total that was actually scored -- so a pair with only
 * co-occurrence evidence shows one full bar and three empty ones, rather than
 * four bars filled from constants. `decay_factor` is shown alongside, because
 * the difference between a strong current association and a strong one from an
 * hour ago is the whole point of keeping an active and a historical score.
 */

import React from 'react';
import { useQuery } from '@tanstack/react-query';
import { X } from 'lucide-react';
import { api } from '../../data/source';
import { buildRelationshipEvidenceViewModel } from '../../lib/transforms/viewModels';
import { Empty, Resolved } from '../common/Panels';
import { dayTime, duration, shortEntity } from '../../lib/format';

const ShareBar: React.FC<{ name: string; value: number; share: number }> = ({
  name,
  value,
  share,
}) => (
  <li>
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-tiny text-warm-white">{name}</span>
      <span className="stamp tabular-nums text-ink-faint">
        {value.toFixed(3)}
        {share > 0 && <span className="text-brass"> · {(share * 100).toFixed(0)}%</span>}
      </span>
    </div>
    <div className="mt-1 h-1.5 w-full overflow-hidden rounded-sm bg-board">
      <div
        className="h-full bg-brass transition-[width] duration-300"
        style={{ width: `${Math.max(0, Math.min(1, share)) * 100}%` }}
      />
    </div>
  </li>
);

export const RelationshipInspectorDrawer: React.FC<{
  entityA: string;
  entityB: string;
  onClose: () => void;
}> = ({ entityA, entityB, onClose }) => {
  const query = useQuery({
    queryKey: ['relationships', entityA],
    queryFn: ({ signal }) => api.relationships(entityA, signal),
    refetchInterval: 5_000,
    retry: false,
  });

  return (
    <aside
      className="folder-pull absolute bottom-0 right-0 top-0 z-40 flex w-full max-w-[460px] flex-col border-l border-brass/25 bg-board-surface shadow-paper-lift"
      role="dialog"
      aria-label="Relationship inspector"
    >
      <header className="flex flex-none items-center justify-between gap-3 border-b border-brass/20 px-4 py-3">
        <div className="min-w-0">
          <p className="stamp text-ink-faint">Association</p>
          <p className="truncate font-mono text-tiny text-brass">
            {shortEntity(entityA)} ↔ {shortEntity(entityB)}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close relationship inspector"
          className="rounded-sm p-1 text-ink-faint transition-colors hover:bg-brass/10 hover:text-warm-white"
        >
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3 custom-scrollbar">
        <Resolved query={query} what="this association">
          {(data) => {
            const match = (data.relationships ?? []).find(
              (rel) =>
                (rel.entity_a === entityA && rel.entity_b === entityB) ||
                (rel.entity_a === entityB && rel.entity_b === entityA),
            );
            if (!match) {
              return (
                <Empty
                  what="No association recorded"
                  hint="These two have not been seen together long enough to score an edge."
                />
              );
            }
            const model = buildRelationshipEvidenceViewModel(match);

            return (
              <div className="flex flex-col gap-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="stamp text-ink-faint">Active strength</p>
                    <p className="font-mono text-lg tabular-nums text-string-red">
                      {model.activeStrength.toFixed(3)}
                    </p>
                  </div>
                  <div>
                    <p className="stamp text-ink-faint">Historical peak</p>
                    <p className="font-mono text-lg tabular-nums text-brass">
                      {model.historicalScore.toFixed(3)}
                    </p>
                  </div>
                </div>

                {model.pattern && (
                  <p className="rounded-sm border border-brass/20 bg-board px-3 py-2 text-tiny text-warm-white">
                    {model.pattern.replace(/_/g, ' ')}
                  </p>
                )}

                <section>
                  <h3 className="stamp mb-2 text-brass">What the score is made of</h3>
                  <ul className="flex flex-col gap-2">
                    {model.contributions.map((contribution) => (
                      <ShareBar key={contribution.name} {...contribution} />
                    ))}
                  </ul>
                  <p className="mt-2 text-micro leading-relaxed text-ink-faint">
                    Decay factor {model.decayFactor.toFixed(2)} — the active strength is the
                    historical score after time-decay since the pair was last seen together.
                  </p>
                </section>

                <section>
                  <h3 className="stamp mb-2 text-brass">Observed</h3>
                  <dl className="grid grid-cols-2 gap-2 text-tiny">
                    <div className="flex justify-between gap-2">
                      <dt className="text-ink-faint">Co-appearances</dt>
                      <dd className="font-mono tabular-nums">{model.coOccurrenceCount}</dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-ink-faint">In proximity</dt>
                      <dd className="font-mono tabular-nums">{model.proximityCount}</dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-ink-faint">Following</dt>
                      <dd className="font-mono tabular-nums">{model.followingCount}</dd>
                    </div>
                    <div className="flex justify-between gap-2">
                      <dt className="text-ink-faint">Interaction</dt>
                      <dd className="font-mono tabular-nums">
                        {duration(model.interactionSeconds)}
                      </dd>
                    </div>
                  </dl>
                  <p className="mt-2 text-micro text-ink-faint">
                    First seen {dayTime(match.first_observed)}, last {dayTime(match.last_observed)}.
                  </p>
                </section>

                {model.evidenceSummary && (
                  <p className="border-t border-brass/15 pt-3 text-tiny leading-relaxed text-ink-faint">
                    {model.evidenceSummary}
                  </p>
                )}
              </div>
            );
          }}
        </Resolved>
      </div>
    </aside>
  );
};
