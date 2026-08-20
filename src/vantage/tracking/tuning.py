"""Empirical parameter search over the ground-truth scenarios.

What this is, and what it is not
---------------------------------
ByteTrack has no learned weights. There is no network here, nothing to
backpropagate through, and no training set - so "training the tracker" in the
gradient-descent sense is not a thing that exists. What the tracker does have is
about ten free parameters (association gates, confidence thresholds, lifecycle
limits, motion-model noise) whose values materially change its accuracy, and
which are conventionally copied from a paper that tuned them on a different
detector, a different frame rate and a different set of videos.

This module replaces that inheritance with measurement. It evaluates candidate
parameter sets against the seeded scenarios in
:mod:`vantage.tracking.scenarios`, scores each with the objective in
:mod:`vantage.tracking.evaluation`, and reports what actually won. The defaults
shipped in :class:`~vantage.tracking.bytetrack.TrackerParams` are the output of
this search, and re-running it is how they get revisited when the detector or
the deployment changes.

Search strategy
---------------
Coordinate descent, not a full grid. A grid over ten parameters at five values
each is roughly ten million evaluations, which is not a thing anyone will
actually run; coordinate descent sweeps one parameter at a time, keeps the best
value, and repeats until a full pass changes nothing. On a search space this
smooth it finds effectively the same answer for a few hundred evaluations, and
it has the property that matters more than optimality here: every step is a
directly interpretable statement of the form "holding everything else fixed,
this value of this parameter measured better".

Two guards against fooling ourselves
------------------------------------
Parameter search on a fixed benchmark overfits, exactly like any other fitting
procedure. Two things are done about it:

* **Multiple detector profiles per evaluation.** Each candidate is scored across
  several simulated detector behaviours - clean, noisy, high-miss - so a
  parameter set that only wins under one specific error profile does not win
  overall.
* **A held-out check.** :func:`validate` re-scores the winner against scenarios
  and seeds the search never selected on. A result that does not survive that is
  reported as overfitted rather than shipped.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

from vantage.core.errors import ConfigError
from vantage.core.logging import get_logger
from vantage.tracking.bytetrack import ByteTracker, TrackerParams
from vantage.tracking.evaluation import (
    TrackingMetrics,
    aggregate,
    evaluate,
    score,
)
from vantage.tracking.scenarios import (
    DetectorProfile,
    Scenario,
    build_suite,
    simulate_detections,
)

log = get_logger(__name__)


TRAINING_PROFILES: tuple[DetectorProfile, ...] = (
    DetectorProfile(seed=101),
    DetectorProfile(
        seed=202, localisation_noise=0.06, miss_rate=0.12, false_positive_rate=0.15
    ),
    DetectorProfile(
        seed=303, localisation_noise=0.02, miss_rate=0.02, false_positive_rate=0.02
    ),
    DetectorProfile(
        seed=404,
        localisation_noise=0.04,
        miss_rate=0.08,
        false_positive_rate=2.5,
        false_positive_confidence=0.55,
    ),
    DetectorProfile(
        seed=505,
        localisation_noise=0.08,
        miss_rate=0.12,
        false_positive_rate=3.0,
        false_positive_confidence=0.7,
        confidence_visible=0.68,
    ),
)
"""Typical, degraded, clean, cluttered - and harsh.

The cluttered profile is not an adversarial extreme, it is the operating
condition this platform actually creates. ByteTrack's second pass only works if
it is *given* low-scoring boxes, so enabling tracking drops the detector's
confidence floor from 0.35 to around 0.1, and everything between those two
numbers is mostly junk. A profile emitting a couple of spurious boxes per frame
is what that looks like.

Its absence was a real defect in an earlier version of this file. Without it the
search chose ``min_hits=1`` - publish every detection instantly, no
corroboration - which scored best on clean input and then collapsed to 4% MOTA
at 3 false positives per frame. The benchmark, not the search, was wrong.

The harsh profile was added for the same reason, one iteration later. With
training capped at moderate localisation error, the search drove the motion
model's process noise down to 2.0, which is optimal for smooth trajectories and
fails badly on sharp ones - held-out IDF1 on the erratic scenario fell to 49.7%.
Training has to contain the hard cases, or the search will confidently optimise
them away.
"""

VALIDATION_PROFILES: tuple[DetectorProfile, ...] = (
    DetectorProfile(seed=9001, localisation_noise=0.045, miss_rate=0.09),
    DetectorProfile(
        seed=9002,
        localisation_noise=0.025,
        miss_rate=0.15,
        false_positive_rate=0.25,
        confidence_visible=0.75,
    ),
    DetectorProfile(
        seed=9003,
        localisation_noise=0.09,
        miss_rate=0.18,
        false_positive_rate=4.5,
        false_positive_confidence=0.75,
        confidence_visible=0.65,
        occluded_miss_scale=4.5,
    ),
)
"""Held out. Different seeds *and* different error magnitudes, so surviving
these means the parameters generalise rather than memorise. The last profile is
deliberately worse than anything in training: four confident false positives per
frame, poor localisation, and a detector that is not sure about anything."""


@dataclass(frozen=True, slots=True)
class Candidate:
    """One parameter set and how it measured."""

    params: TrackerParams
    objective: float
    summary: dict[str, float]
    metrics: tuple[TrackingMetrics, ...]

    def describe(self) -> str:
        return (
            f"score {self.objective:.4f}  IDF1 {self.summary['idf1']:.1%}  "
            f"MOTA {self.summary['mota']:.1%}  IDs {int(self.summary['id_switches'])}"
        )


def run_scenario(
    scenario: Scenario, params: TrackerParams, profile: DetectorProfile
) -> TrackingMetrics:
    """Track one scenario end to end and score the outcome."""
    detections = simulate_detections(scenario, profile)
    tracker = ByteTracker(params)

    started = time.perf_counter()
    results = [tracker.update(result) for result in detections]
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    return evaluate(
        scenario,
        results,
        mean_ms=elapsed_ms / len(detections) if detections else 0.0,
    )


WORST_CASE_WEIGHT = 0.25
"""How much the weakest scenario counts, against the pooled result.

Non-zero for a concrete reason. With a purely pooled objective the search chose
``max_lost_s=0.25``, which scored 0.9142 against 0.9139 for ``1.5`` - a
difference of 0.0003, indistinguishable from noise. But those two settings are
not equivalent: the short buffer drops occlusion IDF1 from 79.0% to 71.5%,
because a track that is only kept for a quarter of a second cannot survive
somebody walking behind a pillar. Surviving occlusion is the single capability
this phase exists to add, and a pooled average let it be traded away for nothing
because the other four scenarios outvoted it.

Scoring the weakest scenario alongside the pooled result states the requirement
directly: a parameter set may not buy an average by failing at one thing.
"""


def assess(
    params: TrackerParams,
    scenarios: list[Scenario],
    profiles: tuple[DetectorProfile, ...] = TRAINING_PROFILES,
) -> Candidate:
    """Score one parameter set across every scenario and detector profile.

    The objective blends the pooled result with the worst individual scenario,
    so a candidate has to be broadly good *and* have no catastrophic weakness.
    """
    metrics = [
        run_scenario(scenario, params, profile)
        for scenario in scenarios
        for profile in profiles
    ]
    summary = aggregate(metrics)

    per_scenario = [
        score(aggregate([m for m in metrics if m.scenario == scenario.name]))
        for scenario in scenarios
    ]
    worst = min(per_scenario) if per_scenario else 0.0
    objective = (1.0 - WORST_CASE_WEIGHT) * score(summary) + WORST_CASE_WEIGHT * worst

    return Candidate(
        params=params,
        objective=objective,
        summary=summary,
        metrics=tuple(metrics),
    )


# Each entry is (name, values). Ordered so the parameters with the largest
# expected effect are swept first, which is what lets a truncated run still
# produce a useful answer.
#
# The grids are fine where the objective is sensitive and coarse where it is
# flat. That is not cosmetic: an earlier coarse grid of (1, 2, 4, 8, 16) for
# the acceleration noise straddled the optimum without containing it, and the
# search returned 2.0 when 3.0 was materially better on held-out data. A
# coordinate search can only return a value it was offered.
SEARCH_SPACE: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("acceleration", (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 12.0)),
    ("max_lost_s", (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)),
    ("high_threshold", (0.3, 0.4, 0.5, 0.6, 0.7)),
    ("init_threshold", (0.5, 0.6, 0.7, 0.8)),
    ("iou_high", (0.1, 0.2, 0.3, 0.45)),
    ("iou_low", (0.2, 0.3, 0.4, 0.5, 0.6)),
    ("iou_tentative", (0.3, 0.4, 0.5, 0.65)),
    ("min_hits", (1, 2, 3, 4, 5)),
    ("low_threshold", (0.05, 0.1, 0.2, 0.3)),
    ("measurement", (0.02, 0.035, 0.05, 0.075, 0.1, 0.15)),
    ("initial_velocity", (1.0, 3.0, 6.0)),
    ("size_drift", (0.1, 0.2, 0.35, 0.6, 1.0)),
)

_NOISE_FIELDS = {"measurement", "acceleration", "initial_velocity", "size_drift"}

MIN_IMPROVEMENT = 0.001
"""Objective gain required to accept a new value.

Without a floor the search will happily adopt a change worth 0.0003, which on a
benchmark of this size is indistinguishable from the seed. That is not merely
wasted effort - it moves a parameter away from its default on no evidence, and
whoever reads the result later has no way to tell that from a real finding.
"""


def _with(params: TrackerParams, name: str, value: float) -> TrackerParams:
    """Return ``params`` with one field changed, noise fields included.

    The motion-noise values live on a nested dataclass, so a flat search space
    needs this indirection to reach them. They are worth reaching: the
    acceleration noise turned out to be the single highest-impact parameter in
    the search, ahead of every association threshold.
    """
    if name in _NOISE_FIELDS:
        return replace(params, noise=replace(params.noise, **{name: value}))
    if name == "min_hits":
        return replace(params, min_hits=int(value))
    return replace(params, **{name: value})


def search(
    scenarios: list[Scenario] | None = None,
    *,
    start: TrackerParams | None = None,
    rounds: int = 3,
    profiles: tuple[DetectorProfile, ...] = TRAINING_PROFILES,
    progress: bool = True,
) -> tuple[Candidate, int]:
    """Coordinate-descent search. Returns the winner and the evaluation count.

    Args:
        rounds: Maximum full sweeps. The search stops early when a whole sweep
            improves nothing, which is the usual outcome by the second or third.
    """
    suite = scenarios if scenarios is not None else build_suite()
    best = assess(start or TrackerParams(), suite, profiles)
    evaluations = 1

    if progress:
        log.info(
            "tuning: baseline",
            extra={"vantage_fields": {"score": round(best.objective, 4)}},
        )

    for round_index in range(rounds):
        improved = False

        for name, values in SEARCH_SPACE:
            round_best = best
            for value in values:
                try:
                    candidate_params = _with(best.params, name, value)
                    if candidate_params == best.params:
                        continue
                    candidate = assess(candidate_params, suite, profiles)
                except ConfigError as exc:
                    # Sweeping one parameter can produce a combination the
                    # tracker rejects outright - a high_threshold below the
                    # low_threshold, say. That is a genuinely invalid point
                    # in the space, not a failure, so it is recorded and
                    # skipped rather than swallowed or allowed to abort the run.
                    log.debug(
                        "tuning: rejected candidate",
                        extra={
                            "vantage_fields": {
                                "parameter": name,
                                "value": value,
                                "error": str(exc),
                            }
                        },
                    )
                    continue
                evaluations += 1
                if candidate.objective > round_best.objective + MIN_IMPROVEMENT:
                    round_best = candidate

            if round_best is not best:
                if progress:
                    log.info(
                        "tuning: improved",
                        extra={
                            "vantage_fields": {
                                "parameter": name,
                                "value": getattr(
                                    round_best.params,
                                    name,
                                    getattr(round_best.params.noise, name, None),
                                ),
                                "score": round(round_best.objective, 4),
                                "delta": round(round_best.objective - best.objective, 4),
                            }
                        },
                    )
                best = round_best
                improved = True

        if not improved:
            if progress:
                log.info(
                    "tuning: converged",
                    extra={"vantage_fields": {"round": round_index + 1}},
                )
            break

    return best, evaluations


def validate(
    params: TrackerParams,
    baseline: TrackerParams | None = None,
    scenarios: list[Scenario] | None = None,
) -> dict[str, dict[str, float]]:
    """Re-score tuned and baseline parameters on the held-out profiles.

    The comparison is the point. An improvement that appears only on the
    profiles the search selected on is overfitting, and the only way to see that
    is to measure both parameter sets somewhere neither was chosen.
    """
    suite = scenarios if scenarios is not None else build_suite()
    reference = baseline if baseline is not None else TrackerParams()

    tuned_result = assess(params, suite, VALIDATION_PROFILES)
    baseline_result = assess(reference, suite, VALIDATION_PROFILES)

    return {
        "tuned": {**tuned_result.summary, "objective": tuned_result.objective},
        "baseline": {**baseline_result.summary, "objective": baseline_result.objective},
    }


def as_config_lines(params: TrackerParams) -> list[str]:
    """Render a parameter set as ``tracking:`` YAML, ready to paste into a config."""
    return [
        "tracking:",
        f"  high_threshold: {params.high_threshold}",
        f"  low_threshold: {params.low_threshold}",
        f"  init_threshold: {params.init_threshold}",
        f"  iou_high: {params.iou_high}",
        f"  iou_low: {params.iou_low}",
        f"  iou_tentative: {params.iou_tentative}",
        f"  min_hits: {params.min_hits}",
        f"  max_lost_s: {params.max_lost_s}",
        f"  measurement_noise: {params.noise.measurement}",
        f"  acceleration_noise: {params.noise.acceleration}",
        f"  initial_velocity_noise: {params.noise.initial_velocity}",
        f"  size_drift_noise: {params.noise.size_drift}",
    ]


def default_params_source(params: TrackerParams) -> str:
    """The winning values, formatted for a report."""
    noise = params.noise
    return (
        f"high_threshold={params.high_threshold}, low_threshold={params.low_threshold}, "
        f"init_threshold={params.init_threshold}, iou_high={params.iou_high}, "
        f"iou_low={params.iou_low}, iou_tentative={params.iou_tentative}, "
        f"min_hits={params.min_hits}, max_lost_s={params.max_lost_s}, "
        f"noise=MotionNoise(measurement={noise.measurement}, "
        f"acceleration={noise.acceleration}, initial_velocity={noise.initial_velocity}, "
        f"size_drift={noise.size_drift})"
    )


__all__ = [
    "SEARCH_SPACE",
    "TRAINING_PROFILES",
    "VALIDATION_PROFILES",
    "Candidate",
    "as_config_lines",
    "assess",
    "run_scenario",
    "search",
    "validate",
]
