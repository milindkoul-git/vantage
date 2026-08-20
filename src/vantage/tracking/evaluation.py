"""Tracking accuracy metrics: CLEAR MOT and IDF1.

Without this module, "the tracker works" means "the boxes looked right when I
watched the window", which is not a claim that survives a parameter change. With
it, every claim in the Phase 3 report is a number produced by a repeatable
procedure, and tuning becomes a search rather than an opinion.

Two families of metric are computed, because each is blind to something the
other catches:

**MOTA** (Multi-Object Tracking Accuracy) counts frame-level mistakes - misses,
false positives, and identity switches - against the number of true objects. It
is the standard headline figure, and its well-known weakness is that identity
switches are counted once, at the instant they happen. A tracker that swaps two
identities and then keeps both consistently forever afterwards is penalised for
exactly one frame, even though every subsequent frame is now wrong about who is
who.

**IDF1** fixes precisely that. It solves a global optimal matching between whole
ground-truth trajectories and whole predicted trajectories, then measures how
much of each trajectory was attributed to the right one. A swap that persists
costs the whole remainder of the trajectory, which is the honest accounting for
a system whose entire purpose is maintaining identity over time.

Both are reported. When they disagree the disagreement is informative, and
:func:`score` weights IDF1 the more heavily of the two for the reason above.

Implementation follows the standard definitions (Bernardin & Stiefelhagen 2008
for CLEAR MOT; Ristani et al. 2016 for IDF1), including the detail that matters
most: **an existing correct match is preserved wherever it remains valid**,
rather than re-solving each frame from scratch. Without that rule, an optimal
per-frame assignment will happily re-pair two overlapping objects for one frame
and report two identity switches where the tracker did nothing wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vantage.tracking.assignment import iou_matrix, linear_sum_assignment, match
from vantage.tracking.contracts import TrackingResult
from vantage.tracking.scenarios import Scenario

DEFAULT_IOU = 0.5
"""Overlap at which a prediction is considered to be the same object as a
ground-truth box. 0.5 is the MOT-challenge convention."""


@dataclass(frozen=True, slots=True)
class TrackingMetrics:
    """The measured accuracy of one tracker run against one scenario."""

    scenario: str
    frames: int
    ground_truth: int
    """Total ground-truth boxes; the denominator of MOTA and recall."""

    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    fragmentations: int
    """Times a ground-truth object went from tracked to untracked and back. High
    fragmentation with low ID switches means identity is being *kept* but
    coverage is patchy - a different problem with a different fix."""

    mostly_tracked: int
    """Ground-truth objects covered for at least 80% of their life."""

    mostly_lost: int
    """Ground-truth objects covered for less than 20% of their life."""

    gt_objects: int
    idtp: int
    idfp: int
    idfn: int
    total_iou: float
    mean_ms: float = 0.0

    @property
    def mota(self) -> float:
        """``1 - (FN + FP + IDSW) / GT``. Can be negative; that is meaningful."""
        if not self.ground_truth:
            return 0.0
        errors = self.false_negatives + self.false_positives + self.id_switches
        return 1.0 - errors / self.ground_truth

    @property
    def motp(self) -> float:
        """Mean IoU over matched pairs - localisation quality, not identity."""
        return self.total_iou / self.true_positives if self.true_positives else 0.0

    @property
    def idf1(self) -> float:
        """Harmonic mean of identity precision and recall."""
        denominator = 2 * self.idtp + self.idfp + self.idfn
        return 2 * self.idtp / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total else 0.0

    @property
    def recall(self) -> float:
        return self.true_positives / self.ground_truth if self.ground_truth else 0.0

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "scenario": self.scenario,
            "frames": self.frames,
            "mota": round(self.mota, 4),
            "motp": round(self.motp, 4),
            "idf1": round(self.idf1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "mostly_tracked": self.mostly_tracked,
            "mostly_lost": self.mostly_lost,
            "gt_objects": self.gt_objects,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "mean_ms": round(self.mean_ms, 3),
        }

    def describe(self) -> str:
        return (
            f"{self.scenario:<10} MOTA {self.mota:6.1%}  IDF1 {self.idf1:6.1%}  "
            f"MOTP {self.motp:5.1%}  IDs {self.id_switches:3d}  "
            f"FM {self.fragmentations:3d}  FP {self.false_positives:4d}  "
            f"FN {self.false_negatives:4d}  MT {self.mostly_tracked}/{self.gt_objects}"
        )


def evaluate(
    scenario: Scenario,
    results: list[TrackingResult],
    *,
    iou_threshold: float = DEFAULT_IOU,
    mean_ms: float = 0.0,
) -> TrackingMetrics:
    """Score ``results`` against ``scenario``'s ground truth.

    Args:
        results: One :class:`TrackingResult` per scenario frame, in order.
        iou_threshold: Overlap required to call a prediction correct.
    """
    if len(results) != len(scenario.frames):
        raise ValueError(
            f"expected one result per frame: {len(scenario.frames)} frames, "
            f"{len(results)} results"
        )

    tp = fp = fn = switches = fragmentations = 0
    total_iou = 0.0

    # gt object id -> track id it was matched to on the previous frame it was
    # matched at all. The persistence rule below depends on this surviving gaps.
    previous_match: dict[int, int] = {}
    matched_frames: dict[int, int] = {}
    gt_frames: dict[int, int] = {}
    was_tracked: dict[int, bool] = {}
    id_overlap: dict[tuple[int, int], int] = {}
    total_predictions = 0

    for frame, result in zip(scenario.frames, results, strict=False):
        truth = list(frame.objects)
        # Coasting tracks are included: a tracker that publishes a predicted box
        # is asserting the object is there, and must be scored on whether it is.
        predictions = list(result.tracks)
        total_predictions += len(predictions)

        for obj in truth:
            gt_frames[obj.object_id] = gt_frames.get(obj.object_id, 0) + 1

        pairs = _match_frame(truth, predictions, previous_match, iou_threshold)

        seen_this_frame: set[int] = set()
        for gt_index, pred_index, overlap in pairs:
            gt_id = truth[gt_index].object_id
            track_id = predictions[pred_index].track_id

            previous = previous_match.get(gt_id)
            if previous is not None and previous != track_id:
                switches += 1
            previous_match[gt_id] = track_id

            # A fragmentation is an *interruption*, so it only counts once the
            # object has been tracked at least once before. Testing this before
            # the counter below is updated is what keeps an object the detector
            # missed on its first frames from being charged for the gap that
            # preceded its first successful track.
            resumed = gt_id in matched_frames and not was_tracked.get(gt_id, False)

            tp += 1
            total_iou += overlap
            matched_frames[gt_id] = matched_frames.get(gt_id, 0) + 1
            key = (gt_id, track_id)
            id_overlap[key] = id_overlap.get(key, 0) + 1
            seen_this_frame.add(gt_id)

            if resumed:
                fragmentations += 1
            was_tracked[gt_id] = True

        for obj in truth:
            if obj.object_id not in seen_this_frame:
                was_tracked[obj.object_id] = False

        fn += len(truth) - len(pairs)
        fp += len(predictions) - len(pairs)

    idtp = _identity_true_positives(id_overlap)
    mostly_tracked = 0
    mostly_lost = 0
    for gt_id, total in gt_frames.items():
        coverage = matched_frames.get(gt_id, 0) / total if total else 0.0
        if coverage >= 0.8:
            mostly_tracked += 1
        elif coverage < 0.2:
            mostly_lost += 1

    return TrackingMetrics(
        scenario=scenario.name,
        frames=len(scenario.frames),
        ground_truth=scenario.instance_count,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        id_switches=switches,
        fragmentations=fragmentations,
        mostly_tracked=mostly_tracked,
        mostly_lost=mostly_lost,
        gt_objects=len(gt_frames),
        idtp=idtp,
        idfp=total_predictions - idtp,
        idfn=scenario.instance_count - idtp,
        total_iou=total_iou,
        mean_ms=mean_ms,
    )


def _match_frame(
    truth: list, predictions: list, previous_match: dict[int, int], iou_threshold: float
) -> list[tuple[int, int, float]]:
    """Pair ground truth to predictions for one frame.

    Existing correspondences win. If a ground-truth object was matched to track
    7 last frame and track 7 still overlaps it acceptably, that pairing is kept
    even when some other track overlaps slightly more. Re-solving freely each
    frame would report identity switches for two objects that merely brushed
    past each other, which would be the benchmark inventing the exact error it
    is supposed to be measuring.
    """
    if not truth or not predictions:
        return []

    overlaps = iou_matrix([obj.box for obj in truth], [track.box for track in predictions])
    track_index = {track.track_id: i for i, track in enumerate(predictions)}

    pairs: list[tuple[int, int, float]] = []
    used_truth: set[int] = set()
    used_pred: set[int] = set()

    for gt_index, obj in enumerate(truth):
        previous = previous_match.get(obj.object_id)
        if previous is None:
            continue
        pred_index = track_index.get(previous)
        if pred_index is None or pred_index in used_pred:
            continue
        overlap = overlaps[gt_index, pred_index]
        if overlap >= iou_threshold:
            pairs.append((gt_index, pred_index, float(overlap)))
            used_truth.add(gt_index)
            used_pred.add(pred_index)

    free_truth = [i for i in range(len(truth)) if i not in used_truth]
    free_pred = [i for i in range(len(predictions)) if i not in used_pred]
    if free_truth and free_pred:
        sub = overlaps[np.ix_(free_truth, free_pred)]
        matched, _, _ = match(1.0 - sub, max_cost=1.0 - iou_threshold)
        for row, col in matched:
            gt_index, pred_index = free_truth[row], free_pred[col]
            pairs.append((gt_index, pred_index, float(overlaps[gt_index, pred_index])))

    return pairs


def _identity_true_positives(id_overlap: dict[tuple[int, int], int]) -> int:
    """Frames correctly attributed under the best global trajectory matching.

    This is the part MOTA cannot see. Each ground-truth trajectory is assigned
    to at most one predicted trajectory for the whole sequence, chosen to
    maximise total agreement, so an identity that gets swapped halfway through
    is credited for only the half it got right.
    """
    if not id_overlap:
        return 0

    gt_ids = sorted({gt for gt, _ in id_overlap})
    track_ids = sorted({track for _, track in id_overlap})
    counts = np.zeros((len(gt_ids), len(track_ids)), dtype=np.float64)
    gt_pos = {gt: i for i, gt in enumerate(gt_ids)}
    track_pos = {track: i for i, track in enumerate(track_ids)}

    for (gt, track), count in id_overlap.items():
        counts[gt_pos[gt], track_pos[track]] = count

    # Maximise agreement by minimising its negation. Pairs with no overlap cost
    # zero, so an unavoidable spare assignment contributes nothing rather than
    # distorting the total.
    rows, cols = linear_sum_assignment(-counts)
    return int(counts[rows, cols].sum())


def aggregate(metrics: list[TrackingMetrics]) -> dict[str, float]:
    """Pool per-scenario results into one set of figures.

    Pooled from raw counts rather than averaged from per-scenario rates: a mean
    of ratios would weight a 120-frame scenario the same as a 1000-frame one and
    quietly let a tracker buy a good headline number on the easy cases.
    """
    if not metrics:
        return {}

    ground_truth = sum(m.ground_truth for m in metrics)
    tp = sum(m.true_positives for m in metrics)
    fp = sum(m.false_positives for m in metrics)
    fn = sum(m.false_negatives for m in metrics)
    switches = sum(m.id_switches for m in metrics)
    idtp = sum(m.idtp for m in metrics)
    idfp = sum(m.idfp for m in metrics)
    idfn = sum(m.idfn for m in metrics)
    total_iou = sum(m.total_iou for m in metrics)

    id_denominator = 2 * idtp + idfp + idfn
    return {
        "mota": 1.0 - (fn + fp + switches) / ground_truth if ground_truth else 0.0,
        "motp": total_iou / tp if tp else 0.0,
        "idf1": 2 * idtp / id_denominator if id_denominator else 0.0,
        "recall": tp / ground_truth if ground_truth else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "id_switches": switches,
        "fragmentations": sum(m.fragmentations for m in metrics),
        "mostly_tracked": sum(m.mostly_tracked for m in metrics),
        "mostly_lost": sum(m.mostly_lost for m in metrics),
        "gt_objects": sum(m.gt_objects for m in metrics),
        "mean_ms": float(np.mean([m.mean_ms for m in metrics])),
    }


def score(summary: dict[str, float]) -> float:
    """Collapse a pooled result into the single number tuning maximises.

    Weighting IDF1 above MOTA is a deliberate statement about what this platform
    is for. Everything built on top of tracking - dwell time, trajectories,
    activity, events - asks questions about a persistent object, and every one
    of them is corrupted by an identity that silently changes hands. A tracker
    with slightly worse per-frame coverage and materially better identity
    retention is the better tracker *here*, and the objective has to say so or
    the search will optimise for the wrong thing.

    The explicit switch penalty on top of IDF1 is small, and exists to break
    ties between parameter sets that score alike: given two equally good
    options, prefer the one that changes its mind less often.
    """
    if not summary:
        return 0.0
    switches_per_object = summary["id_switches"] / max(1, summary["gt_objects"])
    return (
        0.60 * summary["idf1"]
        + 0.30 * summary["mota"]
        + 0.10 * summary["motp"]
        - 0.02 * switches_per_object
    )


def format_table(metrics: list[TrackingMetrics]) -> str:
    """Human-readable per-scenario results plus the pooled summary."""
    lines = [
        f"{'scenario':<10} {'MOTA':>7} {'IDF1':>7} {'MOTP':>6} {'IDs':>4} "
        f"{'FM':>4} {'FP':>5} {'FN':>5} {'MT':>7}",
        "-" * 66,
    ]
    for metric in metrics:
        lines.append(
            f"{metric.scenario:<10} {metric.mota:>7.1%} {metric.idf1:>7.1%} "
            f"{metric.motp:>6.1%} {metric.id_switches:>4d} {metric.fragmentations:>4d} "
            f"{metric.false_positives:>5d} {metric.false_negatives:>5d} "
            f"{metric.mostly_tracked:>3d}/{metric.gt_objects:<3d}"
        )
    summary = aggregate(metrics)
    if summary:
        lines.append("-" * 66)
        lines.append(
            f"{'POOLED':<10} {summary['mota']:>7.1%} {summary['idf1']:>7.1%} "
            f"{summary['motp']:>6.1%} {int(summary['id_switches']):>4d} "
            f"{int(summary['fragmentations']):>4d} "
            f"{'':>5} {'':>5} {int(summary['mostly_tracked']):>3d}/"
            f"{int(summary['gt_objects']):<3d}"
        )
        lines.append(
            f"\nobjective score: {score(summary):.4f}   ({summary['mean_ms']:.2f} ms/frame)"
        )
    return "\n".join(lines)
