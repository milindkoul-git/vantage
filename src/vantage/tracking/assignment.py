"""Optimal one-to-one assignment, and the cost matrices tracking feeds it.

Why this is written here rather than imported
---------------------------------------------
The obvious source of ``linear_sum_assignment`` is SciPy. Adding SciPy to reach
one function costs roughly 30-90 MB of wheel for a dependency whose remaining
99% this platform does not use, and it is a heavy transitive burden on every
deployment target. The alternative package the reference ByteTrack uses (``lap``)
needs a C compiler on Windows, which is a genuinely hostile install step to hand
a collaborator.

The algorithm itself is a hundred lines of well-understood, testable code -
Jonker-Volgenant shortest augmenting paths with dual potentials, O(n^2 m) - and
the matrices tracking produces are tiny (tens of rows). So it is implemented
here, and :mod:`tests.test_tracking` checks it against brute-force enumeration
over every permutation for small matrices, which is a stronger correctness
argument than "we trusted a dependency".

Greedy matching was considered and rejected. It is simpler, but it fails in
exactly the situation tracking cares about: two objects passing close to each
other, where taking the locally best pair first forces the second pair into a
swap. That is an identity switch, the specific error this phase exists to
minimise, so paying O(n^3) on a 20x20 matrix to avoid it is trivially worth it.
"""

from __future__ import annotations

import numpy as np

from vantage.perception.contracts import BoundingBox

_LARGE = 1e6
"""Stand-in for "forbidden". Finite rather than ``inf`` because the dual
potentials arithmetic must stay finite; any pairing that ends up costing this
much is rejected after the solve rather than being made unrepresentable."""


def linear_sum_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(rows, cols)`` minimising the total of ``cost[rows, cols]``.

    A drop-in equivalent of ``scipy.optimize.linear_sum_assignment`` for the
    finite, rectangular case. Every row is matched when ``rows <= cols`` and
    vice versa; the returned pairs are sorted by row index.

    Args:
        cost: 2-D array of finite costs. Non-finite entries are rejected rather
            than silently coerced, because an ``inf`` that arrived by accident
            would otherwise produce a plausible-looking wrong answer.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"cost matrix must be 2-D, got shape {matrix.shape}")
    if matrix.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if not np.isfinite(matrix).all():
        raise ValueError("cost matrix contains non-finite values; use a large finite cost instead")

    # The solver requires rows <= cols; transposing is cheaper than a second
    # implementation, and the pairs are symmetric so only the labels swap.
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T

    assignment = _solve(matrix)

    rows = np.arange(matrix.shape[0], dtype=np.int64)
    cols = assignment
    if transposed:
        rows, cols = cols, rows
        order = np.argsort(rows, kind="stable")
        rows, cols = rows[order], cols[order]
    return rows, cols


def _solve(cost: np.ndarray) -> np.ndarray:
    """Jonker-Volgenant with dual potentials; ``cost`` must have rows <= cols.

    Returns the column assigned to each row. The inner search over unvisited
    columns is vectorised, which is what keeps a pure-Python implementation
    fast enough to run per frame.
    """
    n_rows, n_cols = cost.shape

    # One-based arrays with a sentinel at index 0: this is the formulation the
    # algorithm is stated in, and translating it to zero-based in-flight is a
    # reliable source of off-by-one bugs for no readability gain.
    u = np.zeros(n_rows + 1, dtype=np.float64)  # dual potential per row
    v = np.zeros(n_cols + 1, dtype=np.float64)  # dual potential per column
    parent = np.zeros(n_cols + 1, dtype=np.int64)  # parent[col] = row matched to col
    path = np.zeros(n_cols + 1, dtype=np.int64)  # predecessor column in the augmenting path

    for row in range(1, n_rows + 1):
        parent[0] = row
        col = 0
        min_cost = np.full(n_cols + 1, np.inf, dtype=np.float64)
        visited = np.zeros(n_cols + 1, dtype=bool)

        # Grow a shortest augmenting path until it reaches a free column.
        while True:
            visited[col] = True
            current_row = parent[col]
            free = ~visited[1:]

            reduced = cost[current_row - 1] - u[current_row] - v[1:]
            improved = free & (reduced < min_cost[1:])
            min_cost[1:][improved] = reduced[improved]
            path[1:][improved] = col

            candidates = np.where(free, min_cost[1:], np.inf)
            next_col = int(np.argmin(candidates)) + 1
            delta = candidates[next_col - 1]

            # Shift the potentials so the chosen edge becomes tight. This is
            # what keeps every previously-found match optimal as the path grows.
            u[parent[visited]] += delta
            v[visited] -= delta
            min_cost[1:][free] -= delta

            col = next_col
            if parent[col] == 0:
                break

        # Walk the path back, flipping matched and unmatched edges.
        while col:
            previous = path[col]
            parent[col] = parent[previous]
            col = previous

    assignment = np.full(n_rows, -1, dtype=np.int64)
    for column in range(1, n_cols + 1):
        if parent[column] > 0:
            assignment[parent[column] - 1] = column - 1
    return assignment


def match(
    cost: np.ndarray, max_cost: float
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Solve ``cost`` and split the outcome into matched and unmatched.

    Gating happens *after* the solve, not before. Removing expensive edges up
    front would change which assignment is optimal for the edges that remain;
    solving the full problem and then discarding pairs that exceed ``max_cost``
    keeps the optimality argument intact and is simpler to reason about.

    Args:
        max_cost: Pairs costing more than this are not real matches, and both
            sides are returned as unmatched instead.

    Returns:
        ``(pairs, unmatched_rows, unmatched_cols)``.
    """
    matrix = np.asarray(cost, dtype=np.float64)
    n_rows = matrix.shape[0] if matrix.ndim == 2 else 0
    n_cols = matrix.shape[1] if matrix.ndim == 2 else 0
    if n_rows == 0 or n_cols == 0:
        return [], list(range(n_rows)), list(range(n_cols))

    rows, cols = linear_sum_assignment(matrix)

    pairs: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for row, col in zip(rows.tolist(), cols.tolist()):
        if matrix[row, col] <= max_cost:
            pairs.append((row, col))
            matched_rows.add(row)
            matched_cols.add(col)

    unmatched_rows = [r for r in range(n_rows) if r not in matched_rows]
    unmatched_cols = [c for c in range(n_cols) if c not in matched_cols]
    return pairs, unmatched_rows, unmatched_cols


def iou_matrix(a: list[BoundingBox], b: list[BoundingBox]) -> np.ndarray:
    """Pairwise IoU, vectorised.

    Called once per association pass per frame, so the difference between this
    and a nested Python loop is measurable on a 30 fps stream with a dozen
    objects.
    """
    if not a or not b:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    boxes_a = np.array([box.xyxy for box in a], dtype=np.float64)
    boxes_b = np.array([box.xyxy for box in b], dtype=np.float64)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    intersection = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - intersection

    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def forbid(cost: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Make the ``True`` entries of ``mask`` effectively unmatchable.

    Used for class gating: associating a detection labelled ``car`` to a track
    the system has been calling ``person`` is never correct, however well the
    boxes happen to overlap. A large finite cost rather than ``inf`` keeps the
    dual arithmetic well-defined, and :func:`match` discards the pair afterwards
    because it exceeds any sane gate.
    """
    gated = np.array(cost, dtype=np.float64, copy=True)
    gated[mask] = _LARGE
    return gated
