"""Construct pairwise rewards for equally sized empirical measures.

Rewards are maximized. A callable receives two observation arrays and must
return the matrix of rewards between every observation in the two arrays.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist


Reward = str | Callable[[np.ndarray, np.ndarray], np.ndarray]


def _observations(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim < 1 or raw.shape[0] == 0:
        raise ValueError(f"{name} must have a nonempty observation axis")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric observations")
    with np.errstate(over="ignore", invalid="ignore"):
        result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite observations")
    if result.size == 0:
        raise ValueError(f"{name} observations must have at least one coordinate")
    return result


def _labels(values: np.ndarray | None, n: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for component_mismatch")
    labels = np.asarray(values)
    if labels.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    if labels.dtype.kind not in "biufSU":
        raise ValueError(f"{name} must contain numeric or string labels")
    if labels.dtype.kind in "biuf" and not np.isfinite(labels).all():
        raise ValueError(f"{name} must contain finite labels")
    return labels


def build_reward_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    reward: Reward = "squared_distance",
    *,
    x_labels: np.ndarray | None = None,
    y_labels: np.ndarray | None = None,
    block_size: int | None = None,
) -> np.ndarray:
    """Return finite float64 rewards ``C[i,j] = c(X[i], Y[j])``.

    ``squared_distance`` flattens coordinates, and requires equal coordinate
    counts. ``component_mismatch`` rewards different supplied labels by one.
    Custom callables see the original observation shapes, including any tensor
    axes after the observation axis. ``block_size`` bounds intermediate reward
    evaluation memory; the resulting C still occupies O(n**2) memory.
    """
    x = _observations(X, "X")
    y = _observations(Y, "Y")
    n = len(x)
    if len(y) != n:
        raise ValueError("Only equal sample counts (n == m) are supported")
    if block_size is None:
        block_size = min(n, 256)
    if (
        isinstance(block_size, (bool, np.bool_))
        or not isinstance(block_size, (int, np.integer))
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")

    if callable(reward):
        evaluate = reward
    elif reward == "squared_distance":
        flat_x = x.reshape(n, -1)
        flat_y = y.reshape(n, -1)
        if flat_x.shape[1] != flat_y.shape[1]:
            raise ValueError("squared_distance requires equal coordinate counts")

        def evaluate(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return cdist(a.reshape(len(a), -1), b.reshape(len(b), -1), "sqeuclidean")

    elif reward == "component_mismatch":
        labels_x = _labels(x_labels, n, "x_labels")
        labels_y = _labels(y_labels, n, "y_labels")
        evaluate = None
    else:
        raise ValueError(f"Unknown reward: {reward!r}")

    matrix = np.empty((n, n), dtype=np.float64)
    for start in range(0, n, int(block_size)):
        stop = min(start + int(block_size), n)
        if evaluate is None:
            block = labels_x[start:stop, None] != labels_y[None, :]
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                block = np.asarray(evaluate(x[start:stop], y))
        if block.shape != (stop - start, n):
            raise ValueError(
                f"Reward must return shape {(stop - start, n)}, got {block.shape}"
            )
        if block.dtype.kind not in "biuf":
            raise ValueError("Reward must return real numeric values")
        with np.errstate(over="ignore", invalid="ignore"):
            block = block.astype(np.float64, copy=False)
        if not np.isfinite(block).all():
            raise ValueError("Reward matrix contains nonfinite values or overflow")
        matrix[start:stop] = block
    return matrix
