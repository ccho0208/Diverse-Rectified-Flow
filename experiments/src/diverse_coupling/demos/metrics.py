"""Coupling diversity and marginal fidelity measured separately."""

from __future__ import annotations

import numpy as np

from .datasets import ExperimentSpec, _points, _positive_integer


def _weights(count: int, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        return np.full(count, 1.0 / count, dtype=np.float64)
    raw = np.asarray(weights)
    if raw.shape != (count,) or raw.dtype.kind not in "biuf":
        raise ValueError(f"weights must be a real array of shape ({count},)")
    values = raw.astype(np.float64)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("weights must be finite and nonnegative")
    total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must have positive finite total mass")
    return values / total


def component_frequencies(
    points: np.ndarray,
    spec: ExperimentSpec,
    side: str = "x",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    points = _points(points)
    masses = _weights(len(points), weights)
    return np.bincount(spec.classify(points, side), weights=masses, minlength=spec.components)


def component_pair_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    spec: ExperimentSpec,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted joint component probabilities; rows are X, columns are Y."""
    x = _points(X, "X")
    y = _points(Y, "Y")
    if len(x) != len(y):
        raise ValueError("Paired X and Y must have equal sample counts")
    masses = _weights(len(x), weights)
    labels = spec.classify(x, "x") * spec.components + spec.classify(y, "y")
    return np.bincount(
        labels, weights=masses, minlength=spec.components**2
    ).reshape(spec.components, spec.components)


def empirical_independent_matrix(
    X: np.ndarray, Y: np.ndarray, spec: ExperimentSpec
) -> np.ndarray:
    """Exact component table for the product of two empirical measures."""
    return np.outer(component_frequencies(X, spec, "x"), component_frequencies(Y, spec, "y"))


def sliced_wasserstein(
    reference: np.ndarray,
    samples: np.ndarray,
    *,
    directions: int = 64,
    seed: int = 0,
    max_samples: int = 2048,
) -> float:
    """Approximate 2D sliced-W2 using fixed random projection directions.

    Each one-dimensional empirical W2 is integrated exactly, including when
    the two empirical sample counts differ. Only direction sampling and the
    optional deterministic sample cap are approximations. Returned distances
    are in coordinate units. This compares uniformly weighted marginals.
    """
    x = _points(reference, "reference")
    y = _points(samples, "samples")
    directions = _positive_integer(directions, "directions")
    max_samples = _positive_integer(max_samples, "max_samples")
    rng = np.random.default_rng(seed)
    original_x_count, original_y_count = len(x), len(y)
    x_subset = None
    if len(x) > max_samples:
        x_subset = rng.choice(len(x), max_samples, replace=False)
        x = x[x_subset]
    if len(y) > max_samples:
        # Shared indices preserve distance zero for identical input arrays.
        indices = (
            x_subset
            if original_x_count == original_y_count and x_subset is not None
            else rng.choice(len(y), max_samples, replace=False)
        )
        y = y[indices]
    angles = rng.uniform(0.0, 2 * np.pi, size=directions)
    projections = np.stack((np.cos(angles), np.sin(angles)))
    sorted_x = np.sort(x @ projections, axis=0)
    sorted_y = np.sort(y @ projections, axis=0)
    if len(x) == len(y):
        squared = np.mean((sorted_x - sorted_y) ** 2)
    else:
        breaks = np.unique(
            np.concatenate((np.arange(len(x) + 1) / len(x), np.arange(len(y) + 1) / len(y)))
        )
        intervals = np.diff(breaks)
        midpoints = (breaks[:-1] + breaks[1:]) / 2
        ranks_x = np.minimum((midpoints * len(x)).astype(np.int64), len(x) - 1)
        ranks_y = np.minimum((midpoints * len(y)).astype(np.int64), len(y) - 1)
        distances = (sorted_x[ranks_x] - sorted_y[ranks_y]) ** 2
        squared = np.mean(np.sum(intervals[:, None] * distances, axis=0))
    result = float(np.sqrt(squared))
    if not np.isfinite(result):
        raise ValueError("Sliced Wasserstein calculation overflowed")
    return result


def _moments(points: np.ndarray, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.sum(masses[:, None] * points, axis=0)
    centered = points - mean
    covariance = (masses[:, None] * centered).T @ centered
    return mean, covariance


def evaluate_pairs(
    X: np.ndarray,
    Y: np.ndarray,
    spec: ExperimentSpec,
    *,
    reference_X: np.ndarray | None = None,
    reference_Y: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Evaluate a fixed empirical coupling or generated joint outputs.

    Nonzero coupling edges may be passed directly with their masses. Support
    leakage is reported alongside labels because a point between disks or
    rings still receives a nearest-component label. Sliced-W2 is supplied for
    uniform pairs only; weighted moment errors remain valid for any plan.
    """
    x = _points(X, "X")
    y = _points(Y, "Y")
    if len(x) != len(y):
        raise ValueError("Paired X and Y must have equal sample counts")
    masses = _weights(len(x), weights)
    table = component_pair_matrix(x, y, spec, masses)
    result = {
        "mismatch_reward": float(table.sum() - np.trace(table)),
        "squared_distance_reward": float(np.dot(masses, np.sum((x - y) ** 2, axis=1))),
        "component_pair_matrix": table.tolist(),
        "x_component_frequencies": table.sum(axis=1).tolist(),
        "y_component_frequencies": table.sum(axis=0).tolist(),
    }
    if spec.name in ("two_disks", "rings"):
        result["x_off_support_mass"] = float(np.dot(masses, ~spec.support_mask(x, "x")))
        result["y_off_support_mass"] = float(np.dot(masses, ~spec.support_mask(y, "y")))
    for side, samples, reference in (("x", x, reference_X), ("y", y, reference_Y)):
        if reference is None:
            continue
        target = _points(reference, f"reference_{side.upper()}")
        target_mean, target_cov = _moments(target, _weights(len(target), None))
        sample_mean, sample_cov = _moments(samples, masses)
        result[f"{side}_mean_error"] = float(np.linalg.norm(sample_mean - target_mean))
        result[f"{side}_covariance_error"] = float(np.linalg.norm(sample_cov - target_cov, ord="fro"))
        if np.allclose(masses, 1.0 / len(masses), rtol=1e-12, atol=1e-15):
            result[f"{side}_sliced_wasserstein"] = sliced_wasserstein(target, samples, seed=seed)
    return result
