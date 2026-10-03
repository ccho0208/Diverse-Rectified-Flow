"""The three controlled two-dimensional marginal distributions.

Labels identify mixture components, not conditioning supplied to a model.
Classification is defined everywhere so generated points can be evaluated;
``support_mask`` separately identifies points outside the bounded targets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sources import SourceDistribution


def _positive_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _points(values: np.ndarray, name: str = "points") -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.shape[1] != 2 or len(raw) == 0:
        raise ValueError(f"{name} must have nonempty shape (N, 2)")
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric coordinates")
    with np.errstate(over="ignore", invalid="ignore"):
        points = raw.astype(np.float64, copy=False)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points


def _side(side: str) -> str:
    if side not in ("x", "y", "source"):
        raise ValueError("side must be 'x', 'y', or 'source'")
    return side


@dataclass(frozen=True)
class ExperimentSpec:
    """A serializable definition shared by sampling, rewards, and plots."""

    name: str
    components: int

    def __post_init__(self) -> None:
        _positive_integer(self.components, "components")
        if self.name not in ("gmm8", "two_disks", "rings"):
            raise ValueError(f"Unknown experiment: {self.name!r}")
        if self.name == "gmm8" and self.components != 8:
            raise ValueError("gmm8 has exactly eight components")
        if self.name == "two_disks" and self.components != 2:
            raise ValueError("two_disks has exactly two components")
        if self.name == "rings" and self.components not in (2, 8):
            raise ValueError("rings supports two or eight components")

    def centers(self, side: str = "x") -> np.ndarray | None:
        """Return component centers, or None for concentric annuli."""
        _side(side)
        if self.name == "gmm8":
            angles = 2 * np.pi * np.arange(self.components) / self.components
            return 3.0 * np.column_stack((np.cos(angles), np.sin(angles)))
        if self.name == "two_disks":
            horizontal = -2.0 if side == "source" else 2.0
            return np.array([[horizontal, 1.0], [horizontal, -1.0]])
        return None

    def source_distribution(self, copies: int = 1) -> SourceDistribution:
        """The disks start in left data space; other demos start in Gaussian noise."""
        if self.name == "two_disks":
            return SourceDistribution("uniform_disks", ((-2., 1.), (-2., -1.)), copies=copies)
        return SourceDistribution()

    def radii(self) -> np.ndarray | None:
        """Return the annulus center radii, if applicable."""
        if self.name != "rings":
            return None
        if self.components == 2:
            return np.array([1.0, 3.0])
        return np.arange(1, self.components + 1, dtype=np.float64)

    def sample(
        self, count: int, rng: np.random.Generator, side: str = "x"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Draw IID points and their latent component labels.

        Disks use a square-root radius for uniform area density. Annuli use
        uniform radius and angle, as specified in the experiment outline;
        this is different from uniform area density on an annulus.
        """
        count = _positive_integer(count, "count")
        _side(side)
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        labels = rng.integers(self.components, size=count, dtype=np.int64)
        if self.name == "gmm8":
            points = self.centers(side)[labels] + 0.15 * rng.normal(size=(count, 2))
        else:
            angle = rng.uniform(0.0, 2 * np.pi, size=count)
            if self.name == "two_disks":
                radius = 0.3 * np.sqrt(rng.uniform(size=count))
                centers = self.centers(side)[labels]
            else:
                radius = self.radii()[labels] + rng.uniform(-0.08, 0.08, size=count)
                centers = np.zeros((count, 2), dtype=np.float64)
            points = centers + radius[:, None] * np.column_stack(
                (np.cos(angle), np.sin(angle))
            )
        return np.asarray(points, dtype=np.float64), labels

    def classify(self, points: np.ndarray, side: str = "x") -> np.ndarray:
        """Assign nearest-center/radius labels, including off-support points.

        Nearest centers are maximum-posterior labels for the equal-weight,
        equal-covariance Gaussian mixture. Disk and annulus labels are also
        unambiguous on their disjoint supports.
        """
        points = _points(points)
        _side(side)
        if self.name == "rings":
            distances = np.abs(np.linalg.norm(points, axis=1)[:, None] - self.radii())
        else:
            distances = np.linalg.norm(points[:, None, :] - self.centers(side), axis=2)
        return np.argmin(distances, axis=1).astype(np.int64)

    def support_mask(self, points: np.ndarray, side: str = "x") -> np.ndarray:
        """Identify membership in the true support, independently of labels."""
        points = _points(points)
        _side(side)
        if self.name == "gmm8":
            return np.ones(len(points), dtype=bool)
        tolerance = 1e-12
        if self.name == "two_disks":
            distances = np.linalg.norm(points[:, None, :] - self.centers(side), axis=2)
            return np.min(distances, axis=1) <= 0.3 + tolerance
        distances = np.abs(np.linalg.norm(points, axis=1)[:, None] - self.radii())
        return np.min(distances, axis=1) <= 0.08 + tolerance

    def bounds(self) -> tuple[float, float, float, float]:
        """A common plotting window for both marginals."""
        if self.name == "gmm8":
            bound = 3.8
        elif self.name == "two_disks":
            return (-2.7, 2.7, -1.7, 1.7)
        else:
            bound = float(self.radii()[-1] + 0.6)
        return (-bound, bound, -bound, bound)

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "components": int(self.components),
            "component_weights": [1.0 / self.components] * self.components,
            "bounds": list(self.bounds()),
        }
        if self.name == "gmm8":
            result.update(centers=self.centers().tolist(), standard_deviation=0.15)
        elif self.name == "two_disks":
            result.update(
                setup_version="left_disks_to_right_disks_v2",
                source_centers=self.centers("source").tolist(),
                target_centers=self.centers("x").tolist(),
                x_centers=self.centers("x").tolist(),
                y_centers=self.centers("y").tolist(),
                disk_radius=0.3,
                radial_distribution="R * sqrt(Uniform[0,1])",
            )
        else:
            result.update(
                radii=self.radii().tolist(),
                radial_half_width=0.08,
                radial_distribution="uniform",
                angular_distribution="uniform",
            )
        return result


def build_spec(name: str, *, ring_components: int = 2) -> ExperimentSpec:
    """Build one of the agreed demo distributions."""
    if name == "gmm8":
        return ExperimentSpec(name, 8)
    if name == "two_disks":
        return ExperimentSpec(name, 2)
    if name == "rings":
        return ExperimentSpec(name, ring_components)
    raise ValueError(f"Unknown experiment: {name!r}")
