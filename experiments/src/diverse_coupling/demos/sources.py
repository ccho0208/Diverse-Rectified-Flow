"""Reproducible source distributions for marginal and joint RF training."""

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SourceDistribution:
    kind: str = "normal"
    centers: tuple[tuple[float, float], ...] = ()
    radius: float = 0.3
    copies: int = 1

    def validate(self, dimensions: int) -> None:
        if self.kind == "normal":
            return
        if self.kind != "uniform_disks":
            raise ValueError("unsupported source distribution")
        centers = np.asarray(self.centers, dtype=float)
        if centers.ndim != 2 or centers.shape[1] != 2 or not len(centers) or not np.isfinite(centers).all():
            raise ValueError("source centers must be a finite nonempty 2D array")
        if isinstance(self.copies, bool) or not isinstance(self.copies, int) or self.copies < 1:
            raise ValueError("source copies must be a positive integer")
        if dimensions != 2 * self.copies:
            raise ValueError("disk source dimensions must equal twice the number of copies")
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("source disk radius must be finite and positive")

    def to_dict(self) -> dict:
        if self.kind == "normal":
            return {"kind": "normal"}
        return {"kind": self.kind, "centers": [list(c) for c in self.centers],
                "radius": self.radius, "copies": self.copies,
                "component_weights": [1 / len(self.centers)] * len(self.centers)}

    def sample_numpy(self, count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
        self.validate(dimensions)
        if self.kind == "normal":
            return rng.standard_normal((count, dimensions))
        labels = rng.integers(len(self.centers), size=(count, self.copies))
        angles = rng.uniform(0, 2 * np.pi, size=(count, self.copies))
        radii = self.radius * np.sqrt(rng.uniform(size=(count, self.copies)))
        points = np.asarray(self.centers)[labels] + radii[..., None] * np.stack(
            (np.cos(angles), np.sin(angles)), axis=-1)
        return points.reshape(count, dimensions)

    def sample_torch(self, count: int, dimensions: int, *, generator: torch.Generator,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self.validate(dimensions)
        if self.kind == "normal":
            return torch.randn((count, dimensions), generator=generator, device=device, dtype=dtype)
        shape = (count, self.copies)
        labels = torch.randint(len(self.centers), shape, generator=generator, device=device)
        angles = 2 * torch.pi * torch.rand(shape, generator=generator, device=device, dtype=dtype)
        radii = self.radius * torch.sqrt(torch.rand(shape, generator=generator, device=device, dtype=dtype))
        centers = torch.tensor(self.centers, device=device, dtype=dtype)
        points = centers[labels] + radii[..., None] * torch.stack((angles.cos(), angles.sin()), dim=-1)
        return points.reshape(count, dimensions)

    def antithetic(self, points: np.ndarray) -> np.ndarray:
        """Reflect around the source center, preserving a symmetric mixture."""
        points = np.asarray(points)
        self.validate(points.shape[1])
        if self.kind == "normal":
            return -points
        centers = np.asarray(self.centers)
        center = centers.mean(axis=0)
        distances = np.linalg.norm((2 * center - centers)[:, None] - centers[None], axis=-1)
        if np.any(distances.min(axis=1) > 1e-12):
            raise ValueError("antithetic reflection requires a centrally symmetric disk mixture")
        return (2 * center - points.reshape(len(points), self.copies, 2)).reshape(points.shape)
