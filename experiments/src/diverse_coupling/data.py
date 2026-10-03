"""Fixed, indexed observations for direct empirical coupling objectives.

Each row of a :class:`PairedBatch` is a nonzero edge of a previously solved
coupling. Its mass is the original probability mass in the complete plan;
chunking never renormalizes those masses or resamples pair identities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Iterator

import numpy as np
import torch

from .coupling import CouplingPlan


def _observations(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one observation")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric observations")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite observations")
    if np.prod(array.shape[1:], dtype=np.int64) == 0:
        raise ValueError(f"{name} observations must have at least one coordinate")
    return array


def _observation_ids(values: np.ndarray | None, count: int, name: str) -> np.ndarray:
    if values is None:
        return np.arange(count, dtype=np.int64)
    array = np.asarray(values)
    if array.ndim != 1 or len(array) != count:
        raise ValueError(f"{name} must be a one-dimensional ID array of length {count}")
    return array


@dataclass(frozen=True)
class PairedBatch:
    """One indexed chunk, retaining its original plan masses and atom IDs."""

    targets: torch.Tensor
    masses: torch.Tensor
    x_indices: np.ndarray
    y_indices: np.ndarray
    edge_indices: np.ndarray
    x_ids: np.ndarray
    y_ids: np.ndarray


@dataclass
class PairedDataset:
    """An indexed view of the nonzero entries of a fixed coupling plan.

    Scalar observations are accepted as arrays of shape ``(n,)``. Higher
    dimensional observations retain their trailing coordinate shapes. Input
    arrays are retained rather than eagerly copying every paired target.
    """

    X: np.ndarray
    Y: np.ndarray
    plan: CouplingPlan
    x_ids: np.ndarray | None = None
    y_ids: np.ndarray | None = None
    x_shape: tuple[int, ...] = field(init=False)
    y_shape: tuple[int, ...] = field(init=False)
    x_dim: int = field(init=False)
    y_dim: int = field(init=False)
    joint_dim: int = field(init=False)

    def __post_init__(self) -> None:
        self.X = _observations(self.X, "X")
        self.Y = _observations(self.Y, "Y")
        if not isinstance(self.plan, CouplingPlan):
            raise TypeError("plan must be a CouplingPlan")
        self.plan.validate()
        if self.plan.shape != (len(self.X), len(self.Y)):
            raise ValueError("observation counts must match the coupling plan shape")
        self.x_ids = _observation_ids(self.x_ids, len(self.X), "x_ids")
        self.y_ids = _observation_ids(self.y_ids, len(self.Y), "y_ids")
        self.x_shape = tuple(self.X.shape[1:])
        self.y_shape = tuple(self.Y.shape[1:])
        self.x_dim = int(np.prod(self.x_shape, dtype=np.int64))
        self.y_dim = int(np.prod(self.y_shape, dtype=np.int64))
        self.joint_dim = self.x_dim + self.y_dim

    def __len__(self) -> int:
        return len(self.plan.masses)

    def get_batch(
        self,
        edge_indices: np.ndarray,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> PairedBatch:
        """Materialize requested edges without changing their probability masses."""
        indices = np.asarray(edge_indices)
        if indices.ndim != 1:
            raise ValueError("edge_indices must be a one-dimensional integer array")
        if indices.size and indices.dtype.kind not in "iu":
            raise ValueError("edge_indices must contain integer indices")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("edge index is outside the coupling plan")
        indices = indices.astype(np.int64, copy=True)
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise ValueError("training dtype must be a floating-point torch dtype")

        x_indices = self.plan.row_indices[indices]
        y_indices = self.plan.column_indices[indices]
        x = self.X[x_indices].reshape(len(indices), self.x_dim)
        y = self.Y[y_indices].reshape(len(indices), self.y_dim)
        # Float64 is a supported NumPy bridge even for input dtypes (e.g.
        # extended precision) that torch.as_tensor cannot import directly.
        targets = np.asarray(np.concatenate((x, y), axis=1), dtype=np.float64)
        return PairedBatch(
            targets=torch.as_tensor(targets, dtype=dtype, device=device),
            masses=torch.as_tensor(self.plan.masses[indices], dtype=dtype, device=device),
            x_indices=x_indices,
            y_indices=y_indices,
            edge_indices=indices,
            x_ids=self.x_ids[x_indices],
            y_ids=self.y_ids[y_indices],
        )

    def split_joint(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Restore observation shapes from a batch or a recorded joint trajectory."""
        if not isinstance(states, torch.Tensor):
            raise TypeError("states must be a torch tensor")
        if states.ndim < 1 or states.shape[-1] != self.joint_dim:
            raise ValueError(f"states must have final dimension {self.joint_dim}")
        leading_shape = tuple(states.shape[:-1])
        x = states[..., : self.x_dim].reshape(leading_shape + self.x_shape)
        y = states[..., self.x_dim :].reshape(leading_shape + self.y_shape)
        return x, y

    def epoch_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool = False,
        rng: np.random.Generator | None = None,
    ) -> Iterator[np.ndarray]:
        """Visit every fixed edge once, including a partial final chunk.

        Supply a seeded NumPy generator for reproducible shuffled ordering.
        Both coordinates always follow the same edge index.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if rng is not None and not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a NumPy Generator")
        order = np.arange(len(self), dtype=np.int64)
        if shuffle:
            (rng if rng is not None else np.random.default_rng()).shuffle(order)
        for start in range(0, len(self), batch_size):
            yield order[start : start + batch_size].copy()


def build_paired_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    plan: CouplingPlan,
    *,
    x_ids: np.ndarray | None = None,
    y_ids: np.ndarray | None = None,
) -> PairedDataset:
    """Build a fixed, weighted view of the empirical joint measure defined by P."""
    return PairedDataset(X=X, Y=Y, plan=plan, x_ids=x_ids, y_ids=y_ids)
