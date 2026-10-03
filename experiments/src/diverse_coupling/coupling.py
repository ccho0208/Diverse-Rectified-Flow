"""Maximum-reward couplings of equally sized uniform empirical measures."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment, linprog


def _indices(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if raw.dtype.kind == "u" and raw.size and raw.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the supported integer range")
    return raw.astype(np.int64, copy=True)


def _reward_matrix(C: np.ndarray) -> np.ndarray:
    raw = np.asarray(C)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[0] != raw.shape[1]:
        raise ValueError("C must be a nonempty square matrix; only n == m is supported")
    if raw.dtype.kind not in "biuf":
        raise ValueError("C must contain real numeric rewards")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = raw.astype(np.float64, copy=False)
    if not np.isfinite(matrix).all():
        raise ValueError("C must contain only finite rewards")
    return matrix


@dataclass
class CouplingPlan:
    """A compact probability plan, stored as nonzero indexed edges.

    Each empirical atom has marginal mass 1/n. An assignment plan carries its
    permutation as well; an LP plan does not promise a permutation representation.
    """

    shape: tuple[int, int]
    row_indices: np.ndarray
    column_indices: np.ndarray
    masses: np.ndarray
    permutation: np.ndarray | None = None
    solver: str = "unspecified"
    reward: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            len(self.shape) != 2
            or any(isinstance(v, (bool, np.bool_)) for v in self.shape)
            or any(not isinstance(v, (int, np.integer)) for v in self.shape)
        ):
            raise ValueError("shape must contain two integer dimensions")
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.row_indices = _indices(self.row_indices, "row_indices")
        self.column_indices = _indices(self.column_indices, "column_indices")
        raw_masses = np.asarray(self.masses)
        if raw_masses.ndim != 1 or raw_masses.dtype.kind not in "biuf":
            raise ValueError("masses must be a one-dimensional real numeric array")
        self.masses = raw_masses.astype(np.float64, copy=True)
        if self.permutation is not None:
            self.permutation = _indices(self.permutation, "permutation")

    def validate(self, atol: float = 1e-8) -> dict[str, Any]:
        """Check probability masses, atom indices, and both uniform marginals.

        Raise ValueError on malformed or infeasible plans. Return numerical
        feasibility diagnostics on success. No normalization or repair is done.
        """
        if not np.isfinite(atol) or atol <= 0:
            raise ValueError("atol must be finite and positive")
        n, m = self.shape
        if n <= 0 or n != m:
            raise ValueError("Only nonempty equally sized empirical measures are supported")
        count = len(self.masses)
        if len(self.row_indices) != count or len(self.column_indices) != count:
            raise ValueError("Edge indices and masses must have matching lengths")
        if not np.isfinite(self.masses).all() or (self.masses <= 0).any():
            raise ValueError("Stored coupling edges must have finite strictly positive masses")
        if (
            (self.row_indices < 0).any()
            or (self.row_indices >= n).any()
            or (self.column_indices < 0).any()
            or (self.column_indices >= m).any()
        ):
            raise ValueError("Coupling edge indices are out of range")
        edges = np.column_stack((self.row_indices, self.column_indices))
        if len(np.unique(edges, axis=0)) != count:
            raise ValueError("Coupling edges must be unique")
        rows = np.bincount(self.row_indices, weights=self.masses, minlength=n)
        cols = np.bincount(self.column_indices, weights=self.masses, minlength=m)
        total = float(self.masses.sum())
        row_error = float(np.max(np.abs(rows - 1.0 / n)))
        col_error = float(np.max(np.abs(cols - 1.0 / n)))
        if abs(total - 1.0) > atol or row_error > atol or col_error > atol:
            raise ValueError(
                "Coupling does not have total mass one and uniform marginals "
                f"(total={total}, row error={row_error}, column error={col_error})"
            )
        if self.permutation is not None:
            sigma = self.permutation
            if sigma.shape != (n,) or not np.array_equal(np.sort(sigma), np.arange(n)):
                raise ValueError("permutation must use each column exactly once")
            if (
                count != n
                or not np.array_equal(np.sort(self.row_indices), np.arange(n))
                or not np.array_equal(self.column_indices, sigma[self.row_indices])
                or not np.allclose(self.masses, 1.0 / n, rtol=0, atol=atol)
            ):
                raise ValueError("Plan entries do not agree with its permutation")
        if self.reward is not None and not np.isfinite(self.reward):
            raise ValueError("reward must be finite when provided")
        return {
            "valid": True,
            "total_mass": total,
            "row_marginal_max_error": row_error,
            "column_marginal_max_error": col_error,
            "nonzero_count": int(np.count_nonzero(self.masses)),
            "atol": float(atol),
        }

    def to_sparse(self) -> sparse.csr_matrix:
        """Materialize a SciPy CSR matrix after checking feasibility."""
        self.validate()
        return sparse.csr_matrix(
            (self.masses, (self.row_indices, self.column_indices)), shape=self.shape
        )

    def to_dense(self) -> np.ndarray:
        """Materialize P; assignment storage itself remains O(n)."""
        return self.to_sparse().toarray()

    def save(self, path: str | Path) -> None:
        """Save arrays and JSON metadata in NPZ without pickled objects."""
        self.validate()
        metadata = json.dumps(
            {
                "format_version": 1,
                "has_permutation": self.permutation is not None,
                "solver": self.solver,
                "reward": self.reward,
                "diagnostics": self.diagnostics,
            },
            allow_nan=False,
        )
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as stream:
            np.savez_compressed(
                stream,
                shape=np.asarray(self.shape, dtype=np.int64),
                row_indices=self.row_indices,
                column_indices=self.column_indices,
                masses=self.masses,
                permutation=(
                    self.permutation
                    if self.permutation is not None
                    else np.empty(0, dtype=np.int64)
                ),
                metadata=np.asarray(metadata),
            )

    @classmethod
    def load(cls, path: str | Path) -> CouplingPlan:
        """Load and validate a plan saved by :meth:`save`."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            if metadata.get("format_version") != 1:
                raise ValueError("Unsupported coupling plan format")
            plan = cls(
                shape=tuple(archive["shape"].tolist()),
                row_indices=archive["row_indices"],
                column_indices=archive["column_indices"],
                masses=archive["masses"],
                permutation=(
                    archive["permutation"] if metadata["has_permutation"] else None
                ),
                solver=metadata["solver"],
                reward=metadata["reward"],
                diagnostics=metadata["diagnostics"],
            )
        plan.validate()
        return plan


def solve_assignment(C: np.ndarray) -> np.ndarray:
    """Return one permutation maximizing sum_i C[i, sigma[i]]."""
    matrix = _reward_matrix(C)
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    permutation = np.empty(len(matrix), dtype=np.int64)
    permutation[rows] = columns
    return permutation


def coupling_from_permutation(sigma: np.ndarray) -> CouplingPlan:
    """Construct P[i, sigma[i]] = 1/n from a genuine permutation."""
    permutation = _indices(sigma, "sigma")
    n = len(permutation)
    if n == 0 or not np.array_equal(np.sort(permutation), np.arange(n)):
        raise ValueError("sigma must be a nonempty permutation of 0, ..., n-1")
    plan = CouplingPlan(
        shape=(n, n),
        row_indices=np.arange(n, dtype=np.int64),
        column_indices=permutation,
        masses=np.full(n, 1.0 / n, dtype=np.float64),
        permutation=permutation,
        solver="assignment",
    )
    plan.diagnostics = plan.validate()
    return plan


def solve_optimal_coupling(
    C: np.ndarray, use_permutation: bool = True
) -> CouplingPlan:
    """Maximize <C, P> under equal uniform empirical marginals.

    Assignment is the default compact backend. The reference backend solves an
    unregularized transportation LP with sparse equality constraints. It may
    also select a permutation optimum, but carries the general weighted plan.
    """
    matrix = _reward_matrix(C)
    if not isinstance(use_permutation, (bool, np.bool_)):
        raise ValueError("use_permutation must be a boolean")
    n = len(matrix)
    if use_permutation:
        plan = coupling_from_permutation(solve_assignment(matrix))
    else:
        edge_ids = np.arange(n * n, dtype=np.int64)
        row_ids = np.repeat(np.arange(n, dtype=np.int64), n)
        column_ids = np.tile(np.arange(n, dtype=np.int64), n)
        constraints = sparse.coo_matrix(
            (
                np.ones(2 * n * n, dtype=np.float64),
                (
                    np.concatenate((row_ids, n + column_ids)),
                    np.concatenate((edge_ids, edge_ids)),
                ),
            ),
            shape=(2 * n, n * n),
        ).tocsr()
        # Row offsets contribute a constant because each row has marginal 1/n.
        # Remove them before scaling so a large common reward offset cannot
        # hide meaningful differences behind the LP's numerical tolerances.
        # Half-min plus half-max avoids overflow when extrema have opposite signs.
        offsets = matrix.min(axis=1) * 0.5 + matrix.max(axis=1) * 0.5
        centered = matrix - offsets[:, None]
        scale = float(np.max(np.abs(centered))) or 1.0
        result = linprog(
            -(centered / scale).ravel(),
            A_eq=constraints,
            b_eq=np.full(2 * n, 1.0 / n),
            bounds=(0.0, None),
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        if not result.success:
            raise RuntimeError(f"Transportation LP failed: {result.message}")
        masses = np.asarray(result.x, dtype=np.float64)
        if not np.isfinite(masses).all() or (masses < -1e-9).any():
            raise RuntimeError("Transportation LP returned invalid masses")
        masses = np.maximum(masses, 0.0)
        support = np.flatnonzero(masses > 0.0)
        plan = CouplingPlan(
            shape=(n, n),
            row_indices=row_ids[support],
            column_indices=column_ids[support],
            masses=masses[support],
            solver="transportation_lp",
            diagnostics={
                "lp_status": int(result.status),
                "lp_message": str(result.message),
                "lp_iterations": int(result.nit),
                "reward_scale": scale,
                "reward_centering": "per_row_midpoint",
            },
        )
    plan.reward = float(
        np.sum(matrix[plan.row_indices, plan.column_indices] * plan.masses)
    )
    plan.diagnostics.update(plan.validate())
    return plan
