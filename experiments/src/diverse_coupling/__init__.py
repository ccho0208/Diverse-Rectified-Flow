"""Optimize empirical pair couplings, then learn their joint law."""

from .costs import build_reward_matrix
from .coupling import (
    CouplingPlan,
    coupling_from_permutation,
    solve_assignment,
    solve_optimal_coupling,
)
from .data import PairedDataset, build_paired_dataset

__all__ = [
    "CouplingPlan", "PairedDataset", "build_reward_matrix",
    "coupling_from_permutation", "solve_assignment", "solve_optimal_coupling",
    "build_paired_dataset",
]
