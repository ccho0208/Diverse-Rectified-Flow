"""Straight-path rectified flow on concatenated joint targets."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn


class MLPVelocity(nn.Module):
    """A velocity field accepting a batch of states and their scalar times."""

    def __init__(self, joint_dim: int, hidden_sizes: Sequence[int] = (128, 128, 128)):
        super().__init__()
        if not isinstance(joint_dim, int) or joint_dim < 1:
            raise ValueError("joint_dim must be a positive integer")
        self.joint_dim = joint_dim
        self.hidden_sizes = tuple(hidden_sizes)
        if any(not isinstance(w, int) or w < 1 for w in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        widths = (joint_dim + 1, *self.hidden_sizes, joint_dim)
        layers: list[nn.Module] = []
        for index, (a, b) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(a, b))
            if index < len(widths) - 2:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(self, states: Tensor, times: Tensor) -> Tensor:
        if states.ndim != 2 or states.shape[1] != self.joint_dim:
            raise ValueError("states must have shape (batch, joint_dim)")
        times = torch.as_tensor(times, dtype=states.dtype, device=states.device)
        if times.ndim == 0:
            times = times.expand(len(states))
        if times.shape not in ((len(states),), (len(states), 1)):
            raise ValueError("times must be scalar or have one value per state")
        return self.network(torch.cat((states, times.reshape(-1, 1)), dim=1))


class RectifiedFlowAdapter(nn.Module):
    """Return per-target RF losses; the trainer applies the coupling masses.

    The Gaussian source is independent of the joint target. X and Y dependence
    is carried by the target pairs, rather than by this source-target pairing.
    """

    def __init__(self, joint_dim: int, hidden_sizes: Sequence[int] = (128, 128, 128),
                 *, seed: int | None = None):
        super().__init__()
        # Optional seeded initialization does not consume the caller's RNG state.
        if seed is None:
            self.velocity = MLPVelocity(joint_dim, hidden_sizes)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed)
                self.velocity = MLPVelocity(joint_dim, hidden_sizes)
        self.joint_dim = self.velocity.joint_dim

    def forward(self, states: Tensor, times: Tensor) -> Tensor:
        return self.velocity(states, times)

    def per_example_loss(self, targets: Tensor, *, noise: Tensor,
                         times: Tensor) -> Tensor:
        if targets.ndim != 2 or targets.shape[1] != self.joint_dim:
            raise ValueError("targets must have shape (batch, joint_dim)")
        if noise.shape != targets.shape:
            raise ValueError("noise must have the same shape as targets")
        if times.shape not in ((len(targets),), (len(targets), 1)):
            raise ValueError("times must have one value per target")
        if not torch.isfinite(targets).all() or not torch.isfinite(noise).all():
            raise ValueError("targets and noise must be finite")
        if not torch.isfinite(times).all() or torch.any((times < 0) | (times > 1)):
            raise ValueError("times must be finite and lie in [0, 1]")
        t = times.reshape(-1, 1)
        states = (1 - t) * noise + t * targets
        residual = self(states, times) - (targets - noise)
        return residual.square().sum(dim=1)

    def export_config(self) -> dict:
        return {"kind": "rectified_flow", "joint_dim": self.joint_dim,
                "hidden_sizes": list(self.velocity.hidden_sizes)}
