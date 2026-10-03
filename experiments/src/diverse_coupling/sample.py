"""Generate joint pairs by integrating a learned rectified-flow velocity."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .models import RectifiedFlowAdapter


@dataclass
class SampleResult:
    X: np.ndarray
    Y: np.ndarray
    joint: np.ndarray
    initial_noise: np.ndarray
    times: np.ndarray
    trajectory: np.ndarray | None
    x_shape: tuple[int, ...]
    y_shape: tuple[int, ...]
    metadata: dict = field(default_factory=dict)

    @property
    def x_trajectory(self) -> np.ndarray | None:
        if self.trajectory is None:
            return None
        dim = int(np.prod(self.x_shape))
        return self.trajectory[..., :dim].reshape(
            *self.trajectory.shape[:-1], *self.x_shape)

    @property
    def y_trajectory(self) -> np.ndarray | None:
        if self.trajectory is None:
            return None
        dim = int(np.prod(self.x_shape))
        return self.trajectory[..., dim:].reshape(
            *self.trajectory.shape[:-1], *self.y_shape)

    def save(self, path: str | Path) -> None:
        arrays = {"X": self.X, "Y": self.Y, "joint": self.joint,
                  "initial_noise": self.initial_noise, "times": self.times,
                  "x_shape": np.array(self.x_shape, dtype=np.int64),
                  "y_shape": np.array(self.y_shape, dtype=np.int64),
                  "metadata": np.asarray(json.dumps(self.metadata, allow_nan=False))}
        if self.trajectory is not None:
            arrays["trajectory"] = self.trajectory
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)


def load_joint_model(checkpoint: str | Path, *, device: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model_config = payload["model_config"]
    if model_config.get("kind") != "rectified_flow":
        raise ValueError("automatic sampling currently supports rectified_flow checkpoints")
    x_shape, y_shape = tuple(payload["x_shape"]), tuple(payload["y_shape"])
    if any(not isinstance(d, int) or d < 1 for d in x_shape + y_shape):
        raise ValueError("invalid observation shapes in checkpoint")
    joint_dim = int(np.prod(x_shape)) + int(np.prod(y_shape))
    if joint_dim != model_config["joint_dim"]:
        raise ValueError("checkpoint observation shapes do not match model dimensions")
    dtype_name = payload["train_config"]["dtype"]
    if dtype_name not in ("float32", "float64"):
        raise ValueError("unsupported checkpoint dtype")
    model = RectifiedFlowAdapter(joint_dim, model_config["hidden_sizes"], seed=0)
    model.to(device=device, dtype=getattr(torch, dtype_name))
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


@torch.no_grad()
def integrate_velocity(model: nn.Module, initial_noise: Tensor, *, steps: int = 100,
                       method: str = "heun", record_trajectory: bool = False
                       ) -> tuple[Tensor, Tensor, Tensor | None]:
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    if method not in ("euler", "heun"):
        raise ValueError("method must be euler or heun")
    if initial_noise.ndim != 2 or not torch.isfinite(initial_noise).all():
        raise ValueError("initial_noise must be a finite (count, joint_dim) tensor")
    state = initial_noise.clone()
    times = torch.linspace(0, 1, steps + 1, device=state.device, dtype=state.dtype)
    trajectory = [state.clone()] if record_trajectory else None
    for index in range(steps):
        dt = times[index + 1] - times[index]
        velocity = model(state, times[index])
        if velocity.shape != state.shape:
            raise ValueError("velocity field must preserve the joint state shape")
        proposal = state + dt * velocity
        if method == "heun":
            next_velocity = model(proposal, times[index + 1])
            if next_velocity.shape != state.shape:
                raise ValueError("velocity field must preserve the joint state shape")
            state = state + 0.5 * dt * (velocity + next_velocity)
        else:
            state = proposal
        if not torch.isfinite(state).all():
            raise FloatingPointError("ODE sampling produced nonfinite states")
        if trajectory is not None:
            trajectory.append(state.clone())
    return state, times, None if trajectory is None else torch.stack(trajectory)


def sample_joint_model(checkpoint: str | Path, count: int, record_trajectory: bool = False,
                       *, steps: int = 100, method: str = "heun", seed: int = 0,
                       device: str = "cpu", initial_noise=None) -> SampleResult:
    if not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    model, payload = load_joint_model(checkpoint, device=device)
    dtype = next(model.parameters()).dtype
    dim = model.joint_dim
    if initial_noise is None:
        initial_noise = np.random.default_rng(seed).standard_normal((count, dim))
    source = torch.as_tensor(initial_noise, device=device, dtype=dtype)
    if source.shape != (count, dim):
        raise ValueError("initial_noise must have shape (count, joint_dim)")
    joint, times, trajectory = integrate_velocity(
        model, source, steps=steps, method=method, record_trajectory=record_trajectory)
    joint_array = joint.cpu().numpy()
    x_shape, y_shape = tuple(payload["x_shape"]), tuple(payload["y_shape"])
    dx = int(np.prod(x_shape))
    return SampleResult(
        X=joint_array[:, :dx].reshape(count, *x_shape),
        Y=joint_array[:, dx:].reshape(count, *y_shape), joint=joint_array,
        initial_noise=source.cpu().numpy(), times=times.cpu().numpy(),
        trajectory=None if trajectory is None else trajectory.cpu().numpy(),
        x_shape=x_shape, y_shape=y_shape,
        metadata={"checkpoint": str(Path(checkpoint).resolve()), "seed": seed,
                  "method": method, "steps": steps, "count": count,
                  "record_trajectory": record_trajectory})
