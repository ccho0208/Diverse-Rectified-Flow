"""Shared RF training and forward/reverse integration for the 2D demos.

Every architecture uses the same unconstrained time-conditioned MLP. Training
visits all fixed targets with their global probability masses once per optimizer
step. Noise and times are generated on the training device before chunking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Sequence

import numpy as np
import torch
from torch import nn

from ..models import RectifiedFlowAdapter


@dataclass(frozen=True)
class FlowTrainingConfig:
    epochs: int = 3000
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    chunk_size: int = 2048
    checkpoint_every: int = 250

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for key in ("epochs", "chunk_size", "checkpoint_every"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        device = _available_device(self.device)
        if device.type == "mps" and self.dtype == "float64":
            raise ValueError("MPS requires float32; use CPU for float64")


@dataclass
class FlowResult:
    endpoints: np.ndarray
    times: np.ndarray
    trajectory: np.ndarray | None = None


def _available_device(value: str | torch.device) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; select --device cpu or use a GPU node")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is unavailable")
    elif device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
    elif device.type != "cpu":
        raise ValueError("device must be cpu, cuda, cuda:<index>, or mps")
    return device


def _points(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape (samples, dimensions)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(result)


def _data_identity(targets: np.ndarray, masses: np.ndarray,
                   source_points: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    for label, array in (("targets", targets), ("masses", masses),
                         ("sources", source_points)):
        digest.update(label.encode())
        if array is None:
            digest.update(b"gaussian")
        else:
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _atomic_save(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_flow(checkpoint: str | Path, device: str = "cpu") -> tuple[RectifiedFlowAdapter, dict]:
    """Load our tensor-only checkpoint format, with explicit device selection."""
    destination = _available_device(device)
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format") != "diverse_coupling_demo_flow_v1":
        raise ValueError("checkpoint is not a supported demo-flow checkpoint")
    config = payload["model_config"]
    dtype = getattr(torch, payload["train_config"]["dtype"])
    if destination.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS requires float32; use CPU for this float64 checkpoint")
    model = RectifiedFlowAdapter(config["joint_dim"], config["hidden_sizes"], seed=0)
    model.to(device=destination, dtype=dtype)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def train_flow(
    targets: np.ndarray, config: FlowTrainingConfig, *, output_dir: str | Path,
    source_points: np.ndarray | None = None, masses: np.ndarray | None = None,
    hidden_sizes: Sequence[int] = (64, 64, 64), resume: bool = False,
    metadata: dict | None = None, progress: Callable[[dict], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
) -> RectifiedFlowAdapter:
    """Train a Gaussian-to-target or fixed-source-to-target straight-path RF.

    One epoch is one globally mass-weighted optimizer step. Chunk losses are
    summed without local renormalization, and pair identities are never drawn.
    Resume checks the data hash, architecture, and all optimization settings;
    only the epoch limit and device may change. Epoch seeds make same-device
    resumption equivalent to uninterrupted training. CPU/CUDA random draws can
    differ when the backend changes. A stop callback saves after a complete
    epoch and raises ``InterruptedError`` so a scheduler can terminate safely.
    """
    config.validate()
    target_array = _points(targets, "targets")
    source_array = None if source_points is None else _points(source_points, "source_points")
    if source_array is not None and source_array.shape != target_array.shape:
        raise ValueError("source_points must have the same shape as targets")
    size, dimensions = target_array.shape
    mass_array = (np.full(size, 1 / size, dtype=np.float64) if masses is None
                  else np.asarray(masses, dtype=np.float64))
    if mass_array.shape != (size,) or not np.isfinite(mass_array).all() or (mass_array < 0).any():
        raise ValueError("masses must be a finite nonnegative vector with one entry per target")
    if not np.isclose(mass_array.sum(), 1.0, rtol=0, atol=1e-7):
        raise ValueError("masses must sum to one")
    metadata = {} if metadata is None else json.loads(json.dumps(metadata, allow_nan=False))
    identity = _data_identity(target_array, mass_array, source_array)
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "checkpoint.pt"
    device = _available_device(config.device)
    dtype = getattr(torch, config.dtype)
    hidden_sizes = tuple(hidden_sizes)
    history: list[dict] = []
    previous = None
    if checkpoint.exists():
        if not resume:
            raise FileExistsError(f"{checkpoint} already exists; resume the run or choose another output directory")
        model, previous = load_flow(checkpoint, config.device)
        if previous["data_identity"] != identity:
            raise ValueError("resume data identity differs: targets, masses, or source_points changed")
        if (previous["model_config"]["joint_dim"] != dimensions or
                tuple(previous["model_config"]["hidden_sizes"]) != hidden_sizes):
            raise ValueError("resume model architecture differs")
        allowed_changes = {"epochs", "device"}
        for key, value in config.to_dict().items():
            if key not in allowed_changes and previous["train_config"][key] != value:
                raise ValueError(f"resume training configuration differs: {key}")
        if previous["metadata"] != metadata:
            raise ValueError("resume metadata differs")
        history = previous["history"]
    else:
        model = RectifiedFlowAdapter(dimensions, hidden_sizes, seed=config.seed)
        model.to(device=device, dtype=dtype)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    if previous is not None:
        optimizer.load_state_dict(previous["optimizer_state"])
    target_tensor = torch.as_tensor(target_array, device=device, dtype=dtype)
    mass_tensor = torch.as_tensor(mass_array, device=device, dtype=dtype)
    source_tensor = (None if source_array is None else
                     torch.as_tensor(source_array, device=device, dtype=dtype))
    if not torch.isfinite(target_tensor).all() or (source_tensor is not None and
                                                 not torch.isfinite(source_tensor).all()):
        raise ValueError("training values overflow the requested dtype")
    # MPS does not provide a native Generator; CPU/MPS draws are copied once
    # per epoch. CPU and CUDA use their own generators directly on-device.
    rng_device = device if device.type != "mps" else torch.device("cpu")
    generator = torch.Generator(device=rng_device)
    if previous is not None and previous["rng_device_type"] == rng_device.type:
        generator.set_state(previous["rng_state"].cpu())

    def save() -> None:
        payload = {
            "format": "diverse_coupling_demo_flow_v1", "format_version": 1,
            "model_config": model.export_config(), "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "train_config": config.to_dict(),
            "completed_epochs": len(history), "history": history, "metadata": metadata,
            "data_identity": identity, "source_kind": "normal" if source_tensor is None else "fixed",
            "sample_count": size, "rng_device_type": rng_device.type,
            "rng_state": generator.get_state().cpu(), "rng_policy": "seed_plus_zero_based_epoch",
        }
        _atomic_save(checkpoint, payload)
        _atomic_json(folder / "history.json", history)

    for epoch in range(len(history) + 1, config.epochs + 1):
        generator.manual_seed((config.seed + epoch - 1) % (2**63 - 1))
        if source_tensor is None:
            sources = torch.randn((size, dimensions), device=rng_device, dtype=dtype,
                                  generator=generator).to(device)
        else:
            sources = source_tensor
        times = torch.rand(size, device=rng_device, dtype=dtype, generator=generator).to(device)
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device, dtype=dtype)
        for start in range(0, size, config.chunk_size):
            stop = min(start + config.chunk_size, size)
            src, target = sources[start:stop], target_tensor[start:stop]
            t = times[start:stop].reshape(-1, 1)
            states = (1 - t) * src + t * target
            residual = model(states, times[start:stop]) - (target - src)
            contribution = (residual.square().sum(dim=1) * mass_tensor[start:stop]).sum()
            contribution.backward()
            total.add_(contribution.detach())
        # Aggregate one finite check per complete step instead of synchronizing
        # a CUDA stream once for every layer and chunk.
        gradient_norm = torch.zeros((), device=device, dtype=dtype)
        for parameter in model.parameters():
            if parameter.grad is not None:
                gradient_norm.add_(parameter.grad.square().sum())
        finite = torch.isfinite(torch.stack((total, gradient_norm))).all()
        if not bool(finite):
            raise FloatingPointError("RF training produced nonfinite loss or gradients")
        optimizer.step()
        record = {"epoch": epoch, "loss": float(total.detach().cpu()),
                  "optimizer_steps": epoch, "paired_rows_seen": size}
        history.append(record)
        stopping = stop_requested is not None and stop_requested()
        if epoch % config.checkpoint_every == 0 or epoch == config.epochs or stopping:
            save()
        if progress is not None:
            progress(record.copy())
        if stopping:
            raise InterruptedError(f"Training stopped after epoch {epoch}; checkpoint saved to {checkpoint}")
    # A fully completed resumed run may not enter the loop.
    if not checkpoint.exists():
        save()
    model.eval()
    return model


def _model_device_dtype(model: nn.Module, device: str | torch.device | None,
                        fallback_dtype: torch.dtype = torch.float64) -> tuple[torch.device, torch.dtype]:
    tensors = list(model.parameters()) + list(model.buffers())
    current_device = tensors[0].device if tensors else torch.device("cpu")
    dtype = tensors[0].dtype if tensors else fallback_dtype
    selected = current_device if device is None else _available_device(device)
    model.to(device=selected)
    return selected, dtype


def integrate(
    model: nn.Module, states: np.ndarray, *, steps: int = 200,
    start_time: float = 0., end_time: float = 1., method: str = "heun",
    record_trajectory: bool = False, batch_size: int = 2048,
    device: str | None = None,
) -> FlowResult:
    """Integrate a learned velocity in either direction within [0, 1].

    Recorded trajectories include both the initial and final states. Negative
    time increments implement the inverse ODE without evaluating negative times.
    Batches are independent and each is transferred to the device once.
    """
    array = _points(states, "states")
    for name, value in (("steps", steps), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if method not in ("heun", "euler"):
        raise ValueError("method must be heun or euler")
    if not all(np.isfinite(t) and 0 <= t <= 1 for t in (start_time, end_time)):
        raise ValueError("start_time and end_time must be finite and in [0, 1]")
    if hasattr(model, "joint_dim") and model.joint_dim != array.shape[1]:
        raise ValueError("state dimensions do not match model joint_dim")
    destination, dtype = _model_device_dtype(model, device)
    if destination.type == "mps" and dtype == torch.float64:
        raise ValueError("MPS requires float32")
    times = np.linspace(start_time, end_time, steps + 1, dtype=np.float64)
    output_dtype = np.float64 if dtype == torch.float64 else np.float32
    endpoints = np.empty(array.shape, dtype=output_dtype)
    trajectories = (np.empty((steps + 1, *array.shape), dtype=output_dtype)
                    if record_trajectory else None)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for begin in range(0, len(array), batch_size):
                end = min(begin + batch_size, len(array))
                state = torch.as_tensor(array[begin:end], device=destination, dtype=dtype)
                path = (torch.empty((steps + 1, *state.shape), device=destination, dtype=dtype)
                        if record_trajectory else None)
                if path is not None:
                    path[0] = state
                for index, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
                    delta = float(t1 - t0)
                    t = torch.full((len(state),), float(t0), device=destination, dtype=dtype)
                    velocity = model(state, t)
                    proposal = state + delta * velocity
                    if method == "heun":
                        next_t = torch.full((len(state),), float(t1), device=destination, dtype=dtype)
                        state = state + .5 * delta * (velocity + model(proposal, next_t))
                    else:
                        state = proposal
                    if path is not None:
                        path[index + 1] = state
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError("ODE integration produced nonfinite endpoints")
                endpoints[begin:end] = state.cpu().numpy()
                if path is not None:
                    if not bool(torch.isfinite(path).all()):
                        raise FloatingPointError("ODE integration produced nonfinite trajectories")
                    trajectories[:, begin:end] = path.cpu().numpy()
    finally:
        model.train(was_training)
    return FlowResult(endpoints, times, trajectories)


def velocity_grid(
    model: nn.Module, bounds: Sequence, times: Sequence[float], *,
    grid_size: int = 21, device: str | None = None,
) -> dict:
    """Evaluate an actual 2D marginal field on an XY grid at chosen times."""
    box = np.asarray(bounds, dtype=np.float64)
    if box.shape == (4,):
        box = box.reshape(2, 2)
    if box.shape != (2, 2) or not np.isfinite(box).all() or not (box[:, 1] > box[:, 0]).all():
        raise ValueError("bounds must be ((xmin, xmax), (ymin, ymax)) or (xmin, xmax, ymin, ymax)")
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 2:
        raise ValueError("grid_size must be an integer at least two")
    time_array = np.asarray(times, dtype=np.float64)
    if (time_array.ndim != 1 or not len(time_array) or not np.isfinite(time_array).all()
            or ((time_array < 0) | (time_array > 1)).any()):
        raise ValueError("times must be a nonempty vector in [0, 1]")
    if hasattr(model, "joint_dim") and model.joint_dim != 2:
        raise ValueError("velocity_grid requires a 2D marginal model")
    xx, yy = np.meshgrid(np.linspace(*box[0], grid_size), np.linspace(*box[1], grid_size))
    points = np.stack((xx.ravel(), yy.ravel()), axis=1)
    selected, dtype = _model_device_dtype(model, device)
    state = torch.as_tensor(points, device=selected, dtype=dtype)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            fields = torch.stack([model(state, torch.full((len(state),), float(t),
                                  device=selected, dtype=dtype)) for t in time_array])
            if fields.shape != (len(time_array), len(points), 2):
                raise ValueError("model must return one 2D velocity per grid point")
            if not bool(torch.isfinite(fields).all()):
                raise FloatingPointError("velocity field contains nonfinite values")
            velocities = fields.cpu().numpy()
    finally:
        model.train(was_training)
    return {"points": points, "velocities": velocities, "times": time_array,
            "bounds": box, "grid_size": grid_size}


def round_trip(model: nn.Module, points: np.ndarray, *, steps: int = 200,
               method: str = "heun", batch_size: int = 2048,
               record_trajectory: bool = False, device: str | None = None) -> dict:
    """Reverse endpoints to marginal noise, then report forward reconstruction."""
    original = _points(points, "points")
    reverse = integrate(model, original, steps=steps, start_time=1, end_time=0,
                        method=method, batch_size=batch_size,
                        record_trajectory=record_trajectory, device=device)
    forward = integrate(model, reverse.endpoints, steps=steps, method=method,
                        batch_size=batch_size, device=device)
    errors = np.linalg.norm(forward.endpoints - original, axis=1)
    return {"noise": reverse.endpoints, "reconstructed": forward.endpoints,
            "errors": errors, "mean_error": float(errors.mean()),
            "max_error": float(errors.max()), "rmse": float(np.sqrt(np.mean(errors**2))),
            "reverse_times": reverse.times, "reverse_trajectory": reverse.trajectory}
