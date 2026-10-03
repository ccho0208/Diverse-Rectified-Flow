"""Direct empirical training over the fixed nonzero entries of P."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import Tensor, nn

from .data import PairedDataset


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    dtype: str = "float32"
    mode: str = "full_plan"
    chunk_size: int | None = None
    batch_size: int = 128

    def validate(self) -> None:
        if not isinstance(self.epochs, int) or self.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.mode not in ("full_plan", "permutation_minibatch"):
            raise ValueError("mode must be full_plan or permutation_minibatch")
        for name, value in (("chunk_size", self.chunk_size),
                            ("batch_size", self.batch_size)):
            if value is not None and (not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size is None:
            raise ValueError("batch_size must be a positive integer")
        if torch.device(self.device).type == "mps" and self.dtype == "float64":
            raise ValueError("MPS requires float32; use CPU for float64")


@dataclass
class TrainingResult:
    model_adapter: nn.Module
    history: list[dict]
    checkpoint_path: Path | None


def weighted_objective(per_example_losses: Tensor, masses: Tensor) -> Tensor:
    """Sum globally weighted losses, without renormalizing a batch or chunk."""
    if per_example_losses.ndim != 1 or masses.shape != per_example_losses.shape:
        raise ValueError("losses and masses must be vectors of equal length")
    if not torch.isfinite(per_example_losses).all():
        raise FloatingPointError("model produced nonfinite per-example losses")
    if not torch.isfinite(masses).all() or torch.any(masses < 0):
        raise ValueError("masses must be finite and nonnegative")
    return torch.sum(per_example_losses * masses.to(per_example_losses))


def _losses(model_adapter: nn.Module, targets: Tensor, noise: Tensor,
            times: Tensor) -> Tensor:
    losses = model_adapter.per_example_loss(targets, noise=noise, times=times)
    if losses.shape != (len(targets),):
        raise ValueError("model adapter must return one loss per paired target")
    return losses


def _draws(size: int, dim: int, noise_rng: np.random.Generator,
           time_rng: np.random.Generator, *, device, dtype) -> tuple[Tensor, Tensor]:
    # Separate streams keep the per-edge draws independent of chunk boundaries.
    noise = torch.as_tensor(noise_rng.standard_normal((size, dim)),
                            device=device, dtype=dtype)
    times = torch.as_tensor(time_rng.random(size), device=device, dtype=dtype)
    return noise, times


def accumulate_plan_gradients(
    dataset: PairedDataset, model_adapter: nn.Module, *, chunk_size: int | None = None,
    device: str = "cpu", dtype: torch.dtype = torch.float32,
    noise: Tensor | None = None, times: Tensor | None = None,
    noise_rng: np.random.Generator | None = None,
    time_rng: np.random.Generator | None = None,
) -> float:
    """Backpropagate the complete weighted plan; caller clears/steps gradients.

    Explicit draws make full-vs-chunk comparisons use the same RF realization.
    Otherwise the caller supplies two independent RNG streams. No optimizer step
    is taken here, and there is no random selection of pair identities.
    """
    size = len(dataset)
    if chunk_size is None:
        chunk_size = size
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if (noise is None) != (times is None):
        raise ValueError("provide both noise and times, or neither")
    if noise is not None:
        if noise.shape != (size, dataset.joint_dim) or times.shape != (size,):
            raise ValueError("explicit draws must cover every nonzero pair")
    elif noise_rng is None or time_rng is None:
        raise ValueError("provide noise_rng and time_rng when draws are omitted")
    total = 0.0
    for indices in dataset.epoch_batches(chunk_size):
        batch = dataset.get_batch(indices, device=device, dtype=dtype)
        if noise is None:
            eps, t = _draws(len(indices), dataset.joint_dim, noise_rng, time_rng,
                            device=device, dtype=dtype)
        else:
            select = torch.as_tensor(indices, device=noise.device)
            eps = noise.index_select(0, select).to(device=device, dtype=dtype)
            t = times.index_select(0, select.to(times.device)).to(device=device, dtype=dtype)
        contribution = weighted_objective(
            _losses(model_adapter, batch.targets, eps, t), batch.masses)
        contribution.backward()
        total += float(contribution.detach().cpu())
    return total


def _check_gradients(model_adapter: nn.Module) -> None:
    for parameter in model_adapter.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("training produced nonfinite gradients")


def train_joint_model(
    paired_dataset: PairedDataset, model_adapter: nn.Module, config: TrainConfig,
    *, output_dir: str | Path | None = None, metadata: dict | None = None,
    progress: Callable[[dict], None] | None = None,
) -> TrainingResult:
    """Train a replaceable per-example-loss adapter on the fixed coupling.

    One full_plan epoch is one optimizer step over every weighted pair. In
    permutation_minibatch mode an epoch visits every fixed pair once, with a
    separate optimizer step per batch. Its logged loss is the row-count-weighted
    online loss, evaluated at successive model states.
    """
    config.validate()
    paired_dataset.plan.validate()
    if config.mode == "permutation_minibatch" and paired_dataset.plan.permutation is None:
        raise ValueError("permutation_minibatch requires a permutation plan")
    if getattr(model_adapter, "joint_dim", None) != paired_dataset.joint_dim:
        raise ValueError("model joint_dim must match concatenated observation dimensions")
    if not hasattr(model_adapter, "per_example_loss"):
        raise TypeError("model adapter must implement per_example_loss")
    if output_dir is not None and not hasattr(model_adapter, "export_config"):
        raise TypeError("checkpointed adapters must implement export_config")
    metadata = {} if metadata is None else json.loads(json.dumps(metadata, allow_nan=False))
    dtype = getattr(torch, config.dtype)
    device = torch.device(config.device)
    model_adapter.to(device=device, dtype=dtype).train()
    torch.manual_seed(config.seed)
    streams = np.random.SeedSequence(config.seed).spawn(3)
    noise_rng, time_rng, shuffle_rng = (np.random.default_rng(s) for s in streams)
    optimizer = torch.optim.Adam(model_adapter.parameters(), lr=config.learning_rate)
    history: list[dict] = []
    optimizer_steps = 0

    for epoch in range(1, config.epochs + 1):
        if config.mode == "full_plan":
            optimizer.zero_grad(set_to_none=True)
            loss = accumulate_plan_gradients(
                paired_dataset, model_adapter, chunk_size=config.chunk_size,
                device=device, dtype=dtype, noise_rng=noise_rng, time_rng=time_rng)
            _check_gradients(model_adapter)
            optimizer.step()
            optimizer_steps += 1
        else:
            loss = 0.0
            for indices in paired_dataset.epoch_batches(
                config.batch_size, shuffle=True, rng=shuffle_rng):
                batch = paired_dataset.get_batch(indices, device=device, dtype=dtype)
                eps, t = _draws(len(indices), paired_dataset.joint_dim,
                                noise_rng, time_rng, device=device, dtype=dtype)
                optimizer.zero_grad(set_to_none=True)
                per_example = _losses(model_adapter, batch.targets, eps, t)
                # The only supported minibatch plan has equal masses 1/n.
                objective = weighted_objective(per_example, torch.full_like(
                    per_example, 1 / len(indices)))
                objective.backward()
                _check_gradients(model_adapter)
                optimizer.step()
                optimizer_steps += 1
                loss += float(objective.detach().cpu()) * len(indices) / len(paired_dataset)
        record = {"epoch": epoch, "loss": loss, "optimizer_steps": optimizer_steps,
                  "paired_rows_seen": len(paired_dataset)}
        history.append(record)
        if progress is not None:
            progress(record.copy())

    checkpoint_path = None
    if output_dir is not None:
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = folder / "checkpoint.pt"
        checkpoint = {
            "format_version": 1, "model_config": model_adapter.export_config(),
            "model_state": model_adapter.state_dict(),
            "optimizer_state": optimizer.state_dict(), "train_config": asdict(config),
            "x_shape": list(paired_dataset.x_shape), "y_shape": list(paired_dataset.y_shape),
            "history": history, "metadata": metadata,
            "rng_state": {"noise": noise_rng.bit_generator.state,
                          "time": time_rng.bit_generator.state,
                          "shuffle": shuffle_rng.bit_generator.state,
                          "torch": torch.get_rng_state()},
        }
        torch.save(checkpoint, checkpoint_path)
        (folder / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return TrainingResult(model_adapter, history, checkpoint_path)
