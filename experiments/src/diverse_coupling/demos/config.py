"""JSON configuration shared by all experiments and cluster jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DemoConfig:
    experiments: tuple[str, ...] = ("gmm8", "two_disks", "rings")
    seed: int = 0
    training_samples: int = 2048
    marginal_samples: int = 4096
    evaluation_samples: int = 10000
    marginal_epochs: int = 3000
    joint_epochs: int = 5000
    transport_epochs: int = 3000  # Legacy setting; the disk comparison now reuses the marginal model.
    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    learning_rate: float = 1e-3
    chunk_size: int = 2048
    checkpoint_every: int = 250
    ode_steps: int = 200
    sampling_batch_size: int = 2048
    display_samples: int = 48
    display_frames: int = 41
    grid_size: int = 21
    ring_components: int = 2
    reward: str = "component_mismatch"
    use_permutation: bool = True
    ordinary_transport: bool = True
    device: str = "auto"
    dtype: str = "float32"
    cpu_threads: int = 4

    def validate(self) -> None:
        if not self.experiments or len(set(self.experiments)) != len(self.experiments):
            raise ValueError("experiments must be a nonempty list without duplicates")
        if any(name not in ("gmm8", "two_disks", "rings") for name in self.experiments):
            raise ValueError("experiments must use gmm8, two_disks, or rings")
        positive = ("training_samples", "marginal_samples", "evaluation_samples",
                    "marginal_epochs", "joint_epochs", "transport_epochs", "chunk_size",
                    "checkpoint_every", "ode_steps", "sampling_batch_size", "display_samples",
                    "display_frames", "grid_size", "cpu_threads")
        for name in positive:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.display_frames < 2 or self.display_frames > self.ode_steps + 1:
            raise ValueError("display_frames must lie between 2 and ode_steps + 1")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if self.display_samples > min(self.training_samples, self.evaluation_samples):
            raise ValueError("display_samples cannot exceed training/evaluation counts")
        if not self.hidden_sizes or any(isinstance(w, bool) or not isinstance(w, int) or w < 1
                                        for w in self.hidden_sizes):
            raise ValueError("hidden_sizes must be positive integer widths")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.ring_components not in (2, 8):
            raise ValueError("ring_components must be 2 or 8")
        if self.reward not in ("component_mismatch", "squared_distance"):
            raise ValueError("unsupported reward")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.use_permutation, bool) or not isinstance(self.ordinary_transport, bool):
            raise ValueError("solver and transport flags must be booleans")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["experiments"] = list(self.experiments)
        result["hidden_sizes"] = list(self.hidden_sizes)
        return result

    @classmethod
    def load(cls, path: str | Path) -> "DemoConfig":
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        for key in ("experiments", "hidden_sizes"):
            if key in payload:
                payload[key] = tuple(payload[key])
        config = cls(**payload)
        config.validate()
        return config
