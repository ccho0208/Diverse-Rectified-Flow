"""Array-based entry points without dataset-specific experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from .costs import build_reward_matrix
from .coupling import solve_optimal_coupling
from .data import build_paired_dataset
from .models import RectifiedFlowAdapter
from .sample import sample_joint_model
from .train import TrainConfig, train_joint_model


def _load(path: str) -> np.ndarray:
    result = np.load(path, allow_pickle=False)
    if not isinstance(result, np.ndarray):
        raise ValueError(f"expected a .npy array: {path}")
    return result


def _identity(array: np.ndarray, path: str) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {"source_path": str(Path(path).resolve()), "shape": list(array.shape),
            "dtype": str(array.dtype), "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest()}


def _train(args) -> None:
    folder = Path(args.output)
    if folder.exists() and any(folder.iterdir()):
        raise ValueError("output directory must be empty; choose a new run directory")
    X, Y = _load(args.x), _load(args.y)
    if X.ndim < 1 or Y.ndim < 1 or len(X) != len(Y):
        raise ValueError("initial implementation requires n == m")
    reward_config = {"name": args.reward, "matrix_path": args.reward_matrix,
                     "x_labels_path": args.x_labels, "y_labels_path": args.y_labels}
    if args.reward_matrix is not None:
        if args.x_labels or args.y_labels:
            raise ValueError("reward-matrix cannot be combined with label rewards")
        C = _load(args.reward_matrix)
        reward_config["name"] = "provided_matrix"
    else:
        C = build_reward_matrix(
            X, Y, args.reward,
            x_labels=None if args.x_labels is None else _load(args.x_labels),
            y_labels=None if args.y_labels is None else _load(args.y_labels),
            block_size=args.reward_block_size)
    if C.shape != (len(X), len(Y)):
        raise ValueError("reward matrix shape must match empirical sample counts")
    plan = solve_optimal_coupling(C, use_permutation=args.use_permutation)
    data = build_paired_dataset(
        X, Y, plan,
        x_ids=None if args.x_ids is None else _load(args.x_ids),
        y_ids=None if args.y_ids is None else _load(args.y_ids))
    config = TrainConfig(epochs=args.epochs, learning_rate=args.learning_rate,
                         seed=args.seed, device=args.device, dtype=args.dtype,
                         mode=args.mode, chunk_size=args.chunk_size, batch_size=args.batch_size)
    config.validate()
    if config.mode == "permutation_minibatch" and plan.permutation is None:
        raise ValueError("permutation_minibatch requires --use-permutation")
    model = RectifiedFlowAdapter(data.joint_dim, args.hidden_sizes, seed=args.seed)
    manifest = {
        "format_version": 1, "use_permutation": args.use_permutation,
        "datasets": {"X": _identity(X, args.x), "Y": _identity(Y, args.y)},
        "reward": reward_config, "reward_value": plan.reward,
        "coupling": {"solver": plan.solver, "diagnostics": plan.diagnostics},
        "training": asdict(config), "model": model.export_config(),
        "x_shape": list(data.x_shape), "y_shape": list(data.y_shape),
    }
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "X.npy", X, allow_pickle=False)
    np.save(folder / "Y.npy", Y, allow_pickle=False)
    np.save(folder / "x_ids.npy", data.x_ids, allow_pickle=False)
    np.save(folder / "y_ids.npy", data.y_ids, allow_pickle=False)
    for name, path in (("x_labels", args.x_labels), ("y_labels", args.y_labels)):
        if path is not None:
            np.save(folder / f"{name}.npy", _load(path), allow_pickle=False)
    np.save(folder / "C.npy", C, allow_pickle=False)
    plan.save(folder / "plan.npz")
    (folder / "config.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")

    def progress(record):
        if record["epoch"] == 1 or record["epoch"] % args.log_every == 0 or record["epoch"] == config.epochs:
            print(json.dumps(record), flush=True)

    result = train_joint_model(data, model, config, output_dir=folder,
                               metadata=manifest, progress=progress)
    print(json.dumps({"checkpoint": str(result.checkpoint_path), "reward": plan.reward,
                      "solver": plan.solver, "paired_rows": len(data)}))


def _sample(args) -> None:
    destination = Path(args.output)
    if destination.exists():
        raise ValueError("sample output already exists; choose a new filename")
    result = sample_joint_model(
        args.checkpoint, args.count, args.record_trajectory, steps=args.steps,
        method=args.method, seed=args.seed, device=args.device)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.save(destination)
    print(json.dumps({"samples": str(destination), "X_shape": list(result.X.shape),
                      "Y_shape": list(result.Y.shape), "trajectory": args.record_trajectory}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="construct C, solve P, and train a joint RF")
    train.add_argument("--x", required=True, help="empirical X .npy array")
    train.add_argument("--y", required=True, help="empirical Y .npy array")
    train.add_argument("--output", required=True, help="empty run directory")
    train.add_argument("--reward", choices=("squared_distance", "component_mismatch"),
                       default="squared_distance")
    train.add_argument("--reward-matrix", help="optional precomputed C .npy array")
    train.add_argument("--x-labels")
    train.add_argument("--y-labels")
    train.add_argument("--x-ids")
    train.add_argument("--y-ids")
    train.add_argument("--reward-block-size", type=int)
    train.add_argument("--use-permutation", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    train.add_argument("--hidden-sizes", nargs="+", type=int, default=[128, 128, 128])
    train.add_argument("--mode", choices=("full_plan", "permutation_minibatch"), default="full_plan")
    train.add_argument("--chunk-size", type=int)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--log-every", type=int, default=10)
    train.set_defaults(handler=_train)
    sample = commands.add_parser("sample", help="sample paired outputs from a checkpoint")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--output", required=True, help="new output .npz filename")
    sample.add_argument("--count", type=int, default=256)
    sample.add_argument("--steps", type=int, default=100)
    sample.add_argument("--method", choices=("euler", "heun"), default="heun")
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--device", default="cpu")
    sample.add_argument("--record-trajectory", action="store_true")
    sample.set_defaults(handler=_sample)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if getattr(args, "log_every", 1) < 1:
            raise ValueError("log-every must be positive")
        args.handler(args)
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
