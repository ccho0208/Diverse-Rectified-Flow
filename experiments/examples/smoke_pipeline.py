"""Exercise coupling, direct weighted RF training, and trajectory sampling.

Run from experiments/: PYTHONPATH=src python3 examples/smoke_pipeline.py
This generic small run checks the pipeline rather than sample quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diverse_coupling.cli import main as pipeline_main


def positive_integer(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("epochs must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--epochs", type=positive_integer, default=10)
    parser.add_argument("--no-use-permutation", dest="use_permutation",
                        action="store_false", default=True)
    args = parser.parse_args()
    output = args.output
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("output must be an empty directory; choose a new run directory")

    # Many CPU threads cost more than they save for this small velocity MLP.
    torch.set_num_threads(1)
    rng = np.random.default_rng(37)
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    X = rng.normal(size=(24, 2))
    Y = rng.normal(size=(24, 2)) * np.array([0.7, 1.3]) + np.array([1.0, -0.5])
    np.save(inputs / "X.npy", X, allow_pickle=False)
    np.save(inputs / "Y.npy", Y, allow_pickle=False)
    np.save(inputs / "x_ids.npy", np.arange(1001, 1025), allow_pickle=False)
    np.save(inputs / "y_ids.npy", np.arange(2001, 2025), allow_pickle=False)

    run = output / "model"
    pipeline_main([
        "train", "--x", str(inputs / "X.npy"), "--y", str(inputs / "Y.npy"),
        "--x-ids", str(inputs / "x_ids.npy"), "--y-ids", str(inputs / "y_ids.npy"),
        "--output", str(run), "--reward", "squared_distance",
        "--use-permutation" if args.use_permutation else "--no-use-permutation",
        "--epochs", str(args.epochs), "--hidden-sizes", "32", "32",
        "--mode", "full_plan", "--chunk-size", "7", "--seed", "37",
    ])
    pipeline_main([
        "sample", "--checkpoint", str(run / "checkpoint.pt"),
        "--output", str(output / "samples.npz"), "--count", "16", "--steps", "20",
        "--record-trajectory", "--seed", "38",
    ])
    print(f"Smoke run saved to {output.resolve()}")


if __name__ == "__main__":
    main()
