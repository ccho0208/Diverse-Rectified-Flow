# Empirical optimal coupling experiments

The three complete demonstrations and Vista GPU job instructions are in the
[demo guide](DEMOS.md) and [TACC setup guide](tacc/README.md).

This package implements the training pipeline from document 7,
`7_0928_New_Method.tex`: construct a reward matrix **C**, optimize a coupling
**P**, and train a joint generative model using P's empirical objective directly.
It is a separate Python project in the current workspace; the upstream
rectified-flow repository has not been cloned. See the
[framework design](../outputs/experimental_framework_design.md) for the
mathematical setup.

## Run it

Run the commands below from `experiments/`. Python 3.10 or newer, NumPy, SciPy,
and PyTorch are required. Install the package into your current Python
environment with:

```sh
python3 -m pip install -e .
```

The `diverse-coupling` command and `python3 -m diverse_coupling` then work without
setting `PYTHONPATH`. You can also use the source tree directly when its
dependencies are already installed:

```sh
PYTHONPATH=src python3 examples/smoke_pipeline.py --output runs/smoke --epochs 10
PYTHONPATH=src python3 -m diverse_coupling --help
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The smoke example creates two generic arrays of 24 observations in 2D. It
trains a small joint model and saves generated samples and trajectories. Its
short run checks the pipeline; it does not establish sample quality. Add
`--no-use-permutation` to exercise the transportation LP backend. Use a new
output directory for each run.

With your own `.npy` arrays:

```sh
PYTHONPATH=src python3 -m diverse_coupling train \
  --x X.npy --y Y.npy --output runs/my_run \
  --reward squared_distance --use-permutation \
  --epochs 100 --chunk-size 128 --seed 0

PYTHONPATH=src python3 -m diverse_coupling sample \
  --checkpoint runs/my_run/checkpoint.pt \
  --output runs/my_run/samples.npz \
  --count 256 --steps 100 --record-trajectory --seed 1
```

Training requires an empty output directory, and sampling requires a new output
filename. The sampler supports `--method heun` (default) or `--method euler`.

## Coupling and training semantics

The initial implementation assumes **equal sample counts and uniform empirical
weights**. Arrays have shape `(n, ...)`; trailing coordinate shapes can differ.
Scalar observations of shape `(n,)` are supported. Repeated observation values
remain distinct labeled atoms.

Rewards are maximized: `C[i, j]` measures how desirable the pair `(X[i], Y[j])`
is. The optimizer maximizes `sum(C * P)` with row and column masses `1/n`.
The matrix C is stored densely and requires O(n²) memory.

- `--use-permutation` is the default. It solves a maximum-weight assignment and
  stores the n edges `P[i, permutation[i]] = 1/n` without allocating dense P.
- `--no-use-permutation` solves the unregularized transportation LP on the same
  square problem. It may also return a permutation optimum; it does not force a
  fractional plan or add entropy regularization.
- Unequal counts are rejected in either mode. Any maximizing plan is accepted
  when the optimum is not unique.

Training enumerates the fixed nonzero edges of P and evaluates
`sum_e mass[e] * per_example_loss[e]`. It does not draw new pair identities from
P, recompute assignments per batch, or shuffle the two coordinates independently.
The default `--mode full_plan` makes **one optimizer step per epoch** over the
complete plan. `--chunk-size` limits memory by accumulating globally weighted
gradients before that step; a chunk's weights are never renormalized. Fresh RF
noise and times are still drawn for each target, using random streams that are
independent of chunk boundaries.

`--mode permutation_minibatch --batch-size 128` is an optional mode for
permutation plans. It shuffles fixed paired rows, visits each once per epoch,
and takes one optimizer step per batch. Logged epoch loss is weighted by the
number of rows in each batch, including the partial final batch, and is evaluated
at the successive model states during that epoch.

The rectified-flow model learns the **joint output law** of `(X, Y)`. Each target
is the concatenation of its two observations. Two 2D outputs therefore use a 4D
joint velocity field with a 4D Gaussian source. Training regresses the velocity
along a straight noise-to-joint-target interpolation. Sampling integrates that
joint velocity and restores the original X and Y coordinate shapes. This model
does not use X as the RF source and Y as its target.

## Python API and custom rewards

This complete small example can be run after installation, or with
`PYTHONPATH=src` from this directory:

```python
from pathlib import Path
import numpy as np
import torch

from diverse_coupling import (
    build_reward_matrix, solve_optimal_coupling, build_paired_dataset,
)
from diverse_coupling.models import RectifiedFlowAdapter
from diverse_coupling.train import TrainConfig, train_joint_model
from diverse_coupling.sample import sample_joint_model

torch.set_num_threads(1)  # Useful for this small CPU example.
rng = np.random.default_rng(0)
X = rng.normal(size=(8, 2))
Y = rng.normal(size=(8, 2)) + np.array([1.0, -0.5])
C = build_reward_matrix(X, Y, "squared_distance")
plan = solve_optimal_coupling(C, use_permutation=True)
data = build_paired_dataset(X, Y, plan)

output = Path("runs/api_example")
output.mkdir(parents=True, exist_ok=False)
np.save(output / "X.npy", X)
np.save(output / "Y.npy", Y)
np.save(output / "C.npy", C)
plan.save(output / "plan.npz")
model = RectifiedFlowAdapter(data.joint_dim, [32, 32], seed=0)
result = train_joint_model(
    data, model, TrainConfig(epochs=10, chunk_size=3, seed=0),
    output_dir=output, metadata={"reward": "squared_distance"},
)
samples = sample_joint_model(
    result.checkpoint_path, count=16, steps=20,
    record_trajectory=True, seed=1,
)
samples.save(output / "samples.npz")
print(plan.reward, samples.X.shape, samples.Y.shape, samples.trajectory.shape)
# Output shapes: (16, 2), (16, 2), (21, 16, 4).
```

A custom reward callable receives a block of X and all of Y, preserving their
observation shapes, and must return a finite `(len(x_block), len(Y))` matrix.
For example, with 2D observations:

```python
def weighted_separation(x_block, all_y):
    delta = x_block[:, None, :] - all_y[None, :, :]
    return 2.0 * delta[..., 0] ** 2 + delta[..., 1] ** 2

C = build_reward_matrix(X, Y, weighted_separation, block_size=4)
```

`block_size` limits temporary reward evaluation memory; the completed C is still
dense. For component diversity, supply consistently encoded component labels:

```python
C = build_reward_matrix(
    X, Y, "component_mismatch", x_labels=labels_x, y_labels=labels_y,
)
```

This reward is one for different labels and zero for equal labels. The CLI
equivalent uses `--reward component_mismatch --x-labels x_labels.npy
--y-labels y_labels.npy`. The CLI can also accept `--reward-matrix C.npy`.
Reward construction remains separate from coupling optimization and model
training.

Another model can replace `RectifiedFlowAdapter` by exposing `joint_dim` and
`per_example_loss(targets, noise=..., times=...)`, which returns one loss per
target before weighting. Checkpointed adapters also expose `export_config()`.
Automatic checkpoint loading and sampling currently support the included RF
adapter; another model needs its corresponding loader and sampler.

## Saved artifacts and later visualizations

The CLI training run saves:

| File | Contents |
| --- | --- |
| `X.npy`, `Y.npy` | Original empirical observations. |
| `x_ids.npy`, `y_ids.npy` | Observation IDs; default IDs are original row indices. Supply custom arrays with `--x-ids` and `--y-ids`. |
| `C.npy` | The maximized reward matrix. |
| `plan.npz` | Plan shape, fixed row/column indices, global masses, optional permutation, solver name, reward, and feasibility diagnostics. |
| `config.json` | Reward specification, solver flag, dataset paths/hashes/shapes, model configuration, training configuration, and seed. |
| `history.json` | Epoch losses, optimizer-step counts, and number of paired rows visited. |
| `checkpoint.pt` | Model/optimizer state, coordinate shapes, configuration, metadata, history, and saved random states. |
| `x_labels.npy`, `y_labels.npy` | Supplied component labels, when present. |

`CouplingPlan.load(path)` reloads a plan; `plan.to_sparse()` or `plan.to_dense()`
materializes P explicitly when needed. The API training function itself saves
the checkpoint and history; use the CLI for the complete run manifest above.

Sampling saves an NPZ containing generated `X`, `Y`, their flattened `joint`,
`initial_noise`, integration `times`, and coordinate shapes. With
`--record-trajectory`, `trajectory` has shape `(steps + 1, count, joint_dim)`.
The Python result also exposes `x_trajectory` and `y_trajectory` views with
restored coordinate shapes. A sample keeps the same row index along its whole
trajectory, making pair colors and noise-to-output tracing possible in a later
demo. Generated samples are new draws, so they do not inherit empirical atom IDs.

The core package supplies the reusable solver/training framework. Its `demos`
module adds the eight-component mixture, two-component uniform disks, and
concentric-ring experiments. Solver feasibility and empirical optimal reward
are recorded separately from the neural model's approximation of the joint
law. The short smoke run is not a guarantee that the learned model reproduces
that law or its optimum exactly. General N-particle and nonlinear batch rewards
are outside the current two-marginal linear optimization scope.
