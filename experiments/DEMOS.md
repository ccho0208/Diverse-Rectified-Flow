# Three coupling demonstrations

The demo runner adds the agreed experiments to the shared empirical coupling
framework. Each flow uses the same unconstrained time-conditioned MLP with
three 64-wide hidden layers and SiLU activations. Marginals and conventional
transport have 2D states; the learned pair distribution has a 4D state. Ring
symmetry is learned, not imposed by the architecture.

## Run locally

From `experiments/`, use a Python environment with NumPy, SciPy, PyTorch, and
Matplotlib. Installation with the extra is `python3 -m pip install -e '.[demos]'`.
An existing environment can use the source directly:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_smoke.json --output runs/demo-smoke
```

The three-epoch smoke configuration verifies execution and creates all output
artifacts; it does not measure a trained model's performance. A small CPU
training preview uses:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_local_preview.json --output runs/demo-preview
```

The main configuration uses 2,048 empirical pairs, 4,096 fixed marginal
training targets, and 10,000 generated evaluation pairs. It trains each
marginal for 3,000 full-plan optimizer steps and each joint model for 5,000.
The conventional disk picture reuses the frozen marginal flow. These are starting
settings; inspect marginal fidelity, support leakage, and reverse-integration
error before interpreting diversity scores.

For a GPU run:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_default.json --output runs/comparison --device cuda
```

CUDA requests fail clearly if the selected environment has no available GPU.
The batch launcher in [the TACC guide](tacc/README.md) targets Vista `gh-dev`.
GPU access comes from the allocated node and its environment, rather than from
the presence of GPU-aware code.

## Experiments

| Name | Distribution |
| --- | --- |
| `gmm8` | Eight equal Gaussian components at radius 3, isotropic standard deviation 0.15. Both marginals coincide. |
| `two_disks` | Source: uniform filled disks at (-2,1), (-2,-1). Target: disks at (2,1), (2,-1). Both coupled endpoints X and Y have the right-disk target distribution. Radius 0.3 and equal component weights throughout. |
| `rings` | Two equal annuli with radius uniform in 1±0.08 or 3±0.08, and uniform angle. Both marginals coincide. |

The disk sampler uses radius `0.3 * sqrt(U)` to obtain uniform area density.
For rings, the explicitly specified measure is uniform in radius and angle;
it is not uniform in annular area. Generated endpoints are classified by
nearest Gaussian/disk center or ring radius. Compact-support violations are
reported separately, so a component label cannot conceal points outside the
intended disks or annuli.

The ring configuration can use eight annuli at radii 1,...,8:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_default.json --experiments rings --ring-components 8 \
  --output runs/eight-rings --device cuda
```

Eight components still means a pair coupling. An eight-particle joint coupling
requires a separate multi-marginal optimization and model extension.

## Couplings and measurements

First train and freeze a shared marginal flow F. For GMM and rings, it maps a
standard Gaussian to the target, and the stored endpoint pools are
`X_i=F(z_i)`, `Y_i=F(-z_i)`.

For the disk experiment, F maps the left-disk mixture to the right-disk mixture.
Its training source/target observations are independently paired and fixed.
Draw `z_i` from the left disks and use the measure-preserving reflection
`A(z)=(-4-z[0], -z[1])` around (-2,0). Form `X_i=F(z_i)` and `Y_i=F(A(z_i))`.
Both endpoint pools are on the right; the optimization couples two target
outputs, not a left observation with a right observation. All empirical
baselines use these exact pools:

- **Antithetic:** original identity pairing, with masses 1/n.
- **Independent:** empirical product measure, with exact expected reward and
  component table calculated without sampling pair identities.
- **Empirical optimum:** globally maximize C using the assignment solver, then
  train the 4D joint RF on the fixed paired targets with their plan masses.
- **Learned:** new samples from the trained 4D model, evaluated against
  independent target-distribution references. Its source is the product of
  two independent left-disk mixtures for disks, or a standard 4D Gaussian for
  GMM and rings. Each training step draws fresh independent source samples
  while visiting every fixed P-weighted endpoint pair; it never resamples P.

The default reward is component mismatch. Change JSON `reward` to
`squared_distance` for a separate objective. JSON `use_permutation=false`
selects the transportation LP backend; this can be much larger than assignment
and remains restricted to equal counts and uniform empirical weights.

The two-disk experiment's marginal is itself an ordinary 2D left-to-right RF.
`ordinary_transport.png` compares its independently paired training observations'
straight interpolations with ODE trajectories starting from the same source
points using that same frozen model. This RF can change its induced endpoint
coupling. Its source-to-target diagnostics are stored separately under
`report.json`'s `ordinary_transport`, since they describe a different coupling
from the target-to-target diversity experiment. `ordinary_transport=false`
disables that additional picture and diagnostics; it does not change F.
`transport_epochs` remains accepted for compatibility with old configurations,
but marginal training is governed by `marginal_epochs`.

Reports include component mismatch, squared separation, component-pair
probabilities, marginal component frequencies, compact-support leakage, and
marginal sliced Wasserstein discrepancy to independently sampled true targets.
The latter uses 64 deterministic random projection directions with at most
2,048 points per marginal. This is a finite-sample diagnostic, not an exact
multidimensional Wasserstein distance. The learned reward difference from the
empirical optimum is accompanied by fidelity measurements; approximate
generated marginals mean that difference is not an optimality certificate.

## Reverse mapping and velocity displays

Every generated pair is mapped backward through the frozen 2D marginals.
For disks, both inverse coordinates are in the left-disk source space; the
display explicitly identifies the left source and right target. For GMM and
rings, inverse coordinates are in Gaussian source space. These paired marginal
source coordinates differ from the independent 4D source used to generate the
joint sample. A forward reconstruction
measures the inverse mapping's numerical error.

For a fixed display subset, reverse paths are saved and reordered into forward
marginal time. Pair colors remain attached to the same row throughout. The
slider overlays these paths on the actual 2D marginal velocity fields evaluated
at each time; it does not present a projection of a 4D field as a 2D field.

## Artifacts

Each experiment directory contains:

| File/directory | Contents |
| --- | --- |
| `datasets.npz` | Marginal training targets/labels, independent references, stored source noise. |
| `marginal_x/`, `marginal_y/` | Frozen 2D flow checkpoints and loss histories; identical marginals share X's model. |
| `pool.npz`, `C.npy`, `plan.npz` | Fixed empirical endpoints, IDs, source positions, reward matrix, and probability plan. |
| `joint/` | 4D model checkpoint and weighted training history. |
| `marginal_x/` (disks) | The shared left-to-right RF, also used for the conventional picture. No separate Gaussian-to-disk or ordinary-transport model is trained. |
| `samples.npz` | Generated paired outputs, pair IDs, original 4D noise, and reverse-mapped 2D marginal noise for all pairs. |
| `reverse_mapping.npz` | Full generated marginal noise pairs and the colored display subset's reverse paths. |
| `display.npz` | Marginal fields, sampled paths, pairing matrices, and display points. |
| `report.json` | Coupling rewards, marginal fidelity, solver checks, and reconstruction errors. |
| `overview.png`, `diagnostics.png` | Static scientific figures. |
| `ordinary_transport.png` | Additional two-panel transport comparison for the disks. |
| `demo.html` | Offline interactive display with method selector, time slider, and metrics. |

The run root contains configuration, environment/version metadata, and a
summary linking the three experiments. Copy the HTML and figures back from
TACC for inspection; HTML needs no server or external network dependencies.

Rerender existing arrays without training:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos render --input runs/comparison/rings
```

## Checkpoints and continuation

Checkpoints are atomic and saved periodically. The Vista script forwards an
early wall-time signal; the trainer finishes its current weighted step, saves,
and exits with status 75. Continue using the same configuration/output:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_default.json --output runs/comparison --device cuda --resume
```

Same-device continued training matches uninterrupted training. Switching CPU
and CUDA can change backend-specific random draws. Cached endpoint pools are
preserved when the frozen marginal weights are unchanged. Epoch limits may be
extended; if marginal weights change, the old joint model is preserved under
`previous_joint_models/` before training a new joint model on the new endpoints.
Other data/optimization settings require a new run directory.

Disk runs created before the left-disk source correction cannot be resumed.
The runner checks `setup.json` before modifying an existing run and rejects
the old Gaussian-source setup. Preserve those results and use a fresh directory:

```bash
PYTHONPATH=src python3 -m diverse_coupling.demos run \
  --config configs/demo_two_disks.json --output runs/two-disks-left-source --device cuda
```

## Verification

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
bash -n tacc/vista_gh_dev.slurm
```

Tests cover the data geometry, global plan weighting, exact empirical
baselines, shared model architecture, forward/reverse integration, cached
targets, continued training, field values, display files, and offline rendering.
