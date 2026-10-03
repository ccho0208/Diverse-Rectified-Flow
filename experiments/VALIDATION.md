# Local validation — October 3, 2026

All 58 unit and integration tests passed after the left-disk source correction. Tests cover optimal assignment and
LP rewards, valid probability plans, direct weighted losses/gradients, all
three data geometries, shared model architectures, reverse ODE integration,
marginal velocity grids, exact empirical baselines, full generated noise
mapping, checkpoint continuation, and renderer outputs.

Validation environment: Python 3.11.1, NumPy 1.26.4, SciPy 1.12.0, PyTorch
2.2.2, Matplotlib 3.8.3; CPU execution. CUDA is unavailable on this machine.

The integration suite runs all three experiments with the corrected disk setup.
A new two-disk CPU preview trained the marginal for 600 steps and the joint
model for 1,000 steps, then rendered the source/target fields and reverse
mapping. Disk sampler tests check uniform area density, independent source
copies, and the measure-preserving antithetic reflection. Continuation tests
verify the disk source RNG and reject legacy Gaussian-source checkpoints.

Before this correction, the three-experiment smoke configuration completed, as did the optional
eight-ring smoke variant. The local preview configuration trained all three
experiments and produced HTML, figures, reports, checkpoints, and generated
samples. These small preview runs are for inspecting the implementation;
the default TACC configuration uses larger datasets and training budgets.

The corrected disk static plots were visually inspected. Before the source
correction, the HTML renderer was checked in Chrome:
method selection, slider updates, metrics, and offline JavaScript worked.

The Vista batch script passed Bash syntax checks and local signal-forwarding
and child-exit-status checks. The corrected implementation was validated locally on CPU; it has not been
submitted to TACC or verified with CUDA by this agent. Follow `tacc/README.md` to prepare the
native Vista environment and submit with your default account or allocation.
