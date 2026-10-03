# Local validation — October 3, 2026

All 55 unit and integration tests passed. Tests cover optimal assignment and
LP rewards, valid probability plans, direct weighted losses/gradients, all
three data geometries, shared model architectures, reverse ODE integration,
marginal velocity grids, exact empirical baselines, full generated noise
mapping, checkpoint continuation, and renderer outputs.

Validation environment: Python 3.11.1, NumPy 1.26.4, SciPy 1.12.0, PyTorch
2.2.2, Matplotlib 3.8.3; CPU execution. CUDA is unavailable on this machine.

The three-experiment smoke configuration completed, as did the optional
eight-ring smoke variant. The local preview configuration trained all three
experiments and produced HTML, figures, reports, checkpoints, and generated
samples. These small preview runs are for inspecting the implementation;
the default TACC configuration uses larger datasets and training budgets.

Static plots were visually inspected. The HTML renderer was checked in Chrome:
method selection, slider updates, metrics, and offline JavaScript worked.

The Vista batch script passed Bash syntax checks and local signal-forwarding
and child-exit-status checks. No TACC job was submitted, and CUDA execution
has not been verified on a Vista node. Follow `tacc/README.md` to prepare the
native Vista environment and submit with your allocation.
