# Run the demos on TACC Vista

The supplied batch script targets Vista's `gh-dev` queue with one node, one
Python process, and a 1 hour 50 minute limit. Grace-Hopper nodes have ARM CPUs
and one GPU; `gh-dev` currently allows two-hour jobs. Access requires a TACC
account and an eligible allocation. The script deliberately omits the
unsupported `--gres` and `--gpus-per-task` options. Check current queue limits
with `qlimits`. [TACC Vista guide](https://docs.tacc.utexas.edu/hpc/vista/)

## Prepare an environment on Vista

Copy this project to Vista, keeping the `experiments/` directory intact. Use
an environment created on Vista, rather than copying a Mac or x86 environment.
TACC documents system PyTorch after loading these modules; retaining that
installation is the simplest starting point.
[TACC PyTorch setup](https://docs.tacc.utexas.edu/hpc/vista/#running-pytorch-single-node)

From `experiments/`, prepare the environment once:

```bash
module load gcc cuda python3
python3 -m venv --system-site-packages "$SCRATCH/diverse-demo-env"
source "$SCRATCH/diverse-demo-env/bin/activate"
python3 -c 'import torch; print(torch.__version__)'
python3 -c 'import numpy, scipy, matplotlib'
python3 -m pip install -e '.[demos]' --no-deps
```

If the NumPy/SciPy/Matplotlib import reports a missing package, install the
missing dependency into this same environment before submitting. For example:

```bash
python3 -m pip install 'numpy>=1.24' 'scipy>=1.10' 'matplotlib>=3.7'
```

The editable install uses `--no-deps` to preserve the existing GPU-compatible
PyTorch. If PyTorch itself is unavailable, follow the current TACC instructions
for an ARM-compatible CUDA build. GPU detection occurs on the allocated
compute node when the job starts; a login-node check need not report a GPU.

## Submit

From `experiments/`, set your environment and submit with your allocation:

```bash
export DIVERSE_ENV="$SCRATCH/diverse-demo-env"
sbatch -A YOUR_ALLOCATION tacc/vista_gh_dev.slurm
```

The allocation is a placeholder to replace, not a value stored in the script.
The default configuration is `configs/demo_default.json`. Results go to
`$SCRATCH/diverse-coupling/<job-id>/`; console logs appear in the submission
directory as `diverse-demo-<job-id>.out` and `.err`.

The GPU preflight fails with an actionable message if CUDA is unavailable.
The trainer runs with `--device cuda`; it does not silently switch to CPU.

You can select a different configuration and a persistent run directory:

```bash
export DIVERSE_CONFIG="configs/demo_default.json"
export DIVERSE_RUN_DIR="$SCRATCH/diverse-coupling/my-comparison"
sbatch -A YOUR_ALLOCATION tacc/vista_gh_dev.slurm
```

## Resume an interrupted run

Reuse both the run directory and its configuration:

```bash
export DIVERSE_RUN_DIR="$SCRATCH/diverse-coupling/my-comparison"
export DIVERSE_RESUME=1
sbatch -A YOUR_ALLOCATION tacc/vista_gh_dev.slurm
```

The default configuration saves periodic training checkpoints. The batch
script also requests an early `USR1` signal and forwards it to Python, allowing
the runner to checkpoint and stop between training steps before wall time.
Slurm's `B:` signal prefix targets the batch shell, which is why the script
explicitly forwards it. [Slurm signal documentation](https://slurm.schedmd.com/sbatch.html#OPT_signal)

## Environment options

| Variable | Default | Purpose |
|---|---|---|
| `DIVERSE_PROJECT_DIR` | Submission directory | Location of `experiments/`; a parent containing `experiments/` is also recognized. |
| `DIVERSE_CONFIG` | `configs/demo_default.json` | JSON configuration, relative to `experiments/` or absolute. |
| `DIVERSE_RUN_DIR` | `$SCRATCH/diverse-coupling/<job-id>` | Output directory; set this when resuming. |
| `DIVERSE_ENV` | Unset | Optional virtual environment to activate after loading modules. |
| `DIVERSE_SKIP_MODULES` | `0` | Set to `1` for an already prepared environment. |
| `DIVERSE_RESUME` | `0` | Set to `1` to continue an existing run. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS` | `4` | CPU thread limits for preprocessing and plotting. |

The script creates a private temporary Matplotlib cache and removes that
cache on exit. Training data, checkpoints, figures, and reports remain in the
run directory. See [the demo guide](../DEMOS.md) for experiment configuration
and the resulting visualizations.
