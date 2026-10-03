#!/bin/bash
#SBATCH --job-name=diverse-demo
#SBATCH --partition=gh-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:50:00
#SBATCH --output=diverse-demo-%j.out
#SBATCH --error=diverse-demo-%j.err
#SBATCH --signal=B:USR1@120

# Submit from experiments/: sbatch -A YOUR_ALLOCATION tacc/vista_gh_dev.slurm
# Vista allocates the node's GPU; do not add --gres or --gpus-per-task.
set -euo pipefail

if [[ "${DIVERSE_SKIP_MODULES:-0}" != "1" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo "TACC modules are unavailable. Set DIVERSE_SKIP_MODULES=1 only for a prepared environment." >&2
        exit 1
    fi
    module load gcc cuda python3
fi

if [[ -n "${DIVERSE_ENV:-}" ]]; then
    if [[ ! -f "${DIVERSE_ENV}/bin/activate" ]]; then
        echo "DIVERSE_ENV must name a virtual environment with bin/activate." >&2
        exit 1
    fi
    source "${DIVERSE_ENV}/bin/activate"
fi

demo_project_dir="${DIVERSE_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "${demo_project_dir}/src/diverse_coupling" && -d "${demo_project_dir}/experiments/src/diverse_coupling" ]]; then
    demo_project_dir="${demo_project_dir}/experiments"
fi
if [[ ! -d "${demo_project_dir}/src/diverse_coupling" ]]; then
    echo "Set DIVERSE_PROJECT_DIR to the experiments directory." >&2
    exit 1
fi
cd "$demo_project_dir"
demo_config="${DIVERSE_CONFIG:-configs/demo_default.json}"
if [[ ! -f "$demo_config" ]]; then
    echo "Experiment configuration does not exist: $demo_config" >&2
    exit 1
fi

if [[ -n "${DIVERSE_RUN_DIR:-}" ]]; then
    demo_run_dir="$DIVERSE_RUN_DIR"
else
    : "${SCRATCH:?Set SCRATCH or provide DIVERSE_RUN_DIR.}"
    : "${SLURM_JOB_ID:?Run with sbatch or provide DIVERSE_RUN_DIR.}"
    demo_run_dir="${SCRATCH}/diverse-coupling/${SLURM_JOB_ID}"
fi
mkdir -p "$demo_run_dir"

export PYTHONPATH="${demo_project_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MPLBACKEND=Agg
demo_mpl_cache=$(mktemp -d "${TMPDIR:-/tmp}/diverse-mpl.XXXXXX")
export MPLCONFIGDIR="$demo_mpl_cache"
trap 'rm -rf -- "$demo_mpl_cache"' EXIT

python3 - <<'PY'
import platform
import sys

try:
    import matplotlib
    import numpy
    import scipy
    import torch
except ImportError as exc:
    raise SystemExit(f"Missing demo dependency: {exc}. Prepare the environment before submitting.") from exc

print(f"Python {sys.version.split()[0]}, machine {platform.machine()}, PyTorch {torch.__version__}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Load Vista's GPU-compatible PyTorch environment; this job requires a GPU.")
torch.empty(1, device="cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
PY

demo_arguments=(run --config "$demo_config" --output "$demo_run_dir" --device cuda)
if [[ "${DIVERSE_RESUME:-0}" == "1" ]]; then
    demo_arguments+=(--resume)
fi

echo "Configuration: $demo_config"
echo "Results: $demo_run_dir"

# Slurm signals the batch shell. Forward the early signal to the trainer so
# it can save a checkpoint and stop between steps. Periodic checkpoints also
# support resuming after an unexpected interruption.
demo_python_pid=""
demo_forward_stop() {
    if [[ -n "$demo_python_pid" ]] && kill -0 "$demo_python_pid" 2>/dev/null; then
        kill -USR1 "$demo_python_pid" 2>/dev/null || true
    fi
}
trap demo_forward_stop USR1 TERM

python3 -u -m diverse_coupling.demos "${demo_arguments[@]}" &
demo_python_pid=$!

# A caught signal interrupts wait even while Python remains alive. Continue
# waiting, then retrieve the child's actual status instead of the trap status.
set +e
while true; do
    wait "$demo_python_pid"
    demo_status=$?
    if ! kill -0 "$demo_python_pid" 2>/dev/null; then
        wait "$demo_python_pid"
        demo_status=$?
        break
    fi
done
exit "$demo_status"
