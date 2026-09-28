#!/bin/bash
#SBATCH --job-name=refiner-perf
#SBATCH --partition=gpu-short            # set for the cluster it ran on
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-%j.out

# Environment: a Python 3.11 venv with torch and scipy, built once on the login node:
#   python3.11 -m venv ~/venvs/refiner
#   source ~/venvs/refiner/bin/activate
#   pip install --upgrade pip && pip install torch scipy
VENV="$HOME/venvs/refiner"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No venv at $VENV. Build it on the login node first:"
  echo "  python3.11 -m venv ~/venvs/refiner && source ~/venvs/refiner/bin/activate && pip install --upgrade pip && pip install torch scipy"
  exit 1
fi
source "$VENV/bin/activate"

set -ex
cd "$SLURM_SUBMIT_DIR"
# stop early if python or torch is too old
python - <<'GUARD'
import sys, torch
assert sys.version_info >= (3, 9), f"python {sys.version} is too old"
assert int(torch.__version__.split(".")[0]) >= 2, f"torch {torch.__version__} is too old"
import triton  # noqa: F401
GUARD
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    import triton; print("triton", triton.__version__)
PY

# 1. correctness tests, including the GPU test
python kernels/test_correctness.py

# 2. benchmark sweeps, fp32 and bf16
python kernels/bench.py --out kernels/results/sweep_fp32.csv
python kernels/bench.py --out kernels/results/sweep_bf16.csv --dtype bfloat16

# 3. profile of the original and rewritten layer
python kernels/profile_capture.py | tee kernels/results/profile.txt

# 4. quantization ablations for the k4, k8 and k32 checkpoints (k16 was run separately)
for k in 4 8 32; do
  python quantization/ablate.py --ckpt models/spectral_refiner_k${k}_eps_0_03.pt
done
