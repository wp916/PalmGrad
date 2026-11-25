#!/usr/bin/env bash
# 从任意工作目录启动 PalmGrad；额外参数原样交给训练主程序。
set -euo pipefail
PALMGRAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PALMGRAD_ROOT"
export PYTHONPATH="$PALMGRAD_ROOT/src"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m hand_training.train --horizon 8 --feasibility-report "$PALMGRAD_ROOT/configs/reset_feasibility.json" "$@"
