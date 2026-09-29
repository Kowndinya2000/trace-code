#!/usr/bin/env bash
# Source after activating a Python 3.8 / Isaac Gym Preview 4 environment.
CLUSTER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TRACE_PYTHON="${TRACE_PYTHON:-$(command -v python)}"
TRACE_ENV_PREFIX="$("$TRACE_PYTHON" -c 'import sys; print(sys.prefix)')"
export PATH="$TRACE_ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$TRACE_ENV_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$CLUSTER_REPO_ROOT:$CLUSTER_REPO_ROOT/isaacgymenvs:${ISAACGYM_PYTHON_PATH:+$ISAACGYM_PYTHON_PATH:}${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUBLAS_WORKSPACE_CONFIG=:4096:8
# Preserve the scheduler's GPU allocation. CUDA and Vulkan mapping must pass
# the camera preflight; TRACE_GRAPHICS_DEVICE_ID explicitly overrides Vulkan.
if [[ -z "${VK_ICD_FILENAMES:-}" && -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi
if [[ -n "${VK_ICD_FILENAMES:-}" ]]; then export VK_DRIVER_FILES="$VK_ICD_FILENAMES"; fi
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${TMPDIR:-/tmp}/trace-torch-extensions-${USER:-worker}}"
export MAX_JOBS=2 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TORCH_EXTENSIONS_DIR"
