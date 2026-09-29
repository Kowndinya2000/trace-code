#!/usr/bin/env bash
set -euo pipefail
EVAL_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$EVAL_PACKAGE_ROOT/../cluster/runtime_env.sh"
EVAL_SPEC="${1:?usage: run_scene_qualification.sh SPEC [CUDA_GPU]}"
EVAL_GPU="${2:-0}"
EVAL_GRAPHICS="${TRACE_GRAPHICS_DEVICE_ID:-$EVAL_GPU}"
[[ "$EVAL_GPU" =~ ^[0-9]+$ && "$EVAL_GRAPHICS" =~ ^[0-9]+$ ]] || exit 2
cd "$EVAL_PACKAGE_ROOT"
exec "$TRACE_PYTHON" -u tools/qualify_retrieval_scenes.py \
  task=MoreEvaluation train=MoreOpenLoopSetSCPPO test=True headless=True \
  force_render=False wandb_activate=False sim_device="cuda:$EVAL_GPU" rl_device="cuda:$EVAL_GPU" \
  graphics_device_id="$EVAL_GRAPHICS" \
  +evaluation_spec="$EVAL_SPEC"
