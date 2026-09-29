#!/usr/bin/env bash
set -euo pipefail
EVAL_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$EVAL_PACKAGE_ROOT/../cluster/runtime_env.sh"
EVAL_SPEC="${1:?usage: run_retrieval_evaluation.sh SPEC [CUDA_GPU]}"
EVAL_GPU="${2:-0}"
EVAL_GRAPHICS="${TRACE_GRAPHICS_DEVICE_ID:-$EVAL_GPU}"
[[ "$EVAL_GPU" =~ ^[0-9]+$ && "$EVAL_GRAPHICS" =~ ^[0-9]+$ ]] || exit 2
EVAL_TEACHER="$("$TRACE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["teacher_checkpoint"])' "$EVAL_SPEC")"
EVAL_RECIPE="$("$TRACE_PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("reward_recipe","legacy"))' "$EVAL_SPEC")"
EVAL_OVERRIDES=()
if [[ "$EVAL_RECIPE" == graspability_only ]]; then
  for term in lambdaC lambdaC3 lambdaArc lambdaB lambdaEef lambdaDisturb; do
    EVAL_OVERRIDES+=("task.env.teacher.$term=0.0")
  done
fi
cd "$EVAL_PACKAGE_ROOT"
exec "$TRACE_PYTHON" -u tools/evaluate_retrieval.py \
  task=MoreEvaluation train=MoreOpenLoopSetSCPPO test=True headless=True \
  force_render=False wandb_activate=False sim_device="cuda:$EVAL_GPU" rl_device="cuda:$EVAL_GPU" \
  graphics_device_id="$EVAL_GRAPHICS" \
  checkpoint="$EVAL_TEACHER" "${EVAL_OVERRIDES[@]}" \
  +evaluation_spec="$EVAL_SPEC"
