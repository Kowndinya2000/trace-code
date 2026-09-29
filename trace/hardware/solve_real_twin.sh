#!/bin/bash
# Solve the freshly perceived real scene with the released privileged teacher.
# This is the hardware launchers' simulation-only stage; it never connects to the robot.
set -euo pipefail

MODE=${1:-real}
GPU=${2:-0}
if [ "$MODE" != "real" ]; then
  echo "usage: solve_real_twin.sh real [gpu]" >&2
  exit 2
fi

ROOT=${TRACE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA=${TRACE_DATA:-$ROOT/data}
PY=${PYTHON:-${TRACE_PYTHON:-python}}
IGE=$ROOT/trace/sim/isaacgymenvs
CKPT=${CKPT:-$DATA/checkpoints/teacher_ep210.pth}
OUT=${OUT:-$IGE/open_loop/out}
PHASE_ARGS=()
if [ -n "${PHASE_DIR:-}" ]; then PHASE_ARGS+=("+phase_dir=$PHASE_DIR"); fi

mkdir -p "$OUT/real2sim"
cd "$IGE"
exec "$PY" -u open_loop/solve_in_twin.py \
  task=MoreOpenLoop train=MoreOpenLoopSetSCPPO test=True headless=True \
  wandb_activate=False "checkpoint=$CKPT" num_envs=1 \
  task.env.test_cases.scene_root_dir=test-cases \
  task.env.test_cases.difficulty_choice=real2sim \
  "sim_device=cuda:$GPU" "rl_device=cuda:$GPU" "graphics_device_id=$GPU" \
  +settle_steps=30 "+trajectory_out=$OUT/real2sim" "${PHASE_ARGS[@]}"
