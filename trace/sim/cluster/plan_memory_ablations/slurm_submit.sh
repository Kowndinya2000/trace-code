#!/usr/bin/env bash
# Submit the prepared job graph as three SLURM job arrays: fits and plan batches
# start immediately; cases start after both arrays succeed.
#
#   RUN_DIR=/scratch/$USER/plan-memory-run TRACE_PYTHON=/path/to/py38/bin/python \
#     PARTITION=gpu MAX_PARALLEL=16 bash cluster/plan_memory_ablations/slurm_submit.sh
#
# Optional: ACCOUNT, TIME_FIT (default 00:30:00), TIME_EVAL (00:20:00), EXTRA_SBATCH
# (e.g. "--constraint=ampere"), SETUP (a shell line run before each task, e.g.
# "module load cuda/11.7; source ~/isaacgym-env.sh"). Each task requests one GPU,
# which is index 0 inside the task. If the Vulkan device does not follow the cgroup
# (camera preflight fails), export TRACE_GRAPHICS_DEVICE_ID in SETUP.
set -euo pipefail
: "${RUN_DIR:?set RUN_DIR to the prepared run directory}"
: "${TRACE_PYTHON:?set TRACE_PYTHON to the Isaac Gym Preview 4 Python 3.8 interpreter}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK="$REPO/cluster/plan_memory_ablations/slurm_task.sh"
PARTITION="${PARTITION:-gpu}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
TIME_FIT="${TIME_FIT:-00:30:00}"
TIME_EVAL="${TIME_EVAL:-00:20:00}"
export SETUP="${SETUP:-true}" TRACE_PYTHON RUN_DIR REPO
mkdir -p "$RUN_DIR/slurm"

submit() {
  local kind="$1" time="$2" mem="$3" dependency="$4"
  local list="$RUN_DIR/joblist_${kind}.txt"
  local count
  count=$(grep -c . "$list" || true)
  if [[ "$count" -eq 0 ]]; then
    echo ""
    return
  fi
  local extra=()
  [[ -n "${ACCOUNT:-}" ]] && extra+=(--account "$ACCOUNT")
  [[ -n "$dependency" ]] && extra+=(--dependency "afterok:$dependency")
  # shellcheck disable=SC2206
  [[ -n "${EXTRA_SBATCH:-}" ]] && extra+=(${EXTRA_SBATCH})
  LIST="$list" sbatch --parsable --job-name "abl-$kind" --partition "$PARTITION" \
    --array "0-$((count - 1))%$MAX_PARALLEL" --gres gpu:1 --cpus-per-task 4 --mem "$mem" \
    --time "$time" --output "$RUN_DIR/slurm/%x-%A_%a.out" --export ALL "${extra[@]}" "$TASK"
}

FIT_JOB=$(submit fit "$TIME_FIT" 32G "")
PLAN_JOB=$(submit plan "$TIME_EVAL" 16G "")
DEPS=$(printf '%s\n' "$FIT_JOB" "$PLAN_JOB" | sed '/^$/d' | paste -sd ':' -)
CASE_JOB=$(submit "case" "$TIME_EVAL" 16G "$DEPS")  # quoted: a bare `case` confuses $( ) parsing
echo "fits=${FIT_JOB:-none} plans=${PLAN_JOB:-none} cases=${CASE_JOB:-none}"
echo "When the case array finishes: $TRACE_PYTHON $REPO/cluster/plan_memory_ablations/summarize.py --out $RUN_DIR"
