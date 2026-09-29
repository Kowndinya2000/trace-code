#!/usr/bin/env bash
# One SLURM array task: run line (SLURM_ARRAY_TASK_ID + 1) of $LIST on the task's GPU.
set -euo pipefail
eval "${SETUP:-true}"
ID=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
exec "$TRACE_PYTHON" "$REPO/cluster/plan_memory_ablations/run_job.py" "$ID" --out "$RUN_DIR" --gpu 0
