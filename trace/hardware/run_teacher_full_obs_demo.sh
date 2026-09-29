#!/bin/bash
# End-to-end teacher baseline: retract, fresh full observation, stock-GN
# evaluation, one recurrent-teacher primitive, repeat. No twin solve or sim
# video is generated. The caller supplies the per-trial output directory.
set -e
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTROLLER=teacher_full_obs SKIP_RENDER=1 \
  exec "$SCRIPT_DIR/run_student_demo.sh" "$@"
