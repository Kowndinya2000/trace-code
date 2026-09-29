#!/bin/bash
# One-command recorded circular-spiral baseline using the maintained
# closed-loop D455, stock-GN, staged retraction, and grasp pipeline.
set -e
FORT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTROLLER=spiral SKIP_RENDER=1 exec "$FORT/run_student_demo.sh" "$@"
