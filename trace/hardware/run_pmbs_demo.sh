#!/bin/bash
# One-command recorded PMBS baseline on the real robot: full re-sense from PMBS home
# before every decision, parallel MCTS in PMBS's simulator, shared grasp path and OOW rule.
set -e
FORT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONTROLLER=pmbs SKIP_RENDER=1 exec "$FORT/run_student_demo.sh" "$@"
