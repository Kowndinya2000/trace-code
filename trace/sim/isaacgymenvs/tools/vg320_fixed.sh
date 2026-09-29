#!/bin/bash
# t0-graspability re-verification under the FIXED 320px camera (b96d36d)
bash ${TRACE_ROOT}/trace/sim/isaacgymenvs/tools/vg320.sh
cd ${TRACE_ROOT}/trace/sim/isaacgymenvs
export LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/pmbs/lib:$LD_LIBRARY_PATH
${TRACE_PYTHON} tools/verify_ungraspable.py task=MoreTeacher test=True headless=True '+vg_scenes=test-cases/gen-v1/quarantine' +vg_chunk=8 +vg_start=0 2>&1 | grep "^chunk"
echo VG320_FIXED_DONE
