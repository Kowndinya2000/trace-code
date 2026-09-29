cd ${TRACE_ROOT}/trace/sim/isaacgymenvs
export LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/pmbs/lib:$LD_LIBRARY_PATH
PY=${TRACE_PYTHON}
for s in 0 420 840 1260 1680; do
  $PY tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
    '+vg_scenes=test-cases/gen-v1/train' +vg_chunk=420 +vg_start=$s 2>&1 | grep "^chunk"
done
for s in 0 420; do
  $PY tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
    '+vg_scenes=test-cases/gen-v1/test' +vg_chunk=420 +vg_start=$s 2>&1 | grep "^chunk"
done
echo VG320_DONE
