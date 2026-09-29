#!/bin/bash
# Post-build finalisation for a generated suite: wait for gen_suite to exit,
# re-run the t0-graspability gate on the ASSEMBLED train/ and test/ splits,
# quarantine any flagged scene, then push every tier's annotated render sheet
# to one wandb run.
set -u
cd ${TRACE_ROOT}/trace/sim/isaacgymenvs
export LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/pmbs/lib:${LD_LIBRARY_PATH:-}
PY=${TRACE_PYTHON}
G=test-cases/gen-v2

while pgrep -f "gen_suite_v2.sh" > /dev/null; do sleep 30; done
echo "=== build finished; assembled: train=$(ls $G/train/*.txt 2>/dev/null | wc -l) test=$(ls $G/test/*.txt 2>/dev/null | wc -l)"

for split in train test; do
  n=$(ls $G/$split/*.txt 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && { echo "!! $split empty, aborting"; exit 1; }
  for s in $(seq 0 420 $((n - 1))); do
    $PY tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
        "+vg_scenes=$G/$split" +vg_chunk=420 +vg_start=$s 2>&1 | grep "^chunk"
  done
done

$PY - <<'PYEOF'
import glob, json, os, shutil
G = "test-cases/gen-v2"
qdir = os.path.join(G, "quarantine")
os.makedirs(qdir, exist_ok=True)
moved = []
for split in ("train", "test"):
    for f in glob.glob(f"tools/qa_out/graspable_t0_{split}_*.json"):
        d = json.load(open(f))
        for name in d.get("bad", []):
            src = os.path.join(G, split, os.path.basename(name))
            if os.path.exists(src):
                shutil.move(src, os.path.join(qdir, f"{split}_{os.path.basename(name)}"))
                moved.append(src)
print(f"quarantined {len(moved)} t0-graspable scenes: {moved}")
print("final counts: train=%d test=%d" % (
    len(glob.glob(f"{G}/train/*.txt")), len(glob.glob(f"{G}/test/*.txt"))))
PYEOF

$PY tools/push_suite_wandb.py $G gen-v2-SUITE
echo FINALIZE_DONE
