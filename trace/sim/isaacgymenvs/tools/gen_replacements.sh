cd ${TRACE_ROOT}/trace/sim/isaacgymenvs
export LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/pmbs/lib:$LD_LIBRARY_PATH
PY=${TRACE_PYTHON}
C="--workers 8 --yaw-mode axis --compact-sweeps 2 --max-t3nn-mm 4 --max-t4c-cm 20"
G=test-cases/gen-v1
gen() { $PY tools/build_dataset.py --num $2 --seed $3 --out $G/repl-$1 \
  --gap-mm $4 --target-gap-mm $5 --min-blocked $6 --big-range $7 $8 \
  --motif-prob $9 --max-void-frac ${10} --max-pocket-mm ${11} $C && \
  $PY tools/build_dataset.py --settle --freeze --out $G/repl-$1 && \
  $PY tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
    "+vg_scenes=$G/repl-$1" +vg_start=0 +vg_chunk=420 | grep "^chunk"; }
gen tr-easy 103 501 3.0  1.0 12 2 3 0.0  0.34 32
gen tr-med  103 502 1.5  0.5 14 3 5 0.2  0.34 30
gen tr-hard 103 503 0.75 0.5 16 4 6 0.5  0.33 30
gen te-easy  20 601 3.0  1.0 12 2 3 0.0  0.34 32
gen te-med   20 602 1.5  0.5 14 3 5 0.2  0.34 30
gen te-hard  20 603 0.75 0.5 16 4 6 0.5  0.33 30
$PY - <<'EOF'
import os, shutil, json
G = "test-cases/gen-v1"
for combo, tiers, base in (("train", ["tr-easy","tr-med","tr-hard"], 3000),
                           ("test", ["te-easy","te-med","te-hard"], 3000)):
    sid = base
    for t in tiers:
        d = f"{G}/repl-{t}"
        for f in sorted(x for x in os.listdir(d) if x.endswith(".txt")):
            shutil.copy(os.path.join(d, f), f"{G}/{combo}/{sid:06d}.txt")
            sid += 1
    print(combo, "now:", len([x for x in os.listdir(f'{G}/{combo}') if x.endswith('.txt')]))
EOF
echo REPL_DONE
