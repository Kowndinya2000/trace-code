#!/bin/bash
# Generate + validate the gen-v1 dataset suite (3 train tiers + tiered test).
# Run from isaacgymenvs/ inside the pmbs env with LD_LIBRARY_PATH set.
set -e
G=test-cases/gen-v1
PY=${TRACE_PYTHON}
W=8
COMMON="--workers $W --yaw-mode axis --compact-sweeps 2 --max-t3nn-mm 4 --max-t4c-cm 20"

gen () { # name num seed gap tgap blocked bigmin bigmax motif void pocket
  echo "=== TIER $1 ==="
  if [ $(ls $G/$1/*.txt 2>/dev/null | wc -l) -lt $2 ]; then
  $PY tools/build_dataset.py --num $2 --seed $3 --out $G/$1 \
      --gap-mm $4 --target-gap-mm $5 --min-blocked $6 \
      --big-range $7 $8 --motif-prob $9 \
      --max-void-frac ${10} --max-pocket-mm ${11} $COMMON --wandb
  fi
  $PY tools/build_dataset.py --settle --freeze --out $G/$1
  $PY tools/build_dataset.py --settle --out $G/$1
  # GN gate: no scene may start graspable (quarantine manually if flagged)
  $PY tools/verify_ungraspable.py task=MoreTeacher test=True headless=True \
      "+vg_scenes=$G/$1" +vg_start=0 +vg_chunk=420 | grep "^chunk"
}

gen train-easy   700 100 3.0  1.0 12 2 3 0.0  0.34 32
gen train-medium 700 200 1.5  0.5 14 3 5 0.2  0.34 30
gen train-hard   700 300 0.75 0.5 16 4 6 0.5  0.33 30
gen test-easy    171 900 3.0  1.0 12 2 3 0.0  0.34 32
gen test-medium  171 901 1.5  0.5 14 3 5 0.2  0.34 30
gen test-hard    170 902 0.75 0.5 16 4 6 0.5  0.33 30

echo "=== ASSEMBLE ==="
$PY - <<'EOF'
import json, os, shutil
G = "test-cases/gen-v1"
for combo, tiers in [("train", ["train-easy", "train-medium", "train-hard"]),
                     ("test", ["test-easy", "test-medium", "test-hard"])]:
    out = os.path.join(G, combo)
    os.makedirs(out, exist_ok=True)
    sid, meta = 0, []
    for tier in tiers:
        td = os.path.join(G, tier)
        tm = {m["scene"]: m for m in json.load(open(os.path.join(td, "meta.json")))["scenes"]}
        for f in sorted(x for x in os.listdir(td) if x.endswith(".txt")):
            shutil.copy(os.path.join(td, f), os.path.join(out, f"{sid:06d}.txt"))
            m = dict(tm.get(f, {"scene": f}))
            m.update({"scene": f"{sid:06d}.txt", "tier": tier, "src": f})
            meta.append(m)
            sid += 1
    json.dump({"scenes": meta}, open(os.path.join(out, "meta.json"), "w"), indent=1)
    print(f"{combo}: {sid} scenes")
EOF

echo "=== QA ==="
$PY tools/scene_qa.py $G/train $G/test
echo "=== DONE gen-v1 ==="
