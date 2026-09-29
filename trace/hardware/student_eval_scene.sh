#!/bin/bash
# Teacher / replay / student evaluation with token-visibility videos on one
# perceived real scene.
#   ./student_eval_scene.sh runs/demoN            # both conditions
#   COND="d0" ./student_eval_scene.sh runs/demoN  # arm-shadow occlusion only
# Perceives <run>/d455_topdown_dump0 if <run>/scene/000000.txt is missing, then
# per condition: manifest, one spec per actor, six fresh-simulator runs,
# gallery clips + index.html, 2x3 montage, per-decision frames.
set -e
RUN=$(cd "${1:?usage: student_eval_scene.sh <run_dir>}" && pwd)
FORT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export TRACE_ROOT=${TRACE_ROOT:-$(cd "$FORT/../.." && pwd)}
export TRACE_DATA=${TRACE_DATA:-$TRACE_ROOT/data}
GYM=${TRACE_ROOT}/trace/sim; IGE=$GYM/isaacgymenvs
PY=${TRACE_PYTHON:-python}
export PYTHONPATH=$GYM:$FORT${PYTHONPATH:+:$PYTHONPATH}
B=${TRACE_DATA}/checkpoints
T=$B/teacher_ep210.pth
CALIB=${TRACE_CALIB:-${CALIB:-${TRACE_DATA}/calibration/camera_to_base.txt}}
MASKRCNN=${TRACE_MASKRCNN:-${TRACE_DATA}/segmentation/maskrcnn.pth}
ACTORS="teacher replay bc dagger_r1 dagger_r2 dagger_r3"

cd "$IGE"
if [ ! -f "$RUN/scene/000000.txt" ]; then
  D=$RUN/d455_topdown_dump0
  [ -f "${D}_color.png" ] || D=$RUN/perception/d455_topdown_dump0
  $PY open_loop/perceive_scene.py --color ${D}_color.png --depth ${D}_depth.npy --intrinsics ${D}_K.json \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" --out-dir "$RUN/scene" --scene-id 0 | grep -E 'objects in|scene:'
fi

for cond in ${COND:-default d0}; do
  case $cond in default) R=$RUN/student_eval_exact; PD=0.10; BL=5;; d0) R=$RUN/student_eval_exact_d0; PD=0.0; BL=0;; *) echo "Unknown condition: $cond" >&2; exit 2;; esac
  if [ -f "$R/complete.marker" ]; then echo "[eval] $cond already done"; continue; fi
  mkdir -p "$R/specs" "$R/runs"
  $PY - "$R" "$RUN/scene/000000.txt" "$B" "$PD" "$BL" "$cond" <<'PY'
import json, sys, hashlib
R, scene, B, pd, bl, cond = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sha = hashlib.sha256(open(scene, "rb").read()).hexdigest()
scenes = [dict(path=scene, sha256=sha, tier="test-real")]
json.dump(dict(scenes=scenes), open(f"{R}/manifest.json", "w"))
common = dict(manifest=f"{R}/manifest.json", batch_size=1, offset=0, seed=7, horizon=120,
              pos_noise=0.0, yaw_noise_deg=0.0, video_indices=[0])   # exact perceived layout, no jitter
students = {"bc": "trace_bc_seed0.pt", "dagger_r1": "trace_r1_seed0.pt",
            "dagger_r2": "trace_r2_seed0.pt", "dagger_r3": "trace_r3_seed0.pt"}
cases = [("teacher", dict(actor="teacher")), ("replay", dict(actor="replay"))] + \
        [(n, dict(actor="student", checkpoint=f"{B}/{filename}")) for n, filename in students.items()]
entries = []
for name, case in cases:
    spec = dict(common, output=f"{R}/runs/{name}", cases=[dict(case, name=name, p_drop=pd, blackout_len=bl)])
    if name != "teacher": spec["plan_source"] = f"{R}/runs/teacher/nominal.json"
    json.dump(spec, open(f"{R}/specs/{name}.json", "w"), indent=1)
    entries.append(dict(name=name, policy=name, condition=cond, p_drop=pd, spec=f"{R}/specs/{name}.json"))
json.dump(dict(scenes=scenes, entries=entries), open(f"{R}/gallery_spec.json", "w"), indent=1)
PY
  for c in $ACTORS; do
    [ -f "$R/runs/$c/complete.json" ] && continue
    $PY -u tools/record_student_eval.py task=MoreEvaluation train=MoreOpenLoopSetSCPPO test=True headless=True \
        force_render=False wandb_activate=False checkpoint=$T sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 \
        +evaluation_spec=$R/specs/$c.json > "$R/runs/$c.log" 2>&1 || { echo "[eval] $cond/$c FAILED, see $R/runs/$c.log"; exit 1; }
    $PY - "$R" "$c" <<'PY'
import json, sys; R, c = sys.argv[1:3]
r = json.load(open(f"{R}/runs/{c}/{c}.json"))["rows"][0]
print(f"[eval] {c:10s} {r['reason']:16s} step {r['terminal_step']:3d} q {r['terminal_q']:.2f} vis {r.get('visible_fraction',0):.2f}")
PY
  done
  $PY tools/make_student_video_gallery.py "$R" --refresh > "$R/gallery.log" 2>&1 || { echo "[eval] gallery failed"; exit 1; }
  touch "$R/complete.marker"
done
echo "[eval] done -> $RUN/student_eval_exact{,_d0}"
