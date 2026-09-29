#!/bin/bash
# One command for the whole real-robot demo, recorded end to end.
#
#   ./run_real_demo.sh <out_dir> [--execute]
#     ./run_real_demo.sh runs/demo5 --execute        # moves the robot
#     ./run_real_demo.sh runs/demo5                  # dry run, no motion
#
# The cameras start FIRST and keep rolling across the whole pipeline --
# perception, the twin solve, the pushes, the re-sense, the grasp -- so the
# footage is one continuous take of the actual end-to-end latency instead of
# starting after the interesting decision has already been made.
#
# Consequences of that ordering, both deliberate:
#   * Perception goes through the recorder (--from-dump). A RealSense opens
#     once and the recorder owns it; routing perception through the same dump
#     the re-sense uses also guarantees both see an identically configured
#     sensor; two differently configured sensors disagree on the same scene.
#   * The 4K sim render happens AFTER the recorders stop. It costs ~41 s of
#     frame capture and belongs nowhere near a recorded window; video quality
#     is decoupled from execution latency.
set -e
OUT=${1:?usage: run_real_demo.sh <out_dir> [--execute]}
EXEC=${2:-}
# All recorded teacher open-loop trials use the physics-tick MoveL replay.
# Keep the real-demo default aligned with that protocol; callers can still
# override MODE explicitly for diagnostics.
MODE=${MODE:-physics_movel}
HARDWARE_RATE=${HARDWARE_RATE:-1.2}
# Successful grasps stay in the gripper at PMBS grasp-check home so the
# operator can remove the target by hand. Set KEEP_TARGET=0 only when an
# explicit automatic release is wanted.
KEEP_TARGET=${KEEP_TARGET:-1}
if [ "$MODE" = "cartesian_feedforward" ] && [ -z "${HARDWARE_VALIDATION:-}" ]; then
  echo "[demo] Cartesian hardware replay requires HARDWARE_VALIDATION pointing to a passed full air test." >&2
  exit 1
fi

source "$(dirname "${BASH_SOURCE[0]}")/demo_env.sh"
# Put that interpreter's env on PATH for the whole pipeline. The twin solver
# calls a bare `python`, so without this the solver picks up whatever python the
# CALLER happened to have and dies on `import isaacgym` unless the caller has
# already activated the env by hand.
_ENVBIN=$(dirname "$PY")
export PATH="$_ENVBIN:$PATH"
export LD_LIBRARY_PATH="$_ENVBIN/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
FORT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GYM=${TRACE_ROOT}/trace/sim
IGE=$GYM/isaacgymenvs
CALIB=${TRACE_CALIB:-${CALIB:-${TRACE_DATA}/calibration/camera_to_base.txt}}
MASKRCNN=${TRACE_MASKRCNN}
SOLVER=${SOLVER:-$FORT/solve_real_twin.sh}
BUNDLE=${TRACE_DATA}/checkpoints
CKPT=${CKPT:-$BUNDLE/teacher_ep210.pth}
TRAIN=${TRAIN:-MoreOpenLoopSetSCPPO}
D415=${TRACE_D415_SERIAL:-}
D455=${TRACE_D455_SERIAL:?set TRACE_D455_SERIAL to the top-down RealSense serial number}
EXTERNAL_STORAGE_ROOT=${EXTERNAL_STORAGE_ROOT:-${TRACE_RUNS}}
if [ ! -d "$EXTERNAL_STORAGE_ROOT" ]; then
  echo "[demo] external storage is unavailable: $EXTERNAL_STORAGE_ROOT" >&2
  echo "[demo] reconnect the external media before starting the trial" >&2
  exit 1
fi

mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
# A run dir is per-run. Reusing one mixes this run's dumps, phase log and
# videos with the last one's, and the summary then quotes whichever it finds.
for artifact in "$OUT"/*_dump*.done "$OUT"/*.mp4 "$OUT/phases.json"; do
  if [ -e "$artifact" ]; then
    echo "[demo] $OUT already holds a run. Use a new directory." >&2
    exit 1
  fi
done
LABEL=${SCENE_LABEL:-$(basename "$OUT")}
TRAJ=$IGE/open_loop/out/real2sim/000000.json
CKPT_LABEL=${CKPT_LABEL:-"Released privileged teacher  |  16 push primitives, 4 cm"}
export PYTHONPATH=$GYM:$FORT${PYTHONPATH:+:$PYTHONPATH}
mark() { $PY -m isaacgymenvs.open_loop.recorder_io "$OUT" "$1" "$2" || true; }

T0=$($PY -c "import time;print(time.time())")
# Keep the completed run's small, irreplaceable records outside the camera
# directory while both recorders encode: a recorder-side cleanup of that
# directory can otherwise remove them after successful robot execution. The
# large camera spool is disposable, but the exact trajectory, phase clock and
# measured EEF result are not. Keep crash checkpoints on the external
# experiment disk so a long recording cannot exhaust the root filesystem.
BACKGROUND_ROOT=${BACKGROUND_ROOT:-$EXTERNAL_STORAGE_ROOT/background}
RECOVERY_DIR=${RECOVERY_DIR:-$BACKGROUND_ROOT/run-recovery/${LABEL}-${T0//./_}}
snapshot_core() {
  mkdir -p "$RECOVERY_DIR"
  for artifact in trajectory.json phases.json stage_timing.json real_timing.json \
                  hardware_replay.json d455_topdown_dump*; do
    if [ -e "$OUT/$artifact" ]; then cp -a "$OUT/$artifact" "$RECOVERY_DIR/"; fi
  done
}
restore_core() {
  [ -d "$RECOVERY_DIR" ] || return 0
  mkdir -p "$OUT"
  for artifact in "$RECOVERY_DIR"/*; do
    [ -e "$artifact" ] || continue
    name=$(basename "$artifact")
    if [ ! -e "$OUT/$name" ]; then
      echo "[demo] restoring recorder-lost artifact: $name"
      cp -a "$artifact" "$OUT/"
    fi
  done
}
FINISHED=0
P1=""
P2=""
cleanup() {
  code=$?
  if [ "$FINISHED" != "1" ] && [ -f "$OUT/phases.json" ]; then
    mark failed "pipeline exited with code $code"
  fi
  for camera in webcam_scene d415_scene d455_topdown; do
    if [ -d "$OUT/$camera.ctrl" ]; then touch "$OUT/$camera.ctrl/stop"; fi
  done
  wait $P1 $P2 2>/dev/null || true
  $PY -m isaacgymenvs.open_loop.trial_timing aggregate "$(dirname "$OUT")" >/dev/null || true
}
trap cleanup EXIT
$PY - "$OUT" "$T0" <<'EOF'
import sys
from isaacgymenvs.open_loop.trial_timing import write_json_atomic, write_trial_timing
from pathlib import Path
directory, epoch = Path(sys.argv[1]), float(sys.argv[2])
document = {'t0': epoch, 'events': []}
write_json_atomic(directory/'phases.json', document)
write_trial_timing(directory, document)
EOF
# Only start a recorder for a camera that is actually plugged in. Waiting on a
# ready file from a camera that is not there hangs the whole run.
PRESENT=$($PY -c "import pyrealsense2 as rs; print(' '.join(d.get_info(rs.camera_info.serial_number) for d in rs.context().devices))")
case " $PRESENT " in *" $D455 "*) ;; *) echo "[demo] D455 $D455 not connected - perception needs it" >&2; exit 1;; esac
# Third-person view: the C615 webcam (1920x1080 MJPG @ 30, raw passthrough) is
# preferred; the D415 is only a fallback (USB 2.1 capped it at 1280x720 @ 15).
WEBCAM=${TRACE_WEBCAM:-$(ls /dev/v4l/by-id/usb-046d_HD_Webcam_C615_*-video-index0 2>/dev/null | head -1)}
HAVE_D415=0
if [ -n "$D415" ]; then case " $PRESENT " in *" $D415 "*) HAVE_D415=1;; esac; fi
if [ -n "$WEBCAM" ]; then THIRD=webcam_scene; echo "[demo] third-person: C615 webcam $WEBCAM"
elif [ "$HAVE_D415" = "1" ]; then THIRD=d415_scene; echo "[demo] third-person: D415 (webcam not found)"
else THIRD=""; echo "[demo] no third-person camera - skipping that video"; fi

echo "[demo] starting recorders (t0=$T0)"
P1=""
if [ "$THIRD" = "webcam_scene" ]; then
$PY "$FORT/record_webcam.py" --device "$WEBCAM" --name webcam_scene --out "$OUT" --t0 "$T0" \
    --label "Real robot - open-loop retrieval" --subtitle "$CKPT_LABEL" \
    > "$OUT/webcam.log" 2>&1 &
P1=$!
elif [ "$THIRD" = "d415_scene" ]; then
$PY "$FORT/record_cameras.py" --serial $D415 --name d415_scene   --out "$OUT" --t0 "$T0" \
    --label "Real robot - open-loop retrieval" --subtitle "$CKPT_LABEL" \
    > "$OUT/d415.log" 2>&1 &
P1=$!
fi
# no --calib: the projected workspace box clutters the top-down view, which is
# the one view where the clutter itself is the point
$PY "$FORT/record_cameras.py" --serial $D455 --name d455_topdown --out "$OUT" --t0 "$T0" --depth \
    --label "Perception camera - top-down" --subtitle "$CKPT_LABEL" --layout band \
    > "$OUT/d455.log" 2>&1 &
P2=$!
CAMS="d455_topdown"; [ -n "$THIRD" ] && CAMS="$THIRD $CAMS"
for n in $CAMS; do
  ready_deadline=$((SECONDS + ${CAMERA_READY_TIMEOUT_S:-60}))
  until [ -f "$OUT/$n.ctrl/ready" ]; do
    if [ "$SECONDS" -ge "$ready_deadline" ]; then
      echo "[demo] camera $n did not become ready; inspect the new trial's recorder log" >&2
      exit 1
    fi
    for recorder_pid in $P1 $P2; do
      if ! kill -0 "$recorder_pid" 2>/dev/null; then
        echo "[demo] recorder exited before readiness; stopping before motion" >&2
        exit 1
      fi
    done
    sleep 0.2
  done
done
echo "[demo] recorders ready"

cd "$IGE"
if [ -n "${PREPARED_TRAJECTORY:-}" ]; then
  # Continue an already captured/solved trial without another planning cycle.
  # The source trial retains its original perception and solve timestamps.
  T_PERCEIVE=0
  T_TWIN=0
  cp "$PREPARED_TRAJECTORY" "$OUT/trajectory.json"
  $PY - "$PREPARED_TRAJECTORY" "$OUT" <<'EOF'
import json, sys
from pathlib import Path
from isaacgymenvs.open_loop.trial_timing import write_json_atomic
source, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
write_json_atomic(out/'prepared_reference.json', {
    'trajectory': str(source), 'planning_trial': str(source.parent),
    'perception_and_solve_reused': True,
    'note': 'This recording covers execution; original planning timestamps remain in the source trial.'})
EOF
else
# ---- 1. perceive (through the recorder; it owns the camera) -----------------
tp=$($PY -c "import time;print(time.time())")
mark perceive "capturing scene"
$PY open_loop/perceive_scene.py --from-dump "$OUT" --dump-name d455_topdown \
    --calib "$CALIB" --maskrcnn "$MASKRCNN" --out-dir test-cases/real2sim --scene-id 0
T_PERCEIVE=$($PY -c "import time;print(round(time.time()-$tp,3))")

# ---- 2. solve in the twin --------------------------------------------------
mark twin "loading Isaac Gym"
ts=$($PY -c "import time;print(time.time())")
PHASE_DIR="$OUT" CKPT="$CKPT" "$SOLVER" real 0 > "$OUT/solve.log" 2>&1
T_TWIN=$($PY -c "import time;print(round(time.time()-$ts,3))")
tail -2 "$OUT/solve.log"
# Preserve the exact solve consumed by this trial before shared outputs change.
cp "$TRAJ" "$OUT/trajectory.json"
fi
TRAJ="$OUT/trajectory.json"

# ---- 3. execute on the robot ----------------------------------------------
# MAX_PUSH_CM / FORCE_LONG let the operator waive the plan-length gate for a
# borderline plan without editing the default, which stays evidence-based.
mark execution "loading hardware executor"
HARDWARE_ARGS=()
KEEP_ARGS=()
if [ "$KEEP_TARGET" = "1" ]; then KEEP_ARGS=(--keep); fi
if [ "$MODE" = "cartesian_feedforward" ]; then
  HARDWARE_ARGS=(--hardware-validation "$HARDWARE_VALIDATION" --hardware-rate "$HARDWARE_RATE")
  if [ "${HARDWARE_AIR_CHECK:-0}" = "1" ]; then HARDWARE_ARGS+=(--air-check); fi
fi
$PY open_loop/execute_trajectory.py "$TRAJ" --mode "$MODE" $EXEC --yes --home \
    --calib "$CALIB" --maskrcnn "$MASKRCNN" \
    "${HARDWARE_ARGS[@]}" "${KEEP_ARGS[@]}" \
    ${MAX_PUSH_CM:+--max-push-cm "$MAX_PUSH_CM"} ${FORCE_LONG:+--force-long} \
    --resense-dir "$OUT" --resense-name d455_topdown \
    --timing-out "$OUT/real_timing.json"
if [ "$EXEC" != "--execute" ]; then mark dry-run "plan checked without motion"; fi
FINISHED=1
snapshot_core

echo "[demo] stopping recorders"
for n in $CAMS; do touch "$OUT/$n.ctrl/stop"; done
wait $P1 $P2 2>/dev/null || true
restore_core

# ---- 4. POST-PROCESSING: 4K sim videos, after the run is over --------------
if [ "${SKIP_RENDER:-0}" != "1" ]; then
  echo "[demo] rendering 4K sim videos (post-processing)"
  $PY -u tools/record_solve.py task=MoreOpenLoop train=$TRAIN test=True headless=True \
      wandb_activate=False checkpoint=$CKPT \
      sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 \
      +traj_json="$TRAJ" +out_dir="$OUT/sim" +budget=140 "+scene_label=$LABEL" \
      "+phases_json=$OUT/phases.json" \
      > "$OUT/render.log" 2>&1 || tail -5 "$OUT/render.log"
fi

$PY - "$OUT" "$T_PERCEIVE" "$T_TWIN" <<'EOF'
import json, os, sys, glob, re
out, t_perceive, t_twin = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
def load(p):
    return json.load(open(p)) if os.path.exists(p) else {}
real = load(os.path.join(out, "real_timing.json"))
events = load(os.path.join(out, "phases.json")).get("events", [])
ph = {e["phase"]: e["t"] for e in events}
def span(a, b):
    return round(ph[b] - ph[a], 3) if {a, b} <= set(ph) else None
def rollout_sim():
    event = next((e for e in events if e.get("phase") == "solve-export"), {})
    match = re.search(r"(?:^|;\s*)simulated_s=([0-9.]+)", str(event.get("detail", "")))
    return round(float(match.group(1)), 3) if match else None
new_solver_clock = {"twin-settle", "solve-export"} <= set(ph)
print("\n=== END-TO-END (single recorded take) ===")
rows = [("perceive (camera -> scene)", t_perceive),
        ("twin total wall-clock", t_twin)]
if new_solver_clock:
    rows += [("  Isaac Gym + teacher load", span("twin", "twin-settle")),
             ("  scene settling", span("twin-settle", "solve")),
             ("  PPO rollout simulated", rollout_sim()),
             ("  PPO rollout wall-clock", span("solve", "solve-export")),
             ("  plan finalization", span("solve-export", "solve-done"))]
else:
    rows += [("  legacy combined solve", span("solve", "solve-done"))]
rows += [
        ("push", real.get("push_seconds")),
        ("home -> PMBS", real.get("home_seconds")),
        ("re-sense", real.get("resense_seconds")),
        ("grasp", real.get("grasp_seconds"))]
for k, v in rows:
    print(f"  {k:28s}: {v if v is not None else '-'} s")
tot = t_perceive + t_twin + (real.get("real_total_seconds") or 0)
print(f"  {'TOTAL wall-clock':28s}: {round(tot, 2)} s")
g = real.get("real_grasp") or {}
if g:
    print(f"  re-sensed graspability      : q={g.get('q')} graspable={g.get('graspable')} "
          f"bin={g.get('rotation_idx')}")
if "holding" in real:
    print(f"  gripper                     : {real.get('gripper_position')}/255 "
          f"{'HOLDING' if real['holding'] else 'EMPTY'}")
for n in ("webcam_scene", "d415_scene", "d455_topdown"):
    m = load(os.path.join(out, n + "_rec.json"))
    if m:
        print(f"  video {n:21s}: {m.get('frames')} frames, {m.get('duration_s')} s")
print("  sim videos                  :", len(glob.glob(os.path.join(out, "sim", "*.mp4"))))
stages = load(os.path.join(out, "stage_timing.json"))
print("  measured end-to-end         :", stages.get("end_to_end_seconds"), "s")
print("  per-stage timing JSON       :", os.path.join(out, "stage_timing.json"))
EOF
if [ "${SKIP_POSTPROCESS:-0}" != "1" ]; then
  $PY "$FORT/video_postprocess/compose_open_loop.py" "$OUT" "$OUT/review.mp4"
fi
echo "[demo] done -> $OUT"
