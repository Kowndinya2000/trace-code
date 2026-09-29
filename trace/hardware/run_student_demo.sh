#!/bin/bash
# One command for the STUDENT closed-loop demo, recorded end to end. The same
# recording/perception shell also hosts the spiral and full-observation teacher
# baselines; those paths skip the twin solve.
# Derived from run_real_demo.sh: same recorders, same perception-through-the-
# recorder, same twin solve for the nominal plan -- but the robot is driven by
# the Stage-2 student re-perceiving every primitive (open_loop/run_student.py)
# instead of replaying the twin's trajectory blind.
#
#   ./run_student_demo.sh <out_dir> [--execute]
#     ./run_student_demo.sh runs/student1 --execute   # moves the robot
#     ./run_student_demo.sh runs/student1             # dry run, no motion
#   ./run_spiral_demo.sh runs/spiral1 --execute       # spiral baseline
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
MODE=${MODE:-executed}
KEEP_TARGET=${KEEP_TARGET:-1}
CONTROLLER=${CONTROLLER:-student}
case "$CONTROLLER" in
  student|spiral|teacher_full_obs|pmbs) ;;
  *) echo "CONTROLLER must be 'student', 'spiral', 'teacher_full_obs', or 'pmbs'" >&2; exit 2;;
esac

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
CKPT=${CKPT:-$BUNDLE/teacher_ep210.pth}                # nominal-rollout teacher
STUDENT=${STUDENT:-$BUNDLE/trace_r3.pt}
STUDENT_SHA256=${STUDENT_SHA256:-dda9db7dc06f30bce5aa9ab301e05fc6781641c0f3ae450af3e50948c7ef9d30}
MAX_STEPS=${MAX_STEPS:-0}                               # 0 = sim evaluation horizon (120)
STUDENT_TIMEOUT_S=${STUDENT_TIMEOUT_S:-100}             # wall-clock student execution ceiling
TEACHER_DISTANCE_MARGIN_M=${TEACHER_DISTANCE_MARGIN_M:-0.05}
SPIRAL_MAX_STEPS=${SPIRAL_MAX_STEPS:-240}
SPIRAL_TIMEOUT_S=${SPIRAL_TIMEOUT_S:-180}
SPIRAL_MAX_TRAVEL_M=${SPIRAL_MAX_TRAVEL_M:-2.5}
PMBS_NUM_ENVS=${PMBS_NUM_ENVS:-800}                     # parallel Isaac Gym environments for the search
PMBS_TIME_LIMIT=${PMBS_TIME_LIMIT:-15}                   # search seconds per push
PMBS_MAX_ACTIONS=${PMBS_MAX_ACTIONS:-15}                 # PMBS action cap (pushes + grasp attempts)
PMBS_BACKOFF_M=${PMBS_BACKOFF_M:-0.015}                  # withdraw 1.5 cm back along the push before lifting
SPIRAL_ARC_STEP_M=${SPIRAL_ARC_STEP_M:-0.06}             # spiral protocol: 6 cm command per step
SPIRAL_RADIAL_STEP_M=${SPIRAL_RADIAL_STEP_M:-0.015}      # spiral protocol: 1.5 cm inward per step
SPIRAL_BACKOFF_M=${SPIRAL_BACKOFF_M:-0.01}                # withdraw 1 cm along the own path before lifting
EXPECTED_OBJECTS=${EXPECTED_OBJECTS:-11}                # stop before motion if capture is incomplete
TEACHER_FULL_OBS_MAX_STEPS=${TEACHER_FULL_OBS_MAX_STEPS:-120}
# Withdrawal along the EEF's own path before each lift, in metres. 2 cm always
# clears; 1 cm is marginal, because a concave block's cavity is 23.5 mm deep and
# can ride the tool again. Settable per run, no code edit.
TEACHER_BACKOFF_M=${TEACHER_BACKOFF_M:-0.015}
TEACHER_FULL_OBS_TIMEOUT_S=${TEACHER_FULL_OBS_TIMEOUT_S:-300}
TEACHER_SHA256=${TEACHER_SHA256:-8bb5a5d6d3503e34c5b92ac1d3bff54cae3cdbac111d5fcd71959d8c3d1056b9}
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
for artifact in "$OUT"/*_dump*.done "$OUT"/*.mp4 "$OUT/phases.json" \
                "$OUT/metadata/phases.json"; do
  if [ -e "$artifact" ]; then
    echo "[demo] $OUT already holds a run. Use a new directory." >&2
    exit 1
  fi
done
LABEL=${SCENE_LABEL:-$(basename "$OUT")}
# The student's twin solver reads the shared real2sim scene; every other controller keeps its
# initial perception inside the trial folder on the external drive.
if [ "$CONTROLLER" = "student" ]; then REAL2SIM_DIR=test-cases/real2sim; else REAL2SIM_DIR="$OUT/real2sim"; fi
SHARED_TRAJ=$IGE/open_loop/out/real2sim/000000.json
TRAJ=$OUT/trajectory.json
if [ "$CONTROLLER" = "pmbs" ]; then
  CKPT_LABEL=${CKPT_LABEL:-"PMBS  |  parallel MCTS, ${PMBS_TIME_LIMIT} s over ${PMBS_NUM_ENVS} envs per push"}
  VIDEO_LABEL="Real robot - PMBS baseline"
  EXEC_LABEL="real robot, closed loop (pmbs)"
elif [ "$CONTROLLER" = "spiral" ]; then
  CKPT_LABEL=${CKPT_LABEL:-"Circular spiral  |  closed-loop, 6 cm arc steps"}
  VIDEO_LABEL="Real robot - closed-loop spiral baseline"
  EXEC_LABEL="real robot, closed loop (spiral)"
elif [ "$CONTROLLER" = "teacher_full_obs" ]; then
  CKPT_LABEL=${CKPT_LABEL:-"Teacher  |  full observation before every 4 cm primitive"}
  VIDEO_LABEL="Real robot - full-observation teacher baseline"
  EXEC_LABEL="real robot, teacher full-observation loop"
else
  CKPT_LABEL=${CKPT_LABEL:-"Student DAgger R3  |  closed-loop, 16 primitives, 4 cm"}
  VIDEO_LABEL="Real robot - closed-loop student retrieval"
  EXEC_LABEL="real robot, closed loop (student)"
fi
export PYTHONPATH=$GYM:$FORT${PYTHONPATH:+:$PYTHONPATH}
mark() { $PY -m isaacgymenvs.open_loop.recorder_io "$OUT" "$1" "$2" || true; }

T0=$($PY -c "import time;print(time.time())")
BACKGROUND_ROOT=${BACKGROUND_ROOT:-$EXTERNAL_STORAGE_ROOT/background}
RECOVERY_DIR=${RECOVERY_DIR:-$BACKGROUND_ROOT/run-recovery/${LABEL}-${T0//./_}}
snapshot_core() {
  mkdir -p "$RECOVERY_DIR"
  for artifact in trajectory.json phases.json stage_timing.json real_timing.json \
                  d455_topdown_dump* observations dual_gn; do
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
organize_artifacts() {
  # Keep the trial root presentation-ready: videos at top level, supporting
  # records in named subfolders. Recorders work in the root while live because
  # their control, dump, and annotation paths must be atomic; organization is
  # the final cleanup step after every process has exited.
  mkdir -p "$OUT/metadata" "$OUT/logs" "$OUT/perception" "$OUT/runtime"
  # Move perception bundles before the general JSON rule so each dump's
  # intrinsics file stays beside its color/depth data.
  for artifact in "$OUT"/d455_topdown_dump* "$OUT"/*_frames \
                  "$OUT/observations" "$OUT/dual_gn"; do
    [ -e "$artifact" ] && mv "$artifact" "$OUT/perception/"
  done
  for artifact in "$OUT"/*.json; do
    [ -e "$artifact" ] && mv "$artifact" "$OUT/metadata/"
  done
  for artifact in "$OUT"/*.log; do
    [ -e "$artifact" ] && mv "$artifact" "$OUT/logs/"
  done
  for artifact in "$OUT"/*.ready "$OUT"/*.ctrl; do
    [ -e "$artifact" ] && mv "$artifact" "$OUT/runtime/"
  done
  mkdir -p "$OUT/runtime/misc"
  for artifact in "$OUT"/*; do
    name=$(basename "$artifact")
    case "$name" in
      *.mp4|metadata|logs|perception|runtime|sim) ;;
      *) mv "$artifact" "$OUT/runtime/misc/" ;;
    esac
  done
  rmdir "$OUT/runtime/misc" 2>/dev/null || true
}
FINISHED=0
ORGANIZED=0
P1=""
P2=""
P3=""
cleanup() {
  code=$?
  if [ "$FINISHED" != "1" ] && [ -f "$OUT/phases.json" ]; then
    mark failed "pipeline exited with code $code"
  fi
  snapshot_core
  for camera in webcam_scene d415_scene d455_topdown; do
    if [ -d "$OUT/$camera.ctrl" ]; then touch "$OUT/$camera.ctrl/stop"; fi
  done
  if [ -n "$P3" ] && kill -0 "$P3" 2>/dev/null; then kill "$P3" 2>/dev/null || true; fi
  wait $P1 $P2 2>/dev/null || true
  if [ -n "$P3" ]; then wait "$P3" 2>/dev/null || true; fi
  if [ "$ORGANIZED" != "1" ]; then
    restore_core
    organize_artifacts
  fi
  $PY -m isaacgymenvs.open_loop.trial_timing aggregate "$(dirname "$OUT")" >/dev/null || true
}
trap cleanup EXIT
$PY - "$OUT" "$T0" <<'EOF'
import sys
from pathlib import Path
from isaacgymenvs.open_loop.trial_timing import write_json_atomic, write_trial_timing
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
    --label "$VIDEO_LABEL" --subtitle "$CKPT_LABEL" --exec-label "$EXEC_LABEL" \
    > "$OUT/webcam.log" 2>&1 &
P1=$!
elif [ "$THIRD" = "d415_scene" ]; then
$PY "$FORT/record_cameras.py" --serial $D415 --name d415_scene   --out "$OUT" --t0 "$T0" \
    --label "$VIDEO_LABEL" --subtitle "$CKPT_LABEL" --exec-label "$EXEC_LABEL" \
    > "$OUT/d415.log" 2>&1 &
P1=$!
fi
# no --calib: the projected workspace box clutters the top-down view, which is
# the one view where the clutter itself is the point
$PY "$FORT/record_cameras.py" --serial $D455 --name d455_topdown --out "$OUT" --t0 "$T0" --depth \
    --fast-dumps \
    --label "Perception camera - top-down" --subtitle "$CKPT_LABEL" --exec-label "$EXEC_LABEL" --layout band \
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
# ---- 1. perceive (through the recorder; it owns the camera) -----------------
# The teacher_full_obs runner owns every perception, including decision zero,
# after moving the arm to PMBS home. A separate preflight image would duplicate
# work and would not satisfy that baseline's full-observation contract.
if [ "$CONTROLLER" = "teacher_full_obs" ]; then
  T_PERCEIVE=0
else
  tp=$($PY -c "import time;print(time.time())")
  mark perceive "capturing clean initial scene"
  $PY open_loop/perceive_scene.py --from-dump "$OUT" --dump-name d455_topdown \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" --out-dir "$REAL2SIM_DIR" --scene-id 0
if [ "$EXPECTED_OBJECTS" -gt 0 ]; then
  $PY - "$REAL2SIM_DIR/000000_meta.json" "$EXPECTED_OBJECTS" <<'EOF'
import json, sys
meta_path, expected = sys.argv[1], int(sys.argv[2])
objects = json.load(open(meta_path))["objects"]
observed = [obj for obj in objects if not obj.get("pad", False)]
targets = [obj for obj in observed if obj.get("target", False)]
print(f"[demo] fresh perception: {len(observed)} observed objects, {len(targets)} target")
if len(observed) != expected:
    raise SystemExit(
        f"Fresh perception found {len(observed)} real objects; expected {expected}. "
        "Stopping before twin solve or robot motion."
    )
if len(targets) != 1:
    raise SystemExit(
        f"Fresh perception found {len(targets)} targets; expected exactly one. "
        "Stopping before twin solve or robot motion."
    )
EOF
fi
  T_PERCEIVE=$($PY -c "import time;print(round(time.time()-$tp,3))")
fi

KEEP_ARGS=()
if [ "$KEEP_TARGET" = "1" ]; then KEEP_ARGS=(--keep); fi
if [ "$CONTROLLER" = "spiral" ]; then
  # The baseline has no teacher prior. It enters the same perception/GN and
  # robot runtime immediately after the clean preflight observation.
  mark spiral-load "warming spiral perception and grasp network"
  $PY open_loop/run_spiral.py $EXEC --yes \
      --initial-dump "$OUT/d455_topdown_dump0" \
      --resense-dir "$OUT" --resense-name d455_topdown \
      --max-steps "$SPIRAL_MAX_STEPS" --timeout-s "$SPIRAL_TIMEOUT_S" \
      --max-travel-m "$SPIRAL_MAX_TRAVEL_M" \
      --retract-backoff-m "$SPIRAL_BACKOFF_M" \
      --arc-step-m "$SPIRAL_ARC_STEP_M" --radial-step-m "$SPIRAL_RADIAL_STEP_M" \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" \
      "${KEEP_ARGS[@]}" --trace-out "$OUT/trajectory.json" \
      --timing-out "$OUT/real_timing.json" &
  P3=$!
  T_TWIN=0
elif [ "$CONTROLLER" = "pmbs" ]; then
  # PMBS re-senses the full scene from PMBS home before every decision and searches
  # one push in its own parallel simulator; no teacher twin solve.
  mark pmbs-load "warming PMBS perception, grasp networks and simulator"
  # Lower priority and bounded CPU threads: the 800-environment search otherwise starves
  # the D455 recorder, which then captures ~7 fps against its 30 fps target.
  (cd "$GYM/pmbs_baseline" && OMP_NUM_THREADS=${PMBS_THREADS:-6} MKL_NUM_THREADS=${PMBS_THREADS:-6} \
      nice -n ${PMBS_NICE:-10} $PY -u run_pmbs_real.py $EXEC --yes \
      --initial_dump "$OUT/d455_topdown_dump0" \
      --resense_dir "$OUT" --resense_name d455_topdown \
      --num_envs "$PMBS_NUM_ENVS" --time_limit "$PMBS_TIME_LIMIT" \
      --max_actions "$PMBS_MAX_ACTIONS" --retract_backoff_m "$PMBS_BACKOFF_M" \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" \
      "${KEEP_ARGS[@]}" --trace_out "$OUT/trajectory.json" \
      --timing_out "$OUT/real_timing.json") &
  P3=$!
  T_TWIN=0
elif [ "$CONTROLLER" = "teacher_full_obs" ]; then
  # No initial twin solve and no fixed action list: every teacher action is
  # recomputed from a new 11-object observation captured at PMBS home.
  $PY open_loop/run_teacher_full_obs.py $EXEC --yes \
      --checkpoint "$CKPT" --network-json "$BUNDLE/teacher_network.json" \
      --checkpoint-sha256 "$TEACHER_SHA256" \
      --resense-dir "$OUT" --resense-name d455_topdown \
      --expected-objects "$EXPECTED_OBJECTS" \
      --max-steps "$TEACHER_FULL_OBS_MAX_STEPS" \
      --retract-backoff-m "$TEACHER_BACKOFF_M" \
      --timeout-s "$TEACHER_FULL_OBS_TIMEOUT_S" \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" \
      "${KEEP_ARGS[@]}" --trace-out "$OUT/trajectory.json" \
      --timing-out "$OUT/real_timing.json" &
  P3=$!
  T_TWIN=0
else
  # Warm the student, Mask R-CNN, and stock GN while the digital twin solves.
  # The ready-file handshake prevents the runner from consuming a stale plan.
  WARM_READY="$OUT/student_warm.ready"
  TRAJ_READY="$OUT/trajectory.ready"
  rm -f "$WARM_READY" "$TRAJ_READY"
  $PY open_loop/run_student.py --traj "$TRAJ" --student "$STUDENT" \
      --student-sha256 "$STUDENT_SHA256" $EXEC --yes \
      --traj-ready-file "$TRAJ_READY" --warm-ready-file "$WARM_READY" \
      --initial-dump "$OUT/d455_topdown_dump0" \
      --resense-dir "$OUT" --resense-name d455_topdown --max-steps "$MAX_STEPS" \
      --student-timeout-s "$STUDENT_TIMEOUT_S" \
      --teacher-distance-margin-m "$TEACHER_DISTANCE_MARGIN_M" \
      --calib "$CALIB" --maskrcnn "$MASKRCNN" \
      "${KEEP_ARGS[@]}" --timing-out "$OUT/real_timing.json" &
  P3=$!

  # ---- 2. solve in the twin ------------------------------------------------
  mark twin "loading Isaac Gym"
  ts=$($PY -c "import time;print(time.time())")
  PHASE_DIR="$OUT" CKPT="$CKPT" "$SOLVER" real 0 > "$OUT/solve.log" 2>&1
  T_TWIN=$($PY -c "import time;print(round(time.time()-$ts,3))")
  tail -2 "$OUT/solve.log"
  # solve_in_twin records unsuccessful/OOW rollouts for diagnostics and exits 0.
  # Never publish one to the waiting hardware process.
  $PY - "$SHARED_TRAJ" <<'EOF'
import json, sys
path = sys.argv[1]
doc = json.load(open(path))
meta = doc.get("metadata", {})
if meta.get("solved") is not True or meta.get("oow_violation"):
    raise SystemExit(
        f"Teacher did not produce a valid plan: solved={meta.get('solved')} "
        f"oow={meta.get('oow_violation')} steps={meta.get('num_steps')}. "
        "Stopping before robot motion."
    )
if not doc.get("dense"):
    raise SystemExit("Teacher produced an empty plan. Stopping before robot motion.")
print(f"[demo] teacher plan validated: {len(doc['dense'])} primitives")
EOF
  # Shared solver output may be overwritten by another process. Preserve the
  # exact teacher prior this physical trial will consume before student startup.
  cp "$SHARED_TRAJ" "$TRAJ"
  touch "$TRAJ_READY"
fi

# ---- 3. closed loop on the robot --------------------------------------------
wait "$P3"
P3=""
if [ "$EXEC" != "--execute" ]; then mark dry-run "$CONTROLLER checked without motion"; fi
FINISHED=1
snapshot_core

echo "[demo] stopping recorders"
for n in $CAMS; do touch "$OUT/$n.ctrl/stop"; done
wait $P1 $P2 2>/dev/null || true
restore_core

# ---- 4. POST-PROCESSING: 4K sim videos, after the run is over --------------
if [ "$CONTROLLER" = "student" ] && [ "${SKIP_RENDER:-0}" != "1" ]; then
  echo "[demo] rendering 4K sim videos (post-processing)"
  $PY -u tools/record_solve.py task=MoreOpenLoop train=$TRAIN test=True headless=True \
      wandb_activate=False checkpoint=$CKPT \
      sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 \
      +traj_json="$TRAJ" +out_dir="$OUT/sim" +budget=140 "+scene_label=$LABEL" \
      "+phases_json=$OUT/phases.json" \
      > "$OUT/render.log" 2>&1 || tail -5 "$OUT/render.log"
fi

$PY - "$OUT" "$T_PERCEIVE" "$T_TWIN" "$CONTROLLER" <<'EOF'
import json, os, sys, glob, re
out, t_perceive, t_twin, controller = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
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
print(f"\n=== {controller.upper()} END-TO-END (single recorded take) ===")
if controller == "teacher_full_obs":
    stage = real.get("stage_seconds", {})
    rows = [("model load", stage.get("model_load")),
            ("initial experiment home", stage.get("initial_home")),
            ("retract to PMBS (total)", stage.get("retract")),
            ("full observations (total)", stage.get("capture_perception")),
            ("stock GN calls (total)", stage.get("stock_gn")),
            ("teacher inference (total)", stage.get("teacher_inference")),
            ("audit artifacts (total)", stage.get("audit_artifacts")),
            ("return to push (total)", stage.get("return_to_push")),
            ("teacher pushes (total)", stage.get("push")),
            ("grasp", stage.get("grasp")),
            ("return home", stage.get("return_home")),
            ("actions executed", real.get("steps_run"))]
else:
    rows = [("perceive (camera -> scene)", t_perceive),
            ("twin total wall-clock", t_twin if controller == "student" else None)]
    if controller == "student" and new_solver_clock:
        rows += [("  Isaac Gym + teacher load", span("twin", "twin-settle")),
                 ("  scene settling", span("twin-settle", "solve")),
                 ("  PPO rollout simulated", rollout_sim()),
                 ("  PPO rollout wall-clock", span("solve", "solve-export")),
                 ("  plan finalization", span("solve-export", "solve-done"))]
    elif controller == "student":
        rows += [("  legacy combined solve", span("solve", "solve-done"))]
    rows += [
            (f"{controller} closed loop", real.get("student_seconds",
                                                    real.get("spiral_seconds"))),
            ("  actions executed", real.get("steps_run")),
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
    state = ("HOLDING" if real["holding"] else
             "DROPPED DURING LIFT" if real.get("dropped_during_lift") else "EMPTY")
    print(f"  gripper                     : {real.get('gripper_position')}/255 {state} "
          f"(closed at {real.get('grasp_close_position')}, OBJ "
          f"{real.get('grasp_close_object_status')} -> {real.get('gripper_object_status')})")
print(f"  stock GN threshold         : {real.get('stock_gn_threshold')}")
print(f"  measured planar TCP travel : {real.get('travelled_m')} m")
for n in ("webcam_scene", "d415_scene", "d455_topdown"):
    m = load(os.path.join(out, n + "_rec.json"))
    if m:
        print(f"  video {n:21s}: {m.get('frames')} frames, {m.get('duration_s')} s")
print("  sim videos                  :", len(glob.glob(os.path.join(out, "sim", "*.mp4"))))
stages = load(os.path.join(out, "stage_timing.json"))
print("  measured end-to-end         :", stages.get("end_to_end_seconds"), "s")
print("  per-stage timing JSON       :", os.path.join(out, "metadata", "stage_timing.json"))
EOF
organize_artifacts
ORGANIZED=1
if [ "${SKIP_POSTPROCESS:-0}" != "1" ]; then
  case "$CONTROLLER" in
    student) COMPOSER=compose_closed_loop_memory.py ;;
    teacher_full_obs) COMPOSER=compose_teacher_closed_loop.py ;;
    spiral|pmbs) COMPOSER=compose_spiral_closed_loop.py ;;
  esac
  $PY "$FORT/video_postprocess/$COMPOSER" "$OUT" "$OUT/review.mp4"
fi
echo "[demo] done -> $OUT"
