# open_loop — perceive once, solve in the twin, replay blind

The post-IROS26 pipeline: one external RGB-D camera builds a digital twin
in Isaac Gym, a trained policy solves the whole scene in simulation, and the
recorded end-effector path is replayed on the real UR5e with **no online
feedback** (stock 2F-85, no wrist cameras). It is solver-agnostic: the IROS26
PPO checkpoint (ep 270) is the *blind open-loop lower bound*; the teacher /
student policies plug into the same three stages.

```
 real RGB-D ──1──▶ twin scene file ──2──▶ trajectory JSON ──3──▶ UR5e (moveL, blind)
        perceive_scene.py     solve_in_twin.py      execute_trajectory.py
                                    │
                                    └──2b──▶ replay_in_twin.py   (policy OUT of the loop:
                                             nominal reproduction + perturbed-replica certificate)
```

Shared: `frames.py` (workspace limits, sim/real/pixel/render conversions —
single source of truth), `trajectory.py` (JSON format + Douglas-Peucker
path simplification), `tasks/more_open_loop.py` (the `MoreOpenLoop` task:
per-env recording, settle-freeze, freeze, replay mode, perturbation, final
grasp). Everything below runs from `isaacgymenvs/` in the `pmbs` conda env
with `LD_LIBRARY_PATH=${CONDA_PREFIX}/envs/pmbs/lib`.

## 1. Perceive (`perceive_scene.py`)

PMBS Mask R-CNN (6 block classes; `the PMBS release's logs_image/nmaskrcnn10.pth`
loads strictly with the PMBS anchor/mask-head architecture) → purple HSV band
flags the target → each mask is deprojected through the eye-to-hand
calibration into the sim frame → pose = minAreaRect centre + brute-force
rotation IoU of the **true mesh silhouette** (`assets/.../blocks-more/*.obj`
top face; full 360° for concave/triangle, so flips are resolved) →
**de-penetration** nudges touching footprints apart to ≥ 1 mm clearance
(Isaac Gym hurls interpenetrating blocks at reset) → standard More test-case
file `test-cases/real2sim/000000.txt` (+ `_meta.json`, `_debug.png` overlay).

```bash
# live capture (external RealSense; 1280x720 aligned)
python open_loop/perceive_scene.py --capture --calib open_loop/calib/<cam2base>.txt \
    --maskrcnn the PMBS release parallel_mcts/logs_image/nmaskrcnn10.pth
# offline (a recorded single-camera capture d415-*: color png, depth png in mm, config.json intrinsics)
python open_loop/perceive_scene.py --color $D/color/0000-color.png --depth $D/depth/0000-depth.png \
    --intrinsics $D/config.json --calib "$TRACE_CALIB" --maskrcnn "$TRACE_MASKRCNN"
```

Verified offline on `a recorded single-camera capture d415-initial-scene`
with the PMBS D415 calibration: 11/11 detections, silhouette IoU 0.88–0.97,
de-penetration ≤ 3 mm, twin settle displacement 13 mm.

## 2. Solve (`solve_in_twin.py`)

Deterministic policy rollout in `MoreOpenLoop`, one env per scene file
(`num_envs` is clamped to the number of files: 1 for the real scene, 128 for
a suite). Before the policy sees the scene, a **settle-freeze** pass
(`+settle_steps=30`, gen-v1 recipe) lets residual interpenetration pop and
freezes the settled poses; the per-scene displacement is reported (>2 cm =
perception inconsistent). Each env runs its FIRST episode only (frozen when
done): stop on graspable (Q > 0.9), on an out-of-workspace event (paper OOW
rule → failure; `+oow_rule=False` keeps pushing, as the real robot would),
or at the step budget. Solved envs get the final grasp from the tiled
16-rotation GPN (closed-loop `utils/mtcs_utils` helper; the render uses the
shared heightmap convention, asserted against the target centroid each time).

```bash
CK=runs/16-mps-gpn-512-mb-8192-rew-backtrack_14-21-12-56/nn/last_16-mps-gpn-512-mb-8192-rew-backtrack_ep_270_rew_-119.44291.pth
# the perceived real scene
python open_loop/solve_in_twin.py task=MoreOpenLoop test=True headless=True checkpoint=$CK \
    +trajectory_out=open_loop/out/real2sim
# a sim suite (lower-bound study)
python open_loop/solve_in_twin.py task=MoreOpenLoop test=True headless=True num_envs=128 \
    task.env.test_cases.scene_root_dir=test-cases/dataset/selected task.env.test_cases.difficulty_choice=test-128 \
    checkpoint=$CK +trajectory_out=open_loop/out/test-128-ep270
```

Output: `<trajectory_out>/<scene>.json` per scene + `summary.json`.

### Trajectory JSON (`trajectory.py`, version 2)

| field | meaning | use |
| --- | --- | --- |
| `dense` | ACTUAL EEF sim position every RL step | fidelity reference, `--mode dense` (servoL) |
| `executed` | `dense` simplified to straight segments (Douglas-Peucker, 1.5 mm) | **default** replay path (moveL) |
| `waypoints` | the primitive wp1/wp2 the policy COMMANDED | analysis (`--mode commanded`) |
| `grasp` | GPN grasp: sim xy, render pixel, rotation bin, Q | final grasp |
| `final_target` | target (x, y, yaw) at the end of the solve | replay fidelity metric |
| `metadata` | solved / oow_violation / num_steps / settle_disp_mm / … | bookkeeping |

Why `executed` ≠ `waypoints`: in the twin the IK/PD controller reaches only
~3 mm of every 1 cm primitive phase before the policy issues the next one
(control_freq_inv = 1), so the commanded list overshoots what the arm
actually did. The real robot (moveL) reaches its targets — it must be given
the path the twin *executed*, not the one the policy *asked for*.

## 2b. Replay in the twin (`replay_in_twin.py`) — the test that needs no robot

For each solved trajectory: N replicas of its scene in one batched sim,
replica 0 nominal, replicas 1..N-1 with perception pose noise (±3 mm, ±2°)
+ friction/mass DR; every replica's EEF is servoed through the SAME
absolute path (the policy is bypassed), exactly what `execute_trajectory.py`
sends to the UR5e. Reports per trajectory: `nominal_ok` (does replaying
the recorded motion reproduce the solve?), `target_err_mm`, and
`certificate` = fraction of perturbed replicas ending graspable without OOW
= the lockstep certificate of the method description (§3.A) / predicted blind
open-loop success.

```bash
python open_loop/replay_in_twin.py task=MoreOpenLoop test=True headless=True \
    +replay_dir=open_loop/out/test-128-ep270 +replay_n=16 [+replay_mode=executed|dense|commanded] \
    [+replay_max=32 +replay_skip=0]   # chunk large suites (M*N envs per run)
```

The same machinery certifies a student policy's plan: `MoreOpenLoop.set_replay`
+ `perturb` are the primitives (the method description (§3.10) item 3).

## 3. Execute (`execute_trajectory.py`)

**Dry run by default** — prints the real-frame plan and bounds-checks every
point. `--probe` connects READ-ONLY and compares the live TCP with the
trajectory's expected start pose (the frame-offset check). `--execute`
asks for confirmation, then: [`--home` moveJ] → close fingers → hover over
the start → descend to `--push-z` → blended moveL through the `executed`
path → lift → orient per the grasp rotation bin → open → descend → close →
lift.

```bash
python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json            # dry run
python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json --probe    # read-only TCP check
python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json --execute --home
```

## Hardware checklist (before the first real run)

1. **Eye-to-hand calibration for the mounted camera** — `open_loop/calib/`
   holds the Oct-2025 PMBS D415 file used for the offline test; produce a
   fresh one with a script adapted from `the PMBS release's pmbs_calibrate_eye_to_hand.py`
   (`trace/hardware/calib_utils.calibrate_eye_hand(eye_to_hand=True)`).
2. **Frame offset** — `--probe` at `HOME_JOINTS_DEG`: the delta between the
   live TCP and the trajectory start is `frames.REAL_FRAME_OFFSET` (expected
   ≈ 0: both frames are attached to the robot base; see the CAUTION in
   `frames.py` about the 48 mm tape-vs-sim-box offset, which is NOT a frame
   offset).
3. **`--push-z`** (default 0.020) for the closed stock 2F-85.
4. Eyeball `test-cases/real2sim/000000_debug.png` against a non-headless
   solve (`headless=False`) once.

## Offline tests

```bash
python3 open_loop/test_offline_core.py   # frames / trajectory / executor numerics, no isaacgym needed
```
