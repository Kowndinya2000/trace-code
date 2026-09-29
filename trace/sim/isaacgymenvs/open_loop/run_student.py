"""Closed-loop STUDENT on the real robot: perceive -> student -> one primitive, repeat.

The open-loop pipeline (execute_trajectory.py) replays the twin's trajectory
blind. This runs the Stage-2 student instead: every step it re-perceives the
real scene with the D455, builds the 166-D token observation the student was
trained on (the contract is implemented in open_loop/student_obs.py), asks the student for a primitive, executes
that ONE primitive, and feeds the executed primitive back as the next
previous-action input. The twin's nominal rollout is still needed -- it is the
plan prior the student corrects against -- so the trajectory JSON must carry
per-step object centres in its ``obj_xy`` field.

Every piece of the observation contract was verified against the simulator:
  * rectangle corners from (class, x, y, yaw) match blocks_rect_rotated to the
    settle displacement (0.4-4.7 mm) on all non-round blocks;
  * the 94-D teacher vector reconstructed from those corners + EEF + wall
    distances matches env.obs_buf to 0.000 mm;
  * the numpy port of build_two_step_plan sums to primitive_vectors(0.04) to
    0.000 mm.

The D455 supplies partial observations while the arm remains at push height.
No grasp classifier runs while the student pushes. The student sees the
partial observation at every primitive and continues until its measured
teacher-distance budget or safety horizon ends. The robot then retracts once
to PMBS home and runs the original grasp network on one clean observation.

  python open_loop/run_student.py --traj open_loop/out/real2sim/000000.json \
      --student "$TRACE_DATA/checkpoints/trace_r3.pt" \
      --student-sha256 dda9db7dc06f30bce5aa9ab301e05fc6781641c0f3ae450af3e50948c7ef9d30 \
      --initial-dump <clean-dump-base> \
      --resense-dir <recorder dir> --execute --yes
  python open_loop/run_student.py --traj ... --student ... \
      --offline-dump <base>       # no robot, no camera: obs path only
"""
import argparse
import atexit
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.open_loop import hardware_config
sys.path.insert(0, str(hardware_config.hardware_dir()))
from isaacgymenvs.open_loop import frames, recorder_io, real_grasp
from isaacgymenvs.open_loop import perceive_scene as ps
from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop.evaluation_core import primitive_vectors
from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory
from isaacgymenvs.open_loop.execute_trajectory import (
    HOME_JOINTS_DEG, PMBS_HOME_JOINTS_DEG, STAGING_SIM_XY)

# tasks/more.py name_id2_dims_map, keyed by class NAME: Mask R-CNN's ids
# (1 concave, 2 cube, 3 cylinder, ...) and the sim's (0 concave, 1 cylinder,
# 2 cube, ...) do not agree, so nothing here goes through an integer id.
DIMS = {"concave": (0.090, 0.045), "cylinder": (0.045, 0.045), "cube": (0.045, 0.045),
        "half-cube": (0.0225, 0.045), "rect": (0.045, 0.090), "triangle": (0.045, 0.085)}
WS_X, WS_Y = frames.SIM_WORKSPACE_LIMITS[0], frames.SIM_WORKSPACE_LIMITS[1]


def rect_corners(name, x, y, yaw):
    """(4, 2) sim-frame corners in the sim's order, rotated about the centre."""
    w, h = DIMS[name]
    base = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    c, s = math.cos(yaw), math.sin(yaw)
    return base @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def two_step_plan(a, total=0.04):
    """numpy port of More.build_two_step_plan: action -> (d1, d2), metres."""
    s = total / 2.0
    ds = s / math.sqrt(2)
    card = np.array([[0, s], [s, 0], [0, -s], [-s, 0]])
    if a < 4:
        return card[a].copy(), card[a].copy()
    k = a - 4
    ddir, mode = k // 3, k % 3
    diag = np.array([[ds, ds], [-ds, ds], [ds, -ds], [-ds, -ds]])
    E, W, N, S = np.array([s, 0.0]), np.array([-s, 0.0]), np.array([0.0, s]), np.array([0.0, -s])
    first, second = [E, W, E, W], [N, N, S, S]
    if mode == 0:
        return diag[ddir].copy(), diag[ddir].copy()
    if mode == 1:
        return first[ddir].copy(), second[ddir].copy()
    return second[ddir].copy(), first[ddir].copy()


# self-check: the port must reproduce the library's endpoints exactly
_pv = primitive_vectors(0.04)
for _a in range(16):
    _d1, _d2 = two_step_plan(_a)
    assert np.abs(_d1 + _d2 - _pv[_a]).max() < 1e-9, _a


def teacher_vector(tracks, eef):
    """94-D raw teacher observation from tracked objects (identity permutation)."""
    toks = [(rect_corners(t["name"], t["x"], t["y"], t["yaw"]) - eef).reshape(8) for t in tracks]
    walls = np.array([eef[0] - WS_X[0], WS_X[1] - eef[0], eef[1] - WS_Y[0], WS_Y[1] - eef[1]])
    return np.concatenate(toks + [eef, walls]).astype(np.float32)


def associate(tracks, dets, gate):
    """Greedy nearest-centre, same class, one detection per track. -> vis (11,)"""
    vis = np.zeros(len(tracks), np.float32)
    used = set()
    order = sorted(range(len(tracks)), key=lambda i: 0 if tracks[i]["target"] else 1)
    for i in order:
        tr = tracks[i]
        best, bd = None, gate
        for j, d in enumerate(dets):
            if j in used or d["name"] != tr["name"]:
                continue
            if tr["target"] != bool(d.get("target", False)) and d.get("target") is not None \
                    and any(x["target"] for x in dets):
                continue           # a confirmed target detection only feeds the target track
            dist = math.hypot(d["x"] - tr["x"], d["y"] - tr["y"])
            if dist < bd:
                best, bd = j, dist
        if best is not None:
            used.add(best)
            d = dets[best]
            tr["x"], tr["y"], tr["yaw"] = float(d["x"]), float(d["y"]), float(d["yaw"])
            vis[i] = 1.0
    return vis


def load_plan(traj):
    """Collector convention: plan[t] is the state BEFORE action t."""
    dense = traj.dense
    if not dense or "obj_xy" not in dense[0]:
        sys.exit("trajectory has no per-step obj_xy — re-solve it with the current "
                 "solve_in_twin, which records object centres")
    scene = [l.split() for l in open(traj.scene_file) if l.strip()]
    init_xy = [[float(l[4]), float(l[5])] for l in scene]
    plan_xy = [list(traj.start_eef[:2])] + [list(d["eef"][:2]) for d in dense[:-1]]
    plan_obj = [init_xy] + [d["obj_xy"] for d in dense[:-1]]
    plan_act = [int(d["action"]) for d in dense]
    tracks = [{"name": l[0].split(".")[0], "x": float(l[4]), "y": float(l[5]),
               "yaw": float(l[9]), "target": k == 0} for k, l in enumerate(scene)]
    return plan_xy, plan_obj, plan_act, tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj", required=True, help="twin plan JSON (with obj_xy)")
    ap.add_argument("--traj-ready-file", default=None,
                    help="wait for this marker before loading --traj; lets model warmup "
                         "overlap the twin solve")
    ap.add_argument("--warm-ready-file", default=None,
                    help="write this marker after all perception/policy/GN models are warm")
    ap.add_argument("--student", required=True, help="selected student checkpoint (.pt)")
    ap.add_argument("--student-sha256", default=None,
                    help="required SHA256 for the student; reject a stale or changed file")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--offline-dump", default=None,
                    help="<base> of a saved *_color.png/_depth.npy/_K.json: run the obs "
                         "path with no robot and no camera, EEF follows the plan")
    ap.add_argument("--initial-dump", default=None,
                    help="clean pre-motion D455 dump base used for step 0")
    ap.add_argument("--resense-dir", default=None, help="record_cameras.py dir (owns the D455)")
    ap.add_argument("--resense-name", default="d455_topdown")
    ap.add_argument("--calib", default=str(hardware_config.calibration()))
    ap.add_argument("--maskrcnn", default=str(hardware_config.maskrcnn()))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"),
                    help="inference device; cpu is intended for offline validation")
    ap.add_argument("--object-depth-min-z", type=float, default=0.025,
                    help="minimum calibrated base-frame Z for block-mask pixels")
    ap.add_argument("--object-depth-max-z", type=float, default=0.065,
                    help="maximum calibrated base-frame Z for block-mask pixels")
    ap.add_argument("--object-depth-min-fraction", type=float, default=0.80,
                    help="fraction of mask depth required inside the block-height band")
    ap.add_argument("--robot-ip", default=hardware_config.robot_ip())
    ap.add_argument("--max-steps", type=int, default=0,
                    help="episode horizon; 0 = 120, the simulator's evaluation horizon")
    ap.add_argument("--student-timeout-s", type=float, default=100.0,
                    help="stop continuous student execution after this many seconds; "
                         "the terminal clean grasp evaluation still runs")
    ap.add_argument("--teacher-distance-margin-m", type=float, default=0.05,
                    help="hard planar TCP-travel margin beyond teacher nominal distance")
    ap.add_argument("--assoc-gate", type=float, default=0.06, help="m, per-step association")
    ap.add_argument("--push-z", type=float, default=0.020)
    ap.add_argument("--safe-z", type=float, default=0.22)
    ap.add_argument("--tool-vel", type=float, default=0.28, help="push, = twin EEF speed")
    ap.add_argument("--tool-acc", type=float, default=1.20)
    ap.add_argument("--transit-vel", type=float, default=0.30)
    ap.add_argument("--transit-acc", type=float, default=1.20)
    ap.add_argument("--descend-m", type=float, default=0.040)
    ap.add_argument("--table-z", type=float, default=0.006)
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep a successfully grasped target held at PMBS home")
    args = ap.parse_args()

    max_steps = args.max_steps or 120
    if not np.isfinite(args.student_timeout_s) or args.student_timeout_s <= 0:
        raise SystemExit("--student-timeout-s must be finite and positive")
    if (not np.isfinite(args.teacher_distance_margin_m)
            or args.teacher_distance_margin_m < 0):
        raise SystemExit("--teacher-distance-margin-m must be finite and nonnegative")
    student_sha256 = hashlib.sha256(Path(args.student).read_bytes()).hexdigest()
    if args.student_sha256 and student_sha256.lower() != args.student_sha256.lower():
        raise SystemExit(f"student SHA256 mismatch: expected {args.student_sha256}, "
                         f"got {student_sha256} ({args.student})")

    from isaacgymenvs.tools.verify_checkpoint_bundle import load_student
    net, mask, meta = load_student(args.student)
    P = torch.from_numpy(primitive_vectors(0.04))
    head = meta.get("head", "xy")
    print(f"student: {os.path.basename(args.student)} sha256={student_sha256[:12]} "
          f"head={head} mask={'none' if mask is None else 'yes'}")

    cam2base = np.loadtxt(args.calib)
    dev = torch.device(args.device)
    maskrcnn = ps.load_maskrcnn(args.maskrcnn, dev)
    from isaacgymenvs.utils.mtcs_utils import MCTSHelper
    helper = MCTSHelper("logs_grasp/snapshot-post-020000.reinforcement.pth",
                        "logs_grasp/grasp_model-89.pth", device=args.device)
    print("grasp evaluation: one clean terminal stock GN at PMBS home")

    def load_dump(base):
        import cv2
        color = cv2.imread(base + "_color.png")
        if color is None:
            raise FileNotFoundError(base + "_color.png")
        depth = np.load(base + "_depth.npy")
        with open(base + "_K.json") as f:
            K = json.load(f)
        return color, depth, K, base

    def analyze_frame(color, depth, K, base=None, filter_mid_execution=False):
        """Run Mask R-CNN once and filter only while the arm is in view."""
        raw = ps.segment(color, args.maskrcnn, dev, model=maskrcnn)
        if filter_mid_execution:
            kept, rejected, mask_diag = ps.filter_instances_by_depth(
                raw, depth, K, cam2base,
                min_z=args.object_depth_min_z,
                max_z=args.object_depth_max_z,
                min_fraction=args.object_depth_min_fraction)
        else:
            kept, rejected, mask_diag = raw, [], []
        objects, _ = ps.locate_objects(
            color, depth, K, cam2base, table_z=args.table_z,
            device=dev, require_target=False, instances=kept)
        return {"color": color, "depth": depth, "K": K, "base": base,
                "raw": raw, "kept": kept, "rejected": rejected,
                "mask_filter_applied": bool(filter_mid_execution),
                "mask_diagnostics": mask_diag, "objects": objects}

    def perceive(base=None, filter_mid_execution=False):
        if base:
            color, depth, K, dump_base = load_dump(base)
        elif args.offline_dump:
            color, depth, K, dump_base = load_dump(args.offline_dump)
        elif args.resense_dir:
            color, depth, K, dump_base = recorder_io.request_dump(
                args.resense_dir, args.resense_name)
        else:
            color, depth, K = ps.capture_realsense(warmup=8)
            dump_base = None
        return analyze_frame(color, depth, K, dump_base,
                             filter_mid_execution=filter_mid_execution)

    def stock_grasp(pack):
        return real_grasp.compute_hardware_grasp(
            pack["color"], pack["depth"], pack["K"], cam2base, args.maskrcnn,
            device=args.device, helper=helper, maskrcnn=maskrcnn,
            instances=pack["kept"])

    # ------------------------------------------------------------ robot
    live = args.execute and not args.offline_dump
    if live and not args.initial_dump:
        raise SystemExit("--initial-dump is required for execution")

    # Load and validate the clean initial observation before opening an RTDE
    # connection, so a perception/configuration failure cannot move the robot
    # and then abort.
    initial_pack = perceive(args.initial_dump, filter_mid_execution=False)
    initial_targets = [obj for obj in initial_pack["objects"] if obj.get("target")]
    if len(initial_targets) != 1:
        raise SystemExit("initial clean frame has no unambiguous purple/blue target")

    # OOW failure rule shared with the spiral and PMBS hardware runs: any measured block
    # top-face point beyond the black-mat edge, or (on clean views) a missing block.
    from isaacgymenvs.open_loop import mat_boundary
    mat = mat_boundary.detect_mat_polygon(initial_pack["color"], initial_pack["depth"],
                                          initial_pack["K"], cam2base)

    def mat_measure(pack):
        if "mat_measurements" not in pack:
            blocks, _, _ = ps.filter_instances_by_depth(
                pack["raw"], pack["depth"], pack["K"], cam2base,
                min_z=args.object_depth_min_z, max_z=args.object_depth_max_z,
                min_fraction=args.object_depth_min_fraction)
            pack["mat_measurements"] = mat_boundary.instance_mat_measurements(
                blocks, pack["depth"], pack["K"], cam2base, mat["corners_sim"],
                min_z=args.object_depth_min_z, max_z=args.object_depth_max_z)
        return pack["mat_measurements"]

    mat_exempt = mat_boundary.measured_off_mat_counts(mat_measure(initial_pack))
    initial_block_count = mat_boundary.measured_block_count(mat_measure(initial_pack),
                                                            corners_sim=mat["corners_sim"])

    def off_mat(pack):
        counts = mat_boundary.measured_off_mat_counts(mat_measure(pack))
        return {k: v - mat_exempt.get(k, 0) for k, v in counts.items() if v > mat_exempt.get(k, 0)}

    print(f"mat boundary (sim): {np.round(mat['corners_sim'], 4).tolist()}; "
          f"{initial_block_count} blocks on the mat")

    # The heavy models above do not depend on the teacher trajectory. The
    # end-to-end wrapper starts this process alongside the twin solve, warms
    # every network on the already captured clean frame, then publishes the
    # new trajectory atomically and touches --traj-ready-file. This removes the
    # post-solve model-loading pause before the robot begins execution.
    if args.warm_ready_file:
        ready = Path(args.warm_ready_file)
        ready.parent.mkdir(parents=True, exist_ok=True)
        ready.write_text("ready\n")
        print(f"student models warm; waiting for teacher trajectory: {args.traj_ready_file}",
              flush=True)
    if args.traj_ready_file:
        deadline = time.monotonic() + 600.0
        marker = Path(args.traj_ready_file)
        while not marker.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for trajectory marker {marker}")
            time.sleep(0.02)

    traj = OpenLoopTrajectory.load(args.traj)
    if traj.metadata.get("solved") is not True or traj.metadata.get("oow_violation"):
        raise SystemExit("refusing robot execution: teacher trajectory is not a solved, "
                         "in-workspace rollout")
    plan_xy, plan_obj, plan_act, tracks = load_plan(traj)
    n_plan = len(plan_xy)
    teacher_nominal_distance_m = n_plan * 0.04
    distance_limit_m = teacher_nominal_distance_m + args.teacher_distance_margin_m
    print(f"plan: {n_plan} steps from {os.path.basename(traj.checkpoint)}; "
          f"nominal={teacher_nominal_distance_m:.3f}m, "
          f"hard TCP-travel cap={distance_limit_m:.3f}m; final stock-GN at PMBS")
    if args.resense_dir:
        recorder_io.mark(args.resense_dir, "student-load", "warming robot execution")

    if live and not args.yes:
        if input("Run the STUDENT closed-loop on the robot? type 'yes': ").strip() != "yes":
            print("Aborted."); return
    rtde_c = rtde_r = gripper = None
    push_rot = None
    if live:
        from rtde_control import RTDEControlInterface as C
        from rtde_receive import RTDEReceiveInterface as R
        from dashboard_client import DashboardClient as D
        from robotiq_gripper import RobotiqGripper
        d = D(args.robot_ip); d.connect()
        if d.running():
            d.stop(); time.sleep(1.0)
        try:
            d.unlockProtectiveStop(); time.sleep(0.4)
        except Exception:
            pass
        d.disconnect()
        gripper = RobotiqGripper(args.robot_ip, 63352); gripper.connect()
        rtde_r = R(args.robot_ip); rtde_c = C(args.robot_ip)
        gripper.close_and_wait_for_pos(80, 120)
        rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
        tcp = rtde_r.getActualTCPPose()
        from isaacgymenvs.open_loop.hardware_orientation import fixed_downward_rotation
        push_rot = fixed_downward_rotation(tcp[3:6]).tolist()
        sx, sy = frames.sim_to_real(*plan_xy[0])
        rtde_c.moveL([sx, sy, args.push_z, *push_rot], args.transit_vel, args.transit_acc)

    def pose(x_real, y_real, z):
        return [x_real, y_real, z, push_rot[0], push_rot[1], push_rot[2]]

    def eef_sim():
        if live:
            t = rtde_r.getActualTCPPose()
            return np.array(frames.real_to_sim(t[0], t[1]), np.float32)
        return np.array(plan_xy[min(step, n_plan - 1)], np.float32)

    def mark(phase, detail=""):
        if args.resense_dir:
            recorder_io.mark(args.resense_dir, phase, detail)

    def retract():
        """Straight up, fixed staging pose, then the joint move to PMBS home --
        the sequence that never swings through clutter or into a joint stop."""
        nonlocal rtde_c
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        cur = rtde_r.getActualTCPPose()
        rtde_c.moveL(pose(cur[0], cur[1], args.safe_z), args.transit_vel, args.transit_acc)
        sx_, sy_ = frames.sim_to_real(*STAGING_SIM_XY)
        rtde_c.moveL(pose(sx_, sy_, args.safe_z), args.transit_vel, args.transit_acc)
        rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4)

    def capture_clean():
        """A frame with the arm out of shot (call only at PMBS home)."""
        if args.resense_dir:
            return recorder_io.request_dump(args.resense_dir, args.resense_name)
        color, depth, K = ps.capture_realsense(warmup=30)
        return color, depth, K, None

    recovery = {"armed": bool(live)}

    def emergency_return_home():
        """Recover from an unhandled software error after robot motion begins."""
        nonlocal rtde_c
        if not recovery["armed"]:
            return
        recovery["armed"] = False
        try:
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Emergency return skipped because the robot is safety-stopped",
                      flush=True)
                return
            print("Unhandled execution error: retracting and returning to experiment home",
                  flush=True)
            retract()
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4) is False:
                raise RuntimeError("emergency experiment-home move rejected")
            print("Emergency experiment-home return complete", flush=True)
            rtde_c.stopScript()
            rtde_c.disconnect()
        except Exception as exc:
            print(f"EMERGENCY RETURN FAILED: {exc}", flush=True)

    if live:
        atexit.register(emergency_return_home)

    audit_dir = (os.path.join(args.resense_dir, "observations")
                 if args.resense_dir else None)
    if audit_dir:
        os.makedirs(audit_dir, exist_ok=True)

    def save_observation(pack, t):
        """Save the filtered partial observation used by the video annotation."""
        scene_rgb, _, scene_segm = real_grasp.build_real_heightmap(
            pack["color"], pack["depth"], pack["K"], cam2base,
            pack["kept"], target_idx=None)
        if audit_dir:
            h_cam, w_cam = pack["color"].shape[:2]
            kept_masks = (np.stack([i["mask"] for i in pack["kept"]]).astype(np.uint8)
                          if pack["kept"] else np.zeros((0, h_cam, w_cam), np.uint8))
            np.savez_compressed(
                os.path.join(audit_dir, f"camera_masks_step_{t:03d}.npz"),
                kept=kept_masks, rejected=np.zeros((0, h_cam, w_cam), np.uint8),
                scene_rgb=scene_rgb, scene_segm=scene_segm)

    # ------------------------------------------------------------ episode
    stale = np.zeros(len(tracks), np.float32)
    prev_action, h = -1, None
    log, timing = [], {"student": os.path.basename(args.student),
                      "student_sha256": student_sha256, "student_head": head,
                      "plan_steps": n_plan,
                       "stock_gn_threshold": real_grasp.GRASPABLE_Q_THRESHOLD,
                       "student_timeout_s": args.student_timeout_s,
                       "teacher_nominal_distance_m": teacher_nominal_distance_m,
                       "teacher_distance_margin_m": args.teacher_distance_margin_m,
                       "distance_limit_m": distance_limit_m,
                       "object_depth_band_z": [args.object_depth_min_z,
                                                args.object_depth_max_z],
                       "object_depth_min_fraction": args.object_depth_min_fraction,
                       "mat_corners_sim": np.round(mat["corners_sim"], 5).tolist(),
                       "mat_exempt": mat_exempt, "initial_object_count": initial_block_count,
                       "oow_rule": "any measured block top-face point more than 3 mm beyond the "
                                   "black-mat edge, or fewer blocks on the mat in the clean view"}
    t_run0 = time.time()

    def save_result(elapsed):
        """Persist the trial result; safe to call before and after the return-to-home move."""
        timing["oow_failure"] = oow_failure
        timing["real_total_seconds"] = round(elapsed, 3)
        timing["log"] = log
        if args.timing_out:
            from isaacgymenvs.open_loop.trial_timing import write_json_atomic
            write_json_atomic(args.timing_out, timing)

    mark("student", f"0/{max_steps}")
    oow = False
    stop_reason = "horizon"
    steps_run = 0
    travelled_m, travel_tcp_xy = 0.0, None
    for step in range(max_steps):
        elapsed = time.time() - t_run0
        if elapsed >= args.student_timeout_s:
            stop_reason = "student_timeout"
            mark("student", f"internal: stop; timeout_s={args.student_timeout_s:.1f}")
            print(f"Student execution timeout after {elapsed:.1f}s; terminal re-sense")
            break
        t0 = time.time()
        pack = (initial_pack if step == 0 else
                perceive(filter_mid_execution=True))
        t_perc = time.time() - t0
        eef = eef_sim()
        dets = pack["objects"]
        vis = associate(tracks, dets, args.assoc_gate)
        save_observation(pack, step)
        crossed = off_mat(pack)
        if crossed:
            stop_reason = "off_mat"
            log.append({"kind": "off_mat", "t": step, "evicted": crossed,
                        "blocks_over_edge": [m for m in mat_measure(pack) if m["edge_mm"] < -3.0]})
            mark("student", "internal: stop; OOW")
            print(f"  step {step:2d}: OOW - {crossed} crossed the black-mat edge; stopping the run")
            break
        tobs = teacher_vector(tracks, eef)
        obs = so.build_student_obs(tobs, None, vis, stale, prev_action, plan_xy, step,
                                   plan_act, plan_obj_xy=plan_obj)
        assert obs.shape == (166,) and np.isfinite(obs).all(), "obs contract violated"
        stale = so.step_staleness(stale, vis)
        x = torch.from_numpy(obs).reshape(1, 1, 166)
        if mask is not None:
            x = x * mask
        with torch.no_grad():
            out, _, h = net(x, h)
        if head == "xy":
            action = int(torch.cdist(out[0], P).argmin(-1).item())
        else:
            action = int(out[0].argmax(-1).item())
        d1, d2 = two_step_plan(action)
        wp1 = np.clip(eef + d1, [WS_X[0], WS_Y[0]], [WS_X[1], WS_Y[1]])
        wp2 = np.clip(wp1 + d2, [WS_X[0], WS_Y[0]], [WS_X[1], WS_Y[1]])
        proposed_m = float(np.linalg.norm(wp1 - eef) + np.linalg.norm(wp2 - wp1))
        if travelled_m + proposed_m > distance_limit_m + 1e-9:
            stop_reason = "teacher_distance_limit"
            mark("student", f"internal: stop; TCP travel cap={distance_limit_m:.3f}m")
            print(f"  step {step:2d}: TCP travel budget exhausted "
                  f"({travelled_m:.3f}+{proposed_m:.3f}>{distance_limit_m:.3f}m)")
            break
        drift = float(np.hypot(*(np.array(plan_xy[min(step, n_plan - 1)]) - eef)))
        rec = {"kind": "primitive", "t": step, "eef_sim": eef.round(4).tolist(),
               "action": action,
               "n_raw_masks": len(pack["raw"]), "n_kept_masks": len(pack["kept"]),
               "n_rejected_masks": len(pack["rejected"]),
               "mask_filter_applied": pack["mask_filter_applied"],
               "mask_filter": pack["mask_diagnostics"],
               "n_det": len(dets), "vis": vis.astype(int).tolist(),
               "target_vis": int(vis[0]), "plan_drift_mm": round(1000 * drift, 1),
               "perceive_s": round(t_perc, 3)}
        log.append(rec)
        print(f"  step {step:2d}: {len(dets):2d} det, vis {int(vis.sum()):2d}/11 "
              f"(target {'seen' if vis[0] else 'HIDDEN'}), "
              f"travel {travelled_m:.3f}/{distance_limit_m:.3f}m, "
              f"drift {1000*drift:5.1f} mm -> action {action:2d}  [{t_perc:.2f}s perceive]")
        mark("student", f"internal: student step; travel_m={travelled_m:.4f}")
        if live:
            # perception was seconds of RTDE silence: the control script is gone
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if travel_tcp_xy is None:
                travel_tcp_xy = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
            for wp in (wp1, wp2):
                rx, ry = frames.sim_to_real(*wp)
                rtde_c.moveL(pose(rx, ry, args.push_z), args.tool_vel, args.tool_acc)
                now_xy = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
                travelled_m += float(np.linalg.norm(now_xy - travel_tcp_xy))
                travel_tcp_xy = now_xy
            steps_run += 1
            tcp = rtde_r.getActualTCPPose()
            lo, hi = frames.REAL_WORKSPACE_LIMITS[:2, 0], frames.REAL_WORKSPACE_LIMITS[:2, 1]
            if not (lo[0] - 0.03 <= tcp[0] <= hi[0] + 0.03 and lo[1] - 0.03 <= tcp[1] <= hi[1] + 0.03):
                print("EEF left the real workspace — stopping the episode (oow)")
                oow = True
                stop_reason = "oow"
                break
        else:
            steps_run += 1
            travelled_m += proposed_m
        prev_action = action
    timing["steps_run"] = steps_run
    timing["travelled_m"] = round(travelled_m, 6)
    timing["student_seconds"] = round(time.time() - t_run0, 3)
    timing["oow"] = oow
    timing["stop_reason"] = stop_reason
    oow_failure = stop_reason == "off_mat"

    # ------------------------------------------------------------ terminal
    grasp_res = None
    if live:
        th = time.time()
        mark("home", "clearing to safe height")
        retract()
        timing["home_seconds"] = round(time.time() - th, 3)
        mark("re-sense", "capturing + Mask R-CNN + grasp network")
        ts = time.time()
        color, depth, K, base = capture_clean()
        clean_pack = analyze_frame(color, depth, K, base,
                                   filter_mid_execution=False)
        crossed = off_mat(clean_pack)
        seen = mat_boundary.measured_block_count(mat_measure(clean_pack), corners_sim=mat["corners_sim"])
        if not oow_failure and (crossed or seen < initial_block_count):
            oow_failure = True
            stop_reason = timing["stop_reason"] = "off_mat" if crossed else "object_missing"
            log.append({"kind": stop_reason, "t": "terminal", "evicted": crossed,
                        "blocks_on_mat": seen, "initial_blocks": initial_block_count})
            print(f"Terminal clean view: OOW ({stop_reason}) {crossed or f'{seen}/{initial_block_count} blocks'}")
        g = None if oow_failure else stock_grasp(clean_pack)
        timing["resense_seconds"] = round(time.time() - ts, 3)
        if oow_failure:
            mark("no-grasp", "OOW failure: part of an object left the black mat" if stop_reason == "off_mat"
                 else "OOW failure: an object is no longer on the mat")
        elif g is None:
            print("Re-sense: no purple/blue target detected — not grasping.")
            mark("no-grasp", "target not detected")
        else:
            dbg = g.pop("_debug")
            if base:
                import cv2
                cv2.imwrite(base + "_heightmap.png", dbg["rgb"][:, :, ::-1])
                real_grasp.render_rotation_panel(dbg, g, base + "_gn16.png")
            grasp_res = g
            timing["real_grasp"] = g
            print(f"Re-sense: {g['n_instances']} objects, graspability q={g['q']:.3f} "
                  f"({'GRASPABLE' if g['graspable'] else 'not graspable'})")
            mark("re-sense", f"Original grasp network score: {g['q']:.2f}")
            if g["graspable"]:
                tg = time.time()
                mark("grasp", f"Original grasp network score: {g['q']:.2f}")
                grx, gry = frames.rotation_idx_to_tool_orientation(g["rotation_idx"])
                gz = max(g["surface_z_m"] - args.descend_m, args.table_z + 0.005)
                gx, gy = g["x_real"], g["y_real"]
                rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
                rtde_c.moveL([gx, gy, 0.10, grx, gry, 0.0], args.transit_vel, args.transit_acc)
                gripper.open_and_wait_for_pos(80, 120)
                rtde_c.moveL([gx, gy, gz, grx, gry, 0.0], 0.06, 0.24)
                mark("grasp-close", f"selected orientation bin {g['rotation_idx']} of 16")
                gripper.move_and_wait_for_pos(int(0.9 * gripper.get_max_position()), 80, 120)
                time.sleep(0.2)
                rtde_c.moveL([gx, gy, 0.12, grx, gry, 0.0], args.transit_vel, args.transit_acc)
                rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
                held = gripper.get_current_position()
                timing["gripper_position"] = int(held)
                timing["holding"] = bool(held < int(0.9 * gripper.get_max_position()) - 5)
                print(f"Grasp: gripper {held}/255 ({'HOLDING' if timing['holding'] else 'EMPTY'})")
                if timing["holding"] and not args.keep:
                    gripper.open_and_wait_for_pos(80, 120)   # release off the mat
                    timing["released"] = True
                timing["grasp_seconds"] = round(time.time() - tg, 3)
            else:
                mark("no-grasp", f"Original grasp network score: {g['q']:.2f}")
        # Write the result before the return-to-home move: a rejected or collided homing must
        # not cost a finished run its record, and the recorder needs real_timing.json to annotate.
        save_result(time.time() - t_run0)
        if not (args.keep and timing.get("holding")):
            home_detail = (f"Original grasp network score: {g['q']:.2f}"
                           if g is not None else "returning to home")
            mark("home", home_detail)
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
        else:
            print("--keep: holding the object, parked at PMBS home for manual removal.")
        mark("done", "")
        recovery["armed"] = False
        rtde_c.stopScript()
    save_result(time.time() - t_run0)
    print("\n=== student run ===")
    for k in ("student", "plan_steps", "steps_run", "stop_reason",
              "student_seconds", "home_seconds",
              "resense_seconds", "grasp_seconds", "gripper_position", "holding", "oow",
              "real_total_seconds"):
        if k in timing:
            print(f"  {k:20s} {timing[k]}")
    if args.timing_out:
        print("  written ->", args.timing_out)


if __name__ == "__main__":
    main()
