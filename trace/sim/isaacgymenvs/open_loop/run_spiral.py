"""Closed-loop circular-spiral baseline on the real robot.

This is the spiral counterpart to ``run_student.py`` and ``run_teacher_full_obs.py``.
It uses the same D455 segmentation, calibrated pose reconstruction, stock grasp
network (the teacher closed-loop GN call), staged hardware motion, and terminal
grasp path. The only policy-specific component is ``CircularSpiral``.

Before every 1 cm spiral command:

  partial observation (arm in view) -> out-of-workspace rule ->
    target visible: stock GN in place, no retraction
      GN graspable: back off 1 cm along the tool's own path, retract to PMBS home,
                    confirm on a clean observation, grasp (or resume the spiral)
    target hidden: back off 1 cm, retract to PMBS home, clean observation,
                   out-of-workspace rule, stock GN; grasp if graspable, otherwise
                   return to the saved push pose and continue
  -> one spiral command around the latest target estimate

An occlusion pauses the spiral; it does not reset its orbit progress. The run ends
as a failure when an object is pushed outside the 44.8 cm workspace (the rule the
simulated spiral evaluation applies), when any part of an object's footprint leaves
the black mat (boundary detected on the clean initial frame), or when a clean
observation finds fewer objects than the initial scene.
"""
import argparse
import atexit
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.open_loop import hardware_config
from isaacgymenvs.open_loop import frames, mat_boundary, perceive_scene as ps, real_grasp, recorder_io
from isaacgymenvs.open_loop.execute_trajectory import (
    HOME_JOINTS_DEG, PMBS_HOME_JOINTS_DEG, STAGING_SIM_XY)
from isaacgymenvs.open_loop.run_teacher_full_obs import (
    backoff_point, evicted_objects, outside_counts, public_grasp)
from isaacgymenvs.open_loop.spiral_controller import (
    CircularSpiral, DEFAULT_ARC_STEP_M, DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TRAVEL_M, DEFAULT_MIN_RADIUS_M, DEFAULT_ORBIT_LOOPS,
    DEFAULT_RADIAL_STEP_M, DEFAULT_START_RADIUS_M, DEFAULT_TIMEOUT_S,
)


FAILURE_STOPS = ("out_of_workspace", "off_mat", "object_missing")


def single_target(objects):
    targets = [obj for obj in objects if obj.get("target")]
    return targets[0] if len(targets) == 1 else None


def main():
    t_program0 = time.time()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--offline-dump", default=None,
                    help="saved *_color.png/_depth.npy/_K.json base; no camera or robot")
    ap.add_argument("--initial-dump", default=None,
                    help="clean pre-motion D455 dump base (required for execution)")
    ap.add_argument("--resense-dir", default=None,
                    help="record_cameras.py output directory (owns the D455)")
    ap.add_argument("--resense-name", default="d455_topdown")
    ap.add_argument("--calib", default=str(hardware_config.calibration()))
    ap.add_argument("--maskrcnn", default=str(hardware_config.maskrcnn()))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--object-depth-min-z", type=float, default=0.025)
    ap.add_argument("--object-depth-max-z", type=float, default=0.065)
    ap.add_argument("--object-depth-min-fraction", type=float, default=0.80)
    ap.add_argument("--robot-ip", default=hardware_config.robot_ip())
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
    ap.add_argument("--max-travel-m", type=float, default=DEFAULT_MAX_TRAVEL_M,
                    help="hard planar TCP travel cap for the spiral")
    ap.add_argument("--arc-step-m", type=float, default=DEFAULT_ARC_STEP_M)
    ap.add_argument("--radial-step-m", type=float, default=DEFAULT_RADIAL_STEP_M)
    ap.add_argument("--min-radius-m", type=float, default=DEFAULT_MIN_RADIUS_M)
    ap.add_argument("--start-radius-m", type=float, default=DEFAULT_START_RADIUS_M,
                    help="optional outer spiral radius; enter it radially in arc-step increments")
    ap.add_argument("--orbit-loops", type=float, default=DEFAULT_ORBIT_LOOPS)
    ap.add_argument("--clean-resense-attempts", type=int, default=3,
                    help="clean frames tried at PMBS home after target occlusion")
    ap.add_argument("--retract-backoff-m", type=float, default=0.01,
                    help="withdraw this far back along the tool's own path before lifting, "
                         "so the tool leaves a concave cavity instead of carrying the block")
    ap.add_argument("--push-z", type=float, default=0.020)
    ap.add_argument("--safe-z", type=float, default=0.22)
    ap.add_argument("--tool-vel", type=float, default=0.28)
    ap.add_argument("--tool-acc", type=float, default=1.20)
    ap.add_argument("--transit-vel", type=float, default=0.30)
    ap.add_argument("--transit-acc", type=float, default=1.20)
    ap.add_argument("--descend-m", type=float, default=0.040)
    ap.add_argument("--table-z", type=float, default=0.006)
    ap.add_argument("--start-eef-x", type=float, default=STAGING_SIM_XY[0],
                    help="sim-frame dry-run EEF x")
    ap.add_argument("--start-eef-y", type=float, default=STAGING_SIM_XY[1],
                    help="sim-frame dry-run EEF y")
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--trace-out", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep a successfully grasped target held at PMBS home")
    args = ap.parse_args()

    if args.max_steps <= 0 or args.clean_resense_attempts <= 0:
        raise SystemExit("--max-steps and --clean-resense-attempts must be positive")
    for name, value in (("timeout-s", args.timeout_s), ("max-travel-m", args.max_travel_m)):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name} must be finite and positive")
    if not math.isfinite(args.retract_backoff_m) or args.retract_backoff_m < 0:
        raise SystemExit("--retract-backoff-m must be finite and non-negative")
    spiral = CircularSpiral(
        arc_step_m=args.arc_step_m,
        radial_step_m=args.radial_step_m,
        min_radius_m=args.min_radius_m,
        orbit_loops=args.orbit_loops,
        start_radius_m=args.start_radius_m,
    )

    def mark(phase, detail=""):
        if args.resense_dir:
            recorder_io.mark(args.resense_dir, phase, detail)

    stage = {name: 0.0 for name in ("model_load", "initial_home", "partial_perception",
                                    "stock_gn", "audit_artifacts", "push", "retract",
                                    "clean_perception", "return_to_push", "grasp",
                                    "return_home")}
    t_load = time.time()
    mark("spiral-load", "loading segmentation and grasp networks")
    cam2base = np.loadtxt(args.calib)
    dev = torch.device(args.device)
    maskrcnn = ps.load_maskrcnn(args.maskrcnn, dev)
    from isaacgymenvs.utils.mtcs_utils import MCTSHelper
    helper = MCTSHelper("logs_grasp/snapshot-post-020000.reinforcement.pth",
                        "logs_grasp/grasp_model-89.pth", device=args.device)
    stage["model_load"] += time.time() - t_load
    print("spiral baseline: partial observation + stock GN every step; "
          "retract only for occlusion or grasp confirmation")

    def load_dump(base):
        import cv2
        color = cv2.imread(base + "_color.png")
        if color is None:
            raise FileNotFoundError(base + "_color.png")
        depth = np.load(base + "_depth.npy")
        with open(base + "_K.json") as f:
            K = json.load(f)
        return color, depth, K, base

    def analyze(color, depth, K, base=None, partial=False):
        """Teacher closed-loop analysis; partial frames first drop robot masks by depth."""
        raw = ps.segment(color, args.maskrcnn, dev, model=maskrcnn)
        if partial:
            candidates, rejected, mask_diag = ps.filter_instances_by_depth(
                raw, depth, K, cam2base,
                min_z=args.object_depth_min_z,
                max_z=args.object_depth_max_z,
                min_fraction=args.object_depth_min_fraction)
        else:
            candidates, rejected, mask_diag = raw, [], []
        objects, _ = ps.locate_objects(
            color, depth, K, cam2base, table_z=args.table_z,
            device=dev, require_target=False, instances=candidates)
        # Same post-filter instance list the teacher runner hands to the GN.
        class_id = {name: idx for idx, name in ps.CLASS_ID_TO_NAME.items()}
        kept = [{"mask": obj["mask"], "class": class_id[obj["name"]], "score": obj["score"]}
                for obj in objects]
        return {"color": color, "depth": depth, "K": K, "base": base, "raw": raw,
                "kept": kept, "rejected": rejected, "partial": bool(partial),
                "mask_diagnostics": mask_diag, "objects": objects}

    def capture(partial):
        if args.offline_dump:
            return analyze(*load_dump(args.offline_dump), partial=partial)
        if args.resense_dir:
            return analyze(*recorder_io.request_dump(args.resense_dir, args.resense_name),
                           partial=partial)
        color, depth, K = ps.capture_realsense(warmup=30 if not partial else 8)
        return analyze(color, depth, K, None, partial=partial)

    def stock_grasp(pack):
        return real_grasp.compute_hardware_grasp(
            pack["color"], pack["depth"], pack["K"], cam2base, args.maskrcnn,
            device=args.device, helper=helper, maskrcnn=maskrcnn,
            instances=pack["kept"])

    audit_dir = Path(args.resense_dir) / "observations" if args.resense_dir else None
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)

    observations = []

    def save_observation(pack, kind):
        """Numbered in capture order so the video shows every observation the runner used."""
        n = len(observations)
        observations.append({"n": n, "kind": kind, "base": pack.get("base")})
        mark("spiral-observe", f"internal: observation {n}; kind={kind}")
        if not audit_dir:
            return
        started = time.time()
        target_idx = ps.pick_target(pack["color"], pack["kept"])
        scene_rgb, _, scene_segm = real_grasp.build_real_heightmap(
            pack["color"], pack["depth"], pack["K"], cam2base, pack["kept"],
            target_idx=target_idx)
        h, w = pack["color"].shape[:2]
        kept = (np.stack([i["mask"] for i in pack["kept"]]).astype(np.uint8)
                if pack["kept"] else np.zeros((0, h, w), np.uint8))
        rejected = (np.stack([i["mask"] for i in pack["rejected"]]).astype(np.uint8)
                    if pack["rejected"] else np.zeros((0, h, w), np.uint8))
        np.savez_compressed(audit_dir / f"camera_masks_step_{n:03d}.npz", kept=kept,
                            rejected=rejected, scene_rgb=scene_rgb, scene_segm=scene_segm)
        stage["audit_artifacts"] += time.time() - started

    def render_grasp(pack, grasp):
        debug = grasp.pop("_debug", None) if grasp else None
        if debug is not None and pack.get("base") and args.resense_dir:
            import cv2
            cv2.imwrite(pack["base"] + "_heightmap.png", debug["rgb"][:, :, ::-1])
            real_grasp.render_rotation_panel(debug, grasp, pack["base"] + "_gn16.png")

    live = args.execute and not args.offline_dump
    if live and not args.initial_dump:
        raise SystemExit("--initial-dump is required for execution")
    initial_pack = (analyze(*load_dump(args.initial_dump), partial=False)
                    if args.initial_dump else capture(partial=False))
    if single_target(initial_pack["objects"]) is None:
        raise SystemExit("initial clean frame has no unambiguous purple/blue target")
    mat = mat_boundary.detect_mat_polygon(initial_pack["color"], initial_pack["depth"],
                                          initial_pack["K"], cam2base)

    def measure(pack):
        """Measured top-face distance of every segmented block to the mat edge (cached)."""
        if "mat_measurements" not in pack:
            blocks, _, _ = ps.filter_instances_by_depth(
                pack["raw"], pack["depth"], pack["K"], cam2base,
                min_z=args.object_depth_min_z, max_z=args.object_depth_max_z,
                min_fraction=args.object_depth_min_fraction)
            pack["mat_measurements"] = mat_boundary.instance_mat_measurements(
                blocks, pack["depth"], pack["K"], cam2base, mat["corners_sim"],
                min_z=args.object_depth_min_z, max_z=args.object_depth_max_z)
        return pack["mat_measurements"]

    initial_measurements = measure(initial_pack)
    mat_exempt = mat_boundary.measured_off_mat_counts(initial_measurements)
    initial_count = mat_boundary.measured_block_count(initial_measurements,
                                                      corners_sim=mat["corners_sim"])
    print(f"mat boundary (sim): {np.round(mat['corners_sim'], 4).tolist()}; "
          f"{initial_count} blocks on the mat, closest "
          f"{min(m['edge_mm'] for m in initial_measurements):.1f} mm from the edge; "
          f"already over the edge: {mat_exempt or 'none'}")
    if args.resense_dir and initial_pack.get("base"):
        import cv2
        debug = initial_pack["color"].copy()
        cv2.polylines(debug, [mat["corners_px"].astype(np.int32)], True, (0, 255, 255), 2)
        cv2.imwrite(initial_pack["base"] + "_mat.png", debug)

    if live and not args.yes:
        if input("Run the SPIRAL closed-loop baseline on the robot? type 'yes': ").strip() != "yes":
            print("Aborted.")
            return

    rtde_c = rtde_r = gripper = None
    push_rot = None
    state = {"at_pmbs": False, "armed": bool(live)}
    low = frames.SIM_WORKSPACE_LIMITS[:2, 0]
    high = frames.SIM_WORKSPACE_LIMITS[:2, 1]

    def pose(x_real, y_real, z):
        return [x_real, y_real, z, push_rot[0], push_rot[1], push_rot[2]]

    def retract(back_off_sim=None):
        nonlocal rtde_c
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        # Break contact sideways first: lifting straight out of a concave block's
        # cavity wedges the fingertips against its walls and carries the block.
        if back_off_sim is not None:
            bx, by = frames.sim_to_real(*back_off_sim)
            if not rtde_c.moveL(pose(bx, by, args.push_z), 0.06, 0.24):
                raise RuntimeError("retract back-off move rejected")
        cur = rtde_r.getActualTCPPose()
        if not rtde_c.moveL(pose(cur[0], cur[1], args.safe_z), args.transit_vel, args.transit_acc):
            raise RuntimeError("vertical retract move rejected")
        sx, sy = frames.sim_to_real(*STAGING_SIM_XY)
        if not rtde_c.moveL(pose(sx, sy, args.safe_z), args.transit_vel, args.transit_acc):
            raise RuntimeError("staging move rejected")
        if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("PMBS-home move rejected")
        state["at_pmbs"] = True

    def return_to_push(eef_xy):
        nonlocal rtde_c
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        rx, ry = frames.sim_to_real(*eef_xy)
        if not rtde_c.moveL(pose(rx, ry, args.safe_z), args.transit_vel, args.transit_acc):
            raise RuntimeError("high return-to-push move rejected")
        if not rtde_c.moveL(pose(rx, ry, args.push_z + 0.03), args.transit_vel, args.transit_acc):
            raise RuntimeError("hover return-to-push move rejected")
        if not rtde_c.moveL(pose(rx, ry, args.push_z), 0.06, 0.24):
            raise RuntimeError("push-height descent rejected")
        state["at_pmbs"] = False

    def emergency_return_home():
        nonlocal rtde_c
        if not state["armed"]:
            return
        state["armed"] = False
        try:
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Emergency return skipped because the robot is safety-stopped", flush=True)
                return
            print("Unhandled spiral error: retracting to experiment home", flush=True)
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not state["at_pmbs"]:
                retract()
                rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
            rtde_c.stopScript()
            rtde_c.disconnect()
        except Exception as exc:
            print(f"EMERGENCY RETURN FAILED: {exc}", flush=True)

    if live:
        from dashboard_client import DashboardClient as D
        from robotiq_gripper import RobotiqGripper
        from rtde_control import RTDEControlInterface as C
        from rtde_receive import RTDEReceiveInterface as R

        dashboard = D(args.robot_ip)
        dashboard.connect()
        if dashboard.running():
            dashboard.stop()
            time.sleep(1.0)
        try:
            dashboard.unlockProtectiveStop()
            time.sleep(0.4)
        except Exception:
            pass
        dashboard.disconnect()
        gripper = RobotiqGripper(args.robot_ip, 63352)
        gripper.connect()
        rtde_r, rtde_c = R(args.robot_ip), C(args.robot_ip)
        atexit.register(emergency_return_home)
        mark("initial-home", "moving to experiment home")
        started = time.time()
        gripper.close_and_wait_for_pos(80, 120)
        if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("experiment-home move rejected")
        tcp = rtde_r.getActualTCPPose()
        from isaacgymenvs.open_loop.hardware_orientation import fixed_downward_rotation
        push_rot = fixed_downward_rotation(tcp[3:6]).tolist()
        stage["initial_home"] += time.time() - started

    dry_eef = np.array([args.start_eef_x, args.start_eef_y], dtype=np.float64)

    def eef_sim():
        if live:
            tcp = rtde_r.getActualTCPPose()
            return np.asarray(frames.real_to_sim(tcp[0], tcp[1]), dtype=np.float64)
        return dry_eef.copy()

    log = []
    trace = {"schema_version": 1, "controller": "circular_spiral", "dense": [], "waypoints": []}
    timing = {
        "controller": "circular_spiral",
        "protocol": "partial_obs_stock_gn_every_step_retract_on_occlusion_or_gn_confirmation",
        "stock_gn_threshold": real_grasp.GRASPABLE_Q_THRESHOLD,
        "arc_step_m": args.arc_step_m,
        "radial_step_m": args.radial_step_m,
        "min_radius_m": args.min_radius_m,
        "start_radius_m": args.start_radius_m,
        "orbit_loops": args.orbit_loops,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout_s,
        "max_travel_m": args.max_travel_m,
        "retract_backoff_m": args.retract_backoff_m,
        "object_depth_band_z": [args.object_depth_min_z, args.object_depth_max_z],
        "object_depth_min_fraction": args.object_depth_min_fraction,
        "oow_rule": "any measured block top-face point more than 3 mm beyond the black-mat edge, "
                    "or fewer blocks on the mat in a clean observation than initially",
        "mat_corners_sim": np.round(mat["corners_sim"], 5).tolist(),
        "mat_plane_z": round(mat["plane_z"], 5),
        "mat_exempt": mat_exempt,
        "initial_object_count": initial_count,
    }

    counters = dict(steps_run=0, partial_observations=0, occlusion_retracts=0,
                    confirmation_retracts=0, clean_observations=0, clean_resense_attempts=0)
    travel = {"m": 0.0, "last_xy": None}
    prev_eef = None

    def check_oow(pack, label):
        """-> None or 'off_mat' (OOW failure) for this observation."""
        measurements = measure(pack)
        off = evicted_objects(mat_boundary.measured_off_mat_counts(measurements), mat_exempt)
        if off:
            log.append({"kind": "off_mat", "at": label, "evicted": off,
                        "blocks_over_edge": [m for m in measurements if m["edge_mm"] < -3.0]})
            print(f"  {label}: OOW - {', '.join(f'{n}x{c}' for n, c in off.items())} "
                  f"crossed the black-mat edge; stopping the run.")
            return "off_mat"
        return None

    def clean_check(eef, reason, step_idx):
        """Back off, retract, observe cleanly, apply OOW rule, run stock GN.

        -> (result, pack, grasp) with result in {'graspable', 'resume', 'target_lost', 'oow'}.
        """
        started = time.time()
        mark("spiral-retract", f"{reason} at step {step_idx}")
        if live:
            back_off = (None if prev_eef is None else
                        backoff_point((eef, prev_eef), args.retract_backoff_m, low, high))
            retract(back_off)
        stage["retract"] += time.time() - started
        pack = grasp = None
        attempt = 0
        for attempt in range(1, args.clean_resense_attempts + 1):
            started = time.time()
            mark("re-sense", "clean observation at PMBS home")
            pack = capture(partial=False)
            stage["clean_perception"] += time.time() - started
            counters["clean_resense_attempts"] += 1
            seen = mat_boundary.measured_block_count(measure(pack), corners_sim=mat["corners_sim"])
            if single_target(pack["objects"]) is not None and seen >= initial_count:
                break
            print(f"    clean re-sense {attempt}/{args.clean_resense_attempts}: "
                  f"target {'absent' if single_target(pack['objects']) is None else 'seen'}, "
                  f"{seen}/{initial_count} objects")
            if args.offline_dump:
                break
        if pack is not None and single_target(pack["objects"]) is not None:
            seen = mat_boundary.measured_block_count(measure(pack), corners_sim=mat["corners_sim"])
            if seen < initial_count:
                counters["clean_observations"] += 1
                save_observation(pack, "clean")
                log.append({"kind": "clean_check", "t": step_idx, "reason": reason,
                            "result": "object_missing", "attempts": attempt,
                            "objects": seen, "initial_objects": initial_count})
                print(f"    clean check: {seen}/{initial_count} objects on the mat; stopping the run.")
                return "object_missing", pack, None
        else:
            pack = None
        if pack is None:
            log.append({"kind": "clean_check", "t": step_idx, "reason": reason,
                        "result": "target_lost", "attempts": attempt})
            return "target_lost", None, None
        counters["clean_observations"] += 1
        save_observation(pack, "clean")
        violation = check_oow(pack, f"clean check {counters['clean_observations']}")
        if violation:
            return violation, pack, None
        started = time.time()
        mark("spiral-gn", "evaluating Original Grasp Network on the clean observation")
        grasp = stock_grasp(pack)
        stage["stock_gn"] += time.time() - started
        q = float(grasp["q"]) if grasp is not None else 0.0
        mark("re-sense", f"Original grasp network score: {q:.2f}")
        render_grasp(pack, grasp)
        result = "graspable" if grasp is not None and grasp["graspable"] else "resume"
        log.append({"kind": "clean_check", "t": step_idx, "reason": reason, "result": result,
                    "attempts": attempt, "clean_q": round(q, 4),
                    "base": pack.get("base")})
        print(f"    clean check ({reason}): GN q={q:.3f} -> {result}")
        return result, pack, grasp

    t_run0 = time.time()
    mark("spiral", f"0/{args.max_steps}")
    stop_reason = "horizon"
    g_final = None

    def finalize(elapsed):
        """Write the trial result. Called before the return-to-home move, and again after it,
        so a rejected or collided homing never costs a finished run its record."""
        from isaacgymenvs.open_loop.trial_timing import write_json_atomic
        timing["real_grasp"] = public_grasp(g_final)
        timing["stage_seconds"] = {k: round(v, 3) for k, v in stage.items()}
        timing["real_total_seconds"] = round(elapsed, 3)
        timing["log"] = log
        timing["observations"] = observations
        trace.setdefault("start_eef", [float(args.start_eef_x), float(args.start_eef_y), args.push_z])
        trace.update(stop_reason=timing["stop_reason"], steps_run=counters["steps_run"])
        if args.trace_out:
            write_json_atomic(args.trace_out, trace)
        if args.timing_out:
            write_json_atomic(args.timing_out, timing)

    pack_final = None
    grasp_pack = initial_pack

    initial_g = stock_grasp(initial_pack)
    if initial_g is not None and initial_g["graspable"]:
        stop_reason, g_final, pack_final = "initial_gn_graspable", initial_g, initial_pack
        render_grasp(initial_pack, initial_g)
        if live:
            retract()
    else:
        eef = eef_sim()
        trace["start_eef"] = [float(eef[0]), float(eef[1]), args.push_z]
        if live:
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            tcp = rtde_r.getActualTCPPose()
            if not rtde_c.moveL([tcp[0], tcp[1], args.push_z, *push_rot],
                                args.transit_vel, args.transit_acc):
                raise RuntimeError("initial push-height descent rejected")
        for step_idx in range(args.max_steps):
            elapsed = time.time() - t_run0
            if elapsed >= args.timeout_s:
                stop_reason = "timeout"
                print(f"Spiral timeout after {elapsed:.1f}s")
                break

            eef = eef_sim()
            started = time.time()
            if step_idx == 0:
                pack = initial_pack  # clean frame from experiment home, arm out of the scene
            else:
                mark("spiral-sense", f"partial observation {step_idx}")
                pack = capture(partial=True)
                counters["partial_observations"] += 1
            perceive_s = time.time() - started
            stage["partial_perception"] += perceive_s
            save_observation(pack, "initial" if step_idx == 0 else "partial")
            violation = check_oow(pack, f"step {step_idx}")
            if violation:
                stop_reason = violation
                break

            target_obj = single_target(pack["objects"])
            q_step = None
            recovered = False
            if target_obj is None:
                if not live:
                    stop_reason = "target_occluded_offline"
                    print("Target is hidden in offline input; live recovery requires the robot")
                    break
                counters["occlusion_retracts"] += 1
                print(f"  step {step_idx:3d}: target HIDDEN; retracting for a clean observation")
                result, clean_pack, clean_g = clean_check(eef, "target hidden", step_idx)
                if result == "target_lost":
                    stop_reason = "target_lost_after_clean_resense"
                    break
                if result in FAILURE_STOPS:
                    stop_reason = result
                    break
                if result == "graspable":
                    stop_reason, g_final, pack_final = "clean_gn_graspable", clean_g, clean_pack
                    break
                started = time.time()
                mark("spiral-return", f"returning to saved push pose after step {step_idx}")
                return_to_push(eef)
                stage["return_to_push"] += time.time() - started
                travel["last_xy"] = None
                pack, target_obj, recovered = clean_pack, single_target(clean_pack["objects"]), True
                q_step = float(clean_g["q"]) if clean_g is not None else 0.0
            else:
                started = time.time()
                mark("spiral-gn", f"Evaluating grasp network on partial observation {step_idx}")
                g_step = initial_g if step_idx == 0 else stock_grasp(pack)
                stage["stock_gn"] += time.time() - started
                q_step = float(g_step["q"]) if g_step is not None else 0.0
                mark("spiral-gn", f"Original grasp network score: {q_step:.2f}")
                if g_step is not None and g_step["graspable"]:
                    log.append({"kind": "stop_signal", "t": step_idx, "stock_q": round(q_step, 4),
                                "partial": bool(pack["partial"])})
                    print(f"  step {step_idx:3d}: in-place GN q={q_step:.3f}; confirming on a clean view")
                    if not live:
                        stop_reason, g_final, pack_final = "in_place_gn_graspable_offline", g_step, pack
                        break
                    counters["confirmation_retracts"] += 1
                    result, clean_pack, clean_g = clean_check(eef, "in-place GN graspable", step_idx)
                    if result == "target_lost":
                        stop_reason = "target_lost_after_clean_resense"
                        break
                    if result in FAILURE_STOPS:
                        stop_reason = result
                        break
                    if result == "graspable":
                        stop_reason, g_final, pack_final = "clean_gn_graspable", clean_g, clean_pack
                        break
                    started = time.time()
                    mark("spiral-return", f"returning to saved push pose after step {step_idx}")
                    return_to_push(eef)
                    stage["return_to_push"] += time.time() - started
                    travel["last_xy"] = None
                    pack, target_obj, recovered = clean_pack, single_target(clean_pack["objects"]), True
                    q_step = float(clean_g["q"]) if clean_g is not None else 0.0

            target_xy = np.array([target_obj["x"], target_obj["y"]], dtype=np.float64)
            spiral_step = spiral.next(eef, target_xy)
            waypoint = np.clip(spiral_step.waypoint, low, high)
            proposed_m = float(np.linalg.norm(waypoint - eef))
            if travel["m"] + proposed_m > args.max_travel_m + 1e-9:
                stop_reason = "travel_limit"
                print(f"  step {step_idx:3d}: travel cap exhausted "
                      f"({travel['m']:.3f}+{proposed_m:.3f}>{args.max_travel_m:.3f}m)")
                break

            rec = {"kind": "spiral_step", "t": step_idx,
                   "eef_sim": eef.round(5).tolist(),
                   "target_xy": target_xy.round(5).tolist(),
                   "waypoint": waypoint.round(5).tolist(),
                   "radius_m": round(spiral_step.radius_m, 6),
                   "orbit_progress_rad": round(spiral_step.orbit_progress_rad, 6),
                   "phase": spiral_step.phase,
                   "stock_q": round(q_step, 4),
                   "target_visible_partial": not recovered,
                   "target_recovered": recovered,
                   "n_objects": len(pack["objects"]),
                   "n_raw_masks": len(pack["raw"]),
                   "n_kept_masks": len(pack["kept"]),
                   "n_rejected_masks": len(pack["rejected"]),
                   "perceive_s": round(perceive_s, 3)}
            log.append(rec)
            trace["dense"].append({"t": step_idx, "eef": [float(eef[0]), float(eef[1]), args.push_z],
                                   "target_xy": target_xy.round(5).tolist(), "grasp_q": q_step,
                                   "obj_xy": [[float(o["x"]), float(o["y"])] for o in pack["objects"]]})
            trace["waypoints"].append({"t": step_idx, "wp": [float(waypoint[0]), float(waypoint[1]),
                                                             args.push_z]})
            print(f"  step {step_idx:3d}: GN q={q_step:.3f}, radius "
                  f"{100*spiral_step.radius_m:5.2f} cm, orbit "
                  f"{spiral_step.orbit_progress_rad/(2*math.pi):.2f}/{args.orbit_loops:g}")
            mark("spiral", f"internal: spiral step {step_idx}; radius_m={spiral_step.radius_m:.4f}")

            prev_eef = eef.copy()
            started = time.time()
            if live:
                rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
                if travel["last_xy"] is None:
                    travel["last_xy"] = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
                rx, ry = frames.sim_to_real(*waypoint)
                if not rtde_c.moveL(pose(rx, ry, args.push_z), args.tool_vel, args.tool_acc):
                    raise RuntimeError(f"spiral step {step_idx} move rejected")
                now_xy = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
                travel["m"] += float(np.linalg.norm(now_xy - travel["last_xy"]))
                travel["last_xy"] = now_xy
            else:
                dry_eef = waypoint
                travel["m"] += proposed_m
            stage["push"] += time.time() - started
            counters["steps_run"] += 1
            if spiral_step.complete:
                stop_reason = "spiral_complete"
                break
        else:
            stop_reason = "max_steps"

    timing.update(counters, travelled_m=round(travel["m"], 6), stop_reason=stop_reason,
                  oow_failure=stop_reason in FAILURE_STOPS,
                  spiral_seconds=round(time.time() - t_run0, 3))

    if live:
        # Budget stops still get one clean terminal check, like the student's final
        # grasp check; an out-of-workspace stop is a failure and is never grasped.
        if g_final is None and stop_reason not in FAILURE_STOPS + ("target_lost_after_clean_resense",):
            eef = eef_sim()
            result, clean_pack, clean_g = clean_check(eef, "terminal check", counters["steps_run"])
            if result == "graspable":
                g_final, pack_final = clean_g, clean_pack
            elif result in FAILURE_STOPS:
                stop_reason = timing["stop_reason"] = result
                timing["oow_failure"] = True
            timing["terminal_check"] = result
        elif not state["at_pmbs"] and g_final is None:
            started = time.time()
            eef = eef_sim()
            back_off = (None if prev_eef is None else
                        backoff_point((eef, prev_eef), args.retract_backoff_m, low, high))
            mark("home", "clearing to safe height")
            retract(back_off)
            stage["retract"] += time.time() - started

        if g_final is not None and g_final["graspable"]:
            started = time.time()
            mark("grasp", f"Original grasp network score: {g_final['q']:.2f}")
            grx, gry = frames.rotation_idx_to_tool_orientation(g_final["rotation_idx"])
            gz = max(g_final["surface_z_m"] - args.descend_m, args.table_z + 0.005)
            gx, gy = g_final["x_real"], g_final["y_real"]
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveL([gx, gy, 0.10, grx, gry, 0.0], args.transit_vel, args.transit_acc):
                raise RuntimeError("grasp approach move rejected")
            gripper.open_and_wait_for_pos(80, 120)
            if not rtde_c.moveL([gx, gy, gz, grx, gry, 0.0], 0.06, 0.24):
                raise RuntimeError("grasp descent rejected")
            mark("grasp-close", f"selected orientation bin {g_final['rotation_idx']} of 16")
            closed_pos, closed_status = gripper.move_and_wait_for_pos(
                int(0.9 * gripper.get_max_position()), 80, 120)
            closed_obj = int(getattr(closed_status, "value", closed_status))
            timing["grasp_close_position"] = int(closed_pos)
            timing["grasp_close_object_status"] = closed_obj
            timing["closed_on_object"] = closed_obj in (1, 2)
            time.sleep(0.2)
            if not rtde_c.moveL([gx, gy, 0.12, grx, gry, 0.0], args.transit_vel, args.transit_acc):
                raise RuntimeError("grasp lift rejected")
            if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                raise RuntimeError("post-grasp PMBS-home move rejected")
            status_after = gripper.get_object_status()
            timing["gripper_position"] = int(gripper.get_current_position())
            timing["gripper_object_status"] = status_after
            timing["holding"] = status_after in (1, 2)
            timing["dropped_during_lift"] = bool(timing.get("closed_on_object") and not timing["holding"])
            if timing["holding"] and not args.keep:
                gripper.open_and_wait_for_pos(80, 120)
                timing["released"] = True
            stage["grasp"] += time.time() - started
            timing["grasp_seconds"] = round(time.time() - started, 3)
        elif stop_reason == "out_of_workspace":
            mark("no-grasp", "OOW failure: an object was pushed outside the 44.8 cm workspace")
        elif stop_reason == "off_mat":
            mark("no-grasp", "OOW failure: part of an object left the black mat")
        elif stop_reason == "object_missing":
            mark("no-grasp", "OOW failure: an object is no longer on the mat")
        else:
            mark("no-grasp", "spiral baseline ended without a graspable GN result")

        # The trial result is complete here. Write it before the return-to-home move so a
        # rejected or collided homing cannot destroy the record of a finished run (and so the
        # recorder still finds real_timing.json / trajectory.json when it annotates).
        finalize(time.time() - t_program0)
        if not (args.keep and timing.get("holding")):
            started = time.time()
            mark("home", "returning to home")
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                print("WARNING: final experiment-home move rejected; the trial result is already written")
                timing["final_home"] = "rejected"
            else:
                stage["return_home"] += time.time() - started
            finalize(time.time() - t_program0)
        else:
            print("--keep: holding the object at PMBS home for manual removal.")
        mark("done", "")
        state["armed"] = False
        rtde_c.stopScript()

    finalize(time.time() - t_program0)
    print("\n=== spiral run ===")
    for key in ("controller", "steps_run", "stop_reason", "partial_observations",
                "occlusion_retracts", "confirmation_retracts", "clean_observations",
                "travelled_m", "spiral_seconds", "grasp_seconds", "holding",
                "real_total_seconds"):
        if key in timing:
            print(f"  {key:24s} {timing[key]}")
    print("  stage_seconds            ", timing["stage_seconds"])
    from isaacgymenvs.open_loop.trial_timing import write_json_atomic
    if args.trace_out:
        write_json_atomic(args.trace_out, trace)
    if args.timing_out:
        write_json_atomic(args.timing_out, timing)
        print("  written ->", args.timing_out)


if __name__ == "__main__":
    main()
