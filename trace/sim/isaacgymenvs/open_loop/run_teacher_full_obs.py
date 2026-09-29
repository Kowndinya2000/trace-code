"""Real-robot teacher baseline with a clean full observation after every push.

The first observation is taken at experiment home (the tool sits at the near
edge of the workspace and occludes nothing); every later decision is made at
PMBS home:

  retract -> fresh D455 RGB-D -> Mask R-CNN/pose -> stock GN ->
  recurrent teacher -> return to saved EEF -> one complete 4 cm primitive

Every clean observation is also checked against the 44.8 cm workspace box: an
object pushed out of it ends the run as an out-of-workspace failure, the rule
the twin solve applies (solve_in_twin: an OOW event pre-empts a graspable
result, so the check runs before the grasp network).

If stock GN reports Q >= 0.70, the same clean observation supplies the final
grasp pose and the robot grasps immediately.  PMBS travel is excluded from the
teacher observation and push-distance accounting.  This is a true closed-loop
teacher: it recomputes its action from the newly reconstructed scene instead
of replaying a trajectory solved from the initial scene.
"""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.open_loop import hardware_config
from isaacgymenvs.open_loop import frames, perceive_scene as ps, real_grasp, recorder_io
from isaacgymenvs.open_loop.execute_trajectory import (
    HOME_JOINTS_DEG, PMBS_HOME_JOINTS_DEG, STAGING_SIM_XY,
)
from isaacgymenvs.open_loop.run_student import teacher_vector, two_step_plan
from isaacgymenvs.open_loop.teacher_policy import TeacherPolicy


def ordered_full_scene(objects, expected=11):
    """Return target first; TokenSet makes the remaining order immaterial.

    The target must be visible -- without it there is nothing to retrieve and no
    observation worth acting on. The clutter count is made up to ``expected``
    instead of aborting: surplus detections are dropped worst-silhouette-fit
    first, and a shortfall is padded with ps.PAD_CENTERS_SIM dummy cubes on the
    workspace boundary, the same convention perceive_scene writes into scene
    files. A padded token is a block that is NOT on the table, so every decision
    records how many were used.
    """
    targets = [obj for obj in objects if obj.get("target")]
    if len(targets) != 1:
        return None
    target = targets[0]
    clutter = [obj for obj in objects if obj is not target]
    if len(clutter) > expected - 1:
        clutter.sort(key=lambda obj: -obj.get("fit_iou", 0.0))
        clutter = clutter[:expected - 1]
    # Deterministic output/auditing even though the teacher is permutation invariant.
    clutter.sort(key=lambda obj: (obj["name"], obj["x"], obj["y"], obj["yaw"]))
    while len(clutter) < expected - 1:
        px, py = ps.PAD_CENTERS_SIM[len(clutter) % len(ps.PAD_CENTERS_SIM)]
        clutter.append({"name": "cube", "x": px, "y": py, "yaw": 0.0,
                        "color": ps.PAD_COLOR, "target": False, "pad": True,
                        "score": 0.0, "fit_iou": 0.0, "pixels": 0})
    return [target] + clutter


def outside_counts(objects):
    """{class name: how many of that class sit outside the workspace box}.

    frames.SIM_WORKSPACE_LIMITS is the same 0.448 m square as
    tasks/more_robust.WS_X/WS_Y; it is read from frames because this baseline
    deliberately never loads Isaac Gym. Counting per class instead of tracking
    identities is enough: the rule only asks whether MORE objects are outside
    than were staged outside, and perception gives no stable object ids.
    """
    low, high = frames.SIM_WORKSPACE_LIMITS[:2, 0], frames.SIM_WORKSPACE_LIMITS[:2, 1]
    counts = {}
    for obj in objects:
        if obj.get("pad"):
            # Padded tokens are parked ON the workspace boundary by design
            # (ps.PAD_CENTERS_SIM), i.e. outside the box. They are not blocks on
            # the table, so they cannot be evicted from it.
            continue
        if not (low[0] <= obj["x"] <= high[0] and low[1] <= obj["y"] <= high[1]):
            counts[obj["name"]] = counts.get(obj["name"], 0) + 1
    return counts


def evicted_objects(outside, exempt):
    """Classes with more objects outside the box than the scene started with."""
    return {name: count - exempt.get(name, 0) for name, count in outside.items()
            if count > exempt.get(name, 0)}


def backoff_point(path_reversed, distance, low, high):
    """Retrace the EEF's OWN path backwards by `distance`; sim xy.

    ``path_reversed`` is the primitive walked in reverse -- the pose the tool
    actually reached, then wp1, then where the primitive started. Walking that
    polyline guarantees the withdrawal stays on ground the tool has already
    cleared instead of striking out in a fresh direction, and it stops at the
    primitive's start rather than continuing past it. None disables it.
    """
    if distance <= 0:
        return None
    points = [np.asarray(p, float) for p in path_reversed]
    remaining, here = distance, points[0]
    for nxt in points[1:]:
        segment = nxt - here
        length = float(np.linalg.norm(segment))
        if length < 1e-9:
            continue
        if remaining <= length:
            return np.clip(here + segment / length * remaining, low, high)
        remaining -= length
        here = nxt
    return np.clip(here, low, high)


def public_grasp(grasp):
    if grasp is None:
        return None
    return {key: value for key, value in grasp.items() if key != "_debug"}


def main():
    t_program0 = time.time()

    def save_result(elapsed):
        """Persist the trial result; safe to call before and after the return-to-home move."""
        from isaacgymenvs.open_loop.trial_timing import write_json_atomic
        timing["real_total_seconds"] = round(elapsed, 3)
        timing["stage_seconds"] = {key: round(value, 3)
                                   for key, value in timing["stage_seconds"].items()}
        timing["log"] = log
        if args.trace_out:
            write_json_atomic(args.trace_out, trace)
        if args.timing_out:
            write_json_atomic(args.timing_out, timing)

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--network-json", required=True)
    ap.add_argument("--checkpoint-sha256", default=None)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--offline-dump", default=None,
                    help="saved *_color.png/_depth.npy/_K.json base; no camera or robot")
    ap.add_argument("--resense-dir", default=None,
                    help="record_cameras.py output directory (owns the D455)")
    ap.add_argument("--resense-name", default="d455_topdown")
    ap.add_argument("--calib", default=str(hardware_config.calibration()))
    ap.add_argument("--maskrcnn", default=str(hardware_config.maskrcnn()))
    ap.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    ap.add_argument("--robot-ip", default=hardware_config.robot_ip())
    ap.add_argument("--expected-objects", type=int, default=11)
    ap.add_argument("--resense-attempts", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--timeout-s", type=float, default=300.0,
                    help="wall-clock ceiling on the teacher loop; a run that hits it "
                         "stops with stop_reason=timeout")
    ap.add_argument("--push-total-m", type=float, default=0.04)
    ap.add_argument("--push-z", type=float, default=0.020)
    ap.add_argument("--safe-z", type=float, default=0.22)
    ap.add_argument("--tool-vel", type=float, default=0.28)
    ap.add_argument("--tool-acc", type=float, default=1.20)
    ap.add_argument("--transit-vel", type=float, default=0.30)
    ap.add_argument("--transit-acc", type=float, default=1.20)
    ap.add_argument("--descend-m", type=float, default=0.040)
    ap.add_argument("--retract-backoff-m", type=float, default=0.015,
                    help="withdraw this far along the reverse push direction before "
                         "lifting, so the tool leaves a concave cavity (23.5 mm deep) "
                         "instead of lifting the block with it; 0 disables")
    ap.add_argument("--table-z", type=float, default=0.006)
    ap.add_argument("--timing-out", default=None)
    ap.add_argument("--trace-out", default=None)
    ap.add_argument("--keep", action="store_true",
                    help="keep a successfully grasped target held at PMBS home")
    args = ap.parse_args()

    if args.expected_objects != 11:
        raise SystemExit("the selected teacher requires exactly 11 object tokens")
    if args.resense_attempts <= 0 or args.max_steps <= 0:
        raise SystemExit("--resense-attempts and --max-steps must be positive")
    for name, value in (("timeout-s", args.timeout_s),
                        ("push-total-m", args.push_total_m)):
        if not math.isfinite(value) or value <= 0:
            raise SystemExit(f"--{name} must be finite and positive")
    live = args.execute and not args.offline_dump
    if live and not args.resense_dir:
        raise SystemExit("live execution requires --resense-dir so the recorded D455 owns capture")

    def mark(phase, detail=""):
        if args.resense_dir:
            recorder_io.mark(args.resense_dir, phase, detail)

    timing = {
        "controller": "teacher_full_observation",
        "protocol": "first_obs_at_experiment_home_then_retract_resense_stock_gn_teacher_one_primitive",
        "stock_gn_threshold": real_grasp.GRASPABLE_Q_THRESHOLD,
        "expected_objects": args.expected_objects,
        "max_steps": args.max_steps,
        "timeout_s": args.timeout_s,
        "push_total_m": args.push_total_m,
        "retract_backoff_m": args.retract_backoff_m,
        "teacher_rnn_state": "preserved_across_real_steps",
        "stage_seconds": {name: 0.0 for name in
                          ("model_load", "initial_home", "retract", "capture_perception", "stock_gn",
                           "teacher_inference", "audit_artifacts", "return_to_push", "push",
                           "grasp", "return_home")},
    }
    log = []
    t_load = time.time()
    mark("teacher-load", "loading teacher, segmentation, and grasp networks")
    teacher = TeacherPolicy(args.checkpoint, args.network_json, device=args.device)
    timing["teacher_checkpoint"] = str(Path(args.checkpoint).resolve())
    timing["teacher_sha256"] = teacher.sha256
    timing["teacher_epoch"] = teacher.epoch
    if args.checkpoint_sha256 and teacher.sha256.lower() != args.checkpoint_sha256.lower():
        raise SystemExit(f"teacher SHA256 mismatch: expected {args.checkpoint_sha256}, "
                         f"got {teacher.sha256}")
    cam2base = np.loadtxt(args.calib)
    dev = torch.device(args.device)
    maskrcnn = ps.load_maskrcnn(args.maskrcnn, dev)
    from isaacgymenvs.utils.mtcs_utils import MCTSHelper
    helper = MCTSHelper("logs_grasp/snapshot-post-020000.reinforcement.pth",
                        "logs_grasp/grasp_model-89.pth", device=args.device)
    timing["model_load_seconds"] = round(time.time() - t_load, 3)
    timing["stage_seconds"]["model_load"] = time.time() - t_load
    print(f"teacher: {Path(args.checkpoint).name} epoch={teacher.epoch} "
          f"sha256={teacher.sha256[:12]}")
    print("protocol: full clean re-sense + stock GN before every teacher primitive")

    def load_dump(base):
        import cv2
        color = cv2.imread(base + "_color.png")
        if color is None:
            raise FileNotFoundError(base + "_color.png")
        depth = np.load(base + "_depth.npy")
        with open(base + "_K.json") as stream:
            intrinsics = json.load(stream)
        return color, depth, intrinsics, base

    def analyze(color, depth, intrinsics, base):
        raw_instances = ps.segment(color, args.maskrcnn, dev, model=maskrcnn)
        objects, _ = ps.locate_objects(
            color, depth, intrinsics, cam2base, table_z=args.table_z,
            device=dev, require_target=False, instances=raw_instances,
            expected=args.expected_objects)
        # locate_objects has already rejected masks outside the calibrated mat.
        # Rebuild that exact post-filter instance list from its returned objects
        # so GN and the public mask tile never include a PMBS robot-edge mask.
        class_id = {name: idx for idx, name in ps.CLASS_ID_TO_NAME.items()}
        instances = [{"mask": obj["mask"], "class": class_id[obj["name"]],
                      "score": obj["score"]} for obj in objects]
        return {"color": color, "depth": depth, "K": intrinsics, "base": base,
                "raw": raw_instances, "kept": instances, "objects": objects}

    def capture_and_analyze():
        if args.offline_dump:
            return analyze(*load_dump(args.offline_dump))
        if args.resense_dir:
            return analyze(*recorder_io.request_dump(args.resense_dir, args.resense_name))
        color, depth, intrinsics = ps.capture_realsense(warmup=30)
        return analyze(color, depth, intrinsics, None)

    def stock_grasp(pack):
        return real_grasp.compute_hardware_grasp(
            pack["color"], pack["depth"], pack["K"], cam2base, args.maskrcnn,
            device=args.device, helper=helper, maskrcnn=maskrcnn,
            instances=pack["kept"])

    audit_dir = Path(args.resense_dir) / "observations" if args.resense_dir else None
    if audit_dir:
        audit_dir.mkdir(parents=True, exist_ok=True)

    def save_observation(pack, decision):
        target_idx = ps.pick_target(pack["color"], pack["kept"])
        scene_rgb, _, scene_segm = real_grasp.build_real_heightmap(
            pack["color"], pack["depth"], pack["K"], cam2base,
            pack["kept"], target_idx=target_idx)
        if audit_dir:
            h, w = pack["color"].shape[:2]
            masks = (np.stack([item["mask"] for item in pack["kept"]]).astype(np.uint8)
                     if pack["kept"] else np.zeros((0, h, w), np.uint8))
            np.savez_compressed(
                audit_dir / f"camera_masks_step_{decision:03d}.npz",
                kept=masks, rejected=np.zeros((0, h, w), np.uint8),
                scene_rgb=scene_rgb, scene_segm=scene_segm,
            )

    def render_grasp(pack, grasp):
        debug = grasp.pop("_debug", None) if grasp else None
        # Offline validation must not mutate an archived source dump. Live
        # recorder dumps belong to this run and are the correct audit location.
        if debug is not None and pack.get("base") and args.resense_dir:
            import cv2
            cv2.imwrite(pack["base"] + "_heightmap.png", debug["rgb"][:, :, ::-1])
            real_grasp.render_rotation_panel(debug, grasp,
                                             pack["base"] + "_gn16.png")

    if live and not args.yes:
        if input("Run TEACHER full-observation closed loop on the robot? type 'yes': ").strip() != "yes":
            print("Aborted.")
            return

    rtde_c = rtde_r = gripper = None
    push_rot = None
    at_pmbs = False
    recovery = {"armed": bool(live)}

    def pose(x_real, y_real, z):
        return [x_real, y_real, z, push_rot[0], push_rot[1], push_rot[2]]

    def retract(back_off_sim=None):
        nonlocal rtde_c, at_pmbs
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        # Break contact sideways FIRST. Lifting straight up out of a concave
        # block's cavity wedges the closed fingertips against both cavity walls
        # and carries the block with the tool.
        if back_off_sim is not None:
            bx, by = frames.sim_to_real(*back_off_sim)
            if not rtde_c.moveL(pose(bx, by, args.push_z), 0.06, 0.24):
                raise RuntimeError("retract back-off move rejected")
        cur = rtde_r.getActualTCPPose()
        if not rtde_c.moveL(pose(cur[0], cur[1], args.safe_z),
                            args.transit_vel, args.transit_acc):
            raise RuntimeError("vertical retract move rejected")
        sx, sy = frames.sim_to_real(*STAGING_SIM_XY)
        if not rtde_c.moveL(pose(sx, sy, args.safe_z),
                            args.transit_vel, args.transit_acc):
            raise RuntimeError("staging move rejected")
        if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("PMBS-home move rejected")
        at_pmbs = True

    def return_to_push(eef_xy):
        nonlocal rtde_c, at_pmbs
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        rx, ry = frames.sim_to_real(*eef_xy)
        if not rtde_c.moveL(pose(rx, ry, args.safe_z),
                            args.transit_vel, args.transit_acc):
            raise RuntimeError("high return-to-push move rejected")
        if not rtde_c.moveL(pose(rx, ry, args.push_z + 0.03),
                            args.transit_vel, args.transit_acc):
            raise RuntimeError("hover return-to-push move rejected")
        if not rtde_c.moveL(pose(rx, ry, args.push_z), 0.06, 0.24):
            raise RuntimeError("push-height descent rejected")
        at_pmbs = False

    def emergency_return_home():
        nonlocal rtde_c
        if not recovery["armed"]:
            return
        recovery["armed"] = False
        try:
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Emergency return skipped because the robot is safety-stopped", flush=True)
                return
            print("Unhandled teacher-loop error: returning to experiment home", flush=True)
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not at_pmbs:
                retract()
                rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                raise RuntimeError("emergency experiment-home move rejected")
            rtde_c.stopScript()
            rtde_c.disconnect()
        except Exception as exc:
            print(f"EMERGENCY RETURN FAILED: {exc}", flush=True)

    if live:
        from dashboard_client import DashboardClient
        from robotiq_gripper import RobotiqGripper
        from rtde_control import RTDEControlInterface
        from rtde_receive import RTDEReceiveInterface

        dashboard = DashboardClient(args.robot_ip)
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
        rtde_r = RTDEReceiveInterface(args.robot_ip)
        rtde_c = RTDEControlInterface(args.robot_ip)
        recovery["armed"] = True
        atexit.register(emergency_return_home)

        mark("initial-home", "moving to experiment home")
        started = time.time()
        gripper.close_and_wait_for_pos(80, 120)
        if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("experiment-home move rejected")
        tcp = rtde_r.getActualTCPPose()
        from isaacgymenvs.open_loop.hardware_orientation import fixed_downward_rotation
        push_rot = fixed_downward_rotation(tcp[3:6]).tolist()
        eef = np.asarray(frames.real_to_sim(tcp[0], tcp[1]), dtype=np.float64)
        timing["stage_seconds"]["initial_home"] += time.time() - started

        # Decision 0 is observed FROM experiment home: the tool parks at the
        # near-x edge of the workspace (sim x ~0.318), outside the 5 cm
        # generation inset every scene is built with, so it occludes nothing.
        # Every later observation still comes from PMBS home, because by then
        # the arm is standing wherever its last push ended.
    else:
        eef = np.asarray(STAGING_SIM_XY, dtype=np.float64)

    trace = {"schema_version": 1, "controller": "teacher_full_observation",
             "start_eef": [float(eef[0]), float(eef[1]), args.push_z],
             "checkpoint": str(Path(args.checkpoint).resolve()),
             "checkpoint_sha256": teacher.sha256, "dense": [], "waypoints": []}
    t_run0 = time.time()
    stop_reason = "horizon"
    oow_exempt = {}
    g_final = None
    travelled_m = 0.0
    steps_run = 0
    decisions = 0
    offline_limit = args.max_steps if live else min(args.max_steps, 1)

    for decision in range(offline_limit):
        if time.time() - t_run0 >= args.timeout_s:
            stop_reason = "timeout"
            break

        pack = ordered = None
        attempts = []
        mark("teacher-sense", f"fresh full observation {decision + 1}")
        sense_started = time.time()
        for attempt in range(1, args.resense_attempts + 1):
            pack = capture_and_analyze()
            ordered = ordered_full_scene(pack["objects"], args.expected_objects)
            attempts.append({"attempt": attempt, "objects": len(pack["objects"]),
                             "targets": sum(bool(o.get("target")) for o in pack["objects"])})
            if ordered is not None:
                break
            print(f"  decision {decision:3d}: full re-sense {attempt}/"
                  f"{args.resense_attempts} found no unambiguous target among "
                  f"{len(pack['objects'])} objects")
            if args.offline_dump:
                break
        sense_s = time.time() - sense_started
        timing["stage_seconds"]["capture_perception"] += sense_s
        if ordered is None:
            stop_reason = "target_not_visible"
            log.append({"kind": "decision", "t": decision, "attempts": attempts,
                        "result": stop_reason, "sense_s": round(sense_s, 3)})
            print("Target not visible after retries; stopping before motion.")
            break
        decisions += 1
        artifacts_started = time.time()
        save_observation(pack, decision)
        timing["stage_seconds"]["audit_artifacts"] += time.time() - artifacts_started

        # Out-of-workspace rule, checked on every fresh observation and before
        # the grasp network, exactly as the twin solve orders it. Objects staged
        # outside the box at decision 0 are exempt, the way MoreRobust exempts
        # blocks that started outside.
        outside = outside_counts(ordered)
        if decision == 0:
            oow_exempt = outside
        evicted = evicted_objects(outside, oow_exempt)
        if evicted:
            stop_reason = "out_of_workspace"
            log.append({"kind": "decision", "t": decision, "attempts": attempts,
                        "result": stop_reason, "sense_s": round(sense_s, 3),
                        "evicted": evicted,
                        "objects_outside": [[obj["name"], round(float(obj["x"]), 4),
                                             round(float(obj["y"]), 4)]
                                            for obj in ordered
                                            if outside_counts([obj])]})
            print(f"  decision {decision:3d}: OUT OF WORKSPACE - "
                  f"{', '.join(f'{n}x{c}' for n, c in evicted.items())} "
                  f"outside the 44.8 cm workspace; stopping the run.")
            break

        mark("teacher-gn", "evaluating Original Grasp Network")
        gn_started = time.time()
        grasp = stock_grasp(pack)
        gn_s = time.time() - gn_started
        timing["stage_seconds"]["stock_gn"] += gn_s
        artifacts_started = time.time()
        render_grasp(pack, grasp)
        timing["stage_seconds"]["audit_artifacts"] += time.time() - artifacts_started
        q = float(grasp["q"]) if grasp is not None else 0.0
        mark("teacher-gn", f"Original Grasp Network score: {q:.2f}")
        padded = sum(1 for obj in ordered if obj.get("pad"))
        if padded:
            print(f"  decision {decision:3d}: padded {padded} token(s) at the workspace "
                  f"boundary; {len(pack['objects'])} real detections")
        base_record = {"kind": "decision", "t": decision,
                       "eef_sim": eef.round(6).tolist(), "objects": len(ordered),
                       "padded_tokens": padded,
                       "real_detections": len(pack["objects"]),
                       "attempts": attempts, "sense_s": round(sense_s, 3),
                       "gn_s": round(gn_s, 3), "stock_q": round(q, 6),
                       "graspable": bool(grasp and grasp["graspable"])}
        if grasp is not None and grasp["graspable"]:
            stop_reason = "stock_gn_graspable"
            g_final = grasp
            log.append(base_record)
            print(f"  decision {decision:3d}: GN q={q:.3f} -> GRASP")
            break

        obs = teacher_vector(ordered, eef)
        mark("teacher-policy", f"teacher decision {decision + 1}")
        policy_started = time.time()
        action, logits = teacher.step(obs)
        policy_s = time.time() - policy_started
        timing["stage_seconds"]["teacher_inference"] += policy_s
        d1, d2 = two_step_plan(action, total=args.push_total_m)
        low = frames.SIM_WORKSPACE_LIMITS[:2, 0]
        high = frames.SIM_WORKSPACE_LIMITS[:2, 1]
        wp1 = np.clip(eef + d1, low, high)
        wp2 = np.clip(wp1 + d2, low, high)
        proposed_m = float(np.linalg.norm(wp1 - eef) + np.linalg.norm(wp2 - wp1))
        base_record.update(action=action, logits=np.asarray(logits).round(6).tolist(),
                           policy_s=round(policy_s, 6), proposed_travel_m=proposed_m)
        trace["dense"].append({"t": decision, "eef": [float(eef[0]), float(eef[1]),
                                                         args.push_z],
                               "action": action, "grasp_q": q,
                               "obj_xy": [[float(o["x"]), float(o["y"])] for o in ordered]})
        trace["waypoints"].append({"t": decision, "action": action,
                                   "wp1": [float(wp1[0]), float(wp1[1]), args.push_z],
                                   "wp2": [float(wp2[0]), float(wp2[1]), args.push_z]})
        print(f"  decision {decision:3d}: full {len(pack['objects'])}/11"
              f"{f' (+{padded} pad)' if padded else ''}, GN q={q:.3f}, "
              f"teacher action {action:2d}")

        return_s = push_s = retract_s = 0.0
        eef_before = eef.copy()
        if live:
            mark("teacher-return", f"returning to saved push pose for action {action}")
            started = time.time()
            return_to_push(eef)
            return_s = time.time() - started
            timing["stage_seconds"]["return_to_push"] += return_s

            mark("teacher-push", f"executing teacher primitive {action}")
            started = time.time()
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            previous_real = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
            for waypoint in (wp1, wp2):
                rx, ry = frames.sim_to_real(*waypoint)
                if not rtde_c.moveL(pose(rx, ry, args.push_z),
                                    args.tool_vel, args.tool_acc):
                    raise RuntimeError(f"teacher primitive {action} move rejected")
                actual_real = np.asarray(rtde_r.getActualTCPPose()[:2], dtype=float)
                travelled_m += float(np.linalg.norm(actual_real - previous_real))
                previous_real = actual_real
            tcp = rtde_r.getActualTCPPose()
            eef = np.asarray(frames.real_to_sim(tcp[0], tcp[1]), dtype=np.float64)
            push_s = time.time() - started
            timing["stage_seconds"]["push"] += push_s
            steps_run += 1

            mark("teacher-retract", f"retracting after primitive {action}")
            started = time.time()
            back_off = backoff_point((eef, wp1, eef_before),
                                     args.retract_backoff_m, low, high)
            base_record["backoff_sim"] = (None if back_off is None
                                          else back_off.round(6).tolist())
            retract(back_off)
            retract_s = time.time() - started
            timing["stage_seconds"]["retract"] += retract_s
        else:
            eef = wp2
            travelled_m += proposed_m
            steps_run += 1
        base_record.update(return_s=round(return_s, 3), push_s=round(push_s, 3),
                           retract_s=round(retract_s, 3),
                           eef_after_sim=eef.round(6).tolist())
        log.append(base_record)

    timing.update(decisions=decisions, steps_run=steps_run,
                  travelled_m=round(travelled_m, 6), stop_reason=stop_reason,
                  teacher_loop_seconds=round(time.time() - t_run0, 3))
    trace["stop_reason"] = stop_reason
    trace["steps_run"] = steps_run
    trace["final_eef"] = [float(eef[0]), float(eef[1]), args.push_z]

    if live:
        if g_final is not None and g_final["graspable"]:
            started = time.time()
            mark("grasp", f"Original Grasp Network score: {g_final['q']:.2f}")
            grx, gry = frames.rotation_idx_to_tool_orientation(g_final["rotation_idx"])
            gz = max(g_final["surface_z_m"] - args.descend_m, args.table_z + 0.005)
            gx, gy = g_final["x_real"], g_final["y_real"]
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveL([gx, gy, 0.10, grx, gry, 0.0],
                                args.transit_vel, args.transit_acc):
                raise RuntimeError("grasp approach move rejected")
            gripper.open_and_wait_for_pos(80, 120)
            if not rtde_c.moveL([gx, gy, gz, grx, gry, 0.0], 0.06, 0.24):
                raise RuntimeError("grasp descent rejected")
            mark("grasp-close", f"selected orientation bin {g_final['rotation_idx']} of 16")
            # Keep what the CLOSE itself reported. Robotiq's OBJ says whether the
            # fingers stopped on an object; read only after the lift it cannot
            # tell "closed on air" from "gripped it and dropped it during the
            # lift", because a dropped block lets the fingers run on to the
            # commanded position (scene 3: 227/255, OBJ 3, after the
            # operator watched it lift the block).
            closed_pos, closed_status = gripper.move_and_wait_for_pos(
                int(0.9 * gripper.get_max_position()), 80, 120)
            # move_and_wait_for_pos hands back an ObjectStatus ENUM, not an int.
            closed_obj = int(getattr(closed_status, "value", closed_status))
            timing["grasp_close_position"] = int(closed_pos)
            timing["grasp_close_object_status"] = closed_obj
            timing["closed_on_object"] = closed_obj in (1, 2)
            time.sleep(0.2)
            if not rtde_c.moveL([gx, gy, 0.12, grx, gry, 0.0],
                                args.transit_vel, args.transit_acc):
                raise RuntimeError("grasp lift rejected")
            if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                raise RuntimeError("post-grasp PMBS-home move rejected")
            held = int(gripper.get_current_position())
            status_after = gripper.get_object_status()
            timing["gripper_position"] = held
            timing["gripper_object_status"] = status_after
            timing["holding"] = status_after in (1, 2)
            timing["dropped_during_lift"] = bool(timing.get("closed_on_object")
                                                 and not timing["holding"])
            timing["real_grasp"] = public_grasp(g_final)
            if timing["holding"] and not args.keep:
                gripper.open_and_wait_for_pos(80, 120)
                timing["released"] = True
            timing["stage_seconds"]["grasp"] += time.time() - started
            timing["grasp_seconds"] = round(time.time() - started, 3)
        elif stop_reason == "out_of_workspace":
            mark("no-grasp", "OUT OF WORKSPACE: an object was pushed outside the "
                             "44.8 cm workspace")
        else:
            mark("no-grasp", "teacher baseline ended without a graspable GN result")

        # Written before homing: a rejected or collided return must not cost a finished run its
        # record, and the recorder needs real_timing.json / trajectory.json to annotate.
        save_result(time.time() - t_program0)
        if not (args.keep and timing.get("holding")):
            mark("return-home", "returning to experiment home")
            started = time.time()
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                print("WARNING: final experiment-home move rejected; the trial result is already written")
                timing["final_home"] = "rejected"
            else:
                timing["stage_seconds"]["return_home"] += time.time() - started
            save_result(time.time() - t_program0)
        else:
            print("--keep: holding the object at PMBS home for manual removal.")
        mark("done", "")
        recovery["armed"] = False
        rtde_c.stopScript()

    save_result(time.time() - t_program0)

    print("\n=== teacher full-observation run ===")
    for key in ("steps_run", "decisions", "stop_reason", "travelled_m",
                "teacher_loop_seconds", "grasp_seconds", "holding",
                "real_total_seconds"):
        if key in timing:
            print(f"  {key:24s} {timing[key]}")
    print("  stage_seconds            ", timing["stage_seconds"])


if __name__ == "__main__":
    main()
