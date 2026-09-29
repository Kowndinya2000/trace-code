"""PMBS (parallel Monte-Carlo tree search) on the real UR5e, in the shared hardware harness.

Adapted from the published PMBS release (parallel_mcts/real_robot_main.py) onto
the vendored PMBS policy in this folder. PMBS keeps its own
decision loop: observe the full scene, grasp if the grasp network accepts, otherwise
load the scene into the parallel Isaac Gym environments, search for one 10 cm push
(executed at 90 % of its length, as in PMBS), execute it, retract, re-observe. Up to 15
actions. What is shared with the teacher, student and spiral hardware runs:

  * the recorded D455 owns capture (recorder dumps), and the deployed eye-to-hand calibration
    and 44.8 cm workspace frame are used instead of PMBS's constants;
  * scene reconstruction uses the maintained Mask R-CNN + silhouette pose estimator
    (the PMBS-style pipeline in open_loop/perceive_scene.py) and writes the same More
    scene format PMBS's simulator loads;
  * the grasp decision and grasp motion use the stock-GN real-grasp path (threshold 0.70,
    clearance gate) of the teacher and spiral runs;
  * an object whose measured footprint crosses the black-mat edge, or a missing object,
    ends the run as an OOW failure;
  * phase markers, per-decision observation records and timing JSON drive the annotation.
"""
from __future__ import annotations

import atexit
import json
import math
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)                                  # PMBS loads actions/assets relative to cwd
SIM_ROOT = BASE_DIR.parent
ISAAC = SIM_ROOT / "isaacgymenvs"
# Resolved before the imports below, because they are what puts the package on sys.path.
HARDWARE = Path(os.environ.get("TRACE_HARDWARE", SIM_ROOT.parent / "hardware"))
for path in (str(SIM_ROOT), str(HARDWARE)):
    if path not in sys.path:
        sys.path.append(path)

from isaacgym import gymapi, gymutil               # noqa: E402  (isaacgym before torch)
import numpy as np                                   # noqa: E402
import torch                                         # noqa: E402

from constants import GRASP_Q_GRASP_THRESHOLD, GRASP_Q_PUSH_THRESHOLD, PIXEL_SIZE, WORKSPACE_LIMITS  # noqa: E402
from environment import Environment                  # noqa: E402
from mcts_parallel_new.nodes import PushSearchNode   # noqa: E402
from mcts_parallel_new.push import PushState         # noqa: E402
from mcts_parallel_new.search import MonteCarloTreeSearch  # noqa: E402
from mcts_utils import MCTSHelper as PMBSHelper      # noqa: E402

from isaacgymenvs.open_loop import hardware_config  # noqa: E402
from isaacgymenvs.open_loop import frames, mat_boundary, perceive_scene as ps, real_grasp, recorder_io  # noqa: E402
from isaacgymenvs.open_loop.execute_trajectory import HOME_JOINTS_DEG, PMBS_HOME_JOINTS_DEG, STAGING_SIM_XY  # noqa: E402
from isaacgymenvs.open_loop.run_teacher_full_obs import public_grasp  # noqa: E402

OOW_STOPS = ("off_mat", "object_missing")
CALIB = str(hardware_config.calibration())


def parse_args():
    parameters = [
        {"name": "--controller", "type": str, "default": "ik"},
        {"name": "--num_envs", "type": int, "default": 800},
        {"name": "--time_limit", "type": float, "default": 15.0},
        {"name": "--max_actions", "type": int, "default": 15},
        {"name": "--seed", "type": int, "default": 1234},
        {"name": "--execute", "action": "store_true"},
        {"name": "--yes", "action": "store_true"},
        {"name": "--initial_dump", "type": str, "default": ""},
        {"name": "--resense_dir", "type": str, "default": ""},
        {"name": "--resense_name", "type": str, "default": "d455_topdown"},
        {"name": "--calib", "type": str, "default": CALIB},
        {"name": "--maskrcnn", "type": str, "default": str(hardware_config.maskrcnn())},
        {"name": "--push_scale", "type": float, "default": 0.9},
        {"name": "--push_z", "type": float, "default": 0.020},
        {"name": "--safe_z", "type": float, "default": 0.22},
        {"name": "--retract_backoff_m", "type": float, "default": 0.015},
        {"name": "--descent_force_n", "type": float, "default": 50.0},
        {"name": "--push_force_n", "type": float, "default": 80.0},
        {"name": "--resense_attempts", "type": int, "default": 3},
        {"name": "--timing_out", "type": str, "default": ""},
        {"name": "--trace_out", "type": str, "default": ""},
        {"name": "--keep", "action": "store_true"},
        {"name": "--robot_ip", "type": str, "default": hardware_config.robot_ip()},
    ]
    return gymutil.parse_arguments(description="PMBS on the real UR5e", custom_parameters=parameters,
                                   headless=True)


def write_scene(objects, path: Path):
    ordered = [o for o in objects if o["target"]] + [o for o in objects if not o["target"]]
    with open(path, "w") as stream:
        for o in ordered:
            r, g, b = o["color"]
            stream.write(f"{o['name']}.urdf {r:.6f} {g:.6f} {b:.6f} {o['x']:.6f} {o['y']:.6f} "
                         f"{frames.BLOCK_Z:.6f} 0.0 0.0 {o['yaw']:.6f}\n")


def main():
    t_program0 = time.time()

    def save_result(elapsed):
        """Persist the trial result; safe to call before and after the return-to-home move."""
        from isaacgymenvs.open_loop.trial_timing import write_json_atomic
        timing["stage_seconds"] = {k: round(v, 3) for k, v in stage.items()}
        timing["real_total_seconds"] = round(elapsed, 3)
        timing["log"] = log
        timing["observations"] = observations
        trace.update(start_eef=[float(STAGING_SIM_XY[0]), float(STAGING_SIM_XY[1]), args.push_z],
                     stop_reason=stop_reason, steps_run=pushes)
        if args.trace_out:
            write_json_atomic(args.trace_out, trace)
        if args.timing_out:
            write_json_atomic(args.timing_out, timing)

    args = parse_args()
    args.headless = True                             # the robot run never opens a viewer
    live = bool(args.execute)
    if live and not (args.initial_dump and args.resense_dir):
        raise SystemExit("live execution needs --initial_dump and --resense_dir (recorded D455)")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = Path(args.resense_dir) if args.resense_dir else hardware_config.runs_dir() / "pmbs_offline_runs" / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = run_dir / "pmbs_scenes"
    scene_dir.mkdir(exist_ok=True)
    audit_dir = run_dir / "observations"
    audit_dir.mkdir(exist_ok=True)

    def mark(phase, detail=""):
        if args.resense_dir:
            recorder_io.mark(args.resense_dir, phase, detail)

    stage = {k: 0.0 for k in ("model_load", "sim_setup", "initial_home", "perception", "stock_gn",
                              "scene_load", "search", "audit_artifacts", "push", "retract", "grasp",
                              "return_home")}
    log, observations = [], []
    trace = {"schema_version": 1, "controller": "pmbs", "dense": [], "waypoints": []}
    timing = {"controller": "pmbs", "search": "parallel_mcts", "num_envs": args.num_envs,
              "time_limit_s": args.time_limit, "max_actions": args.max_actions,
              "push_distance_m": 0.1, "push_scale": args.push_scale,
              "stock_gn_threshold": real_grasp.GRASPABLE_Q_THRESHOLD,
              "retract_backoff_m": args.retract_backoff_m}

    # ---- models ------------------------------------------------------------------
    started = time.time()
    mark("pmbs-load", "loading segmentation, grasp networks and PMBS search")
    cam2base = np.loadtxt(args.calib)
    device = torch.device("cuda")
    maskrcnn = ps.load_maskrcnn(args.maskrcnn, device)
    from isaacgymenvs.utils.mtcs_utils import MCTSHelper as GraspHelper
    grasp_helper = GraspHelper(str(ISAAC / "logs_grasp/snapshot-post-020000.reinforcement.pth"),
                               str(ISAAC / "logs_grasp/grasp_model-89.pth"), device="cuda")
    pmbs = PMBSHelper(str(ISAAC / "logs_grasp/snapshot-post-020000.reinforcement.pth"),
                      str(ISAAC / "logs_grasp/grasp_model-89.pth"), args.seed)
    stage["model_load"] += time.time() - started

    def load_dump(base):
        import cv2
        return (cv2.imread(base + "_color.png"), np.load(base + "_depth.npy"),
                json.load(open(base + "_K.json")), base)

    def analyze(color, depth, K, base):
        raw = ps.segment(color, args.maskrcnn, device, model=maskrcnn)
        objects, _ = ps.locate_objects(color, depth, K, cam2base, table_z=0.006, device=device,
                                       require_target=False, instances=raw)
        class_id = {name: idx for idx, name in ps.CLASS_ID_TO_NAME.items()}
        kept = [{"mask": o["mask"], "class": class_id[o["name"]], "score": o["score"]} for o in objects]
        return {"color": color, "depth": depth, "K": K, "base": base, "raw": raw, "kept": kept,
                "objects": objects}

    def capture():
        if args.resense_dir:
            return analyze(*recorder_io.request_dump(args.resense_dir, args.resense_name))
        return analyze(*load_dump(args.initial_dump))

    def measure(pack):
        if "mat_measurements" not in pack:
            blocks, _, _ = ps.filter_instances_by_depth(pack["raw"], pack["depth"], pack["K"], cam2base)
            pack["mat_measurements"] = mat_boundary.instance_mat_measurements(
                blocks, pack["depth"], pack["K"], cam2base, mat["corners_sim"])
        return pack["mat_measurements"]

    def save_observation(pack):
        started = time.time()
        n = len(observations)
        observations.append({"n": n, "kind": "clean", "base": pack.get("base")})
        mark("pmbs-observe", f"internal: observation {n}; kind=clean")
        target_idx = ps.pick_target(pack["color"], pack["kept"])
        scene_rgb, _, scene_segm = real_grasp.build_real_heightmap(
            pack["color"], pack["depth"], pack["K"], cam2base, pack["kept"], target_idx=target_idx)
        h, w = pack["color"].shape[:2]
        kept = (np.stack([i["mask"] for i in pack["kept"]]).astype(np.uint8) if pack["kept"]
                else np.zeros((0, h, w), np.uint8))
        np.savez_compressed(audit_dir / f"camera_masks_step_{n:03d}.npz", kept=kept,
                            rejected=np.zeros((0, h, w), np.uint8), scene_rgb=scene_rgb, scene_segm=scene_segm)
        stage["audit_artifacts"] += time.time() - started

    def render_grasp(pack, grasp):
        debug = grasp.pop("_debug", None) if grasp else None
        if debug is not None and pack.get("base") and args.resense_dir:
            import cv2
            cv2.imwrite(pack["base"] + "_heightmap.png", debug["rgb"][:, :, ::-1])
            real_grasp.render_rotation_panel(debug, grasp, pack["base"] + "_gn16.png")

    # ---- initial clean scene, mat, simulator -------------------------------------
    initial = analyze(*load_dump(args.initial_dump)) if args.initial_dump else capture()
    targets = [o for o in initial["objects"] if o["target"]]
    if len(targets) != 1:
        raise SystemExit(f"initial frame has {len(targets)} targets; expected exactly one")
    mat = mat_boundary.detect_mat_polygon(initial["color"], initial["depth"], initial["K"], cam2base)
    initial_measurements = measure(initial)
    mat_exempt = mat_boundary.measured_off_mat_counts(initial_measurements)
    initial_count = mat_boundary.measured_block_count(initial_measurements, corners_sim=mat["corners_sim"])
    timing.update(mat_corners_sim=np.round(mat["corners_sim"], 5).tolist(), mat_exempt=mat_exempt,
                  initial_object_count=initial_count,
                  oow_rule="any measured block top-face point more than 3 mm beyond the black-mat edge, "
                           "or fewer blocks on the mat than initially")
    print(f"mat boundary (sim): {np.round(mat['corners_sim'], 4).tolist()}; {initial_count} blocks, "
          f"closest {min(m['edge_mm'] for m in initial_measurements):.1f} mm from the edge")
    objects0 = list(initial["objects"])
    ps.relax_penetration(objects0)
    scene0 = scene_dir / "decision_000.txt"
    write_scene(objects0, scene0)
    started = time.time()
    mark("pmbs-load", f"creating {args.num_envs} parallel Isaac Gym environments")
    args.test_case = str(scene0)
    env = Environment(args)
    pmbs.set_env(env)
    env_ids_all = torch.arange(env.num_envs, device=pmbs.device)
    env_ids_main = torch.arange(1, device=pmbs.device)
    env.reset_idx(env_ids_all, is_real=True)
    stage["sim_setup"] += time.time() - started
    block_names = list(env.block_names)

    def load_scene(objects):
        """Re-pose the simulated blocks (all environments) to the perceived objects, by class."""
        pool = {}
        for o in objects:
            pool.setdefault("target" if o["target"] else o["name"], []).append(o)
        states = env.default_block_state.clone()
        for i, name in enumerate(block_names):
            key = "target" if i == 0 else name
            if not pool.get(key):
                return f"no perceived {key} for simulated block {i}"
            o = pool[key].pop(0)
            q = gymapi.Quat.from_euler_zyx(0.0, 0.0, float(o["yaw"]))
            states[:, i, :] = torch.tensor([o["x"], o["y"], frames.BLOCK_Z, q.x, q.y, q.z, q.w, 0, 0, 0, 0, 0, 0],
                                           device=states.device)
        env.default_block_state[:] = states
        for _ in range(2):
            try:
                env.reset_idx(env_ids_all, is_real=True)
            except AssertionError:
                print("  simulator: blocks not static after re-posing; continuing")
        return None

    # ---- robot -------------------------------------------------------------------
    rtde_c = rtde_r = gripper = None
    push_rot = None
    state = {"at_pmbs": False, "armed": False}

    def pose(x, y, z, rot=None):
        return [x, y, z, *(push_rot if rot is None else rot)]

    def push_tool_rotation(sx, sy, ex, ey):
        """PMBS environment_real.push: tool yaw from the real-frame push direction."""
        angle = math.atan2(ey - sy, ex - sx)
        if np.pi / 2 < angle < np.pi * 3 / 2:
            angle -= np.pi
        elif np.pi * 3 / 2 <= angle <= np.pi * 2:
            angle -= np.pi * 2
        half = (angle + np.pi / 2) / 2
        return [float(np.cos(half) * np.pi), float(np.sin(half) * np.pi), 0.0]

    def protected_move(target, vel, acc, force_max, what):
        """PMBS protected_move_to: stop on contact instead of forcing through.

        -> (reached, peak force delta N). The TCP force is compared with its value
        just before the move, as in hardware_cartesian's replay guard.
        """
        nonlocal rtde_c
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        baseline = np.asarray(rtde_r.getActualTCPForce()[:3])
        if not rtde_c.moveL(target, vel, acc, True):
            raise RuntimeError(f"{what} rejected")
        peak, started, goal = 0.0, time.time(), np.asarray(target[:3])
        time.sleep(0.05)
        while True:
            force = float(np.linalg.norm(np.asarray(rtde_r.getActualTCPForce()[:3]) - baseline))
            peak = max(peak, force)
            if force > force_max:
                rtde_c.stopL(2.0)
                time.sleep(0.2)
                return False, peak
            near = np.linalg.norm(np.asarray(rtde_r.getActualTCPPose()[:3]) - goal) < 0.001
            still = np.linalg.norm(np.asarray(rtde_r.getActualTCPSpeed()[:3])) < 0.002
            if (near and still) or rtde_c.getAsyncOperationProgress() < 0 and still:
                return True, peak
            if rtde_r.isProtectiveStopped() or time.time() - started > 20.0:
                raise RuntimeError(f"{what} did not complete")
            time.sleep(0.004)

    def move_l(target, vel=0.30, acc=1.20, what="move"):
        nonlocal rtde_c
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        if not rtde_c.moveL(target, vel, acc):
            raise RuntimeError(f"{what} rejected")

    def to_pmbs_home():
        nonlocal rtde_c
        sx, sy = frames.sim_to_real(*STAGING_SIM_XY)
        cur = rtde_r.getActualTCPPose()
        move_l([cur[0], cur[1], args.safe_z, *cur[3:6]], what="lift")
        move_l(pose(sx, sy, args.safe_z), what="staging move")
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("PMBS-home move rejected")
        state["at_pmbs"] = True

    def emergency():
        nonlocal rtde_c
        if not state["armed"]:
            return
        state["armed"] = False
        try:
            if rtde_r.isProtectiveStopped() or rtde_r.isEmergencyStopped():
                print("Emergency return skipped: robot is safety-stopped", flush=True)
                return
            print("Unhandled PMBS error: returning to experiment home", flush=True)
            if not state["at_pmbs"]:
                to_pmbs_home()
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
            rtde_c.stopScript()
        except Exception as exc:
            print(f"EMERGENCY RETURN FAILED: {exc}", flush=True)

    if live:
        if not args.yes and input("Run PMBS on the robot? type 'yes': ").strip() != "yes":
            return
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
        try:
            rtde_c = RTDEControlInterface(args.robot_ip)
        except RuntimeError:                       # first attempt after an interrupted run always fails
            time.sleep(1.0)
            rtde_c = RTDEControlInterface(args.robot_ip)
        state["armed"] = True
        atexit.register(emergency)
        started = time.time()
        mark("initial-home", "moving to experiment home")
        gripper.close_and_wait_for_pos(80, 120)
        rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
            raise RuntimeError("experiment-home move rejected")
        from isaacgymenvs.open_loop.hardware_orientation import fixed_downward_rotation
        push_rot = fixed_downward_rotation(rtde_r.getActualTCPPose()[3:6]).tolist()
        stage["initial_home"] += time.time() - started

    # ---- PMBS decision loop ------------------------------------------------------
    t_run0 = time.time()
    stop_reason, g_final = "action_limit", None
    pushes = grasps = 0
    holding = False
    for decision in range(args.max_actions):
        started = time.time()
        if decision == 0:
            pack = initial                       # clean frame from experiment home
        else:
            pack = None
            for attempt in range(1, args.resense_attempts + 1):
                mark("re-sense", "full observation at PMBS home")
                pack = capture()
                seen = mat_boundary.measured_block_count(measure(pack), corners_sim=mat["corners_sim"])
                if sum(o["target"] for o in pack["objects"]) == 1 and seen >= initial_count:
                    break
                print(f"  decision {decision}: re-sense {attempt}: "
                      f"{sum(o['target'] for o in pack['objects'])} target(s), {seen}/{initial_count} blocks")
        stage["perception"] += time.time() - started
        save_observation(pack)
        measurements = measure(pack)
        off = {k: v - mat_exempt.get(k, 0) for k, v in mat_boundary.measured_off_mat_counts(measurements).items()
               if v > mat_exempt.get(k, 0)}
        seen = mat_boundary.measured_block_count(measurements, corners_sim=mat["corners_sim"])
        record = {"kind": "decision", "t": decision, "objects": len(pack["objects"]), "blocks_on_mat": seen,
                  "closest_edge_mm": min((m["edge_mm"] for m in measurements), default=None)}
        if off or seen < initial_count:
            stop_reason = "off_mat" if off else "object_missing"
            record.update(result=stop_reason, evicted=off,
                          blocks_over_edge=[m for m in measurements if m["edge_mm"] < -3.0])
            log.append(record)
            print(f"  decision {decision}: OOW ({stop_reason}) {off or f'{seen}/{initial_count} blocks'}")
            break
        if sum(o["target"] for o in pack["objects"]) != 1:
            stop_reason = "target_not_visible"
            log.append(dict(record, result=stop_reason))
            break

        started = time.time()
        mark("pmbs-gn", "evaluating Original Grasp Network")
        grasp = real_grasp.compute_hardware_grasp(
            pack["color"], pack["depth"], pack["K"], cam2base, args.maskrcnn,
            device="cuda", helper=grasp_helper, maskrcnn=maskrcnn,
            instances=pack["kept"])
        stage["stock_gn"] += time.time() - started
        render_grasp(pack, grasp)
        q = float(grasp["q"]) if grasp else 0.0
        mark("pmbs-gn", f"Original grasp network score: {q:.2f}")
        record["stock_q"] = round(q, 4)
        if grasp is not None:
            record.update(
                network_graspable=bool(grasp.get("network_graspable", False)),
                graspable=bool(grasp.get("graspable", False)),
                rotation_idx=grasp.get("rotation_idx"),
                grasp_post_processing=grasp.get("grasp_post_processing"),
                hardware_grasp_checks_passed=bool(
                    grasp.get("hardware_grasp_checks_passed", False)),
                grasp_reject_reason=grasp.get("reject_reason"),
            )

        if grasp is not None and grasp["graspable"]:
            grasps += 1
            record["result"] = "grasp"
            log.append(record)
            if not live:
                stop_reason, g_final = "graspable_offline", grasp
                break
            started = time.time()
            mark("grasp", f"Original grasp network score: {q:.2f}")
            grx, gry = frames.rotation_idx_to_tool_orientation(grasp["rotation_idx"])
            gz = max(grasp["surface_z_m"] - 0.040, 0.011)
            gx, gy = grasp["x_real"], grasp["y_real"]
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            if not rtde_c.moveL([gx, gy, 0.10, grx, gry, 0.0], 0.30, 1.20):
                raise RuntimeError("grasp approach rejected")
            gripper.open_and_wait_for_pos(80, 120)
            if not rtde_c.moveL([gx, gy, gz, grx, gry, 0.0], 0.06, 0.24):
                raise RuntimeError("grasp descent rejected")
            mark("grasp-close", f"selected orientation bin {grasp['rotation_idx']} of 16")
            closed_pos, closed_status = gripper.move_and_wait_for_pos(int(0.9 * gripper.get_max_position()), 80, 120)
            closed_obj = int(getattr(closed_status, "value", closed_status))
            time.sleep(0.2)
            if not rtde_c.moveL([gx, gy, 0.12, grx, gry, 0.0], 0.30, 1.20):
                raise RuntimeError("grasp lift rejected")
            if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(), 1.05, 1.4):
                raise RuntimeError("post-grasp PMBS-home move rejected")
            state["at_pmbs"] = True
            status_after = gripper.get_object_status()
            holding = status_after in (1, 2)
            timing.update(grasp_close_position=int(closed_pos), grasp_close_object_status=closed_obj,
                          closed_on_object=closed_obj in (1, 2), gripper_object_status=status_after,
                          holding=holding, dropped_during_lift=closed_obj in (1, 2) and not holding)
            stage["grasp"] += time.time() - started
            g_final = grasp
            if holding:
                stop_reason = "grasp_success"
                if not args.keep:
                    gripper.open_and_wait_for_pos(80, 120)
                break
            gripper.close_and_wait_for_pos(80, 120)
            print(f"  decision {decision}: grasp failed; PMBS continues")
            continue

        # PMBS search on the reconstructed scene
        started = time.time()
        mark("pmbs-search", "loading the observed scene into the parallel simulator")
        objects = list(pack["objects"])
        ps.relax_penetration(objects)
        write_scene(objects, scene_dir / f"decision_{decision:03d}.txt")
        problem = load_scene(objects)
        stage["scene_load"] += time.time() - started
        if problem:
            stop_reason = "scene_mismatch"
            log.append(dict(record, result=stop_reason, detail=problem))
            print(f"  decision {decision}: {problem}")
            break
        started = time.time()
        mark("pmbs-search", f"parallel MCTS: {args.time_limit:g} s over {args.num_envs} environments")
        color_images, depth_images, _ = env.render_camera(env_ids_main, color=True, depth=True, segm=False)
        grasp_q, _, _ = pmbs.get_grasp_q(color_images[0], depth_images[0], post_checking=True)
        _, focal_depths, _ = env.render_camera(env_ids_main, color=False, depth=True, segm=False, focal_target=True)
        classifier_q = pmbs.grasp_eval(focal_depths[0])
        ok, object_states, _ = env.save_object_states(env_ids_main)
        initial_q = grasp_q if (classifier_q > GRASP_Q_PUSH_THRESHOLD and grasp_q <= GRASP_Q_GRASP_THRESHOLD) \
            else classifier_q
        if not ok:
            print(f"  decision {decision}: simulated scene not static before search; searching anyway")
        root_state = PushState("root", object_states[0], float(initial_q), 0, pmbs)
        pmbs.simulation_recorder["root"] = (object_states[0], color_images[0], depth_images[0], float(classifier_q))
        root = PushSearchNode(root_state)
        search = MonteCarloTreeSearch(root, env.num_envs - 1, args.time_limit)
        if classifier_q > GRASP_Q_PUSH_THRESHOLD and grasp_q <= GRASP_Q_GRASP_THRESHOLD:
            PushState.grasp_method = "dqn"
            PushState.max_level = 1
        best = search.best_action_parallel(eval=True)
        stage["search"] += time.time() - started
        if best is None or best.prev_move is None:
            stop_reason = "no_push"
            log.append(dict(record, result=stop_reason))
            break
        p0, p1 = best.prev_move.pos0, best.prev_move.pos1
        start_sim = np.array([p0[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], p0[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]])
        end_sim = np.array([p1[0] * PIXEL_SIZE + WORKSPACE_LIMITS[0][0], p1[1] * PIXEL_SIZE + WORKSPACE_LIMITS[1][0]])
        end_sim = start_sim + (end_sim - start_sim) * args.push_scale
        record.update(result="push", sim_grasp_q=round(float(grasp_q), 4), classifier_q=round(float(classifier_q), 4),
                      push_start_sim=start_sim.round(5).tolist(), push_end_sim=end_sim.round(5).tolist(),
                      search_s=round(time.time() - started, 3))
        trace["dense"].append({"t": decision, "eef": [float(start_sim[0]), float(start_sim[1]), args.push_z],
                               "obj_xy": [[float(o["x"]), float(o["y"])] for o in objects]})
        trace["waypoints"].append({"t": decision, "wp1": [float(start_sim[0]), float(start_sim[1]), args.push_z],
                                   "wp2": [float(end_sim[0]), float(end_sim[1]), args.push_z]})
        print(f"  decision {decision}: stock GN q={q:.3f}; PMBS push {start_sim.round(3)} -> {end_sim.round(3)} "
              f"({100 * np.linalg.norm(end_sim - start_sim):.1f} cm)")
        pmbs.reset()
        del root, root_state, search, best

        if live:
            started = time.time()
            mark("pmbs-push", f"executing push {pushes + 1}")
            sx, sy = frames.sim_to_real(*start_sim)
            ex, ey = frames.sim_to_real(*end_sim)
            rot = push_tool_rotation(sx, sy, ex, ey)
            record["tool_rotation"] = [round(v, 5) for v in rot]
            if state["at_pmbs"] or decision == 0:
                cur = rtde_r.getActualTCPPose()
                move_l(pose(cur[0], cur[1], args.safe_z), what="lift before push")
            move_l(pose(sx, sy, args.safe_z, rot), what="move above push start")
            move_l(pose(sx, sy, args.push_z + 0.03, rot), what="hover over push start")
            state["at_pmbs"] = False
            down_ok, down_force = protected_move(pose(sx, sy, args.push_z, rot), 0.05, 0.25,
                                                 args.descent_force_n, "push-height descent")
            record.update(descent_reached=down_ok, descent_peak_force_n=round(down_force, 1))
            push_ok, push_force = False, 0.0
            if down_ok:
                push_ok, push_force = protected_move(pose(ex, ey, args.push_z, rot), 0.10, 0.50,
                                                     args.push_force_n, "push")
                record.update(push_reached=push_ok, push_peak_force_n=round(push_force, 1))
            else:
                print(f"  decision {decision}: descent met contact ({down_force:.0f} N); push skipped, as PMBS does")
            stage["push"] += time.time() - started
            started = time.time()
            mark("pmbs-retract", "retracting to PMBS home for the next observation")
            if down_ok:
                cur = np.asarray(frames.real_to_sim(*rtde_r.getActualTCPPose()[:2]))
                direction = end_sim - start_sim
                back = cur - direction / max(np.linalg.norm(direction), 1e-9) * args.retract_backoff_m
                bx, by = frames.sim_to_real(*back)
                protected_move(pose(bx, by, args.push_z, rot), 0.05, 0.25, args.push_force_n, "retract back-off")
            to_pmbs_home()
            stage["retract"] += time.time() - started
        pushes += 1
        log.append(record)

    timing.update(pushes=pushes, grasps=grasps, actions=pushes + grasps, stop_reason=stop_reason,
                  oow_failure=stop_reason in OOW_STOPS, decisions=len(observations),
                  pmbs_loop_seconds=round(time.time() - t_run0, 3), real_grasp=public_grasp(g_final))
    if live:
        if not holding:
            if stop_reason in OOW_STOPS:
                mark("no-grasp", "OOW failure: part of an object left the black mat" if stop_reason == "off_mat"
                     else "OOW failure: an object is no longer on the mat")
            else:
                mark("no-grasp", f"PMBS ended without a successful grasp ({stop_reason})")
        # Written before homing so a rejected or collided return cannot destroy the record.
        save_result(time.time() - t_program0)
        if not (args.keep and holding):
            started = time.time()
            mark("home", "returning to home")
            if not state["at_pmbs"]:
                to_pmbs_home()
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
            rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4)
            stage["return_home"] += time.time() - started
        mark("done", "")
        state["armed"] = False
        rtde_c.stopScript()

    save_result(time.time() - t_program0)
    print("\n=== PMBS real run ===")
    for key in ("stop_reason", "pushes", "grasps", "decisions", "pmbs_loop_seconds", "holding", "real_total_seconds"):
        if key in timing:
            print(f"  {key:22s} {timing[key]}")
    print("  stage_seconds          ", timing["stage_seconds"])
    env.close()


if __name__ == "__main__":
    main()
