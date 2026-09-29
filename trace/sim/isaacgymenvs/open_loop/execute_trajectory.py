"""Stage 3: replay a solved trajectory on the real UR5e — OPEN LOOP.

Loads an OpenLoopTrajectory JSON (from solve_in_twin.py), converts every
point sim -> robot base frame (frames.sim_to_real), and executes with no
online perception:

  executed  (default): blended moveL through the straight-segment
      simplification of the ACTUAL twin EEF trace — the same path
      replay_in_twin.py certifies.
  physics_movel: blended moveL through every physics-tick ACTUAL EEF position.
      This preserves within-primitive bends while holding the tool orientation
      fixed, and uses the same moveL speed profile as executed mode.
  dense: servoL through the raw per-step trace (resampled at --dense-dt).
  commanded: moveL through the policy's commanded wp1/wp2 list (analysis
      only — overshoots the actual sim motion, see trajectory.py).
  cartesian_feedforward: every physics pose, smoothly retimed and streamed
      using the validated Cartesian preview controller. Requires a complete
      raised controller validation artifact. Tool orientation is fixed downward;
      simulator tilt is discarded. Extra air checks are opt-in.

SAFETY: this script is a DRY RUN unless --execute is passed. Dry run prints
the full real-frame plan and bounds-checks every point. --probe connects
READ-ONLY and compares the live TCP with the trajectory's expected start
pose (the frames.REAL_FRAME_OFFSET check, no motion). With --execute it
still asks for confirmation before the first motion. Force limits are the
robot's own; validate the route and cell safety before enabling execution.

Sequence with --execute: [--home: moveJ home] -> close fingers -> hover over
start -> descend to --push-z -> path -> lift -> orient per grasp rotation
bin -> open -> descend to (top face - --descend-m) -> close -> lift -> PMBS home.

TODO(hardware): verify before the first full run —
  - frames.REAL_FRAME_OFFSET (run --probe at home; see the CAUTION in frames.py)
  - --push-z (TCP height while pushing with the stock 2F-85, closed fingers)

Example:
  python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json           # dry run
  python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json --probe
  python open_loop/execute_trajectory.py open_loop/out/real2sim/000000.json --execute --home
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import cv2

from isaacgymenvs.open_loop import hardware_config
from isaacgymenvs.open_loop import frames
from isaacgymenvs.open_loop import recorder_io
from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory

# sim default_joint_angles (cfg/task/MoreOpenLoop.yaml) with the base joint
# +90 deg: the sim<->robot frames differ by Z(-90). Same as trace/hardware/go_home.py's
# "Standard 2F-85" pose. Degrees.
HOME_JOINTS_DEG = [-19.32 + 90.0, -100.29, 147.48, -137.18, -89.72, 160.73]

# PMBS environment_real.py go_home(). Retracted and high (TCP z 0.212),
# well clear of the top-down camera's view of the mat -- unlike HOME_JOINTS_DEG,
# which parks the tool INSIDE the workspace 6 mm off the table. That is the
# pose to re-sense and grasp from; HOME_JOINTS_DEG is only the pre-push start.
PMBS_HOME_JOINTS_DEG = [12.44, -127.35, 127.41, -90.1, -89.51, 102.6]

# Fixed Cartesian staging point (sim frame) used before every moveJ home, so the
# arm always starts that joint-space move from the same configuration. Near the
# workspace edge closest to the base, where the elbow-up solution is unambiguous.
STAGING_SIM_XY = (0.32, 0.03)


def to_real_xy(path_sim):
    """[[x,y,z]...] sim -> [(x_r, y_r)...] robot base frame."""
    return [frames.sim_to_real(p[0], p[1]) for p in path_sim]


def bounds_check(points_real, margin=0.03):
    lo = frames.REAL_WORKSPACE_LIMITS[:2, 0] - margin
    hi = frames.REAL_WORKSPACE_LIMITS[:2, 1] + margin
    bad = [(i, p) for i, p in enumerate(points_real)
           if not (lo[0] <= p[0] <= hi[0] and lo[1] <= p[1] <= hi[1])]
    return bad


def resample_dense(path_real, step_m=0.005):
    """Distance-resample the dense polyline for smooth servoL streaming."""
    pts = np.asarray(path_real, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(s[-1] / step_m))
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, pts[:, 0]), np.interp(si, s, pts[:, 1])], axis=1)


def commanded_primitive_pairs(path_real, has_start_pose):
    """Return the saved wp1->wp2 pair for each commanded primitive.

    ``OpenLoopTrajectory.commanded_path`` includes ``start_eef`` for plotting
    and approach planning. It is not part of the first primitive.
    """
    points = path_real[1:] if has_start_pose else path_real
    if len(points) % 2:
        raise ValueError(f"Commanded path has {len(points)} waypoint entries; expected wp1/wp2 pairs")
    return list(zip(points[::2], points[1::2]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trajectory", help="OpenLoopTrajectory JSON from solve_in_twin.py")
    ap.add_argument("--mode", choices=["executed", "physics_movel", "dense",
                                      "commanded", "cartesian_feedforward"],
                    default="executed")
    ap.add_argument("--hardware-validation", type=Path, help="Passed full raised replay JSON for the selected controller")
    ap.add_argument("--hardware-rate", type=float, default=1.2)
    ap.add_argument("--air-check", action="store_true", help="Opt-in commissioning replay above clutter before this trial")
    ap.add_argument("--execute", action="store_true", help="actually move the robot")
    ap.add_argument("--yes", action="store_true",
                    help="skip the typed confirmation (for scripted runs that were "
                         "already approved by the operator)")
    ap.add_argument("--probe", action="store_true",
                    help="read-only: compare live TCP with the expected start pose")
    ap.add_argument("--home", action="store_true", help="moveJ to HOME_JOINTS_DEG first (with --execute)")
    ap.add_argument("--robot-ip", default=hardware_config.robot_ip())
    ap.add_argument("--push-z", type=float, default=0.020, help="TCP z while pushing (m)")
    ap.add_argument("--hover-z", type=float, default=0.100, help="TCP z for approach/transit (m)")
    ap.add_argument("--keep", action="store_true",
                    help="do not release the object at the end (it then blocks the "
                         "return to home, so the arm parks at PMBS home instead)")
    ap.add_argument("--descend-m", type=float, default=0.040,
                    help="drop this far below the measured top face (PMBS: 40 mm)")
    ap.add_argument("--table-z", type=float, default=0.006, help="table height, real frame")
    ap.add_argument("--lift-z", type=float, default=0.085, help="TCP z after grasping")
    ap.add_argument("--approach-direct-m", type=float, default=0.03,
                    help="if the first waypoint is within this of the live TCP, drop "
                         "straight to push height instead of lifting and traversing")
    ap.add_argument("--safe-z", type=float, default=0.22,
                    help="TCP z for transit ABOVE the clutter; moveJ alone dips through it")
    # PMBS environment_real.py: tool_vel 0.3, tool_acc 1.2, applied at a
    # speed_scale per motion -- 1.0 for free-space transit, 0.3 while pushing
    # (0.09 / 0.36), 0.2 for the final descent (0.06 / 0.24). Free-space motion
    # has no reason to crawl at contact speed; that split is most of the win.
    # The push leg is pinned to the TWIN, not to PMBS: the trajectory being
    # replayed was produced by blocks responding to the sim's EEF, so the
    # contact speed has to be the sim's. Measured on the dense path (one sample
    # per RL env step): 18.4 mm per step at 15.06 Hz = 0.277 m/s mean, 0.288
    # max. PMBS's 0.09 m/s contact speed is a THIRD of that and would be the
    # mismatch, not the match. Acceleration cannot be matched -- the sim reaches
    # speed inside one 16.6 ms physics frame, ~17 m/s^2 -- so it takes the
    # highest rate the arm runs in free space.
    ap.add_argument("--tool-vel", type=float, default=0.28, help="push speed (m/s)")
    ap.add_argument("--tool-acc", type=float, default=1.20, help="push accel (m/s^2)")
    ap.add_argument("--transit-vel", type=float, default=0.30, help="free-space speed")
    ap.add_argument("--transit-acc", type=float, default=1.20, help="free-space accel")
    ap.add_argument("--joint-vel", type=float, default=1.05)  # PMBS go_home
    ap.add_argument("--joint-acc", type=float, default=1.4)
    # The sim never decelerates at a waypoint, so a stop there is itself the
    # infidelity. Blending is capped at 45% of the shorter adjacent segment
    # (~8 mm on the 18 mm steps the policy takes), which is the closest the
    # controller can get to the twin's uninterrupted sweep.
    ap.add_argument("--max-push-cm", type=float, default=50.0,
                    help="refuse a plan whose push path exceeds this (0 = no gate). "
                         "Measured over 13 real runs: 7/7 failures above 50 cm, "
                         "3/3 successes below it.")
    ap.add_argument("--force-long", action="store_true",
                    help="execute a plan that exceeds --max-push-cm anyway")
    ap.add_argument("--max-contact-force-delta", type=float, default=40.0,
                    help="stop Cartesian replay when TCP force changes by more than this many N")
    ap.add_argument("--blend", type=float, default=0.005, help="moveL path blend radius (m)")
    ap.add_argument("--dense-dt", type=float, default=0.05, help="servoL period in dense mode (s)")
    ap.add_argument("--servo", action="store_true",
                    help="dense mode only: stream with servoL instead of a blended "
                         "moveL path. servoL has no velocity cap -- the push runs as "
                         "fast as the servo gain allows. Off by default so every mode "
                         "pushes at the twin's measured EEF speed.")
    ap.add_argument("--resense-dir", default=None,
                    help="record_cameras.py output dir; enables home -> re-sense -> grasp")
    ap.add_argument("--resense-name", default="d455_topdown", help="recorder --name of the D455")
    ap.add_argument("--calib", default=str(hardware_config.calibration()))
    ap.add_argument("--maskrcnn", default=str(hardware_config.maskrcnn()))
    ap.add_argument("--timing-out", default=None, help="write a real-run timing JSON here")
    ap.add_argument("--sim-timing", default=None,
                    help="sim timing JSON from record_solve.py, to report end-to-end time")
    args = ap.parse_args()

    traj = OpenLoopTrajectory.load(args.trajectory)
    if not traj.metadata.get("solved", False):
        # A failed twin rollout is 400+ steps of dithering with no grasp at the
        # end; replaying it only bulldozes the scene (demo39: 637 cm, 65 s).
        # Independent of the plan-length gate, which --max-push-cm 0 waives.
        print("ABORT: twin did not solve this scene (trajectory marked NOT solved); nothing to replay.")
        sys.exit(1)
    print(f"Loaded {args.trajectory}: {len(traj.waypoints)} primitives, "
          f"{len(traj.dense)} dense samples, grasp={'yes' if traj.grasp else 'no'}")

    cartesian = args.mode == 'cartesian_feedforward'
    uses_physics = cartesian or args.mode == 'physics_movel'
    if uses_physics:
        from isaacgymenvs.open_loop.trajectory import validate_episode_continuity
        validate_episode_continuity(traj.dense)
        if not traj.physics:
            raise ValueError(f'{args.mode} hardware replay requires a full physics trace')
        path_sim = [s['eef_state'][:3] for s in [traj.physics['initial']]+traj.physics['samples']]
    else:
        path_sim = traj.path(args.mode)
    path_real = to_real_xy(path_sim)
    if args.mode == "dense":
        path_real = resample_dense(path_real).tolist()

    bad = bounds_check(path_real)
    if bad:
        print(f"ABORT: {len(bad)} point(s) outside the real workspace "
              f"(first: idx {bad[0][0]} at {bad[0][1]}). Check frames.REAL_FRAME_OFFSET.")
        sys.exit(1)

    grasp_real = None
    if traj.grasp:
        gx, gy = frames.sim_to_real(traj.grasp["x_sim"], traj.grasp["y_sim"])
        grx, gry = frames.rotation_idx_to_tool_orientation(traj.grasp["rotation_idx"])
        grasp_real = (gx, gy, grx, gry)
        if bounds_check([(gx, gy)]):
            print(f"ABORT: grasp point ({gx:+.4f}, {gy:+.4f}) outside the real workspace.")
            sys.exit(1)

    total = sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path_real[:-1], path_real[1:]))
    print(f"\n=== open-loop plan ({args.mode} mode, {len(path_real)} points, "
          f"{100 * total:.1f} cm of path) ===")
    # A purposeful solve sweeps out; a dithering one doubles back. net/total
    # near 1 is a directed path, near 0 is the policy shuffling in place.
    net = float(np.hypot(path_real[-1][0] - path_real[0][0],
                         path_real[-1][1] - path_real[0][1]))
    if total > 0:
        direct = net / total
        print(f"path: {100 * total:.0f} cm travelled, {100 * net:.0f} cm net "
              f"displacement (directness {direct:.2f})"
              + ("   <- DITHERING: the policy doubles back a lot" if direct < 0.25 else ""))
    # Plan length predicts open-loop failure, measured over 13 real runs: every
    # plan longer than ~50 cm of push path failed at the re-sense (7 of 7),
    # while all three successes were under it. Divergence accumulates per push,
    # so past roughly 15 primitives the replayed scene no longer resembles the
    # twin's. Warn before spending the run; --max-push-cm 0 disables the gate.
    if args.max_push_cm and 100 * total > args.max_push_cm:
        print(f"\n*** PLAN TOO LONG for open-loop replay: {100 * total:.0f} cm > "
              f"{args.max_push_cm:.0f} cm.\n"
              f"*** Every run above this threshold has failed at the re-sense "
              f"(7/7); all successes were below it.\n"
              f"*** Re-arrange the scene for a shorter solve, or pass "
              f"--max-push-cm 0 to run it anyway.")
        if not args.force_long:
            sys.exit(2)
    print(f"start (real): ({path_real[0][0]:+.4f}, {path_real[0][1]:+.4f})  push z={args.push_z}")
    for i, p in enumerate(path_real[:12]):
        print(f"  wp[{i:03d}] ({p[0]:+.4f}, {p[1]:+.4f})")
    if len(path_real) > 12:
        print(f"  ... {len(path_real) - 12} more")
    if grasp_real:
        print(f"grasp (real): ({grasp_real[0]:+.4f}, {grasp_real[1]:+.4f}) "
              f"tool rot ({grasp_real[2]:+.3f}, {grasp_real[3]:+.3f}), "
              f"q={traj.grasp['q']:.3f}")

    if args.probe:
        from rtde_receive import RTDEReceiveInterface as RTDEReceive
        tcp = RTDEReceive(args.robot_ip).getActualTCPPose()
        sx, sy = path_real[0]
        print(f"\n[probe] live TCP  : ({tcp[0]:+.4f}, {tcp[1]:+.4f}, {tcp[2]:+.4f})")
        print(f"[probe] expected  : ({sx:+.4f}, {sy:+.4f}) at home "
              f"(sim start {np.round(traj.start_eef, 4).tolist()})")
        print(f"[probe] delta     : ({tcp[0] - sx:+.4f}, {tcp[1] - sy:+.4f}) m  <- candidate "
              f"frames.REAL_FRAME_OFFSET if the robot is at HOME_JOINTS_DEG")

    if not args.execute:
        print("\nDry run only. Re-run with --execute to move the robot.")
        return

    if cartesian:
        from isaacgymenvs.open_loop.hardware_cartesian import (
            ContinuousCartesianReference, PlaybackRateReference, stream_cartesian)
        from isaacgymenvs.open_loop.hardware_orientation import FixedDownwardReference
        from isaacgymenvs.tools.validate_cartesian_hardware import (
            preflight_dashboard, stationary, check_route, move_checked, home_checked)
        if not args.hardware_validation or not args.resense_dir:
            raise ValueError('Cartesian execution requires --hardware-validation and --resense-dir')
        if not np.allclose(frames.REAL_FRAME_OFFSET, 0):
            raise ValueError('Selected controller was commissioned for zero XY frame offset')
        proof = json.loads(args.hardware_validation.read_text())
        summary, provenance = proof['summary'], proof['provenance']
        if provenance.get('robot_ip') != args.robot_ip:
            raise ValueError('Raised validation was recorded for a different robot address')
        controller_hash = hashlib.sha256(Path(__file__).with_name('hardware_cartesian.py').read_bytes()).hexdigest()
        source_hash = hashlib.sha256(Path(args.trajectory).read_bytes()).hexdigest()
        if not (summary.get('passed') and summary.get('full_trace_validated')
                and summary.get('shutdown', {}).get('stationary')
                and summary.get('shutdown', {}).get('program_stopped')):
            raise ValueError('Selected raised validation did not pass motion AND shutdown checks')
        if provenance['controller_sha256'] != controller_hash:
            raise ValueError('Controller code changed since the selected raised validation')
        expected = dict(actuator='servol', frequency=250., preview_s=.07, lookahead_s=.05,
                        servo_gain=500., pose_correction=0., playback_rate=args.hardware_rate)
        if any(summary.get(k, 1. if k == 'playback_rate' else None) != v for k, v in expected.items()):
            raise ValueError('Requested controller parameters differ from the raised validation')
        needs_air_check = args.air_check
        args.transit_vel, args.transit_acc = min(args.transit_vel, .06), min(args.transit_acc, .2)
        args.joint_vel, args.joint_acc = min(args.joint_vel, .25), min(args.joint_acc, .4)
        preflight_dashboard(args.robot_ip)

    if not args.yes:
        if input("\nType 'yes' to execute on the robot: ").strip().lower() != "yes":
            print("Aborted.")
            return

    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    from robotiq_gripper import RobotiqGripper

    # gripper first: it needs no robot control program, so a failure here cannot
    # strand an RTDEControl script on the robot (trace/hardware/go_home.py's usage)
    gripper = RobotiqGripper(args.robot_ip, 63352)
    gripper.connect()
    rtde_c = RTDEControl(args.robot_ip, 250. if cartesian else -1.)
    rtde_r = RTDEReceive(args.robot_ip, frequency=500.)
    speed, force = 80, 120

    # Load the re-sense models on a background thread. Building Mask R-CNN and
    # the grasp network put ~3 s of checkpoint I/O in the middle of the timed
    # run; here it overlaps the homing and the whole push phase, so by the time
    # the re-sense needs them the load has long finished.
    model_future = None
    if args.resense_dir:
        from concurrent.futures import ThreadPoolExecutor

        def _load_models():
            import torch
            from isaacgymenvs.open_loop import perceive_scene as _ps
            from isaacgymenvs.utils.mtcs_utils import MCTSHelper
            tw = time.time()
            m = _ps.load_maskrcnn(args.maskrcnn, torch.device("cuda"))
            h = MCTSHelper("logs_grasp/snapshot-post-020000.reinforcement.pth",
                           "logs_grasp/grasp_model-89.pth", device="cuda")
            print(f"[preload] re-sense models ready in {time.time() - tw:.1f} s")
            return m, h

        if not cartesian:
            model_future = ThreadPoolExecutor(1).submit(_load_models)

    from isaacgymenvs.open_loop import real_grasp as _real_grasp
    timing = {"mode": args.mode, "trajectory": os.path.abspath(args.trajectory),
              "max_contact_force_delta_n": args.max_contact_force_delta,
              "stock_gn_threshold": _real_grasp.GRASPABLE_Q_THRESHOLD}
    t_run0 = time.time()
    phases, _t0_prev = recorder_io.read_phases(args.resense_dir) if args.resense_dir else ([], None)

    def mark(phase, detail=""):
        """Timestamped phase marker for the camera recorders to burn in.

        Appends: perception and the twin solver log into the same file earlier
        in the run, and overwriting would erase the part of the story that
        happens before the robot moves."""
        nonlocal phases
        # t0=None: perception logged first and owns the run's epoch. Passing
        # ours would rewrite it and push every earlier stage negative.
        phases = recorder_io.mark(args.resense_dir, phase, detail, events=phases)
    try:
        reference = None
        if cartesian:
            tcp_offset = rtde_c.getTCPOffset()
            if not np.allclose(tcp_offset, [0, 0, .187, 0, 0, 0], atol=1e-4):
                raise RuntimeError('Active gripper TCP offset changed')
            home_pose = np.asarray(rtde_c.getForwardKinematics(np.deg2rad(HOME_JOINTS_DEG).tolist(), tcp_offset))
            reference = PlaybackRateReference(ContinuousCartesianReference(traj.physics, home_pose[3:], args.push_z), args.hardware_rate)
            reference = FixedDownwardReference(reference, home_pose[3:])
            rb = reference.bounds()
            if rb['xyz_min'][2] < .012 or rb['max_speed_m_s'] > .10:
                raise RuntimeError('Contact reference exceeds height or speed limits')
            if needs_air_check:
                mark('validate', 'checking this new physics trace at 15 cm height')
                air = PlaybackRateReference(ContinuousCartesianReference(traj.physics, home_pose[3:], .15), args.hardware_rate)
                air = FixedDownwardReference(air, home_pose[3:])
                initial = stationary(rtde_r)
                lifted = initial['pose'].copy(); lifted[2] = .22
                above = air.initial_pose.copy(); above[2] = .22
                route = [lifted, above, air.initial_pose]
                q = check_route(rtde_c, initial, route)
                grid = np.unique(np.r_[air.times, np.arange(0, air.duration, .04)])
                check_route(rtde_c, dict(pose=air.initial_pose, q=q), air.sample(grid)[0][1:], samples_per_segment=2)
                for point in route: move_checked(rtde_c, rtde_r, point)
                air_output = Path(args.resense_dir)/'hardware_air_validation.json'
                if air_output.exists():
                    raise FileExistsError('Hardware air validation artifact already exists')
                air_result = stream_cartesian(rtde_c, rtde_r, air, air_output, frequency=250.,
                    acceleration=.5, max_speed=.10, max_angular_speed=.20, min_z=.10,
                    max_force_delta=15., max_error=.010, actuator='servol', preview=.07,
                    position_gain=3., pose_correction=0.)
                if not air_result['passed']:
                    raise RuntimeError('New trajectory failed its raised check; no contact replay')
                rtde_c.disconnect()
                preflight_dashboard(args.robot_ip)
                rtde_c = RTDEControl(args.robot_ip, 250.)
            if args.home:
                mark('initial-home', 'returning to experiment home')
                home_checked(rtde_c, rtde_r)
            initial = stationary(rtde_r)
            if np.linalg.norm(initial['pose'][:2]-reference.initial_pose[:2]) > .003:
                raise RuntimeError('Contact replay must start from the checked experiment home')
            q = check_route(rtde_c, initial, [reference.initial_pose])
            grid = np.unique(np.r_[reference.times, np.arange(0, reference.duration, .04)])
            check_route(rtde_c, dict(pose=reference.initial_pose, q=q), reference.sample(grid)[0][1:], samples_per_segment=2)
            timing['hardware_reference'] = rb
            timing['orientation_reference_sha256'] = hashlib.sha256(Path(__file__).with_name('hardware_orientation.py').read_bytes()).hexdigest()
            timing['hardware_validation'] = str(args.hardware_validation.resolve())
            timing['new_trace_air_check'] = needs_air_check
            timing['trajectory_matches_air_validation'] = provenance['trajectory_sha256'] == source_hash
            timing['trajectory_sha256'] = source_hash
        elif args.home:
            rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(), args.joint_vel, args.joint_acc)

        # keep the current tool orientation for the whole push phase
        tcp = rtde_r.getActualTCPPose()
        push_rot = tcp[3:6]

        def pose(x, y, z, rot=push_rot):
            return [x, y, z, rot[0], rot[1], rot[2]]

        # 1. close the fingers for pushing, then get to the first waypoint.
        #    The trajectory starts from the twin's home EEF, so after --home the
        #    arm is already there (~1 mm) and the old lift-to-hover / traverse /
        #    descend was pure dead time. Only route over the clutter when the
        #    approach actually has to travel.
        mark("approach", "closing gripper")
        gripper.close_and_wait_for_pos(speed, force)
        sx, sy = path_real[0]
        approach = float(np.hypot(tcp[0] - sx, tcp[1] - sy))
        if cartesian:
            move_checked(rtde_c, rtde_r, reference.initial_pose)
        elif approach <= args.approach_direct_m:
            print(f"start is {1000 * approach:.1f} mm away — descending straight to push height")
            # fast most of the way down, PMBS descent speed for the last 30 mm
            rtde_c.moveL([pose(sx, sy, args.push_z + 0.03) +
                          [args.transit_vel, args.transit_acc, 0.005],
                          pose(sx, sy, args.push_z) + [0.06, 0.24, 0.0]])
        else:
            print(f"start is {1000 * approach:.1f} mm away — routing over the clutter at hover")
            rtde_c.moveL([pose(tcp[0], tcp[1], args.hover_z) +
                          [args.transit_vel, args.transit_acc, 0.02],
                          pose(sx, sy, args.hover_z) +
                          [args.transit_vel, args.transit_acc, 0.02],
                          pose(sx, sy, args.push_z + 0.03) +
                          [args.transit_vel, args.transit_acc, 0.005],
                          pose(sx, sy, args.push_z) + [0.06, 0.24, 0.0]])

        # 2. the pushing trajectory
        mark("push", f"0/{len(path_real) - 1} waypoints")
        t0 = time.time()
        if cartesian:
            push_output = Path(args.resense_dir)/'hardware_replay.json'
            if push_output.exists():
                raise FileExistsError('Hardware replay artifact already exists')
            result = stream_cartesian(rtde_c, rtde_r, reference, push_output, frequency=250.,
                acceleration=.5, max_speed=.10, max_angular_speed=.20, min_z=.012,
                max_force_delta=args.max_contact_force_delta, max_error=.010,
                actuator='servol', preview=.07,
                position_gain=3., pose_correction=0.)
            timing['hardware_replay'] = result
            if not result['passed']:
                raise RuntimeError('Contact replay failed its tracking criteria; no automatic continuation')
            rtde_c.disconnect()
            preflight_dashboard(args.robot_ip)
            rtde_c = RTDEControl(args.robot_ip, 250.)
        elif args.mode == "dense" and args.servo:
            # servoL tracks as fast as its gain allows -- there is NO velocity
            # limit here. Run this way the dense path executed in 0.485 s and
            # shoved the whole scene aside. Opt-in only.
            print("WARNING: --servo streams servoL with no speed cap.")
            for x, y in path_real[1:]:
                tp = rtde_c.initPeriod()
                rtde_c.servoL(pose(x, y, args.push_z), 0.0, 0.0,
                              args.dense_dt, 0.1, 300)
                rtde_c.waitPeriod(tp)
            rtde_c.servoStop()
        else:
            # ONE blended path move, run asynchronously. A moveL per waypoint
            # decelerates to a full stop at every one of them (~0.3 s each,
            # ~3 s over a 12-point path) for no reason: the whole path is a
            # continuous drag at push_z. moveL's PATH overload takes a LIST OF
            # 9-lists [x,y,z,rx,ry,rz,speed,acc,blend] -- passing a single flat
            # 9-list picks the wrong overload and hangs, which is what bit us
            # before. getAsyncOperationProgress() keeps the per-waypoint phase
            # log the camera overlay reads.
            if args.mode == "commanded":
                # The commanded path is wp1,wp2 PER PRIMITIVE. Between one
                # primitive's wp2 and the next one's wp1 the EEF was never in
                # contact -- in the twin it repositions freely. Dragging a
                # closed gripper along those jumps at push height is what made
                # this mode scatter the scene (demo16: 126 cm travelled for 26
                # cm of net displacement). Lift between primitives, descend to
                # push, sweep, lift -- PMBS's own push, one per primitive.
                primitive_pairs = commanded_primitive_pairs(
                    path_real, has_start_pose=traj.start_eef is not None)
                if len(primitive_pairs) != len(traj.waypoints):
                    raise ValueError(
                        f"Commanded replay recovered {len(primitive_pairs)} primitive pairs "
                        f"from {len(traj.waypoints)} saved primitives")
                n_prim = len(primitive_pairs)
                mark("push", f"0/{n_prim} primitives")
                for k, ((ax, ay), (bx, by)) in enumerate(primitive_pairs):
                    rtde_c.moveL([pose(ax, ay, args.hover_z) +
                                  [args.transit_vel, args.transit_acc, 0.01],
                                  pose(ax, ay, args.push_z) + [0.06, 0.24, 0.0]])
                    rtde_c.moveL(pose(bx, by, args.push_z),
                                 args.tool_vel, args.tool_acc)
                    rtde_c.moveL(pose(bx, by, args.hover_z),
                                 args.transit_vel, args.transit_acc)
                    mark("push", f"{k + 1}/{n_prim} primitives")
                mark("push", f"{n_prim}/{n_prim} primitives")
            else:
                n_wp = len(path_real) - 1
                pts = [np.array([x, y]) for x, y in path_real]
                seg = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(n_wp)]
                path = []
                for k in range(1, n_wp + 1):
                    # blend must fit inside both adjacent segments or the controller
                    # rejects the path; the corner it cuts is <= this radius.
                    nb = min(seg[k - 1], seg[k]) if k < n_wp else 0.0
                    b = 0.0 if k == n_wp else min(args.blend, 0.45 * nb)
                    path.append(pose(*path_real[k], args.push_z) +
                                [args.tool_vel, args.tool_acc, b])
                replay_output = Path(args.resense_dir) / "hardware_replay.json" if args.resense_dir else None
                if replay_output and replay_output.exists():
                    raise FileExistsError("Hardware replay artifact already exists")
                robot_t0 = float(rtde_r.getTimestamp())
                host_t0 = time.monotonic()
                measured = []

                def sample_tcp(progress):
                    stamp = float(rtde_r.getTimestamp())
                    if measured and stamp <= measured[-1]["robot_timestamp_s"]:
                        return
                    measured.append({
                        "time_s": stamp - robot_t0,
                        "host_time_s": time.monotonic() - host_t0,
                        "robot_timestamp_s": stamp,
                        "async_waypoint_progress": int(progress),
                        "tcp_pose": list(rtde_r.getActualTCPPose()),
                        "tcp_speed": list(rtde_r.getActualTCPSpeed()),
                    })

                sample_tcp(0)
                rtde_c.moveL(path, asynchronous=True)
                # ProgressEx, not getAsyncOperationProgress(): the latter returns <0
                # BOTH before the controller has started the move and after it ends,
                # so a plain "break when <0" loop can exit before the arm moves.
                t_a, started, last_progress = time.time(), False, None
                while True:
                    st = rtde_c.getAsyncOperationProgressEx()
                    if st.isAsyncOperationRunning():
                        started = True
                        progress = st.progress()
                        sample_tcp(progress)
                        if progress != last_progress:
                            mark("push", f"{min(progress + 1, n_wp)}/{n_wp} waypoints")
                            last_progress = progress
                    elif started or time.time() - t_a > 1.0:
                        break          # grace only BEFORE the move starts
                    time.sleep(0.004)  # 250 Hz geometry trace; moveL runs onboard
                sample_tcp(n_wp)
                mark("push", f"{n_wp}/{n_wp} waypoints")
                if replay_output:
                    from isaacgymenvs.open_loop.movel_trace import summarize
                    from isaacgymenvs.open_loop.trial_timing import write_json_atomic
                    reference_poses = [pose(x, y, args.push_z) for x, y in path_real]
                    teacher_duration = (traj.physics["samples"][-1]["time_s"]
                                        if args.mode == "physics_movel" and traj.physics else None)
                    replay_summary = summarize(reference_poses, measured, teacher_duration)
                    replay_summary.update({
                        "mode": args.mode,
                        "fixed_orientation_rotvec": list(push_rot),
                        "configured_speed_m_s": args.tool_vel,
                        "configured_acceleration_m_s2": args.tool_acc,
                        "configured_blend_radius_m": args.blend,
                        "requested_measurement_rate_hz": 250,
                    })
                    write_json_atomic(replay_output, {
                        "summary": replay_summary,
                        "reference_poses": reference_poses,
                        "samples": measured,
                    })
                    timing["hardware_replay"] = replay_summary
                    print("MOVEL_HARDWARE_REPLAY_RESULT", json.dumps(replay_summary), flush=True)
        print(f"Push phase done in {time.time() - t0:.1f} s.")

        t_push = time.time() - t0
        timing["push_seconds"] = round(t_push, 3)

        # 3. lift clear, go HOME, and re-sense: after the pushes the twin's
        #    prediction is stale, so graspability is decided on a fresh real
        #    frame (home does not occlude the top-down camera).
        cur = rtde_r.getActualTCPPose()
        if not args.resense_dir:
            rtde_c.moveL(pose(cur[0], cur[1], args.hover_z), args.transit_vel, args.transit_acc)

        if args.resense_dir:
            th = time.time()
            # Straight up, then straight to PMBS home -- the pose the re-sense
            # and the grasp both happen from. Routing via HOME_JOINTS_DEG first
            # was two extra moves to a pose that is inside the workspace and in
            # the camera's way, so it had to be left again immediately.
            # The climb still comes first: moveJ interpolates in joint space and
            # from a pose inside the clutter it swings the tool THROUGH it (it
            # clipped a block on the first real run). From safe_z both endpoints
            # are ~0.21 m up, which is what PMBS's push does too -- lift
            # 10 cm, then go_home().
            cur = rtde_r.getActualTCPPose()
            mark("home", "clearing to safe height")
            if cartesian:
                home_checked(rtde_c, rtde_r, np.deg2rad(PMBS_HOME_JOINTS_DEG))
            else:
                rtde_c.moveL(pose(cur[0], cur[1], args.safe_z),
                             args.transit_vel, args.transit_acc)
            # STAGING POSE, then moveJ. A push can finish anywhere in the
            # workspace, and the IK branch at the far corners leaves a wrist
            # near its limit: moveJ interpolates in JOINT space from there and
            # can run a joint into its stop on the way to PMBS home. Going
            # Cartesian to one fixed pose first makes the joint configuration
            # the same on every reset, so the moveJ that follows is always the
            # same short, known-good interpolation.
            sx_, sy_ = frames.sim_to_real(*STAGING_SIM_XY)
            if not cartesian:
                rtde_c.moveL(pose(sx_, sy_, args.safe_z),
                             args.transit_vel, args.transit_acc)
                rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(),
                             args.joint_vel, args.joint_acc)
            timing["home_seconds"] = round(time.time() - th, 3)
            mark("re-sense", "capturing + Mask R-CNN + grasp network")

            ts = time.time()
            maskrcnn, grasp_helper = _load_models() if cartesian else model_future.result()
            color, depth, K, base = recorder_io.request_dump(args.resense_dir, args.resense_name)
            from isaacgymenvs.open_loop import real_grasp
            cam2base = np.loadtxt(args.calib)
            # OOW failure rule shared with the student, spiral and PMBS hardware runs,
            # applied to the one observation this open-loop baseline makes after pushing:
            # any measured block top face beyond the black-mat edge, or fewer blocks on
            # the mat than in the clean initial frame, ends the trial without a grasp.
            import torch
            from isaacgymenvs.open_loop import mat_boundary, perceive_scene as _ps
            oow_stop, mat = None, None
            initial_base = os.path.join(args.resense_dir, args.resense_name + "_dump0")

            def _measure(c_, d_, K_):
                raw_ = _ps.segment(c_, args.maskrcnn, torch.device("cuda"), model=maskrcnn)
                blocks_, _, _ = _ps.filter_instances_by_depth(raw_, d_, K_, cam2base)
                return mat_boundary.instance_mat_measurements(blocks_, d_, K_, cam2base, mat["corners_sim"])

            try:
                if os.path.exists(initial_base + "_color.png"):
                    c0 = cv2.imread(initial_base + "_color.png")
                    d0 = np.load(initial_base + "_depth.npy")
                    with open(initial_base + "_K.json") as f0:
                        K0 = json.load(f0)
                    mat = mat_boundary.detect_mat_polygon(c0, d0, K0, cam2base)
                    initial_m = _measure(c0, d0, K0)
                else:
                    mat = mat_boundary.detect_mat_polygon(color, depth, K, cam2base)
                    initial_m = None
                final_m = _measure(color, depth, K)
                exempt = mat_boundary.measured_off_mat_counts(initial_m) if initial_m else {}
                off = {k: v - exempt.get(k, 0)
                       for k, v in mat_boundary.measured_off_mat_counts(final_m).items()
                       if v > exempt.get(k, 0)}
                seen = mat_boundary.measured_block_count(final_m, corners_sim=mat["corners_sim"])
                expected = (mat_boundary.measured_block_count(initial_m, corners_sim=mat["corners_sim"])
                            if initial_m else seen)
                timing.update(mat_corners_sim=np.round(mat["corners_sim"], 5).tolist(),
                              initial_object_count=expected, final_blocks_on_mat=seen,
                              closest_edge_mm=min((m["edge_mm"] for m in final_m), default=None))
                if off or seen < expected:
                    oow_stop = "off_mat" if off else "object_missing"
                    timing["oow_evicted"] = off
                    print(f"Re-sense: OOW failure ({oow_stop}) {off or f'{seen}/{expected} blocks on the mat'}")
            except RuntimeError as exc:
                print(f"[oow] mat check unavailable ({exc}); grasp decision unchanged")
            timing["stop_reason"] = oow_stop or "replay_complete"
            timing["oow_failure"] = oow_stop is not None
            g = (None if oow_stop else
                 real_grasp.compute_hardware_grasp(
                     color, depth, K, cam2base, args.maskrcnn,
                     helper=grasp_helper, maskrcnn=maskrcnn))
            timing["resense_seconds"] = round(time.time() - ts, 3)
            if g is None:
                if not oow_stop:
                    print("Re-sense: no purple/blue target detected — not grasping.")
                grasp_real = None
            else:
                dbg = g.pop("_debug")
                cv2.imwrite(base + "_heightmap.png", dbg["rgb"][:, :, ::-1])
                # the 16-rotation search, as a figure. Post-processing only --
                # written after resense_seconds is already stopped.
                real_grasp.render_rotation_panel(dbg, g, base + "_gn16.png")
                timing["real_grasp"] = g
                # The one number that says whether the open-loop premise holds:
                # where the twin said the target would end up after this exact
                # trajectory, against where it actually did. Recorded on every
                # run, success or failure -- a failed run is the informative one.
                ft = getattr(traj, "final_target", None)
                if ft and g.get("target_sim"):
                    err = float(np.hypot(ft[0] - g["target_sim"][0],
                                         ft[1] - g["target_sim"][1]))
                    timing["twin_predicted_target"] = [round(v, 4) for v in ft[:2]]
                    timing["real_target"] = [round(v, 4) for v in g["target_sim"]]
                    timing["twin_vs_real_target_mm"] = round(1000 * err, 1)
                    print(f"twin-vs-real target error: {1000 * err:.0f} mm  "
                          f"(twin said {ft[0]:.3f},{ft[1]:.3f}; "
                          f"real {g['target_sim'][0]:.3f},{g['target_sim'][1]:.3f})")
                print(f"Re-sense: {g['n_instances']} objects, graspability q={g['q']:.3f} "
                      f"({'GRASPABLE' if g['graspable'] else 'not graspable'})")
                grasp_decision = ("GRASPABLE" if g["graspable"] else
                                  g.get("reject_reason", "below 0.70"))
                mark("re-sense", f"{g['n_instances']} objects, graspability {g['q']:.2f} "
                                 f"({grasp_decision})")
                if g["graspable"]:
                    grx, gry = frames.rotation_idx_to_tool_orientation(g["rotation_idx"])
                    # Depth from the MEASURED top face, as grasp_now.py does
                    # (PMBS: surface - 40 mm). A fixed grasp_z was 7 mm higher
                    # than the 11 mm that actually closed on the cylinder.
                    grasp_z = max(g["surface_z_m"] - args.descend_m,
                                  args.table_z + 0.005)
                    print(f"surface {1000 * g['surface_z_m']:.0f} mm -> descend to "
                          f"{1000 * grasp_z:.0f} mm")
                    grasp_real = (g["x_real"], g["y_real"], grx, gry, grasp_z)
                    # Same limit as the other hardware runs: the grasp point must lie on
                    # the black mat (the mat extends past the 44.8 cm workspace).
                    off_mat_point = (mat is not None and cv2.pointPolygonTest(
                        mat["corners_sim"].astype(np.float32).reshape(-1, 1, 2),
                        (float(g["x_sim"]), float(g["y_sim"])), False) < 0)
                    if off_mat_point if mat is not None else bounds_check([(g["x_real"], g["y_real"])], 0.10):
                        print("ABORT grasp: re-sensed point is off the black mat.")
                        grasp_real = None
                    else:
                        print(f"Grasp from REAL camera: ({g['x_real']:+.4f}, {g['y_real']:+.4f}) "
                              f"rot bin {g['rotation_idx']}")
                else:
                    grasp_real = None

        tg = time.time()
        # The re-sense just spent seconds in Mask R-CNN and the grasp network
        # with this control script idle; revive it before commanding the grasp.
        if cartesian:
            stationary(rtde_r)
            if not rtde_c.isProgramRunning():
                raise RuntimeError('Control program stopped before grasp; no automatic recovery')
        else:
            rtde_c = recorder_io.ensure_control(rtde_c, args.robot_ip)
        oow_detail = {"off_mat": "OOW failure: part of an object left the black mat",
                      "object_missing": "OOW failure: an object is no longer on the mat"}
        mark("grasp" if grasp_real else "no-grasp",
             "executing" if grasp_real else
             oow_detail.get(timing.get("stop_reason"), "target not graspable"))
        if grasp_real:
            gx, gy, grx, gry, grasp_z = grasp_real
            # transit and gripper opening are independent -- overlap them
            hover_pose = [gx, gy, args.hover_z, grx, gry, 0.0]
            if not rtde_c.moveL(hover_pose, args.transit_vel,
                                args.transit_acc, asynchronous=True):
                raise RuntimeError("RTDE rejected grasp approach")
            t_a = time.time()
            gripper.open_and_wait_for_pos(speed, force)   # >0.25 s, covers startup
            while rtde_c.getAsyncOperationProgressEx().isAsyncOperationRunning() \
                    or time.time() - t_a < 0.25:
                if not rtde_c.isProgramRunning():
                    raise RuntimeError("RTDE control script stopped during grasp approach")
                time.sleep(0.01)
            actual_hover = np.asarray(rtde_r.getActualTCPPose())
            if np.linalg.norm(actual_hover[:3] - hover_pose[:3]) > .001:
                raise RuntimeError("Robot did not reach the calibrated grasp approach pose")
            grasp_pose = [gx, gy, grasp_z, grx, gry, 0.0]
            if not rtde_c.moveL(grasp_pose, 0.06, 0.24):
                raise RuntimeError("RTDE rejected grasp descent")
            actual_grasp = np.asarray(rtde_r.getActualTCPPose())
            if np.linalg.norm(actual_grasp[:3] - grasp_pose[:3]) > .001:
                raise RuntimeError("Robot did not reach the calibrated grasp pose")
            grasp_pos = int(0.9 * gripper.get_max_position())   # rl_policy convention
            gripper.move_and_wait_for_pos(grasp_pos, speed, force)
            time.sleep(0.2)
            if not rtde_c.moveL([gx, gy, args.lift_z, grx, gry, 0.0],
                                args.transit_vel, args.transit_acc):
                raise RuntimeError("RTDE rejected post-grasp lift")
            if cartesian:
                home_checked(rtde_c, rtde_r, np.deg2rad(PMBS_HOME_JOINTS_DEG))
            else:
                if not rtde_c.moveJ(np.deg2rad(PMBS_HOME_JOINTS_DEG).tolist(),
                                    args.joint_vel, args.joint_acc):
                    raise RuntimeError("RTDE rejected return to PMBS home")
            held = gripper.get_current_position()
            timing["gripper_position"] = int(held)
            timing["holding"] = bool(held < int(0.9 * gripper.get_max_position()) - 5)
            print(f"Grasp sequence complete. gripper {held}/255 "
                  f"({'HOLDING' if timing['holding'] else 'EMPTY'})")
            if timing["holding"] and not args.keep:
                # Release HERE, retracted and off the mat, like PMBS does.
                # It also has to happen before the arm parks: HOME_JOINTS_DEG
                # puts the TCP 6 mm off the table INSIDE the workspace, so
                # carrying the object there would drive it into the clutter.
                gripper.open_and_wait_for_pos(speed, force)
                timing["released"] = True
        else:
            print("Target not graspable after the pushes — lifted clear, no grasp attempted.")

        # The arm's resting pose is OURS, always -- PMBS home is only ever a
        # transient waypoint for the re-sense and the grasp. Every run ends
        # where the next one starts.
        if not (args.keep and timing.get("holding")):
            mark("home", "returning to home")
            if cartesian:
                home_checked(rtde_c, rtde_r)
            else:
                if not rtde_c.moveJ(np.deg2rad(HOME_JOINTS_DEG).tolist(),
                                    args.joint_vel, args.joint_acc):
                    raise RuntimeError("RTDE rejected final experiment-home move")
        else:
            print("--keep: holding the object, parked at PMBS home instead of home.")
        mark("done", "")
        timing["grasp_seconds"] = round(time.time() - tg, 3)
        timing["real_total_seconds"] = round(time.time() - t_run0, 3)
        timing['status'] = 'completed'
    except BaseException as exc:
        timing.update(status='failed', failure=str(exc), real_total_seconds=round(time.time()-t_run0, 3))
        mark('failed', str(exc))
        raise
    finally:
        try:
            if rtde_c.isProgramRunning(): rtde_c.stopScript()
        finally:
            rtde_c.disconnect()
            rtde_r.disconnect()
            if args.timing_out:
                from isaacgymenvs.open_loop.trial_timing import write_json_atomic
                write_json_atomic(args.timing_out, timing)

    # --- timing report ------------------------------------------------------
    if args.sim_timing and os.path.exists(args.sim_timing):
        import json as _json
        sim = _json.load(open(args.sim_timing))
        # record_solve.py is a later visualization replay. Its compute time is
        # useful profiling data, but it did not occur in this robot execution
        # and must not be added to end-to-end or called policy rollout.
        timing["sim_replay_compute_wall_seconds"] = sim.get(
            "replay_compute_wall_seconds_to_graspable",
            sim.get("replay_stepping_seconds_to_graspable"))
        timing["trajectory_simulated_seconds"] = sim.get("trajectory_simulated_seconds")
    print("\n=== timing ===")
    for k, v in timing.items():
        if not isinstance(v, dict):
            print(f"  {k:24s} {v}")
    if args.timing_out:
        import json as _json
        with open(args.timing_out, "w") as f:
            _json.dump(timing, f, indent=2)
        print(f"  written -> {args.timing_out}")


if __name__ == "__main__":
    main()
