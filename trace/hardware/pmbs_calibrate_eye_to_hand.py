# Author: Baichuan Huang; local accuracy and capture improvements.
"""Eye-to-hand calibration with retained observations and held-out validation.

Default: show instructions without connecting to hardware.
--collect: original automatic PMBS pose sampling, with checked motion and capture.
--collect --manual: operator positions robot; this program ONLY receives state.
--replay DATASET: reprocess saved observations without camera or robot access.

The operator confirmed the original sampling envelope. No ranges, orientations,
home targets, or speeds are expanded. Robot safety limits are not a scene model.
"""
import argparse
import os
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from calibration_solver import (
    aggregate_observations, detect_observation, make_board, mean_transform,
    pose_matrix, solve_dataset,
)
from calibration_motion import (AutomaticMotion, HOME_JOINTS, POSITION_MIN, POSITION_MAX,
                                TOOL_SPEED, TOOL_ACCELERATION, JOINT_SPEED,
                                JOINT_ACCELERATION, sample_original_poses)

# Preserve the previous capture envelope. These bounds do not establish
# clearance for the arm, gripper, calibration board, or camera mount.
ROTVEC_MIN = np.array([1.36, -0.2, -0.2])
ROTVEC_MAX = np.array([1.76, 0.2, 0.2])


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_robot_state(receiver):
    if not receiver.isConnected():
        raise RuntimeError("Robot state connection lost")
    if receiver.isEmergencyStopped() or receiver.isProtectiveStopped():
        raise RuntimeError("Robot is emergency/protective stopped")
    if int(receiver.getSafetyMode()) not in (1, 2):
        raise RuntimeError("Robot is not in normal or reduced safety mode")
    state = {
        "tcp_pose": receiver.getActualTCPPose(),
        "tcp_speed": receiver.getActualTCPSpeed(),
        "joints": receiver.getActualQ(),
        "joint_speed": receiver.getActualQd(),
        "robot_timestamp_s": receiver.getTimestamp(),
        "host_monotonic_s": time.monotonic(),
    }
    for name in ("tcp_pose", "tcp_speed", "joints", "joint_speed"):
        array = np.asarray(state[name])
        if array.shape != (6,) or not np.isfinite(array).all():
            raise RuntimeError(f"Invalid robot state: {name}")
        state[name] = array.tolist()
    if not np.isfinite(state["robot_timestamp_s"]):
        raise RuntimeError("Invalid robot timestamp")
    return state


def require_recording_pose(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Invalid TCP pose")
    # Allow small tracking/measurement error when recording boundary targets.
    # Commanded targets still have the exact original bounds in validate_target.
    if np.any(pose[:3] < POSITION_MIN - 0.0005) or np.any(pose[:3] > POSITION_MAX + 0.0005):
        raise ValueError("TCP is outside the previous XYZ recording limits; sample rejected")
    if np.any(pose[3:] < ROTVEC_MIN - 0.005) or np.any(pose[3:] > ROTVEC_MAX + 0.005):
        raise ValueError("TCP rotation vector is outside the previous recording limits")


def is_stationary(state):
    speed = np.asarray(state["tcp_speed"])
    return (np.linalg.norm(speed[:3]) < 0.0005
            and np.linalg.norm(speed[3:]) < 0.003
            and np.max(np.abs(state["joint_speed"])) < 0.005)


def wait_until_settled(receiver, timeout=10.0, settle_seconds=0.75):
    deadline, stable_since, last_timestamp = time.monotonic() + timeout, None, None
    last_update = time.monotonic()
    while time.monotonic() < deadline:
        state = read_robot_state(receiver)
        stamp = state["robot_timestamp_s"]
        if last_timestamp is not None and stamp < last_timestamp:
            raise RuntimeError("Robot state timestamp reversed")
        if stamp != last_timestamp:
            last_timestamp, last_update = stamp, time.monotonic()
        elif time.monotonic() - last_update > 0.5:
            raise RuntimeError("Robot state timestamp stopped advancing")
        require_recording_pose(state["tcp_pose"])
        if is_stationary(state):
            stable_since = time.monotonic() if stable_since is None else stable_since
            if time.monotonic() - stable_since >= settle_seconds:
                return state
        else:
            stable_since = None
        time.sleep(0.05)
    raise ValueError("Robot did not settle; no sample recorded")


def check_burst_motion(states, max_translation_m=0.0002, max_rotation_deg=0.05):
    if not all(is_stationary(state) for state in states):
        raise ValueError("Robot moved during the camera burst")
    times = np.array([state["robot_timestamp_s"] for state in states])
    if np.any(np.diff(times) < 0) or times[-1] <= times[0]:
        raise RuntimeError("Stale or reversed robot timestamps during capture")
    host_times = np.array([state["host_monotonic_s"] for state in states])
    if np.any((host_times - host_times[0]) - (times - times[0]) > 0.5):
        raise RuntimeError("Robot telemetry became stale during capture")
    matrices = np.array([pose_matrix(state["tcp_pose"]) for state in states])
    shifts = np.linalg.norm(matrices[:, :3, 3] - matrices[0, :3, 3], axis=1)
    turns = Rotation.from_matrix(matrices[0, :3, :3].T @ matrices[:, :3, :3]).magnitude()
    if shifts.max() > max_translation_m or np.rad2deg(turns.max()) > max_rotation_deg:
        raise ValueError("Robot drift exceeded 0.2 mm or 0.05 degrees during capture")
    for state in states:
        require_recording_pose(state["tcp_pose"])
    return mean_transform(matrices)


def require_distinct_pose(transform, observations):
    for observation in observations:
        previous = np.asarray(observation["tcp_to_base"])
        distance = np.linalg.norm(previous[:3, 3] - transform[:3, 3])
        angle = Rotation.from_matrix(previous[:3, :3].T @ transform[:3, :3]).magnitude()
        if distance < 0.005 and angle < np.deg2rad(2):
            raise ValueError("This pose repeats an existing sample (within 5 mm and 2 degrees)")


def capture_burst(camera, receiver, board, detector, folder, args):
    initial = wait_until_settled(receiver)
    # Discard queued frames acquired before settling. No images taken in motion
    # enter the corner aggregation below.
    for _ in range(16):
        camera.get_color_frame()
    states = [initial]
    valid, raw, last_frame_number = [], [], None
    for index in range(args.burst_frames):
        before = read_robot_state(receiver)
        image, metadata = camera.get_color_frame()
        after = read_robot_state(receiver)
        states.extend((before, after))
        if last_frame_number is not None and metadata["frame_number"] <= last_frame_number:
            raise RuntimeError("Repeated or reversed camera frame number")
        last_frame_number = metadata["frame_number"]
        name = f"frame_{index:02d}.png"
        if not cv2.imwrite(str(folder / name), image):
            raise OSError("Could not save raw calibration image")
        frame = {"image": name, "camera": metadata, "robot_before": before, "robot_after": after}
        try:
            observation = detect_observation(
                image, board, detector, camera.intrinsics, camera.distortion_coeffs,
                max_rms_px=args.max_pnp_rms)
            valid.append(observation)
            frame["detection"] = observation
        except ValueError as error:
            frame["rejection"] = str(error)
        raw.append(frame)
        # Preserve individual observations even when aggregation later rejects a burst.
        write_json(folder / "frames.json", raw)
        if args.preview:
            display = image.copy()
            if "detection" in frame:
                detected = frame["detection"]
                cv2.aruco.drawDetectedCornersCharuco(
                    display, np.float32(detected["corners"]).reshape(-1, 1, 2),
                    np.int32(detected["ids"]).reshape(-1, 1))
            cv2.imshow("Calibration (q stops recording)", display)
            if cv2.waitKey(1) & 0xff == ord("q"):
                raise KeyboardInterrupt
    ee = check_burst_motion(states)
    observation = aggregate_observations(
        valid, board, camera.intrinsics, camera.distortion_coeffs,
        max_jitter_px=args.max_jitter, max_rms_px=args.max_pnp_rms)
    observation["tcp_to_base"] = ee.tolist()
    observation["frames_file"] = str(folder.name + "/frames.json")
    return observation


def save_solution(dataset, folder, args):
    initial = np.loadtxt(args.initial_camera) if args.initial_camera else None
    report = solve_dataset(dataset, validation_fraction=args.validation_fraction,
                           split_seed=args.split_seed, initial_camera=initial)
    report["source_dataset"] = str(args.replay.resolve() if args.replay else folder / "dataset.json")
    report["board"] = dataset.get("board")
    # These are per-run candidates. Never replace the currently deployed calibration.
    write_json(folder / "report.json", report)
    np.savetxt(folder / "camera_to_base_candidate.txt", report["camera_to_base"])
    np.savetxt(folder / "board_to_tcp_candidate.txt", report["board_to_tcp"])
    for label in ("training", "validation"):
        metrics = report[label]
        print(f"{label}: {metrics['poses']} poses; pixel RMS {metrics['pixel_rms']:.3f} px; "
              f"board-origin mean {metrics['board_origin_mean_mm']:.3f} mm; "
              f"orientation mean {metrics['board_orientation_mean_deg']:.3f} deg")
    print(report["validation_note"])
    for warning in report["warnings"]:
        print("Review:", warning)
    print("Saved observations and candidate calibration:", folder)
    return report


def collect(args):
    # Manual collection never imports or constructs the control interface.
    from rtde_receive import RTDEReceiveInterface
    from realsense_camera import Camera

    board, detector = make_board(args.square_mm / 1000, args.marker_mm / 1000)
    folder = args.output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    receiver = camera = control = motion = None
    dataset = None
    poses = [] if args.manual else sample_original_poses(args.poses, args.pose_seed)
    if args.manual:
        print("Operator positions the robot; this program sends no motion commands.")
    else:
        print("Automatic collection uses the original PMBS sampler, orientations and speeds.")
    print("Recording bounds: X [-0.05, 0.05], Y [-0.70, -0.50], Z [0.22, 0.28] m.")
    print(f"Board: 4x4 legacy pattern, {args.square_mm:g} mm squares, "
          f"{args.marker_mm:g} mm markers. Use the measured print dimensions.")
    if args.manual:
        print("Choose distinct positions and the previously used wrist orientations with clear paths.")
    try:
        camera = Camera(args.camera_serial)
        receiver = RTDEReceiveInterface(args.ip, frequency=30, use_upper_range_registers=False)
        dataset = {"version": 1, "created": datetime.now().isoformat(),
                   "collection_mode": "manual_receive_only" if args.manual else "original_automatic",
                   "robot_ip": args.ip, "planned_poses": poses,
                   "pose_seed": args.pose_seed, "home_joints": HOME_JOINTS.tolist(),
                   "motion": {"tool_speed_m_s": TOOL_SPEED, "tool_acceleration_m_s2": TOOL_ACCELERATION,
                              "joint_speed_rad_s": JOINT_SPEED, "joint_acceleration_rad_s2": JOINT_ACCELERATION,
                              "skip_home": args.skip_home},
                   "camera_serial": camera.serial,
                   "camera_matrix": camera.intrinsics.tolist(),
                   "distortion": camera.distortion_coeffs.tolist(),
                   "object_points": board.getChessboardCorners().tolist(),
                   "board": {"squares": [4, 4], "square_mm": args.square_mm,
                             "marker_mm": args.marker_mm, "legacy": True},
                   "capture_parameters": {"burst_frames": args.burst_frames,
                                          "max_jitter_px": args.max_jitter,
                                          "max_pnp_rms_px": args.max_pnp_rms},
                   "observations": []}
        write_json(folder / "dataset.json", dataset)
        if not args.manual:
            print("Planned TCP poses (xyz metres, rotation vector radians):")
            for index, pose in enumerate(poses):
                print(f"  {index + 1:02d}: {np.round(pose, 5).tolist()}")
            response = input("Clear the original home/calibration paths, verify the board is secure, "
                             "then type RUN to start (anything else exits): ")
            if response.strip() != "RUN":
                return
            if not is_stationary(read_robot_state(receiver)):
                raise RuntimeError("Robot is moving; do not start a second controller program")
            from rtde_control import RTDEControlInterface
            control = RTDEControlInterface(args.ip)
            dataset["active_tcp_offset"] = control.getTCPOffset()
            write_json(folder / "dataset.json", dataset)
            motion = AutomaticMotion(control, receiver, read_robot_state)
            motion.preflight(poses, include_home=not args.skip_home)
            if not args.skip_home:
                print("Moving to the original joint home pose...")
                motion.home()
            else:
                require_recording_pose(read_robot_state(receiver)["tcp_pose"])
        attempt = 0
        while (len(dataset["observations"]) < args.poses if args.manual else attempt < len(poses)):
            if args.manual:
                response = input(f"Pose {len(dataset['observations']) + 1}/{args.poses}: "
                                 "position robot, then Enter to record; q to finish: ").strip().lower()
                if response == "q":
                    break
            else:
                print(f"Moving to pose {attempt + 1}/{len(poses)}: {poses[attempt]}")
                motion.move_to(poses[attempt])
            attempt += 1
            sample_folder = folder / f"capture_{attempt:03d}"
            sample_folder.mkdir()
            try:
                observation = capture_burst(camera, receiver, board, detector, sample_folder, args)
                observation["planned_pose_index"] = attempt - 1 if not args.manual else None
                require_distinct_pose(np.asarray(observation["tcp_to_base"]), dataset["observations"])
                dataset["observations"].append(observation)
                write_json(folder / "dataset.json", dataset)
                print(f"Accepted {len(observation['ids'])}/9 ChArUco corners; "
                      f"jitter {observation['corner_jitter_p95_px']:.3f} px; "
                      f"PnP RMS {observation['pnp_rms_px']:.3f} px")
            except ValueError as error:
                write_json(sample_folder / "rejected.json", {"reason": str(error)})
                print("Sample rejected:", error)
    except (KeyboardInterrupt, EOFError):
        print("\nRecording stopped; collected observations retained.")
    finally:
        cleanup = []
        if motion is not None:
            cleanup.append(("stop motion", motion.stop))
        if control is not None:
            cleanup.extend((("stop control script", control.stopScript),
                            ("disconnect control", control.disconnect)))
        if camera is not None:
            cleanup.append(("stop camera", camera.stop_streaming))
        if receiver is not None:
            cleanup.append(("disconnect receiver", receiver.disconnect))
        if args.preview:
            cleanup.append(("close preview", cv2.destroyAllWindows))
        cleanup_errors = []
        for label, action in cleanup:
            try:
                action()
            except Exception as error:
                cleanup_errors.append(f"{label}: {error}")
        if cleanup_errors:
            raise RuntimeError("Cleanup failed; check robot state: " + "; ".join(cleanup_errors))
    if dataset is not None and len(dataset["observations"]) >= 16:
        save_solution(dataset, folder, args)
    else:
        print("Fewer than 16 accepted poses; observations saved but no transform fitted:", folder)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--collect", action="store_true", help="Collect with the original automatic pose sampler")
    mode.add_argument("--replay", type=Path, metavar="DATASET_JSON", help="Offline solve of saved observations")
    mode.add_argument("--plan", action="store_true", help="Print original sampled targets; no hardware")
    parser.add_argument("--ip", default=os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"))
    parser.add_argument("--camera-serial", default=os.environ.get("TRACE_D455_SERIAL"),
                        help="RealSense serial; defaults to TRACE_D455_SERIAL (first device if unset)")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "calibration_runs")
    parser.add_argument("--poses", type=int, default=36)
    parser.add_argument("--pose-seed", type=int, default=7, help="Reproducible original pose sampler")
    parser.add_argument("--manual", action="store_true", help="Operator positions robot; receive-only capture")
    parser.add_argument("--skip-home", action="store_true", help="Start inside the original capture envelope")
    parser.add_argument("--burst-frames", type=int, default=7)
    parser.add_argument("--square-mm", type=float, default=25.0)
    parser.add_argument("--marker-mm", type=float, default=18.75)
    parser.add_argument("--max-pnp-rms", type=float, default=1.0, help="Per-frame/aggregate rejection limit, pixels")
    parser.add_argument("--max-jitter", type=float, default=0.35, help="95th percentile corner jitter limit, pixels")
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--initial-camera", type=Path, help="Optional previous camera-to-base matrix initializer")
    parser.add_argument("--preview", action="store_true", help="Optional OpenCV GUI; disabled by default")
    args = parser.parse_args(argv)
    if not 16 <= args.poses <= 162 or args.burst_frames < 3:
        parser.error("Use 16..162 poses and at least 3 frames per pose")
    if not 0.15 <= args.validation_fraction <= 0.4:
        parser.error("Validation fraction must be 0.15..0.4")
    if args.poses - max(4, round(args.poses * args.validation_fraction)) < 12:
        parser.error("Need at least twelve training poses after the validation split")
    if not 0 <= args.pose_seed < 2**32 or args.split_seed < 0:
        parser.error("Pose seed must be 0..2**32-1 and split seed nonnegative")
    if not (np.isfinite(args.max_pnp_rms) and args.max_pnp_rms > 0
            and np.isfinite(args.max_jitter) and args.max_jitter > 0):
        parser.error("Rejection thresholds must be finite and positive")
    if args.initial_camera:
        initial = np.loadtxt(args.initial_camera)
        if (initial.shape != (4, 4) or not np.isfinite(initial).all()
                or not np.allclose(initial[3], [0, 0, 0, 1])
                or not np.allclose(initial[:3, :3].T @ initial[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(initial[:3, :3]), 1, atol=1e-5)):
            parser.error("Initial camera matrix must be a finite rigid 4x4 transform")
    make_board(args.square_mm / 1000, args.marker_mm / 1000)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.plan:
        print(json.dumps({"home_joints": HOME_JOINTS.tolist(),
                          "poses": sample_original_poses(args.poses, args.pose_seed)}, indent=2))
    elif args.replay:
        dataset = json.loads(args.replay.read_text())
        folder = args.output / datetime.now().strftime("replay_%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=False)
        save_solution(dataset, folder, args)
    elif args.collect:
        collect(args)
    else:
        print("No hardware connected. Use --collect for the original automatic calibration route,")
        print("--collect --manual for operator-positioned capture, or --replay DATASET_JSON.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as error:
        print("Calibration stopped:", error)
        raise SystemExit(1)
