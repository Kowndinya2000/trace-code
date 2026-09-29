"""Commission full Cartesian trace replay above the clutter, then save errors.

Default is a controller/IK preflight without motion. --execute performs one
raised replay. This tool never descends to push height or moves the gripper.
"""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from isaacgymenvs.open_loop.trajectory import OpenLoopTrajectory
from isaacgymenvs.open_loop.hardware_cartesian import CartesianReference, GentleCartesianReference, ContinuousCartesianReference, PlaybackRateReference, stream_cartesian, pose_error
from isaacgymenvs.open_loop.hardware_orientation import FixedDownwardReference
from isaacgymenvs.open_loop import hardware_config

EXPERIMENT_HOME = np.deg2rad([70.68, -100.29, 147.48, -137.18, -89.72, 160.73])


def preflight_dashboard(robot_ip):
    with socket.create_connection((robot_ip, 29999), timeout=3) as stream:
        stream.recv(4096)
        for cmd, expected in [('running', 'Program running: false'), ('robotmode', 'Robotmode: RUNNING'),
                              ('safetystatus', 'Safetystatus: NORMAL'), ('is in remote control', 'true')]:
            stream.sendall((cmd+'\n').encode()); reply = stream.recv(4096).decode().strip()
            if reply != expected:
                raise RuntimeError(f'{cmd}: {reply}; no takeover or recovery')


def read_state(r):
    if not r.isConnected() or r.getSafetyMode() != 1 or r.getRobotMode() != 7:
        raise RuntimeError('Robot is not in normal running state')
    values = dict(pose=np.asarray(r.getActualTCPPose()), speed=np.asarray(r.getActualTCPSpeed()),
                  force=np.asarray(r.getActualTCPForce()), q=np.asarray(r.getActualQ()), timestamp=r.getTimestamp())
    if not all(np.isfinite(v).all() for v in values.values()):
        raise RuntimeError('Invalid telemetry')
    return values


def stationary(r):
    a = read_state(r); time.sleep(.15); b = read_state(r)
    if b['timestamp'] <= a['timestamp'] or max(np.linalg.norm(a['speed']), np.linalg.norm(b['speed'])) > .005:
        raise RuntimeError('Stale telemetry or moving robot')
    return b


def validate_pose(p):
    if not np.isfinite(p).all() or not (-.23 <= p[0] <= .23 and -.7 <= p[1] <= -.16 and .003 <= p[2] <= .25):
        raise RuntimeError('Route outside the commissioned robot workspace')


def check_route(c, initial, poses, samples_per_segment=25):
    q, previous = initial['q'].copy(), initial['pose'].copy()
    for end in poses:
        rotate = Slerp([0, 1], Rotation.from_rotvec([previous[3:], end[3:]]))
        for fraction in np.linspace(0, 1, samples_per_segment):
            p = np.r_[(1-fraction)*previous[:3]+fraction*end[:3], rotate(fraction).as_rotvec()]
            validate_pose(p)
            if not c.isPoseWithinSafetyLimits(p.tolist()) or not c.getInverseKinematicsHasSolution(p.tolist(), q.tolist()):
                raise RuntimeError('Controller rejects route pose or IK')
            next_q = np.asarray(c.getInverseKinematics(p.tolist(), q.tolist()))
            if np.max(np.abs(next_q-q)) > .2 or not c.isJointsWithinSafetyLimits(next_q.tolist()):
                raise RuntimeError('Unsafe or discontinuous route IK')
            q = next_q
        previous = end
    return q


def move_checked(c, r, target):
    start = stationary(r)
    error = pose_error(start['pose'], target)
    if np.linalg.norm(error[:3]) < .0003 and np.linalg.norm(error[3:]) < .001:
        return
    speed, accel = (.015, .05) if min(start['pose'][2], target[2]) < .06 else (.025, .08)
    old = c.getAsyncOperationProgress(); moving = False
    try:
        if not c.moveL(target.tolist(), speed, accel, True):
            raise RuntimeError('Approach command rejected')
        moving = True
        deadline = time.monotonic()+45; stamp = start['timestamp']; updated = time.monotonic(); saw = False
        while True:
            now = time.monotonic(); current = read_state(r)
            if current['timestamp'] != stamp: stamp, updated = current['timestamp'], now
            if now-updated > .5 or now > deadline or not c.isProgramRunning():
                raise RuntimeError('Approach lost telemetry/control or timed out')
            if np.linalg.norm(current['force'][:3]-start['force'][:3]) > 15:
                raise RuntimeError('Unexpected contact during approach')
            validate_pose(current['pose'])
            progress = c.getAsyncOperationProgress(); saw |= progress >= 0
            if progress < 0 and (saw or progress != old):
                err = pose_error(current['pose'], target)
                if np.linalg.norm(err[:3]) > .001 or np.linalg.norm(err[3:]) > .004:
                    raise RuntimeError('Approach ended away from its target')
                stationary(r)
                return
            time.sleep(.02)
    except BaseException:
        if moving: c.stopL(accel)
        raise


def home_checked(c, r, joints=EXPERIMENT_HOME):
    """Reach the requested home through checked, raised Cartesian transits."""
    joints = np.asarray(joints)
    initial = stationary(r)
    if np.max(np.abs(initial['q']-joints)) < np.deg2rad(.15):
        return
    end = np.asarray(c.getForwardKinematics(joints.tolist(), c.getTCPOffset()))
    validate_pose(end)
    lifted = initial['pose'].copy(); lifted[2] = .22
    staging = np.r_[.03, -.32, .22, end[3:]]
    above = end.copy(); above[2] = .22
    route = [lifted, staging, above]
    if end[2] < .05:
        near = end.copy(); near[2] = .05
        route.append(near)
    route.append(end)
    q = check_route(c, initial, route)
    if np.max(np.abs(q-joints)) > .02:
        raise RuntimeError('Home route reaches a different IK branch')
    for target in route:
        move_checked(c, r, target)
    final = stationary(r)
    if np.max(np.abs(final['q']-joints)) > np.deg2rad(.2):
        raise RuntimeError('Home joint tolerance was not reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--robot-ip', default=hardware_config.robot_ip())
    parser.add_argument('--start-z', type=float, default=.15)
    parser.add_argument('--time-scale', type=float, default=4.)
    parser.add_argument('--timing', choices=('continuous', 'gentle', 'original'), default='continuous')
    parser.add_argument('--speed-fraction', type=float, default=.45,
                        help='Fraction of the earlier quarter-speed test; derivative limits can slow it further')
    parser.add_argument('--playback-rate', type=float, default=1.,
                        help='Uniform speed multiplier of the smooth reference (0.5..1.2)')
    parser.add_argument('--max-physics-ticks', type=int, default=0, help='Prefix only; not a full-trace validation')
    parser.add_argument('--frequency', type=float, default=250.)
    parser.add_argument('--position-gain', type=float, default=3.)
    parser.add_argument('--actuator', choices=('servol', 'speedl'), default='servol')
    parser.add_argument('--preview', type=float, default=.07)
    args = parser.parse_args()
    if not .12 <= args.start_z <= .18:
        parser.error('Commissioning is raised only: start-z must be 0.12..0.18 m')
    if not 60 <= args.frequency <= 500 or not 0 <= args.position_gain <= 20:
        parser.error('Frequency must be 60..500 Hz and position gain 0..20')
    if args.out.exists():
        raise FileExistsError('Use a new output path for each measured attempt')
    trajectory = OpenLoopTrajectory.load(str(args.trajectory))
    if trajectory.physics is None:
        raise ValueError('A full physics trace is required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from rtde_receive import RTDEReceiveInterface
    from rtde_control import RTDEControlInterface
    preflight_dashboard(args.robot_ip)
    r, c = RTDEReceiveInterface(args.robot_ip, frequency=500), None
    try:
        initial = stationary(r)
        c = RTDEControlInterface(args.robot_ip, args.frequency)
        tcp = c.getTCPOffset()
        if not np.allclose(tcp, [0, 0, .187, 0, 0, 0], atol=1e-4):
            raise RuntimeError('Active TCP offset differs from the inspected gripper')
        current_fk = np.asarray(c.getForwardKinematics(initial['q'].tolist(), tcp))
        if np.linalg.norm(pose_error(current_fk, initial['pose'])) > .001:
            raise RuntimeError('FK and robot telemetry disagree')
        home_pose = np.asarray(c.getForwardKinematics(EXPERIMENT_HOME.tolist(), tcp))
        physics = dict(trajectory.physics)
        if args.max_physics_ticks:
            if not 1 <= args.max_physics_ticks <= len(physics['samples']):
                raise ValueError('Invalid prefix length')
            physics['samples'] = physics['samples'][:args.max_physics_ticks]
        if args.timing == 'continuous':
            reference = ContinuousCartesianReference(physics, home_pose[3:], args.start_z, args.speed_fraction)
            if args.playback_rate != 1.:
                reference = PlaybackRateReference(reference, args.playback_rate)
        elif args.timing == 'gentle':
            reference = GentleCartesianReference(physics, home_pose[3:], args.start_z)
        else:
            reference = CartesianReference(physics, home_pose[3:], args.start_z, args.time_scale)
        if args.timing != 'continuous' and args.playback_rate != 1.:
            raise ValueError('Playback-rate override requires the smooth continuous profile')
        reference = FixedDownwardReference(reference, home_pose[3:])
        bounds = reference.bounds()
        if bounds['xyz_min'][2] < .10:
            raise RuntimeError('Raised trace approaches the clutter')
        lifted = initial['pose'].copy(); lifted[2] = .22
        above = reference.initial_pose.copy(); above[2] = .22
        approach = [lifted, above, reference.initial_pose]
        q = check_route(c, initial, approach)
        # Check the actual interpolated curve, not just chords between samples.
        route_times = np.unique(np.r_[reference.times, np.arange(0, reference.duration, .04)])
        poses, _ = reference.sample(route_times)
        check_route(c, dict(pose=reference.initial_pose, q=q), poses[1:], samples_per_segment=2)
        print('RAISED_REPLAY_PREFLIGHT', json.dumps(bounds), flush=True)
        provenance = dict(trajectory=str(args.trajectory.resolve()),
                          trajectory_sha256=hashlib.sha256(args.trajectory.read_bytes()).hexdigest(),
                          robot_ip=args.robot_ip,
                          controller_sha256=hashlib.sha256((Path(__file__).parents[1]/'open_loop/hardware_cartesian.py').read_bytes()).hexdigest(),
                          orientation_reference_sha256=hashlib.sha256((Path(__file__).parents[1]/'open_loop/hardware_orientation.py').read_bytes()).hexdigest(),
                          tcp_offset=tcp, real_home_pose=home_pose.tolist(), bounds=bounds,
                          full_source_ticks=len(trajectory.physics['samples']), selected_source_ticks=reference.source_ticks)
        args.out.with_suffix('.preflight.json').write_text(json.dumps(provenance, indent=2)+'\n')
        if args.execute:
            for pose in approach: move_checked(c, r, pose)
            summary = stream_cartesian(c, r, reference, args.out, frequency=args.frequency,
                                       min_z=.10, max_force_delta=15., position_gain=args.position_gain,
                                       actuator=args.actuator, preview=args.preview,
                                       acceleration=.5, max_speed=.10, max_angular_speed=.20,
                                       max_error=.010, pose_correction=0.)
            artifact = json.loads(args.out.read_text())
            artifact['summary']['full_trace_validated'] = bool(summary['passed'] and reference.source_ticks == len(trajectory.physics['samples']))
            artifact['summary']['validation_scope'] = 'full_trace' if not args.max_physics_ticks else 'prefix_only'
            artifact['provenance'] = provenance
            args.out.write_text(json.dumps(artifact, indent=1)+'\n')
            if not summary['passed']:
                raise RuntimeError('Raised validation failed its tracking/timing criteria; contact remains unvalidated')
        else:
            print('CHECK ONLY: no robot motion or gripper commands', flush=True)
    finally:
        if c is not None:
            try:
                if c.isProgramRunning(): c.stopScript()
            finally: c.disconnect()
        r.disconnect()


if __name__ == '__main__':
    main()
