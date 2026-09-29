"""Full-pose, timed Cartesian feedforward for the physical UR robot.

The physical actuator differs from the simulator's position drives. Never
send simulator joint motor targets directly to hardware. Pose preview supplies
feedforward for the UR servo's lookahead; speedL is a diagnostic alternative.
No object state or learned policy is consulted during replay.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline, make_interp_spline
from scipy.spatial.transform import Rotation


def quat_multiply(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.concatenate((a[..., 3:] * b[..., :3] + b[..., 3:] * a[..., :3]
                           + np.cross(a[..., :3], b[..., :3]),
                           a[..., 3:] * b[..., 3:] - np.sum(a[..., :3] * b[..., :3], axis=-1, keepdims=True)), axis=-1)


class CartesianReference:
    """Hermite interpolation through every pose AND measured velocity sample.

    start_z is an explicit constant height translation (raised commissioning
    or contact datum); vertical variation is preserved. time_scale is explicit
    and recorded: 1 means the original clock. Quaternions are xyzw.
    """
    def __init__(self, physics, real_start_rotation, start_z, time_scale=1.0, xy_offset=(0., 0.)):
        if not np.isfinite(time_scale) or time_scale < 1:
            raise ValueError('time_scale must be finite and >=1; never accelerate a recording')
        states = [physics['initial']] + physics['samples']
        self.times = np.asarray([s['time_s'] for s in states]) * time_scale
        eef = np.asarray([s['eef_state'] for s in states], dtype=np.float64)
        if eef.shape != (len(states), 13) or not np.isfinite(eef).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError('Invalid full physics reference')
        rz = Rotation.from_euler('z', -90, degrees=True)
        xyz = rz.apply(eef[:, :3])
        xyz[:, :2] += np.asarray(xy_offset)
        xyz[:, 2] += float(start_z) - xyz[0, 2]
        vel = rz.apply(eef[:, 7:10]) / time_scale
        sim_rot = Rotation.from_quat(eef[:, 3:7])
        tool_offset = (rz * sim_rot[0]).inv() * Rotation.from_rotvec(real_start_rotation)
        quats = (rz * sim_rot * tool_offset).as_quat()
        omega = rz.apply(eef[:, 10:13]) / time_scale
        for k in range(1, len(quats)):
            if np.dot(quats[k-1], quats[k]) < 0:
                quats[k] *= -1
        qdot = .5 * quat_multiply(np.c_[omega, np.zeros(len(omega))], quats)
        self.position = CubicHermiteSpline(self.times, xyz, vel, extrapolate=False)
        self.quaternion = CubicHermiteSpline(self.times, quats, qdot, extrapolate=False)
        self.duration = float(self.times[-1])
        self.time_scale = float(time_scale)
        self.start_z = float(start_z)
        self.source_dt = float(physics['dt'])
        self.source_ticks = len(physics['samples'])
        self.initial_pose = self.sample(0)[0]

    def sample(self, t):
        t = np.clip(t, 0., self.duration)
        pos, vel = self.position(t), self.position(t, 1)
        q, qd = self.quaternion(t), self.quaternion(t, 1)
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        qn = q / norm
        qdn = (qd - qn * np.sum(qn * qd, axis=-1, keepdims=True)) / norm
        conjugate = qn.copy(); conjugate[..., :3] *= -1
        omega = 2 * quat_multiply(qdn, conjugate)[..., :3]
        pose = np.concatenate((pos, Rotation.from_quat(qn).as_rotvec()), axis=-1)
        return pose, np.concatenate((vel, omega), axis=-1)

    def bounds(self, period=.002):
        times = np.linspace(0, self.duration, int(np.ceil(self.duration/period))+1)
        pose, twist = self.sample(times)
        acceleration = np.diff(twist[:, :3], axis=0) / np.diff(times)[:, None]
        return {'duration_s': self.duration, 'source_ticks': self.source_ticks,
                'time_scale': self.time_scale, 'start_z': self.start_z,
                'xyz_min': pose[:, :3].min(0).tolist(), 'xyz_max': pose[:, :3].max(0).tolist(),
                'max_speed_m_s': float(np.linalg.norm(twist[:, :3], axis=1).max()),
                'max_angular_speed_rad_s': float(np.linalg.norm(twist[:, 3:], axis=1).max()),
                'max_acceleration_m_s2': float(np.linalg.norm(acceleration, axis=1).max())}


class GentleCartesianReference(CartesianReference):
    """Retain all physics poses, with zero speed/acceleration at each corner.

    A quintic phase on each segment bounds speed, acceleration and jerk.
    This deliberately replaces the simulator timing and impulse velocities;
    it is geometric replay under explicit hardware limits, never exact-time
    reproduction. The full pose sequence, including Z and orientation, remains.
    """
    def __init__(self, physics, real_start_rotation, start_z, max_speed=.020,
                 max_acceleration=.035, max_jerk=.10):
        source = CartesianReference(physics, real_start_rotation, start_z)
        self.poses = source.sample(source.times)[0]
        self.rotations = Rotation.from_rotvec(self.poses[:, 3:])
        self.delta_rotation = (self.rotations[1:] * self.rotations[:-1].inv()).as_rotvec()
        self.delta_position = np.diff(self.poses[:, :3], axis=0)
        distances = np.linalg.norm(self.delta_position, axis=1)
        angles = np.linalg.norm(self.delta_rotation, axis=1)
        # max derivatives of 10u^3-15u^4+6u^5: 1.875, 10/sqrt(3), 60.
        segments = np.maximum.reduce((1.875*distances/max_speed,
            np.sqrt((10/np.sqrt(3))*distances/max_acceleration),
            np.cbrt(60*distances/max_jerk),
            1.875*angles/.08, np.sqrt((10/np.sqrt(3))*angles/.15), np.cbrt(60*angles/.3),
            np.full(len(distances), .032)))
        self.times = np.r_[0., np.cumsum(segments)]
        self.duration = float(self.times[-1])
        self.time_scale = None
        self.start_z = float(start_z)
        self.source_dt, self.source_ticks = source.source_dt, source.source_ticks
        self.initial_pose = self.poses[0].copy()
        self.limits = dict(speed_m_s=max_speed, acceleration_m_s2=max_acceleration, jerk_m_s3=max_jerk)

    def sample(self, t):
        t = np.clip(t, 0., self.duration)
        idx = np.minimum(np.searchsorted(self.times, t, side='right')-1, len(self.times)-2)
        duration = self.times[idx+1]-self.times[idx]
        u = (t-self.times[idx])/duration
        phase = 10*u**3-15*u**4+6*u**5
        rate = (30*u**2-60*u**3+30*u**4)/duration
        pose = np.concatenate((self.poses[idx, :3] + self.delta_position[idx]*np.asarray(phase)[..., None],
            (Rotation.from_rotvec(self.delta_rotation[idx]*np.asarray(phase)[..., None])
             * self.rotations[idx]).as_rotvec()), axis=-1)
        twist = np.concatenate((self.delta_position[idx]*np.asarray(rate)[..., None],
                                self.delta_rotation[idx]*np.asarray(rate)[..., None]), axis=-1)
        return pose, twist

    def bounds(self, period=.008):
        result = super().bounds(period)
        result.update(retiming='quintic_rest_to_rest_at_every_physics_pose', limits=self.limits)
        return result


def quaternion_kinematics(spline, times):
    """Unit quaternion and world angular derivatives of a component spline."""
    q, q1, q2, q3 = [spline(times, nu=k) for k in range(4)]
    dot = lambda a, b: np.sum(a*b, axis=-1, keepdims=True)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(n < .5):
        raise ValueError('Quaternion interpolant approaches a singularity')
    n1 = dot(q, q1)/n
    n2 = (dot(q1, q1)+dot(q, q2)-n1*n1)/n
    n3 = (3*dot(q1, q2)+dot(q, q3)-3*n1*n2)/n
    u = q/n
    u1 = (q1-u*n1)/n
    u2 = (q2-2*u1*n1-u*n2)/n
    u3 = (q3-3*u2*n1-3*u1*n2-u*n3)/n
    conjugate = u.copy(); conjugate[..., :3] *= -1
    conjugate1 = u1.copy(); conjugate1[..., :3] *= -1
    omega = 2*quat_multiply(u1, conjugate)[..., :3]
    alpha = 2*quat_multiply(u2, conjugate)[..., :3]
    jerk = 2*(quat_multiply(u3, conjugate)+quat_multiply(u2, conjugate1))[..., :3]
    return u, omega, alpha, jerk


class ContinuousCartesianReference(CartesianReference):
    """Smooth full-pose replay with explicit, nonuniform hardware retiming.

    Quintic splines retain every recorded pose, with zero velocity and
    acceleration at the two endpoints. Interior samples do not require stops.
    Simulator velocities are replaced, not claimed to be reproduced. Derivative
    limits are checked on a dense per-segment grid with a 2% planning margin.
    A separate geometric guard rejects excessive deviation from the polyline.
    """
    def __init__(self, physics, real_start_rotation, start_z, speed_fraction=.45,
                 max_speed=.10, max_acceleration=.50, max_jerk=5.,
                 max_angular_speed=.20, max_angular_acceleration=.60, max_angular_jerk=6.,
                 max_curve_deviation=.0025):
        if not np.isfinite(speed_fraction) or not .1 <= speed_fraction <= .5:
            raise ValueError('speed_fraction must be 0.1..0.5 of the original quarter-speed test')
        limits = np.array([max_speed, max_acceleration, max_jerk,
                           max_angular_speed, max_angular_acceleration, max_angular_jerk])
        if not np.isfinite(limits).all() or np.any(limits <= 0):
            raise ValueError('Positive finite motion limits required')
        source = CartesianReference(physics, real_start_rotation, start_z)
        self.poses = source.sample(source.times)[0]
        xyz = self.poses[:, :3]
        quats = Rotation.from_rotvec(self.poses[:, 3:]).as_quat()
        for k in range(1, len(quats)):
            if np.dot(quats[k-1], quats[k]) < 0:
                quats[k] *= -1
        self.source_dt, self.source_ticks = source.source_dt, source.source_ticks
        self.time_scale, self.start_z = None, float(start_z)
        self.initial_pose = self.poses[0].copy()
        self.speed_fraction = float(speed_fraction)
        self.limits = dict(zip(('speed_m_s', 'acceleration_m_s2', 'jerk_m_s3',
                               'angular_speed_rad_s', 'angular_acceleration_rad_s2',
                               'angular_jerk_rad_s3'), limits.tolist()))
        segments = np.maximum(np.linalg.norm(np.diff(xyz, axis=0), axis=1)/max_speed, .035)
        for iteration in range(100):
            self.times = np.r_[0., np.cumsum(segments)]
            self.position = make_interp_spline(self.times, xyz, k=5,
                bc_type=([(1, np.zeros(3)), (2, np.zeros(3))], [(1, np.zeros(3)), (2, np.zeros(3))]))
            self.quaternion = make_interp_spline(self.times, quats, k=5,
                bc_type=([(1, np.zeros(4)), (2, np.zeros(4))], [(1, np.zeros(4)), (2, np.zeros(4))]))
            grid = self.times[:-1, None]+segments[:, None]*np.linspace(0, 1, 201)
            angular = quaternion_kinematics(self.quaternion, grid)[1:]
            derivatives = [self.position(grid, nu=k) for k in (1, 2, 3)]+list(angular)
            peaks = np.array([np.linalg.norm(x, axis=-1).max(1) for x in derivatives])
            factor = np.max((peaks/(.98*limits[:, None]))**(1/np.array([1, 2, 3, 1, 2, 3])[:, None]), axis=0)
            if factor.max() <= 1:
                break
            segments *= np.maximum(1., factor*1.005)
        else:
            raise ValueError('Continuous retiming did not converge')
        # The requested fraction refers to the measured quarter-speed run.
        # Slower segments may still be necessary to satisfy derivative limits.
        scale = max(1., source.duration*4/speed_fraction/self.times[-1])
        if scale > 1:
            self.times *= scale
            self.position = make_interp_spline(self.times, xyz, k=5,
                bc_type=([(1, np.zeros(3)), (2, np.zeros(3))], [(1, np.zeros(3)), (2, np.zeros(3))]))
            self.quaternion = make_interp_spline(self.times, quats, k=5,
                bc_type=([(1, np.zeros(4)), (2, np.zeros(4))], [(1, np.zeros(4)), (2, np.zeros(4))]))
        self.duration = float(self.times[-1])
        self.actual_average_speed_fraction = source.duration*4/self.duration
        grid = self.times[:-1, None]+np.diff(self.times)[:, None]*np.linspace(0, 1, 201)
        point = self.position(grid)
        delta = np.diff(xyz, axis=0)[:, None, :]
        projection = np.clip(np.sum((point-xyz[:-1, None, :])*delta, axis=-1)
            /np.maximum(np.sum(delta*delta, axis=-1), 1e-20), 0., 1.)
        self.curve_deviation = float(np.linalg.norm(point-xyz[:-1, None, :]-projection[..., None]*delta, axis=-1).max())
        if self.curve_deviation > max_curve_deviation:
            raise ValueError(f'Interpolated path deviates {self.curve_deviation*1000:.2f} mm from its source segments')

    def bounds(self, period=.002):
        result = super().bounds(period)
        grid = self.times[:-1, None]+np.diff(self.times)[:, None]*np.linspace(0, 1, 201)
        _, omega, alpha, angular_jerk = quaternion_kinematics(self.quaternion, grid)
        result.update(retiming='continuous_quintic_through_every_physics_pose', limits=self.limits,
                      requested_fraction_of_quarter_speed=self.speed_fraction,
                      actual_average_fraction_of_quarter_speed=self.actual_average_speed_fraction,
                      max_curve_deviation_mm=self.curve_deviation*1000,
                      max_acceleration_m_s2=float(np.linalg.norm(self.position(grid, nu=2), axis=-1).max()),
                      max_jerk_m_s3=float(np.linalg.norm(self.position(grid, nu=3), axis=-1).max()),
                      max_angular_speed_rad_s=float(np.linalg.norm(omega, axis=-1).max()),
                      max_angular_acceleration_rad_s2=float(np.linalg.norm(alpha, axis=-1).max()),
                      max_angular_jerk_rad_s3=float(np.linalg.norm(angular_jerk, axis=-1).max()))
        return result


def pose_error(actual, desired):
    return np.r_[desired[:3] - actual[:3],
                 (Rotation.from_rotvec(desired[3:]) * Rotation.from_rotvec(actual[3:]).inv()).as_rotvec()]


class PlaybackRateReference:
    """Uniformly change an already smooth reference's clock, preserving its path.

    Record the derivative increases explicitly: velocity scales with rate,
    acceleration with its square, and jerk with its cube.
    """
    def __init__(self, reference, rate=1.):
        if not np.isfinite(rate) or not .5 <= rate <= 1.2:
            raise ValueError('Commissioned playback-rate range is 0.5..1.2')
        self.reference, self.rate = reference, float(rate)
        self.times = reference.times/self.rate
        self.duration = reference.duration/self.rate
        self.source_ticks, self.source_dt = reference.source_ticks, reference.source_dt
        self.initial_pose = reference.initial_pose.copy()

    def sample(self, t):
        pose, twist = self.reference.sample(np.asarray(t)*self.rate)
        return pose, twist*self.rate

    def bounds(self, period=.002):
        result = self.reference.bounds(period*self.rate)
        result['duration_s'] = self.duration
        result['playback_rate'] = self.rate
        result['actual_average_fraction_of_quarter_speed'] *= self.rate
        result['limits'] = dict(result['limits'])
        for key in list(result)+list(result['limits']):
            power = 3 if 'jerk' in key else 2 if 'acceleration' in key else 1 if 'speed' in key else 0
            if key.startswith('max_') and power:
                result[key] *= self.rate**power
            elif key in result['limits'] and power:
                result['limits'][key] *= self.rate**power
        return result


def feedback_twist(actual, desired, feedforward, position_gain=10., rotation_gain=8.):
    error = pose_error(np.asarray(actual), np.asarray(desired))
    return np.asarray(feedforward) + error * np.r_[np.full(3, position_gain), np.full(3, rotation_gain)]


def stop_stream(c, r, actuator, deceleration):
    """Finish the controller program BEFORE doing any blocking log I/O.

    The 20 Hz streaming watchdog must not remain armed while Python serializes
    megabytes of samples. Allow one second for the bounded onboard terminal
    deceleration, then stop the script, which removes its watchdogs. No program
    takeover, watchdog suppression during replay, or automatic fault reset.
    """
    started = time.monotonic()
    if c.isConnected() and c.isProgramRunning():
        try:
            if not c.setWatchdog(1.):
                raise RuntimeError('Could not configure the terminal-stop watchdog')
            accepted = c.servoStop(deceleration) if actuator == 'servol' else c.speedStop(deceleration)
            if not accepted:
                raise RuntimeError('Controller rejected the terminal stop')
        finally:
            c.stopScript()
    else:
        raise RuntimeError('Control program disappeared before the requested shutdown')
    deadline = time.monotonic()+1.
    stopped_samples = 0
    while time.monotonic() < deadline:
        if not r.isConnected() or r.getSafetyMode() != 1 or r.getRobotMode() != 7:
            raise RuntimeError('Robot state changed during shutdown')
        speed = np.asarray(r.getActualTCPSpeed())
        stationary = np.linalg.norm(speed[:3]) < .002 and np.linalg.norm(speed[3:]) < .01
        stopped_samples = stopped_samples+1 if stationary and not c.isProgramRunning() else 0
        if stopped_samples >= 3:
            return dict(program_stopped=True, stationary=True, duration_s=time.monotonic()-started)
        time.sleep(.02)
    raise RuntimeError('Did not verify stationary robot and stopped program after replay')


def stream_cartesian(c, r, reference, output, *, frequency=125., acceleration=1.2,
                     max_speed=.65, max_angular_speed=.6, max_error=.015,
                     max_force_delta=20., min_z=.010, position_gain=10., rotation_gain=8.,
                     actuator='servol', preview=.05, servo_gain=500., lookahead=.05,
                     pose_correction=0.):
    """Replay once with live guards; stop and persist evidence on every exit.

    Configure RTDEControl at frequency before calling. Startup/approach are
    separate. No clipping or hidden retiming: limit violations abort the run.
    """
    period = 1./frequency
    if actuator not in ('servol', 'speedl') or not 0 <= preview <= .1 or not .03 <= lookahead <= .2 or not 100 <= servo_gain <= 800:
        raise ValueError('Unsupported actuator or servo parameters')
    initial = np.asarray(r.getActualTCPPose())
    error = pose_error(initial, reference.initial_pose)
    if np.linalg.norm(error[:3]) > .001 or np.linalg.norm(error[3:]) > np.deg2rad(.5):
        raise RuntimeError('Robot is not at the reference start; refusing streaming')
    if np.linalg.norm(r.getActualTCPSpeed()) > .005:
        raise RuntimeError('Robot must be stationary before replay')
    bounds = reference.bounds()
    if bounds['xyz_min'][2] < min_z or bounds['max_speed_m_s'] > max_speed:
        raise ValueError('Reference exceeds the declared height or speed envelope')
    baseline_force = np.asarray(r.getActualTCPForce())[:3]
    rows, completed, failure = [], False, None
    shutdown = None
    start_stamp = last_stamp = float(r.getTimestamp())
    started = last_update = time.monotonic()
    max_gap = 0.
    try:
        if not c.setWatchdog(20.):
            raise RuntimeError('Could not enable RTDE stream watchdog')
        while True:
            cycle = c.initPeriod()
            now = time.monotonic()
            stamp = float(r.getTimestamp())
            if stamp > last_stamp:
                max_gap = max(max_gap, stamp-last_stamp)
                last_stamp, last_update = stamp, now
            if stamp < start_stamp or now-last_update > .04 or now-started > reference.duration+1.:
                raise RuntimeError('Stale/reversed robot clock or streaming timeout')
            if not c.isConnected() or not c.isProgramRunning() or r.getSafetyMode() != 1 or r.getRobotMode() != 7:
                raise RuntimeError('Control connection or robot safety state changed')
            actual = np.asarray(r.getActualTCPPose())
            speed = np.asarray(r.getActualTCPSpeed())
            force = np.asarray(r.getActualTCPForce())[:3]
            elapsed = stamp-start_stamp
            desired, feedforward = reference.sample(elapsed)
            error = pose_error(actual, desired)
            command = feedback_twist(actual, desired, feedforward, position_gain, rotation_gain)
            command_pose = reference.sample(elapsed+preview)[0]
            command_pose[:3] += pose_correction*error[:3]
            command_pose[3:] = (Rotation.from_rotvec(pose_correction*error[3:])
                               * Rotation.from_rotvec(command_pose[3:])).as_rotvec()
            rows.append({'time_s': elapsed, 'host_time_s': now-started, 'tcp_pose': actual.tolist(),
                         'tcp_speed': speed.tolist(), 'tcp_force': force.tolist(),
                         'joints': r.getActualQ(), 'reference_pose': desired.tolist(),
                         'reference_twist': feedforward.tolist(), 'command_twist': command.tolist(),
                         'command_pose': command_pose.tolist(),
                         'position_error_mm': float(np.linalg.norm(error[:3])*1000),
                         'orientation_error_deg': float(np.rad2deg(np.linalg.norm(error[3:])))})
            if not np.isfinite(np.r_[actual, speed, force, command]).all():
                raise RuntimeError('Non-finite telemetry or command')
            if actual[2] < min_z-.002 or actual[2] > bounds['xyz_max'][2]+.025:
                raise RuntimeError('Tool left the declared height envelope')
            if np.linalg.norm(force-baseline_force) > max_force_delta:
                raise RuntimeError('Unexpected contact force during replay')
            if np.linalg.norm(error[:3]) > max_error or np.linalg.norm(error[3:]) > np.deg2rad(3.):
                raise RuntimeError('EEF tracking error exceeded the abort limit')
            if np.linalg.norm(speed[:3]) > max_speed+.05 or np.linalg.norm(speed[3:]) > max_angular_speed+.1:
                raise RuntimeError('Measured speed exceeded the abort limit')
            if elapsed >= reference.duration:
                completed = True
                break
            if actuator == 'speedl':
                if np.linalg.norm(command[:3]) > max_speed or np.linalg.norm(command[3:]) > max_angular_speed:
                    raise RuntimeError('Required command exceeds declared speed limits')
                accepted = c.speedL(command.tolist(), acceleration, period)
            else:
                if np.linalg.norm(command_pose[:3]-actual[:3]) > .025 or command_pose[2] < min_z:
                    raise RuntimeError('Preview target exceeds Cartesian lead/height limits')
                accepted = c.servoL(command_pose.tolist(), 0., 0., period, lookahead, servo_gain)
            if not accepted:
                raise RuntimeError('Robot rejected Cartesian streaming command')
            c.waitPeriod(cycle)
    except BaseException as exc:
        failure = str(exc)
        raise
    finally:
        try:
            shutdown = stop_stream(c, r, actuator, acceleration)
        except BaseException as exc:
            failure = (failure+'; ' if failure else '')+'shutdown: '+str(exc)
            raise
        finally:
            p = np.asarray([row['position_error_mm'] for row in rows])
            a = np.asarray([row['orientation_error_deg'] for row in rows])
            summary = dict(bounds, completed=completed, failure=failure, shutdown=shutdown,
                           controller=('cartesian_pose_preview_feedforward_with_eef_correction'
                                       if actuator == 'servol' else 'cartesian_velocity_feedforward_plus_pose_feedback'),
                           actuator=actuator, preview_s=preview, servo_gain=servo_gain,
                           lookahead_s=lookahead, pose_correction=pose_correction,
                           frequency=frequency, stop_deceleration_m_s2=acceleration,
                           speedl_acceleration_m_s2=acceleration if actuator == 'speedl' else None,
                           position_gain=position_gain, rotation_gain=rotation_gain,
                           max_position_error_mm=float(p.max()) if len(p) else None,
                           rms_position_error_mm=float(np.sqrt(np.mean(p*p))) if len(p) else None,
                           max_orientation_error_deg=float(a.max()) if len(a) else None,
                           max_sample_gap_s=max_gap, samples=len(rows),
                           final_pose_after_stop=r.getActualTCPPose())
            summary['passed'] = bool(completed and shutdown and failure is None and len(rows)
                                     and p.max() <= 2.0 and a.max() <= .5 and max_gap <= .032)
            path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'summary': summary, 'samples': rows}, indent=1)+'\n')
            print('HARDWARE_REPLAY_RESULT', json.dumps(summary), flush=True)
    return summary
