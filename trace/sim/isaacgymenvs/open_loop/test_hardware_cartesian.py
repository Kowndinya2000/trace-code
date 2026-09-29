"""Offline contracts for the physical Cartesian replay adapter."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np
from scipy.spatial.transform import Rotation


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


hw = load('hardware_cartesian')
orientation = load('hardware_orientation')


def trace():
    rows = []
    for k in range(3):
        q = Rotation.from_euler('z', .03*k).as_quat().tolist()
        rows.append({'time_s': .02*k, 'eef_state': [.3+.002*k, .01+.001*k, .02+.0002*k]
                     + q + [.1, .05, .01, 0, 0, 1.5]})
    return dict(dt=.02, initial=rows[0], samples=rows[1:])


class HardwareReferenceTest(unittest.TestCase):
    def test_hardware_reference_holds_exactly_downward_without_changing_path_or_timing(self):
        home = [3.140807612573337, .01762512439953542, -.0009565601316576635]
        physics = trace()
        for k, state in enumerate([physics['initial']]+physics['samples']):
            state['eef_state'][3:7] = Rotation.from_euler('xyz', [.03*k, -.02*k, .01*k]).as_quat().tolist()
        source = hw.PlaybackRateReference(hw.ContinuousCartesianReference(physics, home, .02), 1.2)
        fixed = orientation.FixedDownwardReference(source, home)
        times = np.linspace(0, source.duration, 1001)
        original_pose, original_twist = source.sample(times)
        pose, twist = fixed.sample(times)
        np.testing.assert_array_equal(pose[:, :3], original_pose[:, :3])
        np.testing.assert_array_equal(twist[:, :3], original_twist[:, :3])
        np.testing.assert_array_equal(fixed.times, source.times)
        self.assertEqual(fixed.duration, source.duration)
        np.testing.assert_array_equal(pose[:, 3:], np.tile(fixed.rotation, (len(times), 1)))
        np.testing.assert_array_equal(twist[:, 3:], 0.)
        axes = Rotation.from_rotvec(pose[:, 3:]).apply(np.tile([0., 0., 1.], (len(times), 1)))
        np.testing.assert_allclose(axes, np.tile([0., 0., -1.], (len(times), 1)), atol=1e-14)
        # Servo preview samples, including beyond the end, must remain locked.
        for t in [-.01, 0., .07, source.duration, source.duration+.07]:
            np.testing.assert_array_equal(fixed.sample(t)[0][3:], fixed.rotation)

    def test_all_positions_velocities_and_heights_survive_coordinate_mapping(self):
        ref = hw.CartesianReference(trace(), [np.pi, 0, 0], .15)
        p, v = ref.sample(ref.times)
        np.testing.assert_allclose(p[:, :3], [[.01, -.3, .15], [.011, -.302, .1502], [.012, -.304, .1504]], atol=1e-12)
        np.testing.assert_allclose(v[:, :3], np.tile([.05, -.1, .01], (3, 1)), atol=1e-12)
        np.testing.assert_allclose(v[:, 3:], np.tile([0, 0, 1.5], (3, 1)), atol=1e-12)
        self.assertLess(hw.pose_error(np.r_[p[0, :3], np.pi, 0, 0], p[0])[3:].dot(
            hw.pose_error(np.r_[p[0, :3], np.pi, 0, 0], p[0])[3:]), 1e-20)

    def test_quaternion_signs_do_not_create_spins(self):
        a = trace(); b = copy.deepcopy(a)
        b['samples'][0]['eef_state'][3:7] = [-x for x in b['samples'][0]['eef_state'][3:7]]
        ar, br = [hw.CartesianReference(t, [np.pi, 0, 0], .15) for t in (a, b)]
        ts = np.linspace(0, ar.duration, 101)
        for x, y in zip(ar.sample(ts), br.sample(ts)):
            np.testing.assert_allclose(x, y, atol=1e-12)

    def test_declared_time_scaling_changes_velocity_not_geometry(self):
        a, b = [hw.CartesianReference(trace(), [np.pi, 0, 0], .15, scale) for scale in (1, 4)]
        for t in np.linspace(0, a.duration, 13):
            ap, av = a.sample(t); bp, bv = b.sample(4*t)
            np.testing.assert_allclose(ap, bp, atol=1e-12)
            np.testing.assert_allclose(av, 4*bv, atol=1e-12)
        self.assertEqual(b.duration, 4*a.duration)

    def test_feedback_is_zero_when_tracking_and_corrects_in_base_frame(self):
        pose = np.array([.02, -.4, .1, np.pi, 0, 0])
        ff = np.array([.1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(hw.feedback_twist(pose, pose, ff), ff, atol=1e-12)
        target = pose.copy(); target[0] += .001
        self.assertAlmostEqual(hw.feedback_twist(pose, target, ff)[0], .11)

    def test_speedup_and_nonfinite_clock_are_rejected(self):
        for scale in (.5, 0, float('nan')):
            with self.assertRaises(ValueError):
                hw.CartesianReference(trace(), [np.pi, 0, 0], .15, scale)

    def test_gentle_timing_keeps_every_corner_and_stops_smoothly(self):
        physics = trace()
        physics['samples'][1]['eef_state'][:3] = [.302, .013, .0204]
        original = hw.CartesianReference(physics, [np.pi, 0, 0], .15)
        gentle = hw.GentleCartesianReference(physics, [np.pi, 0, 0], .15)
        poses, speeds = gentle.sample(gentle.times)
        original_poses = original.sample(original.times)[0]
        np.testing.assert_allclose(poses[:, :3], original_poses[:, :3], atol=1e-12)
        for a, b in zip(poses, original_poses):
            np.testing.assert_allclose(hw.pose_error(a, b), 0., atol=1e-12)
        np.testing.assert_allclose(speeds, 0., atol=1e-12)
        times = np.linspace(0, gentle.duration, 10001)
        _, speeds = gentle.sample(times)
        accel = np.diff(speeds[:, :3], axis=0)/np.diff(times)[:, None]
        jerk = np.diff(accel, axis=0)/np.diff(times)[1:, None]
        self.assertLessEqual(np.linalg.norm(speeds[:, :3], axis=1).max(), .020001)
        self.assertLessEqual(np.linalg.norm(accel, axis=1).max(), .035001)
        self.assertLessEqual(np.linalg.norm(jerk, axis=1).max(), .10001)

    def test_continuous_replay_preserves_poses_and_has_smooth_bounded_motion(self):
        physics = trace()
        a = hw.CartesianReference(physics, [np.pi, 0, 0], .15)
        b = hw.ContinuousCartesianReference(physics, [np.pi, 0, 0], .15)
        for x, y in zip(a.sample(a.times)[0], b.sample(b.times)[0]):
            np.testing.assert_allclose(hw.pose_error(x, y), 0., atol=1e-12)
        np.testing.assert_allclose(b.sample([0, b.duration])[1], 0., atol=1e-12)
        np.testing.assert_allclose(b.position([0, b.duration], nu=2), 0., atol=1e-12)
        self.assertGreater(np.linalg.norm(b.sample(b.times[1])[1]), .001)
        bounds = b.bounds()
        for key, limit in [('max_speed_m_s', .10), ('max_acceleration_m_s2', .50),
                           ('max_jerk_m_s3', 5.), ('max_angular_speed_rad_s', .20),
                           ('max_angular_acceleration_rad_s2', .60), ('max_angular_jerk_rad_s3', 6.)]:
            self.assertLessEqual(bounds[key], limit)

    def test_quaternion_angular_derivatives_match_finite_differences(self):
        b = hw.ContinuousCartesianReference(trace(), [np.pi, 0, 0], .15)
        t = np.linspace(.001, b.duration-.001, 1001)
        q, omega, alpha, jerk = hw.quaternion_kinematics(b.quaternion, t)
        dt = 1e-5
        qm = hw.quaternion_kinematics(b.quaternion, t-dt)
        qp = hw.quaternion_kinematics(b.quaternion, t+dt)
        numerical_omega = (Rotation.from_quat(qp[0])*Rotation.from_quat(qm[0]).inv()).as_rotvec()/(2*dt)
        np.testing.assert_allclose(omega, numerical_omega, atol=1e-7)
        np.testing.assert_allclose(alpha, (qp[1]-qm[1])/(2*dt), atol=1e-7)
        np.testing.assert_allclose(jerk, (qp[2]-qm[2])/(2*dt), atol=1e-6)

    def test_shutdown_stops_program_before_return_and_verifies_stationary_robot(self):
        c, r = Mock(), Mock()
        c.isConnected.return_value = True
        c.isProgramRunning.side_effect = [True, False, False, False]
        c.setWatchdog.return_value = True
        c.servoStop.return_value = True
        r.isConnected.return_value = True
        r.getSafetyMode.return_value = 1
        r.getRobotMode.return_value = 7
        r.getActualTCPSpeed.return_value = np.zeros(6)
        result = hw.stop_stream(c, r, 'servol', .5)
        self.assertTrue(result['program_stopped'])
        self.assertTrue(result['stationary'])
        names = [call[0] for call in c.mock_calls]
        self.assertLess(names.index('servoStop'), names.index('stopScript'))

    def test_failed_stop_still_terminates_program_and_cannot_pass(self):
        c, r = Mock(), Mock()
        c.isConnected.return_value = True
        c.isProgramRunning.return_value = True
        c.setWatchdog.return_value = True
        c.servoStop.return_value = False
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            hw.stop_stream(c, r, 'servol', .5)
        c.stopScript.assert_called_once()

    def test_playback_increase_preserves_path_and_reports_derivative_increases(self):
        a = hw.ContinuousCartesianReference(trace(), [np.pi, 0, 0], .15)
        b = hw.PlaybackRateReference(a, 1.2)
        for t in np.linspace(0, a.duration, 101):
            pa, va = a.sample(t)
            pb, vb = b.sample(t/1.2)
            np.testing.assert_allclose(hw.pose_error(pa, pb), 0., atol=1e-12)
            np.testing.assert_allclose(vb, va*1.2, atol=1e-12)
        self.assertAlmostEqual(b.duration, a.duration/1.2)
        x, y = a.bounds(), b.bounds()
        self.assertAlmostEqual(y['max_jerk_m_s3'], x['max_jerk_m_s3']*1.2**3)
        self.assertAlmostEqual(y['limits']['acceleration_m_s2'], x['limits']['acceleration_m_s2']*1.2**2)


if __name__ == '__main__':
    unittest.main()
