"""Hardware-only orientation constraint: keep the pushing tool perpendicular."""
import numpy as np
from scipy.spatial.transform import Rotation


def fixed_downward_rotation(home_rotation):
    """Preserve home heading while setting the tool Z axis exactly downward."""
    matrix = Rotation.from_rotvec(home_rotation).as_matrix()
    if matrix[2, 2] > -np.cos(np.deg2rad(10)):
        raise ValueError('Experiment-home tool orientation is not downward')
    heading = np.arctan2(matrix[1, 0], matrix[0, 0])
    return (Rotation.from_euler('z', heading)
            * Rotation.from_euler('x', np.pi)).as_rotvec()


class FixedDownwardReference:
    """Keep the reference's XYZ path and clock, replacing all tool rotations.

    Preserve the experiment-home heading and set the tool Z axis exactly down.
    The simulator's orientation wobble is not part of the hardware push command.
    This is a reference constraint; the commissioned streaming controller and
    its live guards are unchanged.
    """
    def __init__(self, reference, home_rotation):
        self.rotation = fixed_downward_rotation(home_rotation)
        self.reference = reference
        self.times, self.duration = reference.times, reference.duration
        self.source_ticks, self.source_dt = reference.source_ticks, reference.source_dt
        self.initial_pose = reference.initial_pose.copy()
        self.initial_pose[3:] = self.rotation

    def sample(self, t):
        pose, twist = self.reference.sample(t)
        pose, twist = pose.copy(), twist.copy()
        pose[..., 3:] = self.rotation
        twist[..., 3:] = 0.
        return pose, twist

    def bounds(self, period=.002):
        result = self.reference.bounds(period)
        result['orientation_mode'] = 'fixed_perpendicular_to_table'
        result['fixed_rotation_vector'] = self.rotation.tolist()
        for key in ('max_angular_speed_rad_s', 'max_angular_acceleration_rad_s2', 'max_angular_jerk_rad_s3'):
            if key in result:
                result['source_'+key] = result[key]
                result[key] = 0.
        return result
