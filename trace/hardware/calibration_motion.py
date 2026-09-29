"""Original PMBS pose sampler and supervised controller execution.

The operator has confirmed the original sampling envelope is clear. Controller
checks below enforce configured robot limits; they do not model scene obstacles.
No new poses, enlarged ranges, or automatic recovery/home moves are introduced.
"""
import time

import numpy as np


POSITION_MIN = np.array([-0.05, -0.7, 0.22])
POSITION_MAX = np.array([0.05, -0.5, 0.28])
ROTATIONS = np.array([[1.723, 0, 0], [1.56, 0.2, 0.2], [1.56, -0.2, -0.2],
                      [1.56, -0.2, 0.2], [1.56, 0.2, -0.2],
                      [1.76, 0, 0], [1.36, 0, 0]])
# Preserve the exact original joint target, including its original conversion.
HOME_JOINTS = np.array([52.11, -130, 102, -89, -71, 148]) * 3.14 / 180
TOOL_SPEED, TOOL_ACCELERATION = 0.1, 0.5
JOINT_SPEED, JOINT_ACCELERATION = 0.5, 0.5


def sample_original_poses(count=36, seed=7):
    if not 1 <= count <= 162:
        raise ValueError("Original sampler has 162 candidates; choose 1..162 poses")
    rng = np.random.RandomState(seed)
    xs, ys = np.meshgrid(np.linspace(-0.05, 0.05, 9), np.linspace(-0.7, -0.5, 9))
    xy = list(zip(xs.ravel(), ys.ravel()))
    candidates = [[x, y, 0.22, *ROTATIONS[0]] for x, y in xy]
    for x, y in xy:
        z = [0.22, 0.25, 0.28][rng.randint(1, 3)]
        rotation = ROTATIONS[rng.randint(1, len(ROTATIONS))]
        candidates.append([x, y, z, *rotation])
    indices = rng.choice(len(candidates), count, replace=False)
    return [np.asarray(candidates[index]).tolist() for index in indices]


def validate_target(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Invalid motion target")
    if np.any(pose[:3] < POSITION_MIN) or np.any(pose[:3] > POSITION_MAX):
        raise ValueError("Motion target exceeds original XYZ limits")
    if not np.any(np.all(np.isclose(ROTATIONS, pose[3:], atol=1e-10, rtol=0), axis=1)):
        raise ValueError("Motion target uses an orientation outside the original sampler")
    if not np.any(np.isclose(pose[2], [0.22, 0.25, 0.28], atol=1e-10, rtol=0)):
        raise ValueError("Motion target uses a height outside the original sampler")


class AutomaticMotion:
    def __init__(self, control, receiver, read_state):
        self.control, self.receiver, self.read_state = control, receiver, read_state
        self.active_kind = None

    def preflight(self, poses, include_home=True):
        # A completed moveJ/moveL reports done while the joints still creep for
        # a few hundred ms; a one-shot reading here refuses the very next move
        # right after the home pose. Wait, bounded, for the arm to settle before
        # refusing.
        deadline = time.monotonic() + 3.0
        while True:
            state = self.read_state(self.receiver)
            speed = np.asarray(state["tcp_speed"])
            moving = (np.linalg.norm(speed[:3]) >= 0.0005 or np.linalg.norm(speed[3:]) >= 0.003
                      or np.max(np.abs(state["joint_speed"])) >= 0.005)
            if not moving:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("Robot is still moving; next motion refused")
            time.sleep(0.05)
        if not self.control.isConnected():
            raise RuntimeError("Robot control connection lost")
        if include_home and not self.control.isJointsWithinSafetyLimits(HOME_JOINTS.tolist()):
            raise RuntimeError("Original home target rejected by controller safety limits")
        for pose in poses:
            validate_target(pose)
            if not self.control.isPoseWithinSafetyLimits(pose):
                raise RuntimeError(f"Controller safety limits reject target {pose}")
            if not self.control.getInverseKinematicsHasSolution(pose):
                raise RuntimeError(f"No inverse-kinematics solution for target {pose}")

    def stop(self):
        kind = self.active_kind
        if kind == "joint":
            self.control.stopJ(JOINT_ACCELERATION)
        elif kind == "linear":
            self.control.stopL(TOOL_ACCELERATION)
        self.active_kind = None

    def _wait(self, target, kind, previous_progress, timeout=60.0):
        deadline, last_timestamp, last_update = time.monotonic() + timeout, None, time.monotonic()
        saw_running = False
        while True:
            state = self.read_state(self.receiver)
            if not self.control.isConnected():
                raise RuntimeError("Robot control connection lost during motion")
            stamp = state["robot_timestamp_s"]
            if last_timestamp is not None and stamp < last_timestamp:
                raise RuntimeError("Robot timestamp reversed during motion")
            if stamp != last_timestamp:
                last_timestamp, last_update = stamp, time.monotonic()
            if time.monotonic() - last_update > 0.5:
                raise RuntimeError("Robot telemetry is stale during motion")
            progress = self.control.getAsyncOperationProgress()
            saw_running = saw_running or progress >= 0
            if progress < 0 and (saw_running or progress != previous_progress):
                if kind == "joint":
                    reached = np.max(np.abs(np.asarray(state["joints"]) - target)) < 0.01
                else:
                    # SO(3) distance handles equivalent rotation-vector representations.
                    from scipy.spatial.transform import Rotation
                    actual = np.asarray(state["tcp_pose"])
                    rotation_error = (Rotation.from_rotvec(actual[3:]).inv()
                                      * Rotation.from_rotvec(target[3:])).magnitude()
                    reached = (np.linalg.norm(actual[:3] - target[:3]) < 0.001
                               and rotation_error < np.deg2rad(0.2))
                if not reached:
                    raise RuntimeError("Controller ended motion without reaching its target")
                self.active_kind = None
                return
            if time.monotonic() > deadline:
                raise RuntimeError("Robot motion timed out")
            time.sleep(0.05)

    def home(self):
        self.preflight([], include_home=True)
        previous_progress = self.control.getAsyncOperationProgress()
        if previous_progress >= 0:
            raise RuntimeError("A previous asynchronous robot operation is still running")
        self.active_kind = "joint"
        try:
            if not self.control.moveJ(HOME_JOINTS.tolist(), JOINT_SPEED, JOINT_ACCELERATION, True):
                raise RuntimeError("Controller rejected the original home move")
            self._wait(HOME_JOINTS, "joint", previous_progress)
        except BaseException:
            self.stop()
            raise

    def move_to(self, pose):
        self.preflight([pose], include_home=False)
        previous_progress = self.control.getAsyncOperationProgress()
        if previous_progress >= 0:
            raise RuntimeError("A previous asynchronous robot operation is still running")
        self.active_kind = "linear"
        try:
            if not self.control.moveL(pose, TOOL_SPEED, TOOL_ACCELERATION, True):
                raise RuntimeError("Controller rejected calibration move")
            self._wait(np.asarray(pose), "linear", previous_progress)
        except BaseException:
            self.stop()
            raise
