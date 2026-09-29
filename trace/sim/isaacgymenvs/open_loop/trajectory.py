"""Open-loop trajectory file format (JSON).

Version 3 preserves the original controller inputs and full motion reference:

  - physics     : initial scene/state plus timestamped EEF pose and velocity,
                   joint states, applied joint targets, and resulting object
                   states at every gym.simulate tick. The fidelity reference.
  - dense trace : legacy EEF XYZ samples at RL-step boundaries. These omit
                   the turns inside two-phase primitives; retained for readers
                   and students that index the plan at the RL control rate.
  - executed     : the dense trace simplified to straight segments
                   (Douglas-Peucker, `EXECUTED_TOL`) — the DEFAULT moveL path.
  - commanded    : the primitive waypoints (wp1, wp2) the policy COMMANDED.
                   These are not necessarily reached before the next phase;
                   kept for analysis rather than exact motion reconstruction.
  - grasp        : final grasp pose from the grasp network (sim frame + raw
                   rotation bin so the executor rebuilds the exact tool angle)
  - final_target : target (x, y, yaw) at the end of the twin rollout, so a
                   replay can measure how faithfully it reproduced the plan.

All coordinates are SIM frame; conversion to the robot frame happens only in
the executor via frames.sim_to_real().
"""
import json
import math
import os
import time

FORMAT_VERSION = 3
EXECUTED_TOL = float(os.environ.get("EXECUTED_TOL", 0.0015))   # m, Douglas-Peucker tolerance
                          # for the executed (moveL) path. Env-var override so the
                          # fidelity-vs-waypoint-count trade can be swept without
                          # re-solving: the dense trace is stored in every JSON.


def simplify_path(points, tol=EXECUTED_TOL):
    """Douglas-Peucker on [[x, y, z], ...] (xy distance). Keeps endpoints."""
    if len(points) < 3:
        return [list(p) for p in points]
    (x0, y0), (x1, y1) = points[0][:2], points[-1][:2]
    dx, dy = x1 - x0, y1 - y0
    seg = (dx * dx + dy * dy) ** 0.5
    best, idx = -1.0, 0
    for i in range(1, len(points) - 1):
        px, py = points[i][0] - x0, points[i][1] - y0
        d = abs(dx * py - dy * px) / seg if seg > 1e-9 else (px * px + py * py) ** 0.5
        if d > best:
            best, idx = d, i
    if best <= tol:
        return [list(points[0]), list(points[-1])]
    left = simplify_path(points[:idx + 1], tol)
    right = simplify_path(points[idx:], tol)
    return left[:-1] + right


class OpenLoopTrajectory:
    def __init__(self, scene_file="", checkpoint="", metadata=None):
        self.scene_file = scene_file
        self.scene_text = None         # immutable scene snapshot; real2sim paths are reused
        self.checkpoint = checkpoint
        self.metadata = metadata or {}
        self.start_eef = None          # [x, y, z] sim, after reset
        self.dense = []                # per RL step: {"t", "eef", "action", "grasp_q"}
        self.waypoints = []            # per primitive: {"step", "action", "wp1", "wp2"} (commanded)
        self.grasp = None              # {"x_sim","y_sim","px","py","rotation_idx","q"}
        self.final_target = None       # [x, y, yaw] sim at the end of the rollout
        self.physics = None            # timestamped physics-tick states + applied joint targets

    # ---- recording -------------------------------------------------------
    def record_start(self, eef_xyz):
        self.start_eef = [float(v) for v in eef_xyz]

    def record_step(self, t, eef_xyz, action_idx, grasp_q, obj_xy=None):
        d = {
            "t": int(t),
            "eef": [float(v) for v in eef_xyz],
            "action": int(action_idx),
            "grasp_q": float(grasp_q),
        }
        if obj_xy is not None:            # (11, 2) twin-predicted centres, sim order
            d["obj_xy"] = [[float(a), float(b)] for a, b in obj_xy]
        self.dense.append(d)

    def plan_obj_xy(self):
        """(T, 11, 2) twin-predicted object centres per step, or None for a
        trajectory recorded before this field existed."""
        if not self.dense or "obj_xy" not in self.dense[0]:
            return None
        return [d["obj_xy"] for d in self.dense]

    def record_waypoints(self, t, action_idx, wp1_xyz, wp2_xyz):
        self.waypoints.append({
            "step": int(t),
            "action": int(action_idx),
            "wp1": [float(v) for v in wp1_xyz],
            "wp2": [float(v) for v in wp2_xyz],
        })

    def record_grasp(self, x_sim, y_sim, px, py, rotation_idx, q):
        self.grasp = {
            "x_sim": float(x_sim), "y_sim": float(y_sim),
            "px": int(px), "py": int(py),
            "rotation_idx": int(rotation_idx), "q": float(q),
        }

    # ---- views -----------------------------------------------------------
    def set_physics(self, trace):
        """Attach a complete physics trace; legacy dense/action fields stay intact.

        initial is state at t=0. Sample k contains the command applied during
        [k*dt, (k+1)*dt) and the resulting state at (k+1)*dt. Quaternions are
        xyzw; EEF state is xyz, xyzw, linear velocity, angular velocity.
        """
        validate_physics_trace(trace)
        self.physics = trace

    def physics_path(self):
        if self.physics is None:
            raise ValueError("No physics trace: regenerate this legacy trajectory from its recorded actions")
        return [list(s["eef_state"][:3]) for s in
                [self.physics["initial"]] + self.physics["samples"]]

    def cartesian_trace(self):
        """Full timed Cartesian reference, without simplification or retiming."""
        if self.physics is None:
            raise ValueError("No physics trace: regenerate this legacy trajectory from its recorded actions")
        return [{"time_s": s["time_s"], "position": list(s["eef_state"][:3]),
                 "quaternion_xyzw": list(s["eef_state"][3:7]),
                 "linear_velocity": list(s["eef_state"][7:10]),
                 "angular_velocity": list(s["eef_state"][10:13])}
                for s in [self.physics["initial"]] + self.physics["samples"]]

    def dense_path(self):
        path = [list(self.start_eef)] if self.start_eef is not None else []
        return path + [list(d["eef"]) for d in self.dense]

    def executed_path(self, tol=EXECUTED_TOL):
        """Straight-segment simplification of the ACTUAL trace (default replay)."""
        return simplify_path(self.dense_path(), tol)

    def commanded_path(self):
        """Start pose then wp1, wp2 of every primitive (what was commanded)."""
        path = [list(self.start_eef)] if self.start_eef is not None else []
        for wp in self.waypoints:
            path.append(list(wp["wp1"]))
            path.append(list(wp["wp2"]))
        return path

    def path(self, mode="executed"):
        return {"executed": self.executed_path, "dense": self.dense_path,
                "commanded": self.commanded_path, "physics": self.physics_path}[mode]()

    # ---- (de)serialisation ----------------------------------------------
    def write_scene(self, path):
        """Materialize the embedded scene, falling back to the legacy source."""
        text = self.scene_text
        if text is None:
            with open(self.scene_file) as f:
                text = f.read()
        with open(path, "w") as f:
            f.write(text)

    def to_dict(self):
        return {
            "version": FORMAT_VERSION,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "scene_file": self.scene_file,
            "scene_text": self.scene_text,
            "checkpoint": self.checkpoint,
            "metadata": self.metadata,
            "start_eef": self.start_eef,
            "dense": self.dense,
            "waypoints": self.waypoints,
            "executed": self.executed_path(),
            "grasp": self.grasp,
            "final_target": self.final_target,
            "physics": self.physics,
        }

    def save(self, path):
        payload = json.dumps(self.to_dict(), indent=1)
        with open(path, "w") as f:
            f.write(payload)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        if d.get("version") not in (1, 2, FORMAT_VERSION):
            raise ValueError(f"Unsupported trajectory version: {d.get('version')}")
        traj = cls(d.get("scene_file", ""), d.get("checkpoint", ""), d.get("metadata"))
        traj.start_eef = d.get("start_eef")
        traj.scene_text = d.get("scene_text")
        traj.dense = d.get("dense", [])
        traj.waypoints = d.get("waypoints", [])
        traj.grasp = d.get("grasp")
        traj.final_target = d.get("final_target")
        if d.get("physics") is not None:
            traj.set_physics(d["physics"])
        return traj


def validate_episode_continuity(dense):
    """Reject recordings spliced across simulator episode resets."""
    previous = 0
    for index, sample in enumerate(dense):
        step = int(sample['t'])
        if step <= previous:
            raise ValueError(f'Simulator episode reset at dense sample {index}: '
                             f'progress {previous} -> {step}; regenerate the trajectory')
        previous = step


def validate_physics_trace(trace):
    """Reject incomplete/retimed traces before they become replay commands."""
    dt = float(trace["dt"])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("physics.dt must be finite and positive")
    if trace.get("frame") != "sim" or trace.get("quaternion_order") != "xyzw":
        raise ValueError("Physics trace requires sim coordinates and xyzw quaternions")
    initial, samples = trace["initial"], trace["samples"]
    if not samples or not math.isfinite(float(initial["time_s"])) or abs(float(initial["time_s"])) > 1e-12:
        raise ValueError("Physics trace requires an initial state at zero and at least one sample")
    ndof = len(initial["joint_pos"])
    if ndof < 6:
        raise ValueError("Physics trace requires the complete robot joint state")
    for k, state in enumerate([initial] + samples):
        if not math.isfinite(float(state["time_s"])) or abs(float(state["time_s"]) - k * dt) > max(1e-9, dt * 1e-6):
            raise ValueError("Physics timestamps must be contiguous and match dt")
        for name, count in (("eef_state", 13), ("joint_pos", ndof), ("joint_vel", ndof)):
            values = state[name]
            if len(values) != count or not all(math.isfinite(float(v)) for v in values):
                raise ValueError("Invalid physics state: " + name)
        qnorm = sum(float(v)**2 for v in state["eef_state"][3:7])
        if abs(qnorm - 1) > 1e-3:
            raise ValueError("Physics EEF quaternion must be normalized")
        if k:
            cmd = state["joint_targets"]
            if len(cmd) != ndof or not all(math.isfinite(float(v)) for v in cmd):
                raise ValueError("Invalid physics joint targets")
