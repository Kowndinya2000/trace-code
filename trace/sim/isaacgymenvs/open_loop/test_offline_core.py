"""Offline sanity tests for open_loop.frames, trajectory and the executor's geometry.

Nothing here talks to a robot or to Isaac Gym; it checks the frame conversions, the
trajectory file format and the waypoint pairing that the executor depends on."""
import importlib.util
import os
import sys
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(BASE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


frames = load("frames")
trajectory = load("trajectory")

# --- frames: round trips -------------------------------------------------
for x, y in [(0.3, -0.1), (0.5, 0.0), (0.724, 0.224), (0.276, -0.224)]:
    xr, yr = frames.sim_to_real(x, y)
    xs, ys = frames.real_to_sim(xr, yr)
    assert abs(xs - x) < 1e-12 and abs(ys - y) < 1e-12, (x, y, xs, ys)

px, py = frames.sim_to_pix(0.5, 0.0)
xs, ys = frames.pix_to_sim(px, py)
assert abs(xs - 0.5) < frames.PIXEL_SIZE and abs(ys - 0.0) < frames.PIXEL_SIZE
assert frames.sim_to_pix(*frames.pix_to_sim(100, 50)) == (100, 50)

# rl_policy convention spot check: sim (0.5, 0.1) -> real (0.1, -0.5)
assert frames.sim_to_real(0.5, 0.1) == (0.1, -0.5)
assert frames.sim_to_pix(0.276, -0.224) == (0, 0) and frames.sim_to_pix(0.5, 0.0) == (112, 112)

# rotation-bin chain matches rl_policy for bin 0: ang=0 -> ang'=pi/2 -> half=pi/4
import math
rx, ry = frames.rotation_idx_to_tool_orientation(0, apply_frame_offset=False)
assert abs(rx - math.cos(math.pi / 4) * math.pi) < 1e-9
assert abs(ry - math.sin(math.pi / 4) * math.pi) < 1e-9
# the executor applies the sim->real bin offset by default, which takes bin 0 to (pi, 0)
rx, ry = frames.rotation_idx_to_tool_orientation(0)
assert abs(rx - math.pi) < 1e-9 and abs(ry) < 1e-9
assert frames.in_sim_workspace(0.5, 0.0) and not frames.in_sim_workspace(0.9, 0.0)

# --- trajectory: build, views, save/load round trip ----------------------
t = trajectory.OpenLoopTrajectory("scene.txt", "ckpt.pth", {"solved": True})
t.record_start([0.5, 0.0, 0.05])
t.record_waypoints(1, 3, [0.51, 0.0, 0.05], [0.52, 0.0, 0.05])
t.record_step(1, [0.515, 0.001, 0.05], 3, 0.4)
t.record_waypoints(2, 7, [0.52, 0.01, 0.05], [0.52, 0.02, 0.05])
t.record_step(2, [0.52, 0.015, 0.05], 7, 0.95)
t.record_grasp(0.5, 0.1, 112, 162, 4, 0.97)

wp = t.commanded_path()
assert len(wp) == 5 and wp[0] == [0.5, 0.0, 0.05] and wp[-1] == [0.52, 0.02, 0.05]
assert len(t.dense_path()) == 3            # start + 2 recorded steps
assert t.path("executed") == t.executed_path()

# Douglas-Peucker: collinear points collapse, a corner survives
line = [[0.3, 0.0, 0.05], [0.31, 0.0004, 0.05], [0.32, -0.0003, 0.05], [0.33, 0.0, 0.05]]
assert trajectory.simplify_path(line, tol=0.0015) == [line[0], line[-1]]
corner = [[0.3, 0.0, 0.05], [0.32, 0.0, 0.05], [0.32, 0.02, 0.05]]
assert trajectory.simplify_path(corner, tol=0.0015) == corner
assert trajectory.simplify_path(corner[:1]) == corner[:1]

with tempfile.TemporaryDirectory() as d:
    p = t.save(os.path.join(d, "t.json"))
    t2 = trajectory.OpenLoopTrajectory.load(p)
    assert t2.to_dict()["waypoints"] == t.to_dict()["waypoints"]
    assert t2.grasp == t.grasp and t2.start_eef == t.start_eef
    assert t2.executed_path() == t.executed_path()
    assert t2.scene_file == "scene.txt"

# --- executor helpers: bounds + resampling (import by path too) ----------
sys.modules.setdefault("isaacgymenvs", type(sys)("isaacgymenvs"))
# execute_trajectory imports its siblings through the package, so stand a stub package
# up from the modules already loaded by path. Keep this in step with its imports.
# The real entries are restored at the end: under `unittest discover` this module shares
# an interpreter with the tests that do import the real package.
_stub_keys = ["isaacgymenvs", "isaacgymenvs.open_loop", "isaacgymenvs.open_loop.frames",
              "isaacgymenvs.open_loop.trajectory", "isaacgymenvs.open_loop.hardware_config",
              "isaacgymenvs.open_loop.recorder_io", "isaacgymenvs.open_loop.execute_trajectory"]
_saved = {key: sys.modules.get(key) for key in _stub_keys}
ol_pkg = type(sys)("isaacgymenvs.open_loop")
siblings = {"frames": frames, "trajectory": trajectory}
for _name in ("hardware_config", "recorder_io"):
    siblings[_name] = load(_name)
for _name, _mod in siblings.items():
    setattr(ol_pkg, _name, _mod)
    sys.modules["isaacgymenvs.open_loop." + _name] = _mod
sys.modules["isaacgymenvs.open_loop"] = ol_pkg
ex = load("execute_trajectory")

real_path = ex.to_real_xy(wp)
assert ex.bounds_check(real_path) == [], f"in-workspace path flagged: {real_path}"
assert ex.bounds_check([(0.5, 0.5)]) != []      # far outside real workspace
pairs = ex.commanded_primitive_pairs(real_path, has_start_pose=True)
assert pairs == [(real_path[1], real_path[2]), (real_path[3], real_path[4])]
try:
    ex.commanded_primitive_pairs(real_path[:-1], has_start_pose=True)
    raise AssertionError("odd commanded waypoint count was accepted")
except ValueError:
    pass
rs = ex.resample_dense([(0.0, -0.5), (0.02, -0.5)], step_m=0.005)
assert 3 <= len(rs) <= 6

print("ALL OPEN-LOOP CORE TESTS PASSED")

for _key, _mod in _saved.items():
    if _mod is None:
        sys.modules.pop(_key, None)
    else:
        sys.modules[_key] = _mod
