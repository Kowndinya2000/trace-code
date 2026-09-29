"""Re-sense and execute ONE grasp. No pushing, no trajectory.

  python open_loop/grasp_now.py            # dry run, prints the pose
  python open_loop/grasp_now.py --execute
"""
import argparse, sys, time
from pathlib import Path
import numpy as np

from isaacgymenvs.open_loop import hardware_config
sys.path.insert(0, str(hardware_config.hardware_dir()))
from isaacgymenvs.open_loop import frames, perceive_scene as ps, real_grasp

_ASSET_DIR = Path(real_grasp.__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--execute", action="store_true")
ap.add_argument("--robot-ip", default=hardware_config.robot_ip())
ap.add_argument("--calib", default=str(hardware_config.calibration()))
ap.add_argument("--maskrcnn", default=str(hardware_config.maskrcnn()))
ap.add_argument("--hover-z", type=float, default=0.10)
ap.add_argument("--grasp-z", type=float, default=0.018)
ap.add_argument("--lift-z", type=float, default=0.12)
ap.add_argument("--tool-vel", type=float, default=0.15)   # PMBS tool_vel 0.3 * speed_scale 0.5
ap.add_argument("--tool-acc", type=float, default=0.6)    # PMBS tool_acc 1.2 * 0.5
ap.add_argument("--captures", type=int, default=5)
ap.add_argument("--table-z", type=float, default=0.006)
ap.add_argument("--no-return", action="store_true", help="keep the object, skip the release")
ap.add_argument("--descend-mm", type=float, default=40.0, help="drop below the top face")
ap.add_argument("--rot-offset-bins", type=int, default=0,  # now applied in frames.py
                help="constant bin offset applied to the jaw axis. Our heightmap is "
                     "row=sim x/col=sim y; rotation_idx_to_tool_orientation is PMBS's "
                     "chain for row=real x/col=real y, and the two frames differ by the "
                     "Z(-90) swap-negate = 4 bins.")
ap.add_argument("--ws-margin", type=float, default=0.02,
                help="allowed grasp distance outside the 44.8 cm workspace (the mat extends further)")
ap.add_argument("--panel", default=None,
                help="write the grasp network's 16-rotation search here (PNG)")
a = ap.parse_args()

HOME_JOINTS_DEG = [-19.32 + 90.0, -100.29, 147.48, -137.18, -89.72, 160.73]
PMBS_HOME = np.deg2rad([12.44, -127.35, 127.41, -90.1, -89.51, 102.6]).tolist()

# Retract before capture so the arm cannot occlude the target or clutter.
cam2base = np.loadtxt(a.calib)
if a.execute:
    from rtde_control import RTDEControlInterface as _C
    from rtde_receive import RTDEReceiveInterface as _R
    from dashboard_client import DashboardClient as _D
    _d = _D(a.robot_ip); _d.connect()
    if _d.running():
        _d.stop(); time.sleep(1.0)      # a loaded .urp blocks the RTDE upload
    _d.disconnect()
    _rr = _R(a.robot_ip); _cc = _C(a.robot_ip)
    _p = _rr.getActualTCPPose()
    _cc.moveL([_p[0], _p[1], 0.22, _p[3], _p[4], _p[5]], 0.20, 0.8)
    _cc.moveJ(PMBS_HOME, 1.05, 1.4)
    print("retracted to PMBS home before sensing")   # keep _cc: see below

# graspability varies frame to frame (0.81-1.45 measured on one static scene),
# so take the best of N captures rather than betting on a single frame
best = None
for i in range(a.captures):
    color, depth, K = ps.capture_realsense(warmup=10)
    if a.panel:
        # Retain every source capture even if target detection fails or a
        # different frame wins the multi-capture search below.
        panel = Path(a.panel)
        capture_path = panel.with_name(panel.stem + "_captures") / f"capture_{i:02d}.png"
        real_grasp.save_grasp_record(real_grasp.camera_debug(color, depth, K, cam2base),
                                     None, capture_path)
    gi = real_grasp.compute_hardware_grasp(
        color, depth, K, cam2base, a.maskrcnn)
    if gi is None:
        continue
    print(f"  capture {i}: q={gi['q']:.3f} rot={gi['rotation_idx']}")
    if best is None or gi["q"] > best["q"]:
        best = gi
assert best is not None, "no purple/blue target detected"
g = best
_dbg = g.pop("_debug")
if a.panel:
    print("16-rotation panel ->", real_grasp.render_rotation_panel(_dbg, g, a.panel))
print(f"graspability q={g['q']:.3f}  "
      f"{'GRASPABLE' if g['graspable'] else 'REJECTED'}"
      f"  (threshold {real_grasp.GRASPABLE_Q_THRESHOLD:.2f})")
if not g["graspable"]:
    sys.exit(g.get("reject_reason", "not graspable") + " - refusing to close on it")
print(f"grasp {1000*g.get('grasp_offset_m', 0):.0f} mm from target centroid")
gx, gy = g["x_real"], g["y_real"]
if "grasp_xy_correction_real_m" in g:
    _corr_mm = 1000 * np.asarray(g["grasp_xy_correction_real_m"])
    print(f"calibrated real-XY correction ({_corr_mm[0]:+.3f}, {_corr_mm[1]:+.3f}) mm")
rot_used = (g["rotation_idx"] + a.rot_offset_bins) % 16
grx, gry = frames.rotation_idx_to_tool_orientation(rot_used)
print(f"grasp real ({gx:+.4f}, {gy:+.4f})  rot bin {g['rotation_idx']}"
      f"{f' +{a.rot_offset_bins} -> {rot_used}' if a.rot_offset_bins else ''}"
      f"  tool ({grx:+.3f}, {gry:+.3f})")
lo, hi = frames.REAL_WORKSPACE_LIMITS[:2, 0] - a.ws_margin, frames.REAL_WORKSPACE_LIMITS[:2, 1] + a.ws_margin
assert lo[0] <= gx <= hi[0] and lo[1] <= gy <= hi[1], "grasp point outside the real workspace"
if not a.execute:
    sys.exit("dry run - pass --execute to move")

from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from robotiq_gripper import RobotiqGripper

gr = RobotiqGripper(a.robot_ip, 63352); gr.connect()
# REUSE the interfaces opened for the retract. Opening a second RTDEControl
# fails with "One of the RTDE input registers are already in use" -- the first
# one still holds them even after stopScript.
c, r = _cc, _rr
# The captures above ran Mask R-CNN and the grasp network several times over,
# which is long enough for the watchdog to drop the control script uploaded for
# the retract. Revive it or every move below silently does nothing.
from isaacgymenvs.open_loop import recorder_io as _rio
c = _rio.ensure_control(c, a.robot_ip)

def checked_move(name, command, target=None, tolerance_m=0.001):
    """Require both RTDE acceptance and arrival before the grasp can close."""
    ok = command()
    if ok is False:
        raise RuntimeError(f"{name} was rejected by RTDE")
    if target is not None:
        actual = np.asarray(r.getActualTCPPose()[:3], dtype=float)
        error = float(np.linalg.norm(actual - np.asarray(target[:3], dtype=float)))
        print(f"{name}: TCP error {1000*error:.2f} mm")
        if error > tolerance_m:
            raise RuntimeError(f"{name} missed commanded TCP by {1000*error:.2f} mm")
# PMBS home: retracted clear of the workspace and high, so the approach is a
# clean top-down descent that cannot clip a block on the way in, and the
# retraction afterwards is one fast moveJ (environment_real.go_home).
try:
    t0 = time.time()
    surf = g.get("surface_z_m", 0.048)
    gz = max(surf - a.descend_mm / 1000.0, a.table_z + 0.005)
    print(f"surface {1000*surf:.0f} mm -> descend to {1000*gz:.0f} mm (surface - {a.descend_mm:.0f} mm)")

    t = r.getActualTCPPose()
    checked_move("vertical clearance", lambda: c.moveL(
        [t[0], t[1], 0.22, t[3], t[4], t[5]], 0.15, 0.6))
    checked_move("PMBS home", lambda: c.moveJ(PMBS_HOME, 1.05, 1.4))
    open_pos, open_status = gr.open_and_wait_for_pos(80, 120)
    print(f"gripper fully open: position {open_pos}, status {open_status}")
    if open_pos > 10:
        raise RuntimeError(f"gripper did not fully expand: position {open_pos}")
    hover = [gx, gy, gz + 0.10, grx, gry, 0.0]
    grasp = [gx, gy, gz, grx, gry, 0.0]
    checked_move("grasp hover", lambda: c.moveL(hover, a.tool_vel, a.tool_acc), hover)
    checked_move("grasp descent", lambda: c.moveL(grasp, 0.06, 0.24), grasp)
    gr.close_and_wait_for_pos(80, 120)
    time.sleep(0.3)
    pos = gr.get_current_position()
    held = pos < 0.88 * gr.get_max_position()
    lift = [gx, gy, gz + 0.12, grx, gry, 0.0]
    checked_move("grasp lift", lambda: c.moveL(lift, a.tool_vel, a.tool_acc), lift)
    print(f"grasp in {time.time()-t0:.1f}s | gripper {pos}/{gr.get_max_position()} "
          f"({'HOLDING' if held else 'closed empty'})")

    checked_move("PMBS grasp-check home", lambda: c.moveJ(PMBS_HOME, 1.05, 1.4))
    if held and not a.no_return:
        gr.open_and_wait_for_pos(80, 120)              # release clear of the mat
        print("released at PMBS home")
    if not (held and a.no_return):
        # Park at OUR home, the arm's resting pose. Only skipped while still
        # holding: that pose puts the TCP 6 mm off the table inside the
        # workspace, which would drive the object into the clutter.
        checked_move("experiment home", lambda: c.moveJ(
            np.deg2rad(HOME_JOINTS_DEG).tolist(), 1.05, 1.4))
        print("parked at home")
    print(f"total {time.time()-t0:.1f}s | protective_stop={r.isProtectiveStopped()}")
finally:
    c.stopScript()
