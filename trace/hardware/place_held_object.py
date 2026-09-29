"""Put the object the gripper is holding down at a chosen workspace point.

Ends a trial that finished with --keep: the target is still in the fingers at
PMBS home and the next scene needs it on the mat. The tool orientation is
whatever the grasp used -- it is read from the live TCP, not recomputed.

Refuses to descend onto occupied table. One D455 depth frame is deprojected
into the sim frame and the disc around the placement point is checked for
anything standing above the mat; a 45 mm block there would be struck by the
held object on the way down.

    python place_held_object.py --sim-xy 0.515189,0.019672            # check only
    python place_held_object.py --sim-xy 0.515189,0.019672 --execute
"""
import argparse
import sys

import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from isaacgymenvs.open_loop import frames, perceive_scene as ps  # noqa: E402

CALIB = (os.environ.get("TRACE_CALIB", "data/calibration/camera_to_base.txt"))


def occupancy(sim_xy, radius, calib):
    """(points examined, tallest thing found) in the disc around sim_xy."""
    color, depth, K = ps.capture_realsense(warmup=30)
    x_s, y_s, z = ps.deproject_to_sim(depth, K, np.loadtxt(calib))
    # depth == 0 means "no return", and those pixels deproject to the camera
    # origin itself -- which lands inside the disc, 64 cm tall, and makes every
    # cell look occupied. Only lit pixels say anything about the table.
    near = (depth > 0) & (np.hypot(x_s - sim_xy[0], y_s - sim_xy[1]) < radius) & np.isfinite(z)
    return int(near.sum()), (float(np.percentile(z[near], 99.5)) if near.any() else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-xy", required=True, help="placement point, sim frame: x,y")
    ap.add_argument("--z", type=float, default=0.011,
                    help="TCP z to release at; the grasp descended to this")
    ap.add_argument("--radius", type=float, default=0.05,
                    help="clearance disc around the placement point (m)")
    ap.add_argument("--max-height", type=float, default=0.012,
                    help="tallest thing tolerated in that disc (m above the mat)")
    ap.add_argument("--transit-z", type=float, default=0.15)
    ap.add_argument("--calib", default=CALIB)
    ap.add_argument("--ip", default=os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"))
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    sim_xy = tuple(float(v) for v in args.sim_xy.split(","))
    px, py = frames.sim_to_real(*sim_xy)
    print(f"placement  sim ({sim_xy[0]:+.4f},{sim_xy[1]:+.4f})  real ({px:+.4f},{py:+.4f})  z={args.z:.3f}")

    n, tallest = occupancy(sim_xy, args.radius, args.calib)
    print(f"clearance  {n} depth points within {args.radius * 100:.0f} cm, "
          f"tallest {tallest * 1000:.1f} mm above the mat "
          f"(limit {args.max_height * 1000:.0f} mm)")
    if n < 100:
        sys.exit("Too few depth points to judge the cell; refusing to place.")
    if tallest > args.max_height:
        sys.exit("Something is standing at the placement point; refusing to place.")

    if not args.execute:
        print("Cell is clear. Re-run with --execute to place.")
        return

    from robotiq_gripper import RobotiqGripper
    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    rtde_c, rtde_r = RTDEControl(args.ip), RTDEReceive(args.ip)
    gripper = RobotiqGripper(args.ip, 63352)
    gripper.connect()
    if gripper.get_current_position() < 10:
        rtde_c.stopScript()
        sys.exit("Gripper is open -- nothing is being held.")
    rot = rtde_r.getActualTCPPose()[3:6]          # the orientation it grasped with

    def pose(x, y, z):
        return [x, y, z, rot[0], rot[1], rot[2]]

    tcp = rtde_r.getActualTCPPose()
    ok = rtde_c.moveL(pose(tcp[0], tcp[1], max(tcp[2], args.transit_z)), 0.25, 1.0)
    ok = ok and rtde_c.moveL(pose(px, py, args.transit_z), 0.25, 1.0)
    ok = ok and rtde_c.moveL(pose(px, py, args.z), 0.06, 0.24)
    if not ok:
        rtde_c.stopScript()
        sys.exit("A placement move was rejected; the object is still held.")
    gripper.open_and_wait_for_pos(80, 120)
    rtde_c.moveL(pose(px, py, args.transit_z), 0.12, 0.5)
    tcp = rtde_r.getActualTCPPose()
    print(f"released at ({tcp[0]:+.4f}, {tcp[1]:+.4f})  "
          f"err=({tcp[0] - px:+.4f}, {tcp[1] - py:+.4f}) m  gripper {gripper.get_current_position()}")
    rtde_c.stopScript()
    print("Placed. Run 'python goto_corner.py HOME' to park the arm.")


if __name__ == "__main__":
    main()
