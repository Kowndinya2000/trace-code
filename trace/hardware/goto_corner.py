"""Park the UR5e at one workspace point and release the control program.

Lifts to hover at the CURRENT xy first, travels at hover, then descends —
so calling it repeatedly steps corner to corner without dragging the tool
across the table. Geometry comes from reach_test, so it lands on exactly the
points the tour visits.

    python goto_corner.py NW
    python goto_corner.py C --hover-only
    python goto_corner.py 0.515189,0.019672 --hover-z 0.05 --hover-only --dwell 10
"""
import argparse
import sys
import time

import numpy as np

import reach_test as rt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("point", help="SW/S/SE/E/NE/N/NW/W/C, HOME, or 'x_sim,y_sim'")
    ap.add_argument("--ip", default=os.environ.get("TRACE_ROBOT_IP", "192.168.1.102"))
    ap.add_argument("--side", type=float, default=0.448)
    ap.add_argument("--center-sim", default="0.5,0.0")
    ap.add_argument("--offset", default="0.0,0.0")
    ap.add_argument("--push-z", type=float, default=0.020)
    ap.add_argument("--hover-z", type=float, default=0.10)
    ap.add_argument("--safe-z", type=float, default=0.22,
                    help="transit height above the clutter when homing")
    ap.add_argument("--tool-vel", type=float, default=0.08)
    ap.add_argument("--tool-acc", type=float, default=0.1)
    ap.add_argument("--hover-only", action="store_true", help="stay at hover, no touchdown")
    ap.add_argument("--dwell", type=float, default=0.0, help="hold at the point for S seconds")
    args = ap.parse_args()

    center = tuple(float(v) for v in args.center_sim.split(","))
    off = tuple(float(v) for v in args.offset.split(","))

    from rtde_control import RTDEControlInterface as RTDEControl
    from rtde_receive import RTDEReceiveInterface as RTDEReceive
    rtde_c = RTDEControl(args.ip)
    rtde_r = RTDEReceive(args.ip, use_upper_range_registers=False)

    if args.point.upper() == "HOME":
        # moveJ interpolates in JOINT space, so from inside the workspace the
        # tool swings THROUGH the clutter -- it clipped a block on the first real
        # run at hover height. Climb to safe_z, cross in a straight Cartesian
        # line above the objects, and only then moveJ down onto the home joints.
        tcp = rtde_r.getActualTCPPose()
        rtde_c.moveL(tcp[:2] + [args.safe_z] + tcp[3:6], args.tool_vel, args.tool_acc)
        hx, hy = rt.sim_to_real(0.32, 0.03, off)
        rtde_c.moveL([hx, hy, args.safe_z] + tcp[3:6], args.tool_vel, args.tool_acc)
        rtde_c.moveJ(np.deg2rad(rt.HOME_JOINTS_DEG).tolist(), 0.2, 0.2)
        tcp = rtde_r.getActualTCPPose()
        print(f"parked HOME: ({tcp[0]:+.4f}, {tcp[1]:+.4f}) z={tcp[2]:.4f}")
        rtde_c.stopScript()
        return 0

    pts = dict(rt.perimeter_points(center, args.side))
    if "," in args.point:
        xs, ys = (float(v) for v in args.point.split(","))
    elif args.point in pts:
        xs, ys = pts[args.point]
    else:
        sys.exit(f"unknown point {args.point!r}; pick from {list(pts)}, HOME, or 'x_sim,y_sim'")
    xr, yr = rt.sim_to_real(xs, ys, off)
    print(f"{args.point}  sim ({xs:+.3f},{ys:+.3f})  real ({xr:+.3f},{yr:+.3f})")

    tcp = rtde_r.getActualTCPPose()
    rot = tcp[3:6]

    def pose(x, y, z):
        return [x, y, z, rot[0], rot[1], rot[2]]

    ok = True
    if tcp[2] < args.safe_z - 1e-3:                       # lift where we stand
        ok = rtde_c.moveL(pose(tcp[0], tcp[1], args.safe_z), args.tool_vel, args.tool_acc)
    ok = ok and rtde_c.moveL(pose(xr, yr, args.safe_z), args.tool_vel, args.tool_acc)
    ok = ok and rtde_c.moveL(pose(xr, yr, args.hover_z), args.tool_vel, args.tool_acc)
    if not args.hover_only:
        ok = ok and rtde_c.moveL(pose(xr, yr, args.push_z), args.tool_vel, args.tool_acc)
    time.sleep(0.3 + max(args.dwell, 0.0))

    tcp = rtde_r.getActualTCPPose()
    margin = 360.0 - np.abs(np.rad2deg(rtde_r.getActualQ()))
    print(f"moveL ok={ok}  protective_stop={rtde_r.isProtectiveStopped()}")
    print(f"parked TCP: ({tcp[0]:+.4f}, {tcp[1]:+.4f}) z={tcp[2]:.4f}   "
          f"err=({tcp[0]-xr:+.4f}, {tcp[1]-yr:+.4f}) m")
    print(f"min joint margin {margin.min():.1f} deg (j{int(margin.argmin())})")

    rtde_c.stopScript()
    print("stopScript() — control program terminated")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
