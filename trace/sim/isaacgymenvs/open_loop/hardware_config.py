"""Where the hardware scripts find machine-specific things.

Nothing about one robot cell is baked into the source. A different cell needs at most
three environment variables:

    TRACE_CALIB      4x4 camera-to-base transform (eye-to-hand), as a text matrix
    TRACE_HARDWARE   the robot-side drivers (gripper, cameras, calibration helpers)
    TRACE_ROBOT_IP   the UR5e address

Produce a calibration with trace/hardware/pmbs_calibrate_eye_to_hand.py; the board and
mounting requirements are in docs/HARDWARE.md. No calibration is shipped: it is specific
to one camera mount and table, and using someone else's would silently misplace grasps.
"""
import os
from pathlib import Path

SIM_ROOT = Path(__file__).resolve().parents[2]          # trace/sim
REPO_ROOT = SIM_ROOT.parents[1]                         # repository root
DATA = Path(os.environ.get("TRACE_DATA", REPO_ROOT / "data"))
DEFAULT_ROBOT_IP = "192.168.1.102"


def calibration() -> Path:
    return Path(os.environ.get("TRACE_CALIB", DATA / "calibration" / "camera_to_base.txt"))


def maskrcnn() -> Path:
    return Path(os.environ.get("TRACE_MASKRCNN", DATA / "segmentation" / "maskrcnn.pth"))


def hardware_dir() -> Path:
    return Path(os.environ.get("TRACE_HARDWARE", REPO_ROOT / "trace" / "hardware"))


def runs_dir() -> Path:
    return Path(os.environ.get("TRACE_RUNS", REPO_ROOT / "runs"))


def robot_ip() -> str:
    """Robot address, configured per lab rather than embedded in the release."""
    return os.environ.get("TRACE_ROBOT_IP", DEFAULT_ROBOT_IP)
