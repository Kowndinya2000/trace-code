"""Filesystem layout for TRACE.

Nothing in this repository hard-codes a machine-specific path. Every location is derived
from the repository root, or overridden by an environment variable:

    TRACE_ROOT      repository root (default: two levels above this file)
    TRACE_DATA      downloaded assets, scenes, checkpoints (default: $TRACE_ROOT/data)
    TRACE_RUNS      where runs write their outputs      (default: $TRACE_ROOT/runs)
    TRACE_PYTHON    interpreter used for child processes (default: the current one)
    TRACE_CALIB     camera-to-base calibration matrix   (hardware only)
    TRACE_ROBOT_IP  UR5e address                        (hardware only)

`python -m trace.common.paths` prints the resolved layout and flags anything missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("TRACE_ROOT", Path(__file__).resolve().parents[2]))
SIM = ROOT / "trace" / "sim"          # put this on PYTHONPATH: it holds the isaacgymenvs package
DATA = Path(os.environ.get("TRACE_DATA", ROOT / "data"))
RUNS = Path(os.environ.get("TRACE_RUNS", ROOT / "runs"))
PYTHON = os.environ.get("TRACE_PYTHON", sys.executable)

# Downloaded payloads (scripts/download_data.py writes them here).
ASSETS = SIM / "assets"                       # URDF and meshes for the blocks, workspace and UR5e
SCENES = DATA / "scenes"                      # the 511 evaluation scenes, 2,097 training scenes
CHECKPOINTS = DATA / "checkpoints"            # teacher, students, label controls, ablation fits
GRASP_MODELS = SIM / "isaacgymenvs" / "logs_grasp"   # grasp network, under the name the code expects
COLLECTIONS = DATA / "collections"            # expert and DAgger labels (retraining only)
SEGMENTATION = DATA / "segmentation"          # Mask R-CNN weights (hardware perception)

TEACHER = CHECKPOINTS / "teacher_ep210.pth"
TEACHER_NETWORK = CHECKPOINTS / "teacher_network.json"   # rl-games network spec for the teacher
STUDENT = CHECKPOINTS / "trace_r3.pt"
MANIFEST_DEV = SCENES / "development.json"
MANIFEST_TRAIN = SCENES / "training.json"
BUDGET_REFERENCE = DATA / "budget.json"
MASKRCNN = Path(os.environ.get("TRACE_MASKRCNN", SEGMENTATION / "maskrcnn.pth"))

# Hardware-only settings; harmless defaults so simulation never needs them.
ROBOT_IP = os.environ.get("TRACE_ROBOT_IP", "192.168.1.102")
CALIBRATION = Path(os.environ.get("TRACE_CALIB", DATA / "calibration" / "camera_to_base.txt"))


def require(path: Path, hint: str = "run scripts/download_data.py") -> Path:
    if not Path(path).exists():
        raise SystemExit(f"Missing {path}\n  {hint}")
    return Path(path)


def ensure_runtime() -> None:
    """Make Isaac Gym importable without the caller exporting anything.

    Its bindings link against libpython and the CUDA runtime that ship inside the environment,
    so the dynamic loader has to know about <env>/lib before the process starts. When that is
    missing we re-exec this program once with the variable set, which is invisible to the user
    and cheaper than asking everyone to remember an export line.
    """
    library = str(Path(sys.prefix) / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if library in current.split(os.pathsep) or os.environ.get("TRACE_RUNTIME_READY"):
        os.environ.setdefault("TRACE_RUNTIME_READY", "1")
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([library, current]).strip(os.pathsep)
    os.environ["TRACE_RUNTIME_READY"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.execv(sys.executable, [sys.executable, *sys.argv])


def child_env() -> dict:
    """Environment for child processes: the simulation package plus the Isaac Gym runtime."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(SIM), str(SIM / "isaacgymenvs"), environment.get("PYTHONPATH", "")]).strip(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [str(Path(sys.prefix) / "lib"), environment.get("LD_LIBRARY_PATH", "")]).strip(os.pathsep)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    environment.setdefault("TRACE_PYTHON", PYTHON)      # the shell launchers spawn this
    return environment


def main() -> None:
    rows = [("TRACE_ROOT", ROOT), ("TRACE_DATA", DATA), ("TRACE_RUNS", RUNS),
            ("assets", ASSETS), ("scenes", SCENES), ("checkpoints", CHECKPOINTS),
            ("grasp models", GRASP_MODELS), ("segmentation", SEGMENTATION),
            ("calibration", CALIBRATION)]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        mark = "ok " if Path(value).exists() else "MISSING"
        print(f"{name:<{width}}  {mark}  {value}")
    print(f"{'python':<{width}}  ok       {PYTHON}")
    print(f"{'robot ip':<{width}}  ok       {ROBOT_IP}   (hardware only)")


if __name__ == "__main__":
    main()
