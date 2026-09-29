"""Workspace geometry and frame conversions — single source of truth.

Three coordinate systems appear in this project:

  sim frame   : Isaac Gym world of the More task.
                Workspace x in [0.276, 0.724], y in [-0.224, 0.224].
  real frame  : UR5e base frame.
                Workspace x in [-0.227, 0.221], y in [-0.676, -0.228]
                (trace/hardware/constants.py REAL_WORKSPACE_LIMITS).
  pixel frame : 2 mm/px, but TWO origins exist and mixing them is a 48 px
                (9.6 cm) error:
                  * workspace-relative (sim_to_pix / pix_to_sim): origin at
                    the workspace corner, 224x224 — the legacy heightmap.
                  * CANVAS (sim_to_canvas_pix / canvas_pix_to_sim): origin at
                    the camera canvas corner (0.180, -0.320), 320x320 over
                    0.64 m — what the More camera actually renders now, and
                    what the grasp network consumes. The workspace occupies
                    the CENTRAL 224 px of it (offset CANVAS_WS_OFFSET = 48).
                Row = sim x, col = sim y in both.

sim <-> real is the Z(-90 deg) swap-negate used throughout the closed-loop
code (rl_policy.py: ``grasp_cx, grasp_cy = grasp_cy, -grasp_cx``):

  x_real = y_sim          x_sim = -y_real
  y_real = -x_sim         y_sim =  x_real

REAL_FRAME_OFFSET defaults to (0, 0), meaning both frames are rigidly attached
to the robot base. Every new installation must verify that assumption with the
read-only ``execute_trajectory.py --probe`` check before enabling motion. The
physical mat placement does not define this transform and may extend beyond
the simulator's task workspace.
"""
import numpy as np

# --- canonical constants (match rl_policy.py / trace/hardware/constants.py) ---------
SIM_WORKSPACE_LIMITS = np.asarray([[0.276, 0.724], [-0.224, 0.224], [-0.0001, 0.4]])
REAL_WORKSPACE_LIMITS = np.asarray([[-0.227, 0.221], [-0.676, -0.228], [0.18, 0.4]])
PIXEL_SIZE = 0.002          # metres per pixel (unchanged by the wider canvas)
IMAGE_SIZE = 224            # legacy workspace-relative heightmap (square)
CANVAS_SIZE = 320           # camera render: 320 px x 2 mm = 0.64 m
CANVAS_ORIGIN = np.asarray([0.180, -0.320])   # sim xy of canvas pixel (0, 0)
CANVAS_WS_OFFSET = 48       # workspace corner in canvas pixels ((320-224)/2)
NUM_ROTATION = 16           # grasp-network rotation bins
BLOCK_Z = 0.024             # rest height used by every test-case scene file

# Additive correction applied after sim->real swap-negate (see CAUTION above).
# (0, 0) reproduces rl_policy.py's behaviour exactly.
REAL_FRAME_OFFSET = np.asarray([0.0, 0.0])


def sim_to_pix(x_sim, y_sim):
    """Sim world (m) -> heightmap pixel (px, py)."""
    px = int(round((x_sim - SIM_WORKSPACE_LIMITS[0, 0]) / PIXEL_SIZE))
    py = int(round((y_sim - SIM_WORKSPACE_LIMITS[1, 0]) / PIXEL_SIZE))
    return px, py


def pix_to_sim(px, py):
    """Heightmap pixel -> sim world (m). Mirrors rl_policy.py pix2world."""
    return (SIM_WORKSPACE_LIMITS[0, 0] + px * PIXEL_SIZE,
            SIM_WORKSPACE_LIMITS[1, 0] + py * PIXEL_SIZE)


def sim_to_canvas_pix(x_sim, y_sim):
    """Sim world (m) -> CAMERA canvas pixel (row, col), 320x320 @ 2 mm/px.

    This is the convention the grasp network sees. Use it for anything derived
    from a rendered image; sim_to_pix is the legacy workspace-relative frame.
    """
    row = int(round((x_sim - CANVAS_ORIGIN[0]) / PIXEL_SIZE))
    col = int(round((y_sim - CANVAS_ORIGIN[1]) / PIXEL_SIZE))
    return row, col


def canvas_pix_to_sim(row, col):
    """Camera canvas pixel -> sim world (m)."""
    return (CANVAS_ORIGIN[0] + row * PIXEL_SIZE,
            CANVAS_ORIGIN[1] + col * PIXEL_SIZE)


def sim_to_real(x_sim, y_sim):
    """Sim world -> robot base frame (swap-negate + optional offset)."""
    return (y_sim + REAL_FRAME_OFFSET[0], -x_sim + REAL_FRAME_OFFSET[1])


def real_to_sim(x_real, y_real):
    """Robot base frame -> sim world (inverse of sim_to_real)."""
    return (-(y_real - REAL_FRAME_OFFSET[1]), x_real - REAL_FRAME_OFFSET[0])


# Our heightmap is row = sim x, col = sim y; the angle chain below is PMBS's,
# written for row = real x, col = real y. The two frames differ by the Z(-90)
# swap-negate, i.e. 4 of the 16 bins. Without this the jaws are 90 deg out --
# harmless on a symmetric target, but it lands the fingers on a neighbour
# instead of the gap (measured on the real robot: with the offset
# the gripper closed to 121/255 on the 44 mm cylinder and lifted it; without it
# every attempt hit the adjacent concave and protective-stopped).
ROT_BIN_SIM_TO_REAL = 4


def rotation_idx_to_tool_orientation(rot_idx, apply_frame_offset=True):
    """Grasp-network rotation bin -> UR axis-angle tool orientation (rx, ry).

    Verbatim port of the angle chain in rl_policy.py lines 771-802 so the
    open-loop executor grasps with exactly the closed-loop convention.
    Returns (rx, ry); rz is 0 in the TCP pose.
    """
    if apply_frame_offset:
        rot_idx = (rot_idx + ROT_BIN_SIM_TO_REAL) % NUM_ROTATION
    ang = np.deg2rad(rot_idx * (360.0 / NUM_ROTATION))
    if np.pi / 2 < ang < np.pi * 3 / 2:
        ang = ang - np.pi
    elif np.pi * 3 / 2 <= ang <= np.pi * 2:
        ang = ang - np.pi * 2
    ang = -ang + np.pi / 2
    half = ang / 2
    rx = (np.cos(half)) * np.pi     # grasp_orientation = [1, 0] rotated
    ry = (np.sin(half)) * np.pi
    return float(rx), float(ry)


def in_sim_workspace(x_sim, y_sim, margin=0.0):
    return (SIM_WORKSPACE_LIMITS[0, 0] - margin <= x_sim <= SIM_WORKSPACE_LIMITS[0, 1] + margin
            and SIM_WORKSPACE_LIMITS[1, 0] - margin <= y_sim <= SIM_WORKSPACE_LIMITS[1, 1] + margin)
