"""Geometry-only controller for the real-robot circular spiral baseline.

The controller deliberately has no camera, grasp-network, or robot dependencies.
The closed-loop runner owns sensing and recovery; this module only advances a
continuous inward spiral around the latest visible target position.
"""
from dataclasses import dataclass
import math

import numpy as np


# One source of truth for the hardware runner and the simulator evaluator.
DEFAULT_MAX_STEPS = 240
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_MAX_TRAVEL_M = 2.5
DEFAULT_ARC_STEP_M = 0.01
DEFAULT_RADIAL_STEP_M = 0.0025
DEFAULT_MIN_RADIUS_M = 0.06
DEFAULT_START_RADIUS_M = None
DEFAULT_ORBIT_LOOPS = 3.0


@dataclass
class SpiralStep:
    waypoint: np.ndarray
    radius_m: float
    angle_rad: float
    orbit_progress_rad: float
    phase: str
    complete: bool


class CircularSpiral:
    """Advance by fixed arc-length and radial increments around a target.

    Recomputing the polar phase from the measured EEF pose prevents a target
    refresh or an interrupted retract/re-sense cycle from producing a jump.
    Orbit progress is accumulated separately, so the requested number of final
    loops is measured from entry into the minimum-radius orbit rather than from
    the absolute polar angle.
    """

    def __init__(self, *, arc_step_m=DEFAULT_ARC_STEP_M,
                 radial_step_m=DEFAULT_RADIAL_STEP_M,
                 min_radius_m=DEFAULT_MIN_RADIUS_M,
                 orbit_loops=DEFAULT_ORBIT_LOOPS,
                 start_radius_m=DEFAULT_START_RADIUS_M):
        values = (arc_step_m, radial_step_m, min_radius_m, orbit_loops)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("spiral parameters must be finite and positive")
        if start_radius_m is not None and (
                not math.isfinite(start_radius_m) or start_radius_m < min_radius_m):
            raise ValueError("start_radius_m must be finite and at least min_radius_m")
        self.arc_step_m = float(arc_step_m)
        self.radial_step_m = float(radial_step_m)
        self.min_radius_m = float(min_radius_m)
        self.orbit_loops = float(orbit_loops)
        self.start_radius_m = (None if start_radius_m is None
                               else float(start_radius_m))
        self.spiral_started = start_radius_m is None
        self.orbit_progress_rad = 0.0

    @property
    def complete(self):
        return self.orbit_progress_rad >= 2.0 * math.pi * self.orbit_loops

    def next(self, eef_xy, target_xy):
        eef = np.asarray(eef_xy, dtype=np.float64)
        target = np.asarray(target_xy, dtype=np.float64)
        if eef.shape != (2,) or target.shape != (2,) \
                or not np.isfinite(eef).all() or not np.isfinite(target).all():
            raise ValueError("EEF and target must be finite planar positions")

        relative = eef - target
        measured_radius = float(np.linalg.norm(relative))
        angle = math.atan2(relative[1], relative[0]) if measured_radius > 1e-9 else 0.0

        # Skip broad outer orbits when a bounded start is requested. The entry
        # remains continuous: each command moves radially inward by at most one
        # arc-step length, rather than jumping directly onto the start circle.
        if (not self.spiral_started and
                measured_radius > self.start_radius_m + self.radial_step_m):
            radius = max(self.start_radius_m,
                         measured_radius - self.arc_step_m)
            waypoint = target + radius * np.array(
                [math.cos(angle), math.sin(angle)], dtype=np.float64)
            return SpiralStep(
                waypoint=waypoint,
                radius_m=radius,
                angle_rad=angle,
                orbit_progress_rad=self.orbit_progress_rad,
                phase="radial_entry",
                complete=False,
            )
        self.spiral_started = True

        if measured_radius > self.min_radius_m:
            radius = max(self.min_radius_m, measured_radius - self.radial_step_m)
        else:
            # A refreshed target estimate can place the EEF inside the final
            # orbit. Approach that orbit at the configured radial rate instead
            # of jumping outward by as much as min_radius_m in one command.
            radius = min(self.min_radius_m, measured_radius + self.radial_step_m)
        radius = max(radius, 1e-9)
        angle_delta = self.arc_step_m / radius

        # Count only travel on the final orbit. Crossing the boundary on this
        # step contributes its angular increment and cannot lose a whole step.
        if abs(radius - self.min_radius_m) <= 1e-12:
            self.orbit_progress_rad += angle_delta

        next_angle = angle + angle_delta
        waypoint = target + radius * np.array(
            [math.cos(next_angle), math.sin(next_angle)], dtype=np.float64)
        return SpiralStep(
            waypoint=waypoint,
            radius_m=radius,
            angle_rad=next_angle,
            orbit_progress_rad=self.orbit_progress_rad,
            phase="spiral",
            complete=self.complete,
        )
