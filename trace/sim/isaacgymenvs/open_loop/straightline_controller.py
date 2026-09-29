"""Geometry-only controller for the target-oriented straight-line baseline."""
from dataclasses import dataclass
import math

import numpy as np


# Selected on the development set as the longest 1 cm-grid cap with zero
# observed workspace exits. The nominal 15 cm calibration first exited at 12 cm.
DEFAULT_DISTANCE_M = 0.11
DEFAULT_STEP_M = 0.01


@dataclass
class StraightLineStep:
    waypoint: np.ndarray
    progress_m: float
    direction: np.ndarray
    complete: bool


class StraightLine:
    """Move a fixed path length, steering each step toward the target."""

    def __init__(self, start_xy, target_xy, *, distance_m=DEFAULT_DISTANCE_M,
                 step_m=DEFAULT_STEP_M):
        start = np.asarray(start_xy, dtype=np.float64)
        target = np.asarray(target_xy, dtype=np.float64)
        if (start.shape != (2,) or target.shape != (2,) or
                not np.isfinite(start).all() or not np.isfinite(target).all()):
            raise ValueError("start and target must be finite planar positions")
        if not all(math.isfinite(value) and value > 0
                   for value in (distance_m, step_m)):
            raise ValueError("straight-line parameters must be finite and positive")
        delta = target - start
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-9:
            raise ValueError("straight-line heading is undefined at the target")
        self.start = start.copy()
        self.direction = delta / norm
        self.distance_m = float(distance_m)
        self.step_m = float(step_m)
        self.progress_m = 0.0

    @property
    def complete(self):
        return self.progress_m >= self.distance_m - 1e-12

    def next(self, eef_xy, target_xy):
        eef = np.asarray(eef_xy, dtype=np.float64)
        target = np.asarray(target_xy, dtype=np.float64)
        if (eef.shape != (2,) or target.shape != (2,) or
                not np.isfinite(eef).all() or not np.isfinite(target).all()):
            raise ValueError("EEF and target must be finite planar positions")
        delta = target - eef
        norm = float(np.linalg.norm(delta))
        if norm > 1e-9:
            self.direction = delta / norm
        command_m = min(self.step_m, self.distance_m - self.progress_m)
        if command_m > 0:
            self.progress_m += command_m
        return StraightLineStep(
            waypoint=eef + self.direction * max(command_m, 0.0),
            progress_m=self.progress_m,
            direction=self.direction.copy(),
            complete=self.complete,
        )
