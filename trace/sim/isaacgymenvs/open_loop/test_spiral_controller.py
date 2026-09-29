import math
import unittest

import numpy as np

from isaacgymenvs.open_loop.spiral_controller import CircularSpiral


class CircularSpiralTest(unittest.TestCase):
    def test_spiral_shrinks_and_preserves_arc_direction(self):
        spiral = CircularSpiral(arc_step_m=.01, radial_step_m=.0025,
                                min_radius_m=.06, orbit_loops=3)
        target = np.array([.5, 0.])
        step = spiral.next(np.array([.6, 0.]), target)

        self.assertAlmostEqual(step.radius_m, .0975)
        self.assertAlmostEqual(np.linalg.norm(step.waypoint - target), .0975)
        self.assertAlmostEqual(step.angle_rad, .01 / .0975)
        self.assertEqual(step.orbit_progress_rad, 0.0)
        self.assertEqual(step.phase, "spiral")

    def test_bounded_start_uses_continuous_radial_entry(self):
        spiral = CircularSpiral(arc_step_m=.01, radial_step_m=.0025,
                                min_radius_m=.06, orbit_loops=3,
                                start_radius_m=.10)
        target = np.array([.5, 0.])
        step = spiral.next(np.array([.7, 0.]), target)

        self.assertEqual(step.phase, "radial_entry")
        self.assertAlmostEqual(step.radius_m, .19)
        np.testing.assert_allclose(step.waypoint, [.69, 0.])
        self.assertEqual(step.orbit_progress_rad, 0.0)

        eef = step.waypoint
        for _ in range(9):
            step = spiral.next(eef, target)
            eef = step.waypoint

        self.assertEqual(step.phase, "radial_entry")
        self.assertAlmostEqual(step.radius_m, .10)
        spiral_step = spiral.next(eef, target)
        self.assertEqual(spiral_step.phase, "spiral")
        self.assertAlmostEqual(spiral_step.radius_m, .0975)

        residual_step = CircularSpiral(
            min_radius_m=.06, start_radius_m=.10
        ).next([.6015, 0.], target)
        self.assertEqual(residual_step.phase, "spiral")

    def test_start_radius_must_cover_minimum_orbit(self):
        with self.assertRaises(ValueError):
            CircularSpiral(min_radius_m=.06, start_radius_m=.05)

    def test_resense_target_update_continues_without_jump(self):
        spiral = CircularSpiral(arc_step_m=.01, radial_step_m=.0025,
                                min_radius_m=.06, orbit_loops=3)
        eef = np.array([.60, .02])
        refreshed_target = np.array([.51, -.01])
        step = spiral.next(eef, refreshed_target)

        expected_radius = np.linalg.norm(eef - refreshed_target) - .0025
        self.assertAlmostEqual(step.radius_m, expected_radius)
        self.assertLess(np.linalg.norm(step.waypoint - eef), .011)

    def test_completion_counts_full_minimum_radius_orbits(self):
        spiral = CircularSpiral(arc_step_m=.01, radial_step_m=.0025,
                                min_radius_m=.06, orbit_loops=1)
        target = np.zeros(2)
        eef = np.array([.06, 0.])
        expected_steps = math.ceil(2 * math.pi * .06 / .01)

        for _ in range(expected_steps - 1):
            result = spiral.next(eef, target)
            eef = result.waypoint
            self.assertFalse(result.complete)
        result = spiral.next(eef, target)
        self.assertTrue(result.complete)

    def test_target_refresh_inside_final_orbit_does_not_jump_outward(self):
        spiral = CircularSpiral(arc_step_m=.01, radial_step_m=.0025,
                                min_radius_m=.06, orbit_loops=1)
        target = np.array([.5, 0.])
        eef = np.array([.505, 0.])
        result = spiral.next(eef, target)

        self.assertAlmostEqual(result.radius_m, .0075)
        self.assertLessEqual(np.linalg.norm(result.waypoint - eef), .013)
        self.assertEqual(result.orbit_progress_rad, 0.0)

    def test_invalid_parameters_are_rejected(self):
        cases = ({"arc_step_m": 0}, {"radial_step_m": -1},
                 {"min_radius_m": float("nan")}, {"orbit_loops": 0})
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CircularSpiral(**kwargs)


if __name__ == "__main__":
    unittest.main()
