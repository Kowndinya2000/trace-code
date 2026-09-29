import unittest

import numpy as np

from isaacgymenvs.open_loop.straightline_controller import StraightLine


class StraightLineTest(unittest.TestCase):
    def test_moves_fifteen_centimetres_toward_a_stationary_target(self):
        controller = StraightLine([0.3, 0.0], [0.5, 0.2], distance_m=.15, step_m=.01)
        direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
        eef = np.array([.3, 0.0])
        results = []
        for _ in range(15):
            result = controller.next(eef, [.5, .2])
            results.append(result)
            eef = result.waypoint

        np.testing.assert_allclose(results[0].waypoint,
                                   np.array([.3, 0.0]) + .01 * direction)
        np.testing.assert_allclose(results[-1].waypoint,
                                   np.array([.3, 0.0]) + .15 * direction)
        self.assertAlmostEqual(results[-1].progress_m, .15)
        self.assertTrue(results[-1].complete)

    def test_last_step_is_shortened_to_exact_path_length(self):
        controller = StraightLine([0.0, 0.0], [1.0, 0.0],
                                  distance_m=.025, step_m=.01)
        eef = np.array([0.0, 0.0])
        results = []
        for _ in range(3):
            result = controller.next(eef, [1.0, 0.0])
            results.append(result)
            eef = result.waypoint

        self.assertEqual([round(result.progress_m, 3) for result in results],
                         [.01, .02, .025])
        np.testing.assert_allclose(results[-1].waypoint, [.025, 0.0])

    def test_heading_follows_a_moving_target(self):
        controller = StraightLine([.3, .1], [.5, .1])
        first = controller.next([.3, .1], [.5, .1])
        second = controller.next(first.waypoint, first.waypoint + [0.0, .2])

        np.testing.assert_allclose(first.direction, [1.0, 0.0])
        np.testing.assert_allclose(second.direction, [0.0, 1.0])
        np.testing.assert_allclose(second.waypoint, first.waypoint + [0.0, .01])

    def test_invalid_inputs_are_rejected(self):
        cases = [
            (([0, 0], [0, 0]), {}),
            (([0, 0], [1, 0]), {"distance_m": 0}),
            (([0, 0], [1, 0]), {"step_m": float("nan")}),
        ]
        for args, kwargs in cases:
            with self.subTest(args=args, kwargs=kwargs), self.assertRaises(ValueError):
                StraightLine(*args, **kwargs)

    def test_invalid_live_state_is_rejected(self):
        controller = StraightLine([0, 0], [1, 0])
        with self.assertRaises(ValueError):
            controller.next([float("nan"), 0], [1, 0])


if __name__ == "__main__":
    unittest.main()
