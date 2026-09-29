"""The teacher baseline's out-of-workspace rule."""
import unittest

from isaacgymenvs.open_loop import frames
from isaacgymenvs.open_loop.run_teacher_full_obs import (
    backoff_point, evicted_objects, outside_counts)


def block(name, x, y):
    return {"name": name, "x": x, "y": y, "yaw": 0.0}


class OutOfWorkspaceRuleTest(unittest.TestCase):
    def test_box_matches_the_canonical_workspace(self):
        # Same square as tasks/more_robust.WS_X / WS_Y (0.448 m at (0.5, 0)).
        low, high = frames.SIM_WORKSPACE_LIMITS[:2, 0], frames.SIM_WORKSPACE_LIMITS[:2, 1]
        self.assertAlmostEqual(float(high[0] - low[0]), 0.448, places=6)
        self.assertAlmostEqual(float(high[1] - low[1]), 0.448, places=6)

    def test_inside_scene_has_no_violation(self):
        scene = [block("cylinder", 0.5, 0.0), block("rect", 0.72, 0.22)]
        self.assertEqual(outside_counts(scene), {})
        self.assertEqual(evicted_objects(outside_counts(scene), {}), {})

    def test_object_pushed_out_is_a_violation(self):
        before = [block("cylinder", 0.5, 0.0), block("rect", 0.70, 0.20)]
        after = [block("cylinder", 0.5, 0.0), block("rect", 0.74, 0.20)]
        exempt = outside_counts(before)
        self.assertEqual(exempt, {})
        self.assertEqual(evicted_objects(outside_counts(after), exempt), {"rect": 1})

    def test_object_staged_outside_is_exempt_until_another_leaves(self):
        before = [block("rect", 0.74, 0.0), block("triangle", 0.5, 0.0)]
        exempt = outside_counts(before)
        self.assertEqual(exempt, {"rect": 1})
        # the same block still outside: not a new violation
        still = [block("rect", 0.75, 0.0), block("triangle", 0.5, 0.0)]
        self.assertEqual(evicted_objects(outside_counts(still), exempt), {})
        # a second one leaves: violation
        worse = [block("rect", 0.75, 0.0), block("triangle", 0.5, -0.30)]
        self.assertEqual(evicted_objects(outside_counts(worse), exempt), {"triangle": 1})


if __name__ == "__main__":
    unittest.main()


class BackOffPathTest(unittest.TestCase):
    """The post-push withdrawal retraces the EEF's own path."""

    LOW = frames.SIM_WORKSPACE_LIMITS[:2, 0]
    HIGH = frames.SIM_WORKSPACE_LIMITS[:2, 1]

    def back(self, path, distance):
        import numpy as np
        point = backoff_point(path, distance, self.LOW, self.HIGH)
        return None if point is None else np.round(point, 6).tolist()

    def test_disabled_by_zero(self):
        self.assertIsNone(self.back(((0.5, 0.0), (0.48, 0.0), (0.46, 0.0)), 0.0))

    def test_short_back_off_stays_on_the_last_segment(self):
        # primitive ran 0.46 -> 0.48 -> 0.50 in x; 2 cm back is exactly wp1
        self.assertEqual(self.back(((0.50, 0.0), (0.48, 0.0), (0.46, 0.0)), 0.02),
                         [0.48, 0.0])
        self.assertEqual(self.back(((0.50, 0.0), (0.48, 0.0), (0.46, 0.0)), 0.005),
                         [0.495, 0.0])

    def test_long_back_off_turns_the_corner_instead_of_leaving_the_path(self):
        # the two segments are perpendicular: 3 cm must follow the bend, not
        # continue straight back along the last one
        path = ((0.50, 0.02), (0.50, 0.0), (0.46, 0.0))
        self.assertEqual(self.back(path, 0.03), [0.49, 0.0])

    def test_never_walks_past_the_start_of_the_primitive(self):
        path = ((0.50, 0.0), (0.48, 0.0), (0.46, 0.0))
        self.assertEqual(self.back(path, 0.50), [0.46, 0.0])


class PaddedTokenTest(unittest.TestCase):
    """Padded tokens sit outside the box by design and are not evictions."""

    def test_pad_tokens_are_not_out_of_workspace(self):
        from isaacgymenvs.open_loop import perceive_scene as ps
        pad_x, pad_y = ps.PAD_CENTERS_SIM[1]          # (0.850, -0.108), the one that fired
        scene = [block("cylinder", 0.5, 0.0),
                 {"name": "cube", "x": pad_x, "y": pad_y, "yaw": 0.0, "pad": True}]
        self.assertEqual(outside_counts(scene), {})
        self.assertEqual(evicted_objects(outside_counts(scene), {}), {})

    def test_a_real_block_out_there_still_counts(self):
        scene = [block("cylinder", 0.5, 0.0), block("cube", 0.850, -0.108)]
        self.assertEqual(evicted_objects(outside_counts(scene), {}), {"cube": 1})
