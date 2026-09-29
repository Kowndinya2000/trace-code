import unittest

import numpy as np

from isaacgymenvs.open_loop import mat_boundary as mb

MAT = np.array([[0.33, -0.25], [0.33, 0.29], [0.79, 0.29], [0.79, -0.25]])


class MatBoundaryTest(unittest.TestCase):
    def test_centred_block_is_on_mat(self):
        cube = {"name": "cube", "x": 0.5, "y": 0.0, "yaw": 0.0}
        self.assertEqual(mb.footprint_off_mat(cube, MAT), 0.0)
        self.assertEqual(mb.off_mat_counts([cube], MAT), {})

    def test_small_overhang_is_detected_before_centre_leaves(self):
        # Cube centre 1 cm inside the far edge: its 45 mm footprint pokes ~12.5 mm out.
        cube = {"name": "cube", "x": 0.78, "y": 0.0, "yaw": 0.0}
        self.assertAlmostEqual(mb.footprint_off_mat(cube, MAT), 0.0125, places=3)
        self.assertEqual(mb.off_mat_counts([cube], MAT), {"cube": 1})

    def test_rotation_changes_overhang(self):
        rect = {"name": "rect", "x": 0.33 + 0.03, "y": 0.0, "yaw": 0.0}
        flat = mb.footprint_off_mat(rect, MAT)
        rect["yaw"] = np.pi / 2
        self.assertNotAlmostEqual(flat, mb.footprint_off_mat(rect, MAT))

    def test_padding_tokens_ignored(self):
        pad = {"name": "cube", "x": 0.0, "y": 0.0, "yaw": 0.0, "pad": True}
        self.assertEqual(mb.off_mat_counts([pad], MAT), {})


if __name__ == "__main__":
    unittest.main()
