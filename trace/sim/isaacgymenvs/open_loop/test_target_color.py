"""Purple and royal blue are target colours; cyan distractors are not."""
import importlib.util
from pathlib import Path
import unittest

import cv2
import numpy as np

spec = importlib.util.spec_from_file_location(
    'perceive_scene', Path(__file__).with_name('perceive_scene.py'))
ps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ps)


def bgr_image(hsv_values):
    hsv = np.array(hsv_values, dtype=np.uint8).reshape(1, len(hsv_values), 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TargetColorTest(unittest.TestCase):
    def test_measured_blue_is_accepted_and_cyan_is_rejected(self):
        image = bgr_image([(103, 255, 151), (84, 255, 135)])
        blue = np.array([[True, False]])
        cyan = np.array([[False, True]])
        self.assertEqual(ps.purple_score(image, blue), 1.0)
        self.assertEqual(ps.purple_score(image, cyan), 0.0)
        self.assertEqual(ps.pick_target(image, [{'mask': cyan}, {'mask': blue}]), 1)

    def test_measured_purple_remains_accepted(self):
        image = bgr_image([(116, 107, 123), (84, 255, 135)])
        purple = np.array([[True, False]])
        cyan = np.array([[False, True]])
        self.assertEqual(ps.purple_score(image, purple), 1.0)
        self.assertEqual(ps.pick_target(image, [{'mask': cyan}, {'mask': purple}]), 1)


if __name__ == '__main__':
    unittest.main()
