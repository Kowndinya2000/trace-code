"""CPU checks for physical-object identity in shuffled observation videos."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_student_video_gallery import verify_object_mapping


class ObjectMappingTests(unittest.TestCase):
    def fixture(self):
        # Deliberately not a self-inverse permutation; using argsort is wrong.
        order = np.array([0, 7, 3, 1, 9, 6, 2, 5, 8, 10, 4])
        initial = np.zeros((1, 11, 13))
        initial[0, :, 0] = np.linspace(.3, .6, 11)
        initial[0, :, 1] = np.linspace(-.2, .2, 11)
        states = np.stack([initial.copy(), initial.copy()])
        states[0, 0, :, 0] += .012
        eef0 = np.array([[.25, .02, .1]])
        eef = np.array([[[.27, -.01, .1]], [[.29, -.03, .1]]])
        world_visible = np.zeros((2, 1, 11))
        world_visible[0, 0, [0, 2, 9]] = 1
        world_visible[1, 0, [1, 3, 8]] = 1
        obs = np.zeros((2, 1, 166))
        corners = np.array([[-.01, -.01], [-.01, .01], [.01, .01], [.01, -.01]])
        for t in range(2):
            state = initial[0] if t == 0 else states[t-1, 0]
            hand = eef0[0] if t == 0 else eef[t-1, 0]
            tokens = obs[t, 0, :110].reshape(11, 10)
            for cell, world in enumerate(order):
                tokens[cell, 8] = world_visible[t, 0, world]
                if tokens[cell, 8]:
                    tokens[cell, :8] = (state[world, :2] + corners - hand[:2]).ravel()
        return dict(student_obs=obs, visibility_world=world_visible,
                    initial_eef=eef0, eef=eef, initial_state=initial,
                    block_state=states, permutation=(order[1:]-1)[None])

    def test_scene_ids_follow_forward_permutation(self):
        mapping = verify_object_mapping(self.fixture(), 0)
        self.assertEqual([m['label'] for m in mapping],
                         ['T', '7', '3', '1', '9', '6', '2', '5', '8', '10', '4'])
        self.assertEqual([m['cell'] for m in mapping], list(range(11)))

    def test_unpermuted_visibility_is_rejected(self):
        arrays = self.fixture()
        arrays['student_obs'][:, 0, :110].reshape(2, 11, 10)[:, :, 8] = arrays['visibility_world'][:, 0]
        with self.assertRaises(AssertionError):
            verify_object_mapping(arrays, 0)

    def test_wrong_visible_object_geometry_is_rejected(self):
        arrays = self.fixture()
        arrays['student_obs'][0, 0, :8] += .05
        with self.assertRaises(AssertionError):
            verify_object_mapping(arrays, 0)

    def test_blackout_retains_identity_without_imputing_geometry(self):
        arrays = self.fixture()
        arrays['student_obs'][:] = 0
        arrays['visibility_world'][:] = 0
        mapping = verify_object_mapping(arrays, 0)
        self.assertEqual(mapping[4], dict(cell=4, world_index=9, label='9'))
        self.assertFalse(arrays['student_obs'].any())


if __name__ == '__main__':
    unittest.main()
