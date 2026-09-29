"""A replay must never interpolate across an automatic episode reset."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('trajectory', Path(__file__).with_name('trajectory.py'))
trajectory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trajectory)


class EpisodeContinuityTest(unittest.TestCase):
    def test_continuous_or_subsampled_episode(self):
        for steps in ([], [1], [1, 2, 3], [2, 4, 8]):
            trajectory.validate_episode_continuity([{'t': t} for t in steps])

    def test_reset_cannot_be_hidden_by_increasing_physics_timestamps(self):
        # The failed hardware demo retained a monotonic synthetic physics
        # clock while dense progress restarted after its twelfth step.
        steps = list(range(1, 13)) + [0, 1, 2]
        with self.assertRaisesRegex(ValueError, 'progress 12 -> 0'):
            trajectory.validate_episode_continuity([{'t': t} for t in steps])

    def test_repeated_progress_is_rejected(self):
        with self.assertRaises(ValueError):
            trajectory.validate_episode_continuity([{'t': t} for t in [1, 2, 2]])


if __name__ == '__main__':
    unittest.main()
