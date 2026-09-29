"""Offline tests for lossless trace serialization and timing-aware validation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name, str(Path(__file__).with_name(name + ".py")))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


trajectory = module("trajectory")
metrics = module("trace_metrics")


def trace():
    initial = {"time_s": 0.0, "eef_state": [0.3, 0.0, 0.02, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               "joint_pos": [0.0]*8, "joint_vel": [0.0]*8,
               "block_state": [[0.5, 0, 0.024, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]}
    samples = []
    for k, xyz in enumerate([[0.31, 0, 0.021], [0.31, 0.01, 0.02]]):
        s = copy.deepcopy(initial)
        s["time_s"] = (k+1)*0.0166
        s["eef_state"][:3] = xyz
        s["joint_targets"] = [0.125*(k+1)]*8
        samples.append(s)
    return dict(dt=0.0166, frame="sim", quaternion_order="xyzw", initial=initial, samples=samples)


class PhysicsTraceTest(unittest.TestCase):
    def test_roundtrip_retains_turn_height_timing_and_command_alignment(self):
        t = trajectory.OpenLoopTrajectory()
        t.record_start([0.3, 0, 0.02])
        t.record_step(1, [0.31, 0.01, 0.02], 6, 0.95)
        t.set_physics(trace())
        with tempfile.TemporaryDirectory() as d:
            p = t.save(str(Path(d)/"trace.json"))
            loaded = trajectory.OpenLoopTrajectory.load(p)
        self.assertEqual(t.physics, loaded.physics)
        self.assertEqual(loaded.physics_path()[1], [0.31, 0, 0.021])
        self.assertEqual(len(loaded.dense_path()), 2)
        self.assertEqual([s["time_s"] for s in loaded.cartesian_trace()], [0, 0.0166, 0.0332])
        self.assertEqual(loaded.physics["samples"][0]["joint_targets"], [0.125]*8)

    def test_legacy_files_load_but_cannot_claim_full_trace(self):
        for version in [1, 2]:
            with tempfile.TemporaryDirectory() as d:
                p = Path(d)/"old.json"
                p.write_text(json.dumps(dict(version=version,start_eef=[0.3,0,0.02],dense=[])))
                t = trajectory.OpenLoopTrajectory.load(str(p))
                self.assertEqual(t.dense_path(), [[0.3,0,0.02]])
                with self.assertRaises(ValueError):
                    t.physics_path()

    def test_embedded_scene_survives_source_overwrite(self):
        t = trajectory.OpenLoopTrajectory("missing_or_reused_scene.txt")
        t.scene_text = "cube.urdf original scene\n"
        with tempfile.TemporaryDirectory() as d:
            p = t.save(str(Path(d)/"trajectory.json"))
            loaded = trajectory.OpenLoopTrajectory.load(p)
            dest = Path(d)/"000000.txt"
            loaded.write_scene(str(dest))
            self.assertEqual(dest.read_text(), t.scene_text)

    def test_bad_timing_and_commands_are_rejected(self):
        for value in [float('nan'), 0.0, 0.0166, 0.1]:
            t = trace()
            t['samples'][1]['time_s'] = value
            with self.assertRaises(ValueError):
                trajectory.validate_physics_trace(t)
        for bad in [[], [1]*6, [float('inf')]*8]:
            t = trace()
            t['samples'][0]['joint_targets'] = bad
            with self.assertRaises(ValueError):
                trajectory.validate_physics_trace(t)

    def test_metrics_measure_tracking_not_just_final_endpoint(self):
        ref, actual = trace(), trace()
        actual['samples'][0]['eef_state'][0] += 0.01
        m = metrics.compare_physics(ref, actual)
        self.assertAlmostEqual(m['max_eef_position_mm'], 10)
        self.assertEqual(m['final_eef_position_mm'], 0)
        self.assertFalse(m['eef_state_exact'])
        actual = trace()
        actual['samples'].pop()
        self.assertFalse(metrics.compare_physics(ref, actual)['complete'])

    def test_quaternion_sign_is_not_orientation_error(self):
        ref, actual = trace(), trace()
        for s in actual['samples']:
            s['eef_state'][3:7] = [-v for v in s['eef_state'][3:7]]
        self.assertEqual(metrics.compare_physics(ref, actual)['max_eef_orientation_deg'], 0)


if __name__ == '__main__':
    unittest.main()
