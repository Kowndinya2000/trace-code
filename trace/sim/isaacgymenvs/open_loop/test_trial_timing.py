import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('trial_timing', Path(__file__).with_name('trial_timing.py'))
timing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timing)


def document(rows):
    return dict(t0=100., events=[dict(t=100.+t, phase=p, detail='') for t, p in rows])


class TimingTest(unittest.TestCase):
    def test_teacher_full_observation_stages_repeat_and_reconcile(self):
        rows = [(1, 'teacher-load'), (2, 'initial-home'),
                (3, 'teacher-retract'), (5, 'teacher-sense'),
                (7, 'teacher-gn'), (11, 'teacher-policy'),
                (12, 'teacher-return'), (15, 'teacher-push'),
                (17, 'teacher-retract'), (19, 'teacher-sense'),
                (21, 'teacher-gn'), (24, 'grasp'), (29, 'done')]
        result = timing.summarize_events(document(rows), 'trial1')
        self.assertEqual(result['status'], 'completed')
        self.assertAlmostEqual(result['end_to_end_seconds'], 29)
        self.assertAlmostEqual(sum(result['stage_seconds'].values()), 29)
        self.assertEqual(result['stage_seconds']['teacher_retract'], 4)
        self.assertEqual(result['stage_seconds']['teacher_full_observation'], 4)
        self.assertEqual(result['stage_seconds']['teacher_grasp_evaluation'], 7)
        self.assertEqual(result['stage_seconds']['grasp'], 5)

    def test_progress_and_nested_solver_events_reconcile_without_double_counting(self):
        data = timing.summarize_events(document([(2, 'perceive'), (5, 'twin'), (9, 'solve'),
            (10, 'solve-done'), (11, 'approach'), (12, 'push'), (13, 'push'), (15, 'home'),
            (18, 're-sense'), (20, 'grasp'), (23, 'home'), (25, 'done'), (100, 'render')]), 'trial')
        self.assertEqual(data['end_to_end_seconds'], 25)
        self.assertEqual(sum(data['stage_seconds'].values()), 25)
        self.assertEqual(data['stage_seconds']['push'], 3)
        self.assertEqual(data['stage_seconds']['home_for_grasp_check'], 3)
        self.assertEqual(data['stage_seconds']['return_home'], 2)
        self.assertEqual(data['stage_seconds']['teacher_solve'], 1)
        self.assertEqual(data['status'], 'completed')

    def test_solver_boundaries_separate_load_settle_rollout_and_finalization(self):
        doc = document([
            (2, 'perceive'), (5, 'twin'), (9, 'twin-settle'),
            (11, 'solve'), (14, 'solve-export'), (16, 'solve-done'),
            (17, 'approach'), (19, 'done')])
        doc['events'][4]['detail'] = 'simulated_s=2.25; wall_s=3.0; steps=34; finalizing plan'
        data = timing.summarize_events(doc, 'trial')
        self.assertEqual(data['stage_seconds']['simulator_and_teacher_load'], 4)
        self.assertEqual(data['stage_seconds']['scene_settle'], 2)
        self.assertEqual(data['stage_seconds']['ppo_rollout_wall'], 3)
        self.assertEqual(data['stage_seconds']['plan_finalization'], 2)
        self.assertEqual(data['stage_seconds']['twin_handoff'], 1)
        self.assertEqual(data['solver_clocks']['ppo_rollout_simulated_seconds'], 2.25)
        self.assertEqual(data['solver_clocks']['ppo_rollout_wall_seconds'], 3.0)
        self.assertEqual(sum(data['stage_seconds'].values()), 19)

    def test_failure_closes_active_stage_and_incomplete_run_is_not_completed(self):
        data = timing.summarize_events(document([(1, 'push'), (4, 'failed')]), 'trial')
        self.assertEqual(data['status'], 'failed')
        self.assertEqual(data['stage_seconds']['push'], 3)
        data = timing.summarize_events(document([(1, 'push')]), 'trial')
        self.assertEqual(data['status'], 'running')
        self.assertIsNone(data['end_to_end_seconds'])

    def test_reversed_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            timing.summarize_events(document([(2, 'push'), (1, 'done')]), 'trial')

    def test_aggregate_keeps_failures_separate_and_excludes_incomplete_trials(self):
        with tempfile.TemporaryDirectory() as root:
            for i, rows in enumerate([[(1, 'push'), (4, 'done')], [(2, 'push'), (7, 'failed')], [(1, 'push')]]):
                d = Path(root)/f'scene001-trial{i}'
                timing.write_trial_timing(d, document(rows))
            result = timing.aggregate_trials(root)
            self.assertEqual(result['totals']['completed_trials']['end_to_end_seconds'], 4)
            self.assertEqual(result['totals']['all_finished_trials']['end_to_end_seconds'], 11)
            self.assertEqual(result['excluded_incomplete_or_dry_run_count'], 1)
            self.assertEqual(result['trial_count'], 3)

    def test_archived_retry_is_counted_once_as_a_separate_attempt(self):
        with tempfile.TemporaryDirectory() as root:
            d = Path(root)/'scene001-trial1'
            old = timing.summarize_events(document([(1, 'solve'), (4, 'failed')]), d.name)
            timing.write_json_atomic(d/'attempts/attempt001/stage_timing.json', old)
            timing.write_trial_timing(d, document([(1, 'push'), (5, 'done')]))
            result = timing.aggregate_trials(root)
            self.assertEqual(result['trial_count'], 1)
            self.assertEqual(result['attempt_count'], 2)
            self.assertEqual(result['totals']['all_finished_trials']['end_to_end_seconds'], 9)


if __name__ == '__main__':
    unittest.main()
