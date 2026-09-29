"""Persist stage durations from the exact event clock used by video overlays.

Each interval belongs to exactly one stage, so stage sums equal elapsed time.
Repeated progress annotations do not double-count, and failed/incomplete trials
remain distinguishable when aggregating. Rendering happens after the trial's
terminal marker and is excluded from its end-to-end execution duration.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path


TERMINAL = {'done': 'completed', 'failed': 'failed', 'dry-run': 'dry_run'}
LEGACY_STAGES = {'twin': 'twin_startup', 'solve': 'teacher_solve',
                 'solve-done': 'twin_finalize'}
STAGES = {'perceive': 'perception',
          'twin': 'simulator_and_teacher_load',
          'twin-settle': 'scene_settle',
          'solve': 'ppo_rollout_wall',
          'solve-export': 'plan_finalization',
          'solve-done': 'twin_handoff', 'execution': 'execution_setup',
          'student-load': 'student_policy_load',
          'spiral-load': 'spiral_policy_load',
          'teacher-load': 'teacher_model_load',
          'teacher-retract': 'teacher_retract',
          'teacher-sense': 'teacher_full_observation',
          'teacher-gn': 'teacher_grasp_evaluation',
          'teacher-policy': 'teacher_inference',
          'teacher-return': 'teacher_return_to_push',
          'teacher-push': 'teacher_push',
          'validate': 'hardware_validation', 'approach': 'approach', 'push': 'push',
          'student': 'student_execution', 'spiral': 'spiral_execution',
          'spiral-retract': 'occlusion_recovery', 're-sense': 'grasp_evaluation',
          'grasp': 'grasp', 'grasp-close': 'grasp',
          'no-grasp': 'grasp_skipped', 'initial-home': 'initial_home',
          'return-home': 'return_home'}


def write_json_atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(temporary, path)


def summarize_events(document, trial_id):
    events = document.get('events', [])
    # Old recordings only bracketed the whole solve. Keep their historical
    # stage names instead of pretending the missing internal boundaries exist.
    phase_names = {event.get('phase') for event in events}
    stages = dict(STAGES)
    if not phase_names.intersection({'twin-settle', 'solve-export'}):
        stages.update(LEGACY_STAGES)
    start = document.get('t0')
    if start is None or not math.isfinite(start):
        raise ValueError('A finite video epoch t0 is required')
    previous = start
    active = 'camera_startup'
    status = 'running'
    seen_grasp_check = False
    intervals = []
    terminal = None
    for event in events:
        now, phase = event['t'], event['phase']
        if not math.isfinite(now) or now < previous:
            raise ValueError('Phase clock is non-finite or reversed')
        if now > previous:
            if intervals and intervals[-1]['stage'] == active:
                intervals[-1]['end_offset_s'] = now-start
                intervals[-1]['duration_s'] = now-start-intervals[-1]['start_offset_s']
            else:
                intervals.append(dict(stage=active, start_offset_s=previous-start,
                                      end_offset_s=now-start, duration_s=now-previous))
        previous = now
        if phase in TERMINAL:
            status, terminal = TERMINAL[phase], now
            break
        if phase in ('re-sense', 'grasp', 'no-grasp'):
            seen_grasp_check = True
        if phase == 'home':
            active = 'return_home' if seen_grasp_check else 'home_for_grasp_check'
        else:
            active = stages.get(phase, phase)
    totals = {}
    for interval in intervals:
        stage = interval['stage']
        totals[stage] = totals.get(stage, 0.)+interval['duration_s']
    elapsed = previous-start
    if not math.isclose(sum(totals.values()), elapsed, abs_tol=1e-6):
        raise ValueError('Stage durations do not reconcile with elapsed time')
    solver_clocks = None
    for event in events:
        if event.get('phase') != 'solve-export':
            continue
        detail = str(event.get('detail', ''))
        sim = re.search(r'(?:^|;\s*)simulated_s=([0-9.]+)', detail)
        wall = re.search(r'(?:^|;\s*)wall_s=([0-9.]+)', detail)
        if sim:
            simulated = float(sim.group(1))
            wall_clock = float(wall.group(1)) if wall else None
            solver_clocks = {
                'ppo_rollout_simulated_seconds': simulated,
                'ppo_rollout_wall_seconds': wall_clock,
                'wall_to_sim_ratio': (wall_clock / simulated
                                      if wall_clock is not None and simulated else None),
            }
        break
    return dict(schema_version=1, trial_id=trial_id, status=status,
                clock='phases.json video annotation epoch, Unix seconds',
                started_at_epoch_s=start, ended_at_epoch_s=terminal,
                observed_until_epoch_s=previous, elapsed_seconds=elapsed,
                end_to_end_seconds=elapsed if terminal is not None else None,
                stage_seconds=totals, intervals=intervals,
                solver_clocks=solver_clocks,
                includes_camera_startup=True, excludes_postprocessing=True,
                completion_means='pipeline finished; grasp outcome is recorded separately in real_timing.json')


def write_trial_timing(directory, document=None):
    directory = Path(directory)
    if document is None:
        document = json.loads((directory/'phases.json').read_text())
    result = summarize_events(document, directory.name)
    write_json_atomic(directory/'stage_timing.json', result)
    return result


def aggregate_trials(root):
    root = Path(root)
    trials = []
    for path in sorted(root.rglob('stage_timing.json')):
        data = json.loads(path.read_text())
        trials.append(dict(trial_id=data['trial_id'], timing_file=str(path.resolve()),
                           status=data['status'], end_to_end_seconds=data['end_to_end_seconds'],
                           stage_seconds=data['stage_seconds']))
    groups = {}
    for name, accepted in [('completed_trials', {'completed'}), ('failed_trials', {'failed'}),
                           ('all_finished_trials', {'completed', 'failed'})]:
        selected = [t for t in trials if t['status'] in accepted]
        totals = {}
        for trial in selected:
            for stage, duration in trial['stage_seconds'].items():
                totals[stage] = totals.get(stage, 0.)+duration
        groups[name] = dict(count=len(selected),
                           end_to_end_seconds=sum(t['end_to_end_seconds'] for t in selected),
                           stage_seconds=totals)
    return dict(schema_version=1, root=str(root.resolve()),
                trial_count=len({t['trial_id'] for t in trials}), attempt_count=len(trials),
                excluded_incomplete_or_dry_run_count=sum(t['status'] not in {'completed', 'failed'} for t in trials),
                totals=groups, trials=trials)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    summarize = commands.add_parser('summarize')
    summarize.add_argument('trial_dir', type=Path)
    aggregate = commands.add_parser('aggregate')
    aggregate.add_argument('experiments_root', type=Path)
    aggregate.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.command == 'summarize':
        result = write_trial_timing(args.trial_dir)
    else:
        result = aggregate_trials(args.experiments_root)
        write_json_atomic(args.out or args.experiments_root/'timing_totals.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
