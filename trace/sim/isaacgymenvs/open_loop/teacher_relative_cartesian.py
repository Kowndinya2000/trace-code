"""Continuous teacher-relative Cartesian student controller (simulator only).

The learned head proposes one endpoint or both two-leg waypoints in metres,
relative to the current EEF.  During the nominal plan, the proposal is projected
onto a ball around the teacher's commanded waypoint path.  After the plan, each
recovery command is capped and kept in a disk around the nominal final EEF.

This module is NumPy-only so the geometric contract can be unit tested without
Isaac Gym, CUDA, or a robot connection.
"""
import numpy as np

from isaacgymenvs.open_loop.plan_constraint import primitive_paths


PROTOCOL = 'teacher-relative-cartesian-v1'
HEADS = ('eef_endpoint', 'eef_waypoints')
RECOVERY_ACTORS = ('student', 'teacher')


def config(head, residual_m=.03, recovery_travel_m=.12, recovery_radius_m=.13,
           progress_window=2):
    if head not in HEADS:
        raise ValueError('Unknown Cartesian head')
    values = [residual_m, recovery_travel_m, recovery_radius_m]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError('Cartesian bounds must be positive and finite')
    if (not isinstance(progress_window, int) or isinstance(progress_window, bool)
            or progress_window < 0):
        raise ValueError('Progress window must be a non-negative integer')
    return dict(protocol=PROTOCOL, head=head, output_dim=2 if head == 'eef_endpoint' else 4,
                primitive_distance_m=.04, max_teacher_path_residual_m=float(residual_m),
                max_recovery_travel_m=float(recovery_travel_m),
                max_recovery_radius_m=float(recovery_radius_m),
                progress_window=int(progress_window), progress='monotone_nearest_forward',
                recovery_anchor='nominal_final_eef',
                recovery_accounting='measured physics-frame TCP xy arc length',
                orientation='unchanged', height='unchanged', hardware_validated=False)


def validate(settings):
    if not isinstance(settings, dict) or settings.get('head') not in HEADS:
        raise ValueError('Missing Cartesian policy settings')
    expected = config(settings['head'], settings['max_teacher_path_residual_m'],
                      settings['max_recovery_travel_m'], settings['max_recovery_radius_m'],
                      settings['progress_window'])
    if settings != expected:
        raise ValueError('Unknown or modified Cartesian policy settings')
    return expected


def label_paths(actions, head, total=.04):
    """Convert categorical teacher actions to Cartesian regression targets."""
    actions = np.asarray(actions)
    if not np.issubdtype(actions.dtype, np.integer) or ((actions < 0) | (actions >= 16)).any():
        raise ValueError('Expected action indices in [0, 16)')
    paths = primitive_paths(total)[actions, 1:]
    return paths[:, -1] if head == 'eef_endpoint' else paths.reshape(*actions.shape, 4)


def controller_proposal(student_raw, teacher_action, progress, action_count, settings,
                        recovery_actor='student'):
    """Choose who supplies the Cartesian proposal without changing its bounds.

    The privileged-teacher option is a simulator-only diagnostic: the student
    remains in control throughout the nominal reference, and only plan-exhausted
    recovery proposals are replaced by the teacher's current primitive path.
    ``decode`` still applies the checkpoint's per-command, cumulative-travel,
    and recovery-radius limits.
    """
    settings = validate(settings)
    if recovery_actor not in RECOVERY_ACTORS:
        raise ValueError('Unknown Cartesian recovery actor')
    raw = np.asarray(student_raw, dtype=np.float32).reshape(-1)
    if raw.shape != (settings['output_dim'],) or not np.isfinite(raw).all():
        raise ValueError('Invalid student Cartesian proposal')
    progress, action_count = int(progress), int(action_count)
    if progress < 0 or action_count < 0 or progress > action_count:
        raise ValueError('Invalid plan progress/action count')
    if progress < action_count or recovery_actor == 'student':
        phase = 'nominal' if progress < action_count else 'recovery'
        return raw.copy(), 'student_' + phase
    action = int(teacher_action)
    if action != teacher_action or not 0 <= action < 16:
        raise ValueError('Privileged recovery requires a valid teacher action')
    proposal = label_paths(np.asarray([action], dtype=np.int64), settings['head'])[0]
    return np.asarray(proposal, dtype=np.float32), 'teacher_recovery'


def advance_progress(eef_xy, plan_xy, current, window=2):
    """Monotone local nearest-point progress; it cannot jump across the plan."""
    eef = np.asarray(eef_xy, dtype=np.float64)
    xy = np.asarray(plan_xy, dtype=np.float64)
    if eef.shape != (2,) or xy.ndim != 2 or xy.shape[1] != 2 or not len(xy):
        raise ValueError('Invalid EEF or nominal plan geometry')
    if not np.isfinite(eef).all() or not np.isfinite(xy).all():
        raise ValueError('Non-finite EEF or nominal plan geometry')
    current = int(current)
    if current < 0 or current >= len(xy):
        raise ValueError('Progress lies outside nominal plan')
    hi = min(len(xy) - 1, current + int(window))
    candidates = np.arange(current, hi + 1)
    # np.argmin deliberately keeps the earlier point on exact ties.
    return int(candidates[np.linalg.norm(xy[candidates] - eef, axis=1).argmin()])


def _scale_polyline(path, maximum_length):
    path = np.asarray(path, dtype=np.float64)
    vertices = np.concatenate([np.zeros((1, 2)), path], axis=0)
    length = np.linalg.norm(np.diff(vertices, axis=0), axis=1).sum()
    scale = min(1., float(maximum_length) / max(length, 1e-12))
    return path * scale, float(length * scale)


def _project_disk(points, anchor, radius):
    delta = points - anchor
    norm = np.linalg.norm(delta, axis=1)
    scale = np.minimum(1., radius / np.maximum(norm, 1e-12))
    return anchor + delta * scale[:, None]


def decode(raw, eef_xy, plan, progress, recovery_used_m, settings):
    """Return relative two-waypoint XY commands and traceable projection data.

    ``None`` is returned only after the measured post-plan recovery budget is
    exhausted.  Nominal-plan deviations project back toward the teacher path;
    they never become the old fail-closed ``constraint_blocked`` outcome.
    """
    settings = validate(settings)
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    eef = np.asarray(eef_xy, dtype=np.float64)
    xy = np.asarray(plan['xy'], dtype=np.float64)
    actions = np.asarray(plan['actions'])
    if (raw.shape != (settings['output_dim'],) or eef.shape != (2,)
            or xy.shape != (len(actions) + 1, 2) or not np.isfinite(raw).all()
            or not np.isfinite(eef).all() or not np.isfinite(xy).all()
            or not np.isfinite(recovery_used_m) or recovery_used_m < 0):
        raise ValueError('Invalid Cartesian decoder input')
    progress = int(progress)
    if progress < 0 or progress > len(actions):
        raise ValueError('Invalid nominal progress')

    if settings['head'] == 'eef_endpoint':
        proposed = np.stack([raw / 2., raw], axis=0)
    else:
        proposed = raw.reshape(2, 2)

    if progress < len(actions):
        action = int(actions[progress])
        if not 0 <= action < 16:
            raise ValueError('Invalid nominal primitive')
        nominal_rel = primitive_paths(settings['primitive_distance_m'])[action, 1:]
        nominal_abs = xy[progress] + nominal_rel
        proposed_abs = eef + proposed
        residual = proposed_abs - nominal_abs
        maximum = np.linalg.norm(residual, axis=1).max()
        alpha = min(1., settings['max_teacher_path_residual_m'] / max(maximum, 1e-12))
        executed_abs = nominal_abs + alpha * residual
        executed = executed_abs - eef
        return executed.astype(np.float32), dict(
            reason=None, recovery=False, nominal_action=action, progress=progress,
            projection_alpha=float(alpha), actual_plan_drift_m=float(np.linalg.norm(eef - xy[progress])),
            commanded_max_teacher_residual_m=float(np.linalg.norm(executed_abs - nominal_abs, axis=1).max()),
            commanded_polyline_length_m=float(np.linalg.norm(np.diff(
                np.concatenate([eef[None], executed_abs], axis=0), axis=0), axis=1).sum()))

    remaining = settings['max_recovery_travel_m'] - float(recovery_used_m)
    if remaining <= 1e-8:
        return None, dict(reason='extension_budget', recovery=True, progress=progress)
    executed, length = _scale_polyline(proposed, min(settings['max_teacher_path_residual_m'], remaining))
    absolute = _project_disk(eef + executed, xy[-1], settings['max_recovery_radius_m'])
    executed = absolute - eef
    # Projection to the recovery disk can lengthen a command only if the actual
    # EEF already lies outside it. Re-cap so one recovery decision is never more
    # than 3 cm of commanded polyline, while still pointing toward the anchor.
    executed, length = _scale_polyline(executed, min(settings['max_teacher_path_residual_m'], remaining))
    absolute = eef + executed
    return executed.astype(np.float32), dict(
        reason=None, recovery=True, nominal_action=-1, progress=progress,
        projection_alpha=None, actual_plan_drift_m=float(np.linalg.norm(eef - xy[-1])),
        # Same trace field as the nominal phase, but in recovery this is the
        # maximum commanded correction from the CURRENT EEF (never the up-to
        # 13 cm distance from the nominal-final anchor).
        commanded_max_teacher_residual_m=float(np.linalg.norm(executed, axis=1).max()),
        commanded_max_recovery_radius_m=float(np.linalg.norm(absolute - xy[-1], axis=1).max()),
        commanded_polyline_length_m=float(length), remaining_recovery_m=float(remaining))
