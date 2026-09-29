"""Small-offset, fail-closed action selection. No robot connections.

Bounds cover ideal commanded two-leg XY paths, not contact dynamics or collision
clearance. Actual EEF deviation is checked at every policy decision. This pilot
uses existing primitives; it is NOT a continuous residual controller.
"""
import numpy as np
from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop.evaluation_core import primitive_vectors

PROTOCOL = 'teacher-command-path-constrained-categorical-v2'
PLAN_START = so.N_TOKENS * so.TOKEN_DIM + so.N_GLOBAL + so.N_ACTIONS


def config(radius=.020, correction_step=.020):
    if not np.isfinite([radius, correction_step]).all() or min(radius, correction_step) <= 0:
        raise ValueError('Positive finite path and correction limits required')
    return dict(protocol=PROTOCOL, radius_m=float(radius), correction_step_m=float(correction_step),
                primitive_distance_m=.04, no_feasible_action='constraint_blocked',
                plan_exhausted='constraint_exhausted', hardware_validated=False)


def primitive_paths(total=.04):
    end = primitive_vectors(total).astype(np.float64)
    mid = end / 2
    s = total / 2
    for k, (x, y) in enumerate(((1,1),(-1,1),(1,-1),(-1,-1))):
        mid[5+3*k] = (x*s, 0)
        mid[6+3*k] = (0, y*s)
    return np.stack([np.zeros_like(end), mid, end], axis=1)


def extension_config(max_steps=5, max_length=.05, step_length=.01, shorten_to_bounds=False):
    if (not isinstance(max_steps,int) or isinstance(max_steps,bool) or max_steps<1
            or not np.isfinite([max_length,step_length]).all()
            or not 0<step_length<=.04 or max_length<=0
            or not isinstance(shorten_to_bounds, bool)):
        raise ValueError('Invalid extension step/travel budget')
    result = dict(protocol='bounded-short-step-extension-v1',max_steps=max_steps,
                max_length_m=float(max_length),step_length_m=float(step_length),
                anchor='nominal_final_eef',radius='checkpoint_radius',
                distance_accounting='commanded two-leg path length (conservative if workspace-clipped)')
    if shorten_to_bounds:
        result.update(protocol='bounded-adaptive-step-extension-v2', shorten_to_bounds=True,
                      extension_step_bound='commanded path length; nominal correction bound unchanged')
    return result


def _shortened_extension_paths(eef_offset, total, radius):
    """Largest scale per action whose two line segments stay inside a disk.

    The disk is convex, so checking start, intermediate and final waypoints
    covers the entire ideal commanded polyline. This is not a physics bound.
    """
    paths = primitive_paths(total)
    if np.linalg.norm(eef_offset) > radius + 1e-7:
        return paths, np.zeros(16)
    vertices = paths[:, 1:]
    a = np.square(vertices).sum(-1)
    b = (vertices * eef_offset).sum(-1)
    c = min(float(np.square(eef_offset).sum() - radius * radius), 0.)
    # Positive root of ||offset + scale * vertex||^2 = radius^2.
    roots = (-b + np.sqrt(np.maximum(b * b - a * c, 0.))) / a
    scales = np.clip(roots.min(-1), 0., 1.)
    # Leave a small numerical margin only for boundary-limited commands.
    scales = np.where(scales < 1., scales * (1. - 1e-6), scales)
    return paths * scales[:, None, None], total * scales


def select(logits, eef, plan, step, settings, extension=None, used_length=0.):
    """Return action or -1 with an explicit terminal failure, never an unsafe fallback."""
    expected = config(settings['radius_m'], settings['correction_step_m'])
    if settings != expected:
        raise ValueError('Unknown/modified constraint specification')
    logits, eef = np.asarray(logits), np.asarray(eef)
    if logits.shape != (16,) or eef.shape != (2,) or not np.isfinite(logits).all() or not np.isfinite(eef).all():
        raise ValueError('Expected finite 16 logits and XY EEF')
    if step < 0 or step != int(step):
        raise ValueError('Invalid plan progress')
    xy = np.asarray(plan['xy'], dtype=float)
    actions = np.asarray(plan['actions'])
    if xy.shape != (len(actions)+1, 2) or not np.isfinite(xy).all():
        raise ValueError('Invalid nominal plan')
    if step >= len(actions):
        if extension is None:
            return -1, dict(reason='constraint_exhausted', feasible=0)
        checked=extension_config(extension['max_steps'],extension['max_length_m'],extension['step_length_m'],
                                 extension.get('shorten_to_bounds', False))
        if checked!=extension or not np.isfinite(used_length) or used_length<0:
            raise ValueError('Invalid extension specification/usage')
        remaining=extension['max_length_m']-used_length
        if step-len(actions)>=extension['max_steps'] or remaining<=1e-8:
            return -1,dict(reason='extension_budget',feasible=0)
        total=min(extension['step_length_m'],remaining)
        paths=primitive_paths(total)
        lengths=np.full(16,total)
        correction_bound=settings['correction_step_m']
        minimum_length=0.
        if extension.get('shorten_to_bounds'):
            paths,lengths=_shortened_extension_paths(eef-xy[-1],total,settings['radius_m'])
            minimum_length=1e-6
            # Extension command length is governed by the explicit extension
            # cap; the checkpoint's nominal-plan correction limit stays intact.
            correction_bound=extension['step_length_m']
        error=np.linalg.norm(eef[None,None,:]+paths-xy[-1][None,None,:],axis=-1)
        change=np.linalg.norm(paths,axis=-1)
        allowed=((error.max(-1)<=settings['radius_m']+1e-7) &
                 (change.max(-1)<=correction_bound+1e-7) & (lengths>=minimum_length))
        if not allowed.any():
            return -1,dict(reason='constraint_blocked',feasible=0)
        action=int(np.where(allowed,logits,-np.inf).argmax())
        return action,dict(reason=None,feasible=int(allowed.sum()),raw_action=int(logits.argmax()),
            actual_drift_m=float(error[0,0]),commanded_max_offset_m=float(error[action].max()),
            commanded_max_correction_step_m=float(change[action].max()),
            action_scale=float(lengths[action]/.04),extension_length_m=float(lengths[action]))
    nominal_action = int(actions[step])
    if nominal_action != actions[step] or not 0 <= nominal_action < 16:
        raise ValueError('Invalid nominal primitive')
    paths = primitive_paths()
    # Compare like with like: both are COMMANDED waypoints. The measured next
    # nominal EEF can under-travel during the finite physics interval and must
    # not be substituted for the teacher's command endpoint.
    reference = xy[step][None,:] + paths[nominal_action]
    candidates = eef[None,None,:] + paths
    error = np.linalg.norm(candidates-reference[None,:,:], axis=-1)
    delta_error = np.linalg.norm((candidates[:,1:]-reference[None,1:]) -
                                (eef-xy[step])[None,None,:], axis=-1)
    allowed = ((error.max(-1) <= settings['radius_m']+1e-7) &
               (delta_error.max(-1) <= settings['correction_step_m']+1e-7))
    if not allowed.any():
        return -1, dict(reason='constraint_blocked', feasible=0,
                        actual_drift_m=float(error[0,0]))
    action = int(np.where(allowed, logits, -np.inf).argmax())
    return action, dict(reason=None, feasible=int(allowed.sum()),
                        raw_action=int(logits.argmax()), actual_drift_m=float(error[0,0]),
                        commanded_max_offset_m=float(error[action].max()),
                        commanded_max_correction_step_m=float(delta_error[action].max()))


def training_cost(obs, nominal_actions, radius=.020):
    """Torch endpoint/offset-change costs available in existing cached observations.

Training is soft regularization; the independent runtime selector enforces the
hard constraints, including the intermediate command and actual current drift.
"""
    import torch
    if (obs.shape[-1] != so.OBS_DIM or nominal_actions.shape != obs.shape[:-1]
            or not np.isfinite(radius) or radius <= 0):
        raise ValueError('Invalid observations/radius')
    z = obs[..., PLAN_START:]
    p = torch.as_tensor(primitive_paths()[:,1:], device=obs.device, dtype=obs.dtype)
    delta = p-p[nominal_actions.clamp(min=0)][...,None,:,:]
    error = ((delta-z[...,:2][...,None,None,:])/radius).square().sum(-1).amax(-1)
    change = (delta/radius).square().sum(-1).amax(-1)
    active = nominal_actions >= 0
    return (error.clamp(max=25)+.5*change.clamp(max=25))*active[...,None]
