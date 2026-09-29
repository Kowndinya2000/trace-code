"""Plan-window and blackout-schedule variants for representation ablations.

The released 166-D student observation exposes a 4-pose look-ahead of the
nominal EEF path. A variant keeps every other feature bit-identical -- tokens,
globals, previous action, and the plan tail (drift, progress, exhaustion flag,
predicted object centres) -- and changes only the look-ahead block:

  window K       nominal EEF at min(t+k, n-1), k = 0..K-1   (K = 4 is released)
  full_static    nominal EEF at min(j, n-1),   j = 0..119   (whole plan, every step)

Blackout schedules:

  horizon_uniform  released: start ~ U{0..horizon-len}, drawn from the visibility
                   stream before any dropout draw (most short episodes end first)
  within_plan      start = floor(u * max(1, L - len + 1)) with u from a separate
                   scene-keyed stream, L = nominal action count; dropout draws are
                   identical for every blackout length
"""
import numpy as np

from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop.evaluation_core import scene_seed, token_order

TOKEN_END = so.N_TOKENS * so.TOKEN_DIM                       # 110
BASE_DIM = TOKEN_END + so.N_GLOBAL + so.N_ACTIONS            # 132
TAIL_DIM = 2 + 2 + 2 * so.N_TOKENS                           # 26
FULL_SLOTS = 120
DEFAULT_LAYOUT = dict(plan_mode='window', plan_k=so.PLAN_LOOKAHEAD)
SCHEDULES = ('horizon_uniform', 'within_plan')
assert BASE_DIM + 2 * so.PLAN_LOOKAHEAD + TAIL_DIM == so.OBS_DIM


def layout(value=None):
    """Normalized layout. plan_objects='window' also windows predicted object centres
    (indices t..t+K-1); the released form keeps only the current index."""
    value = dict(DEFAULT_LAYOUT if value is None else value)
    mode = value.get('plan_mode')
    objects = value.get('plan_objects', 'current')
    if objects not in ('current', 'window'):
        raise ValueError(f'Unknown object mode {objects!r}')
    if mode == 'window':
        k = int(value.get('plan_k', -1))
        if not 0 <= k <= FULL_SLOTS:
            raise ValueError('Window length must be in [0, 120]')
        if objects == 'window':
            if k < 1:
                raise ValueError('Windowed objects require at least one index')
            return dict(plan_mode='window', plan_k=k, plan_objects='window')
        return dict(plan_mode='window', plan_k=k)
    if mode == 'full_static':
        if objects != 'current':
            raise ValueError('Full-plan object windows are not supported')
        return dict(plan_mode='full_static', plan_k=FULL_SLOTS)
    raise ValueError(f'Unknown plan mode {mode!r}')


def object_slots(value=None):
    lay = layout(value)
    return lay['plan_k'] if lay.get('plan_objects') == 'window' else 1


PLAN_COMPONENTS = ('lookahead', 'drift', 'progress', 'objects')


def plan_components(value=None):
    """Index ranges of each plan component in a variant observation."""
    lay = layout(value)
    look = BASE_DIM + 2 * lay['plan_k']
    return dict(lookahead=(BASE_DIM, look), drift=(look, look + 2), progress=(look + 2, look + 4),
                objects=(look + 4, look + 4 + 2 * so.N_TOKENS * object_slots(lay)))


def validate_drop(drop):
    drop = sorted(set(drop or ()))
    if any(c not in PLAN_COMPONENTS for c in drop):
        raise ValueError(f'Unknown plan component in {drop}')
    return drop


def component_mask(value, drop):
    """Multiplicative mask that zeroes the listed plan components, or None."""
    drop = validate_drop(drop)
    if not drop:
        return None
    mask = np.ones(obs_dim(value), np.float32)
    ranges = plan_components(value)
    for component in drop:
        lo, hi = ranges[component]
        mask[lo:hi] = 0.
    return mask


def is_default(value):
    return layout(value) == DEFAULT_LAYOUT


def obs_dim(value=None):
    return BASE_DIM + 2 * layout(value)['plan_k'] + TAIL_DIM + 2 * so.N_TOKENS * (object_slots(value) - 1)


def _indices(lay, t, n):
    k = np.arange(lay['plan_k'])
    j = t[..., None] + k if lay['plan_mode'] == 'window' else np.broadcast_to(k, t.shape + k.shape)
    return np.minimum(j, n - 1)


def lookahead(plan_xy, t, eef, value):
    """Look-ahead block for decision indices t (any shape) and matching EEF (..., 2).

    Reproduces encode_plan's arithmetic: Python-float plan coordinates minus a
    float32 EEF, stored as float32.
    """
    lay = layout(value)
    t = np.asarray(t, np.int64)
    eef = np.asarray(eef, np.float32)
    out = np.zeros(t.shape + (2 * lay['plan_k'],), np.float32)
    n = len(plan_xy)
    if n == 0 or lay['plan_k'] == 0:
        return out
    pts = np.asarray(plan_xy, np.float64)[_indices(lay, t, n)]          # (..., K, 2)
    out[:] = (pts - eef[..., None, :].astype(np.float64)).reshape(out.shape)
    return out


def object_window(plan_objects, t, eef, value):
    """Predicted centres at min(t+k, n-1), k=0..K-1, minus the float32 EEF (encode_plan arithmetic)."""
    slots = object_slots(value)
    t = np.asarray(t, np.int64)
    eef = np.asarray(eef, np.float32)
    out = np.zeros(t.shape + (slots, so.N_TOKENS, 2), np.float32)
    n = len(plan_objects) if plan_objects is not None else 0
    if n:
        pred = np.asarray(plan_objects, np.float32)[:, :so.N_TOKENS]           # (n, <=11, 2)
        j = np.minimum(t[..., None] + np.arange(slots), n - 1)
        out[..., :pred.shape[1], :] = pred[j] - eef[..., None, None, :]
    return out.reshape(t.shape + (slots * 2 * so.N_TOKENS,))


def convert(base, plan_xy, t, value, plan_objects=None):
    """Variant observation(s) from released 166-D observation(s) at decision index t."""
    lay = layout(value)
    base = np.asarray(base, np.float32)
    if lay == DEFAULT_LAYOUT:
        return base
    eef = base[..., TOKEN_END:TOKEN_END + 2]
    look = lookahead(plan_xy, t, eef, lay)
    tail = base[..., BASE_DIM + 2 * so.PLAN_LOOKAHEAD:]
    if lay.get('plan_objects') == 'window':
        if plan_objects is None:
            raise ValueError('Windowed objects require the nominal object trajectory')
        tail = np.concatenate([tail[..., :4], object_window(plan_objects, t, eef, lay)], axis=-1)
    return np.concatenate([base[..., :BASE_DIM], look, tail], axis=-1).astype(np.float32)


def initial_scene_context(base, plan_objects):
    """Initial-scene-only control from released 166-D observation(s).

    Zeroes every nominal-EEF feature (look-ahead, drift, progress, exhaustion flag) and
    replaces the twin's PREDICTED object centres at decision t with its INITIAL object
    centres (plan index 0, the perceived scene the plan was built from), in the same
    current-EEF-relative frame and object order as encode_plan. The student therefore
    keeps access to the initial scene estimate but receives no predicted future.
    """
    out = np.array(base, np.float32, copy=True)
    if plan_objects is None or not len(plan_objects):
        raise ValueError('Initial-scene context requires the twin object trajectory')
    initial = np.asarray(plan_objects[0], np.float32)
    count = min(so.N_TOKENS, initial.shape[0])
    objects = BASE_DIM + 2 * so.PLAN_LOOKAHEAD + 4
    out[..., BASE_DIM:objects] = 0.
    eef = out[..., TOKEN_END:TOKEN_END + 2]
    relative = initial[:count] - eef[..., None, :]
    out[..., objects:objects + 2 * so.N_TOKENS] = 0.
    out[..., objects:objects + 2 * count] = relative.reshape(eef.shape[:-1] + (2 * count,))
    return out


def blackout_start(schedule, scene_hash, seed, length, horizon, plan_actions, visibility_rng):
    """Start decision of the single blackout window, or -1 when length is zero.

    horizon_uniform consumes the visibility stream exactly as the released code.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f'Unknown blackout schedule {schedule!r}')
    if schedule == 'horizon_uniform':
        return int(visibility_rng.integers(0, horizon - length + 1)) if length else -1
    if not length:
        return -1
    u = np.random.default_rng(scene_seed(scene_hash, seed, 'blackout-within-plan')).random()
    return int(u * max(1, int(plan_actions) - length + 1))


def remask_tokens(shard, p_drop, length, schedule):
    """Recompute token visibility/staleness for every trajectory of a collection shard.

    Mirrors evaluate_retrieval.run_case: a trajectory draws dropout only at observed
    decisions, geometry is zeroed where hidden, staleness is min(age, 10) / 10.
    Returns a copy of the shard's clean observations with the new token block.
    """
    if str(shard.get('capture_schema', '')) != 'clean-observation-and-arm-mask-v1':
        raise ValueError('Shard lacks clean observations and arm masks')
    if str(shard['age_mode']) != 'current':
        raise ValueError('Only current-decision staleness is supported')
    x = np.array(shard['clean_obs'], np.float32, copy=True)
    T, N, _ = x.shape
    horizon, seed = int(shard['execution_limit']), int(shard['visibility_seed'])
    for i, identity in enumerate(shard['scene_hash']):
        rng = np.random.default_rng(scene_seed(str(identity), seed, 'visibility'))
        start = blackout_start(schedule, str(identity), seed, int(length), horizon,
                               int(shard['step_budget'][i]), rng)
        age = np.zeros(so.N_TOKENS, np.float32)
        order = token_order(shard['permutation'][i])
        for t in range(T):
            vis = np.zeros(so.N_TOKENS, np.float32)
            if shard['decision_observed'][t, i]:
                vis = so.apply_random_occlusion(shard['geometric_visibility_world'][t, i], rng, p_drop,
                                                blackout=length > 0 and start <= t < start + length)
                age = so.step_staleness(age, vis)
            tokens = x[t, i, :TOKEN_END].reshape(so.N_TOKENS, so.TOKEN_DIM)
            tokens[:, :so.GEOM_DIM] = np.where(vis[order, None] > .5, tokens[:, :so.GEOM_DIM], 0.)
            tokens[:, so.GEOM_DIM] = vis[order]
            tokens[:, so.GEOM_DIM + 1] = np.clip(age[order] / so.STALE_CAP, 0., 1.)
    return np.nan_to_num(x, nan=0, posinf=0, neginf=0)
