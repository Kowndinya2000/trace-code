"""Student observation o_t^S in TOKEN space (the method description).

The doc's §12 body writes the student's view as pixels (I~_t plus a per-pixel
mask V_t(u,v)); "visible-object tokens with per-object visibility flags" is
listed as design option 1(c). We take 1(c), for three reasons:

  * the teacher IS a set encoder over 11 object tokens, so student and teacher
    share a representation and distillation compares like with like;
  * occlusion becomes a modelled property of each object instead of something
    the renderer has to produce -- and it cannot: more.py clips at
    near_plane 999.75, so the arm body never appears and the sim camera measures
    99.95% visibility while the real D455 shows the arm across much of the frame;
  * it is ~1000x cheaper than 4x128x128 image stacks per step per env.

Teacher token (8): 4 rotated bbox corners RELATIVE to the EEF, token 0 = target,
clutter order shuffled per episode. Student token (10) = those 8, gated by
visibility, plus [visible, staleness]. An unseen object reports ZERO geometry,
never a stale pose silently presented as current -- "unknown" and "here it is"
must not look alike. The GRU is what carries belief; staleness tells it how much
to trust that belief.

Occlusion model: padded projected regions of every physical arm/gripper body,
using CURRENT link poses and whole object mesh footprints (arm_occlusion.py).
Any footprint intersection hides the entire object token. Normal BC/DAgger
collection adds 5% independent detection dropout and one scheduled five-decision
blackout. Both remain configurable, including an explicit zero/zero ablation.
"""
import numpy as np

N_ACTIONS = 16
N_TOKENS = 11
GEOM_DIM = 8                     # 4 corners x (x, y), EEF-relative
TOKEN_DIM = GEOM_DIM + 2         # + [visible, staleness]
N_GLOBAL = 6                     # EEF xy + 4 wall distances (same as teacher)
PLAN_LOOKAHEAD = 4
INCLUDE_PLAN_ACTION_ID = False   # doc open-decision 3: the ID form is the BASELINE
# geometry (K lookahead poses) + drift + progress/exhausted, optionally + action id
PLAN_PRED_OBJ = True             # carry the twin's PREDICTED object layout
# lookahead EEF poses + EEF drift + progress/exhausted + predicted object xy
PLAN_DIM = (2 * PLAN_LOOKAHEAD + 2 + 2
            + (2 * N_TOKENS if PLAN_PRED_OBJ else 0)
            + (N_ACTIONS if INCLUDE_PLAN_ACTION_ID else 0))
OBS_DIM = N_TOKENS * TOKEN_DIM + N_GLOBAL + N_ACTIONS + PLAN_DIM

STALE_CAP = 10.0                 # steps, for normalising staleness


def arm_shadow_visibility(object_points, body_points, margin_m=.010, mode="link_union"):
    """Whole-footprint overlap; body points are required, not merely the EEF."""
    from isaacgymenvs.open_loop.arm_occlusion import footprint_visibility, projected_regions
    return footprint_visibility(object_points, projected_regions(body_points, margin_m, mode))


def apply_random_occlusion(vis, rng, p_drop=0.0, blackout=False):
    """Apply configured independent dropout or a full blackout (V_t = 0)."""
    if not 0 <= p_drop <= 1:
        raise ValueError("Dropout probability must be in [0,1]")
    if blackout:
        return np.zeros_like(vis)
    if p_drop > 0:
        vis = vis * (rng.random(vis.shape) > p_drop).astype(np.float32)
    return vis


def encode_plan(plan_xy, t, eef_xy, plan_actions=None,
                include_action_id=INCLUDE_PLAN_ACTION_ID, plan_obj_xy=None):
    """z_t^tau: the nominal twin plan as GEOMETRY, not as primitive labels.

    Passing the teacher's action ID for step t is a bad prior for two reasons.
    It is degenerate -- the student also emits action IDs, so echoing the prior
    is a shortcut the warm-start phase actively teaches. And it is ambiguous:
    the 16 primitives are EEF-RELATIVE pushes, so "direction k" issued from the
    nominal pose means something else from wherever the student actually is.
    An ID is only meaningful together with the pose it was issued from.

    So the prior is the nominal EEF path itself:
      - the next K nominal poses, relative to the student's current EEF;
      - DRIFT: nominal pose at THIS step minus the actual EEF. This is the
        quantity the student exists to correct, and it is zero exactly when the
        open-loop plan is still valid;
      - progress and plan-exhausted flags.
    The action one-hot stays available behind include_action_id, because the doc
    (open decision 3) asks for it as the comparison baseline.
    """
    z = np.zeros(PLAN_DIM, np.float32)
    n = len(plan_xy)
    if n == 0:
        return z
    eef = np.asarray(eef_xy, np.float32)
    for k in range(PLAN_LOOKAHEAD):
        j = min(t + k, n - 1)
        z[2 * k] = plan_xy[j][0] - eef[0]
        z[2 * k + 1] = plan_xy[j][1] - eef[1]
    j_now = min(t, n - 1)                       # drift against the nominal pose NOW
    z[2 * PLAN_LOOKAHEAD] = plan_xy[j_now][0] - eef[0]
    z[2 * PLAN_LOOKAHEAD + 1] = plan_xy[j_now][1] - eef[1]
    z[2 * PLAN_LOOKAHEAD + 2] = min(1.0, t / max(1, n - 1))
    z[2 * PLAN_LOOKAHEAD + 3] = float(t >= n)
    base = 2 * PLAN_LOOKAHEAD + 4
    if PLAN_PRED_OBJ:
        # Where the TWIN predicted every object would be at this step, in the
        # same EEF-relative frame as the observation tokens, so the student can
        # subtract the two directly. This is the error signal that matters:
        # in open-loop replay the arm tracks its commanded waypoints to well
        # under a millimetre (measured 0.1 mm on this UR5e), so EEF drift stays
        # ~0 even when the plan has already failed -- the real
        # target finished 60 mm from prediction with graspability 0.12 against a
        # predicted 0.91, and EEF drift would have reported nothing wrong.
        # Under blackout these predictions are also the student's only estimate
        # of the layout: twin trajectory = prior, memory + surviving
        # observations = correction.
        if plan_obj_xy is not None and len(plan_obj_xy):
            pred = np.asarray(plan_obj_xy[min(t, len(plan_obj_xy) - 1)], np.float32)
            for i in range(min(N_TOKENS, pred.shape[0])):
                z[base + 2 * i] = pred[i, 0] - eef[0]
                z[base + 2 * i + 1] = pred[i, 1] - eef[1]
        base += 2 * N_TOKENS
    if include_action_id and plan_actions is not None and len(plan_actions):
        z[base + int(plan_actions[min(t, len(plan_actions) - 1)])] = 1.0
    return z


def build_student_obs(teacher_obs, obj_xy, vis, stale, prev_action, plan_xy, t,
                      plan_actions=None, plan_obj_xy=None):
    """(OBS_DIM,) float32 from one env's teacher obs + a visibility vector.

    teacher_obs: (N_TOKENS*8 + 6,) the privileged 94-D vector
    vis:         (N_TOKENS,) 1 visible / 0 occluded, target included
    stale:       (N_TOKENS,) steps since last seen
    """
    geom = np.asarray(teacher_obs[:N_TOKENS * GEOM_DIM], np.float32).reshape(N_TOKENS, GEOM_DIM)
    glob = np.asarray(teacher_obs[N_TOKENS * GEOM_DIM:N_TOKENS * GEOM_DIM + N_GLOBAL], np.float32)
    v = np.asarray(vis, np.float32).reshape(N_TOKENS, 1)
    s = np.clip(np.asarray(stale, np.float32).reshape(N_TOKENS, 1) / STALE_CAP, 0.0, 1.0)
    toks = np.concatenate([np.where(v > .5, geom, 0.), v, s], axis=1)
    a_prev = np.zeros(N_ACTIONS, np.float32)
    if prev_action is not None and prev_action >= 0:
        a_prev[int(prev_action)] = 1.0
    eef = glob[:2]
    z = encode_plan(plan_xy, t, eef, plan_actions, plan_obj_xy=plan_obj_xy)
    return np.concatenate([toks.reshape(-1), glob, a_prev, z]).astype(np.float32)


def step_staleness(stale, vis):
    """Reset staleness where seen, otherwise age it by one step."""
    return np.where(np.asarray(vis) > 0.5, 0.0, np.asarray(stale, np.float32) + 1.0)


def demo():
    rng = np.random.default_rng(0)
    arm = np.array([[[0, -.05], [.55, -.05], [.55, .05], [0, .05]]])
    square = np.array([[-.01, -.01], [.01, -.01], [.01, .01], [-.01, .01]])
    objects = square + np.array([[.40, .015], [.50, .20], [.60, -.15]])[:, None, :]
    np.testing.assert_array_equal(arm_shadow_visibility(objects, arm), [0, 1, 1])

    tobs = rng.normal(size=N_TOKENS * GEOM_DIM + N_GLOBAL).astype(np.float32)
    v = np.ones(N_TOKENS, np.float32); v[3] = 0.0
    st = np.zeros(N_TOKENS, np.float32); st[3] = 4.0
    plan = [(0.50, 0.00), (0.55, 0.02), (0.60, 0.04)]
    z_on = encode_plan(plan, 1, (0.55, 0.02))          # exactly on plan -> zero drift
    assert abs(z_on[2 * PLAN_LOOKAHEAD]) < 1e-6 and abs(z_on[2 * PLAN_LOOKAHEAD + 1]) < 1e-6
    z_off = encode_plan(plan, 1, (0.53, 0.02))         # 2 cm behind -> drift shows it
    assert abs(z_off[2 * PLAN_LOOKAHEAD] - 0.02) < 1e-6, z_off[2 * PLAN_LOOKAHEAD]
    assert PLAN_DIM == 2 * PLAN_LOOKAHEAD + 4 + 2 * N_TOKENS, "predicted-object block"
    # predicted object layout must land EEF-relative and be recoverable
    pobj = [np.tile(np.array([[0.60, 0.10]], np.float32), (N_TOKENS, 1))] * 3
    zp = encode_plan(plan, 1, (0.55, 0.02), plan_obj_xy=pobj)
    b = 2 * PLAN_LOOKAHEAD + 4
    assert abs(zp[b] - 0.05) < 1e-6 and abs(zp[b + 1] - 0.08) < 1e-6, zp[b:b + 2]

    o = build_student_obs(tobs, None, v, st, 5, [(0.5, 0.0), (0.55, 0.02)], 0, [2, 7])
    assert o.shape == (OBS_DIM,), o.shape
    tok3 = o[3 * TOKEN_DIM:4 * TOKEN_DIM]
    assert np.allclose(tok3[:GEOM_DIM], 0.0), "occluded token must report no geometry"
    assert tok3[GEOM_DIM] == 0.0 and abs(tok3[GEOM_DIM + 1] - 0.4) < 1e-6, "flag + staleness"
    tok0 = o[0:TOKEN_DIM]
    assert tok0[GEOM_DIM] == 1.0 and not np.allclose(tok0[:GEOM_DIM], 0.0), "visible token intact"

    # blackout zeroes everything; staleness then grows
    vb = apply_random_occlusion(np.ones(N_TOKENS, np.float32), rng, blackout=True)
    assert vb.sum() == 0.0
    assert step_staleness(np.zeros(N_TOKENS), vb).min() == 1.0
    print(f"student_obs (token space) OK: obs dim {OBS_DIM} "
          f"= {N_TOKENS}x{TOKEN_DIM} tokens + {N_GLOBAL} global + {N_ACTIONS} prev-action + {PLAN_DIM} plan")


if __name__ == "__main__":
    demo()
