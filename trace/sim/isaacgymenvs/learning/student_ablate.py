"""Observation- and architecture-level ablations for the Stage-2 student.

Reviewer-facing question these answer: WHICH parts of the 166-D student
observation, and which architectural component, actually carry the task?

The occlusion sweep shows tokens are necessary (success collapses from 86% to
15.6% when every token is dropped). It does NOT show the plan prior or the
recurrence are necessary -- a policy could in principle ignore either. These
ablations zero individual observation blocks, or remove the recurrence, and
retrain from the same shards so the comparison is controlled.

Channel layout (open_loop/student_obs.py), OBS_DIM = 166:
    [0   : 110)  11 tokens x 10 dims  (8 geometry + [visible, staleness])
    [110 : 116)  6 globals (EEF xy + 4 wall distances)
    [116 : 132)  16-D one-hot of the previous action
    [132 : 166)  34-D plan encoding:
                   [132:140) 4 lookahead EEF poses
                   [140:142) EEF drift
                   [142:144) progress / exhausted flags
                   [144:166) 22-D PREDICTED object layout
"""
import torch

N_TOKENS, TOKEN_DIM, GEOM = 11, 10, 8
TOK_END = N_TOKENS * TOKEN_DIM          # 110
GLOB, PREV = slice(110, 116), slice(116, 132)
PLAN = slice(132, 166)
PLAN_POSE, PLAN_PREDOBJ = slice(132, 144), slice(144, 166)

ABLATIONS = {
    "none":        "full observation, full architecture (control)",
    "no_stale":    "zero the staleness channel of every token",
    "no_vis":      "zero the visibility flag (geometry already zeroed when occluded)",
    "no_plan":     "zero the entire 34-D plan encoding",
    "no_predobj":  "zero only the 22-D predicted object layout, keep plan poses",
    "no_prev":     "zero the previous-action one-hot",
    "no_gru":      "replace the GRU with a memoryless linear map",
    "no_plan_no_gru": "zero the entire 34-D plan encoding and replace the GRU with a memoryless map",
    "initial_scene": ("zero the nominal EEF block and replace predicted object centres with the "
                      "twin's initial centres (obs_variants.initial_scene_context)"),
}


def obs_mask(name, device, dtype=torch.float32):
    """Multiplicative (166,) mask, or None for architecture-only ablations.

    initial_scene is a feature substitution, not a mask: callers apply
    obs_variants.initial_scene_context to the observation instead.
    """
    if name in ("none", "no_gru", "initial_scene"):
        return None
    m = torch.ones(166, device=device, dtype=dtype)
    if name == "no_stale":
        for t in range(N_TOKENS):
            m[t * TOKEN_DIM + GEOM + 1] = 0.0
    elif name == "no_vis":
        for t in range(N_TOKENS):
            m[t * TOKEN_DIM + GEOM] = 0.0
    elif name in ("no_plan", "no_plan_no_gru"):
        m[PLAN] = 0.0
    elif name == "no_predobj":
        m[PLAN_PREDOBJ] = 0.0
    elif name == "no_prev":
        m[PREV] = 0.0
    else:
        raise KeyError(f"unknown ablation {name!r}; choices: {sorted(ABLATIONS)}")
    return m


def strip_recurrence(net, width=None):
    """Swap the GRU for a memoryless map.

    Default (width=None): a 2-layer MLP of the GRU's hidden width (~132k
    parameters versus ~395k for the GRU). width=768 gives ~394k parameters,
    a capacity-matched memoryless control. The returned module keeps the
    (input, h) -> (output, h) signature; h is passed through unchanged.
    """
    import torch.nn as nn

    class NoMemory(nn.Module):
        def __init__(self, d_in, d_hid, d_out):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d_in, d_hid), nn.ELU(),
                                     nn.Linear(d_hid, d_out))
        def forward(self, x, h=None):
            return self.net(x), h

    hidden = net.gru_hidden if width is None else int(width)
    net.gru = NoMemory(256, hidden, net.gru_hidden).to(next(net.parameters()).device)
    return net
