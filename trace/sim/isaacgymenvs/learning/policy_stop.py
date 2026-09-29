"""Policy-conditioned recurrent stopping and deterministic budget progress."""
import numpy as np
import torch
from torch import nn

from isaacgymenvs.open_loop import student_obs as so
from isaacgymenvs.open_loop.teacher_relative_budget import validate as validate_budget


PROTOCOL = 'policy-conditioned-budgeted-query-v1'
Q_ANCHORED_PROTOCOL = 'policy-conditioned-q-anchored-query-v2'
HAZARD_WINDOW_PROTOCOL = 'policy-conditioned-hazard-window-query-v3'
MAX_QUERIES = 3
LOOKUP_CONTEXT_DIM = 6
HAZARD_WINDOW_AUX_DIM = 9


def budget_progress(step, travelled_m, step_budget, settings):
    """Return monotone progress toward the first teacher-relative hard cap."""
    settings = validate_budget(settings)
    step = np.asarray(step, dtype=np.float64)
    travelled = np.asarray(travelled_m, dtype=np.float64)
    budget = np.asarray(step_budget, dtype=np.float64)
    if (step < 0).any() or (travelled < 0).any() or (budget < 0).any():
        raise ValueError('Progress inputs must be non-negative')
    step_limit = budget + settings['extra_steps']
    length_limit = budget * settings['primitive_length_m'] + settings['extra_length_m']
    progress = np.maximum(step / step_limit, travelled / length_limit)
    return np.clip(progress, 0., 1.)


class BudgetedLearnedStopAttempt:
    """Resolve up to three learned queries while enforcing both hard limits."""
    def __init__(self, step_budget, horizon, settings, threshold=.9,
                 max_queries=MAX_QUERIES):
        self.settings = validate_budget(settings)
        raw = np.asarray(step_budget)
        self.step_budget = raw.astype(np.int64)
        if (raw.ndim != 1 or not len(raw) or not np.array_equal(raw, self.step_budget)
                or (self.step_budget < 0).any()):
            raise ValueError('Invalid teacher step budgets')
        self.step_limit = np.minimum(self.step_budget + self.settings['extra_steps'], horizon)
        self.length_limit_m = (self.step_budget * self.settings['primitive_length_m']
                               + self.settings['extra_length_m'])
        self.horizon, self.threshold = int(horizon), float(threshold)
        if max_queries != MAX_QUERIES:
            raise ValueError('Unknown lookup budget')
        self.max_queries = int(max_queries)
        self.remaining_queries = np.full(len(raw), self.max_queries, dtype=np.int64)
        self.query_count = np.zeros(len(raw), dtype=np.int64)
        self.failed_queries = np.zeros(len(raw), dtype=np.int64)
        self.reason = np.full(len(raw), 'running', dtype='U24')
        self.step = np.full(len(raw), -1, dtype=np.int64)
        self.terminal_q = np.full(len(raw), np.nan)
        self.terminal_travel_m = np.full(len(raw), np.nan)

    @property
    def live(self):
        return self.reason == 'running'

    def update(self, step, q, oow, invalid=None, travelled_m=None,
               reserve_next_m=0., stop=None):
        q = np.asarray(q)
        oow = np.asarray(oow, dtype=bool)
        travelled = np.asarray(travelled_m, dtype=np.float64)
        n = len(self.reason)
        if (q.shape != (n,) or oow.shape != (n,) or travelled.shape != (n,)
                or not np.isfinite(travelled).all() or (travelled < 0).any()
                or not np.isfinite(reserve_next_m) or reserve_next_m < 0):
            raise ValueError('Invalid budgeted learned-stop update')
        bad = ~np.isfinite(q)
        if invalid is not None:
            bad |= np.asarray(invalid, dtype=bool)
        reason = np.full(n, 'running', dtype='U24')
        reason[step >= self.step_limit] = 'step_timeout'
        distance_due = travelled >= self.length_limit_m - 1e-9
        if reserve_next_m:
            distance_due |= travelled + reserve_next_m > self.length_limit_m + 1e-9
        reason[distance_due] = 'travel_timeout'
        if stop is not None:
            stop = np.asarray(stop, dtype=bool)
            if stop.shape != (n,):
                raise ValueError('Stop batch mismatch')
            stop &= self.live & (self.remaining_queries > 0)
            self.query_count[stop] += 1
            failed = stop & (q <= self.threshold)
            self.failed_queries[failed] += 1
            self.remaining_queries[stop] -= 1
            reason[stop & (q > self.threshold)] = 'success'
        reason[q == -2.] = 'out_of_view'
        reason[bad] = 'invalid_state'
        reason[oow] = 'oow'
        newly = self.live & (reason != 'running')
        self.reason[newly], self.step[newly] = reason[newly], int(step)
        self.terminal_q[newly] = q[newly]
        self.terminal_travel_m[newly] = travelled[newly]
        return np.flatnonzero(newly)

    def rows(self):
        if self.live.any():
            raise RuntimeError('Cannot summarize unfinished attempts')
        return [dict(reason=str(reason), terminal_step=int(step),
                     terminal_q=float(q) if np.isfinite(q) else None,
                     terminal_travel_m=float(travel), success=reason == 'success',
                     lookups=int(lookups), failed_lookups=int(failed),
                     step_budget=int(budget), step_limit=int(step_limit),
                     length_limit_m=float(length_limit))
                for reason, step, q, travel, lookups, failed, budget, step_limit, length_limit in zip(
                    self.reason, self.step, self.terminal_q, self.terminal_travel_m,
                    self.query_count, self.failed_queries,
                    self.step_budget, self.step_limit, self.length_limit_m)]


class StopBackbone(nn.Module):
    """Target-preserving set encoder initialized from the frozen action torso.

    The action encoder pools all tokens before its recurrent layer.  That is
    appropriate for action selection, but a conservative stop decision needs
    to retain the target identity even when its current geometry is blanked by
    arm occlusion.  This torso therefore concatenates the explicitly encoded
    target token with the original pooled set and lets its GRU carry that
    target-specific evidence through occlusion.
    """
    def __init__(self, action_state=None, embed=64, gru_hidden=256):
        super().__init__()
        if action_state is not None:
            embed = action_state['embed.0.weight'].shape[0]
            gru_hidden = action_state['gru.weight_ih_l0'].shape[0] // 3
        mods, width = [], so.TOKEN_DIM + 1
        for _ in range(2):
            mods += [nn.Linear(width, embed), nn.LayerNorm(embed), nn.ELU()]
            width = embed
        self.embed = nn.Sequential(*mods)
        self.pool_norm = nn.LayerNorm(2 * embed)
        torso_in = 3 * embed + so.N_GLOBAL + so.N_ACTIONS + so.PLAN_DIM
        self.trunk = nn.Sequential(nn.Linear(torso_in, 256), nn.LayerNorm(256), nn.ELU(),
                                   nn.Linear(256, 256), nn.LayerNorm(256), nn.ELU())
        self.gru = nn.GRU(256, gru_hidden, batch_first=False)
        self.gru_hidden = gru_hidden
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=.5)
                nn.init.zeros_(module.bias)
        if action_state is not None:
            self._initialize_from_action(action_state, embed)

    def _initialize_from_action(self, action_state, embed):
        own = self.state_dict()
        for name in own:
            if name.startswith(('embed.', 'pool_norm.', 'gru.')):
                own[name].copy_(action_state[name])
        # The new target-specific block is inserted between the original pool
        # and extras.  Zero initialization makes the initial torso exactly the
        # released action torso; fitting may then add target evidence safely.
        old = action_state['trunk.0.weight']
        own['trunk.0.weight'].zero_()
        own['trunk.0.weight'][:, :2 * embed].copy_(old[:, :2 * embed])
        own['trunk.0.weight'][:, 3 * embed:].copy_(old[:, 2 * embed:])
        for name in ('trunk.0.bias', 'trunk.1.weight', 'trunk.1.bias',
                     'trunk.3.weight', 'trunk.3.bias', 'trunk.4.weight',
                     'trunk.4.bias'):
            own[name].copy_(action_state[name])
        self.load_state_dict(own, strict=True)

    def encode(self, observations):
        batch = observations.shape[0]
        token_values = observations[:, :so.N_TOKENS * so.TOKEN_DIM].view(
            batch, so.N_TOKENS, so.TOKEN_DIM)
        flags = torch.zeros(batch, so.N_TOKENS, 1, device=observations.device,
                            dtype=observations.dtype)
        flags[:, 0, 0] = 1.
        encoded = self.embed(torch.cat([token_values, flags], dim=-1))
        pooled = self.pool_norm(torch.cat([encoded.mean(1), encoded.amax(1)], dim=-1))
        extras = observations[:, so.N_TOKENS * so.TOKEN_DIM:]
        return torch.cat([pooled, encoded[:, 0], extras], dim=-1)

    def forward(self, observations, recurrent_state=None):
        time_steps, batch, _ = observations.shape
        encoded = self.encode(observations.reshape(time_steps * batch, -1))
        features = self.trunk(encoded).view(time_steps, batch, -1)
        return self.gru(features, recurrent_state)


class PolicyStopNet(nn.Module):
    """A separate recurrent critic for the continue and stop decisions."""
    def __init__(self, action_state=None, hidden=128, embed=64, gru_hidden=256):
        super().__init__()
        self.backbone = StopBackbone(action_state, embed=embed, gru_hidden=gru_hidden)
        gru_hidden = self.backbone.gru_hidden
        self.stop = nn.Sequential(nn.LayerNorm(gru_hidden), nn.Linear(gru_hidden, hidden),
                                  nn.ELU(), nn.Linear(hidden, MAX_QUERIES * 2))
        nn.init.orthogonal_(self.stop[1].weight, gain=.5)
        nn.init.zeros_(self.stop[1].bias)
        nn.init.orthogonal_(self.stop[3].weight, gain=.5)
        nn.init.zeros_(self.stop[3].bias)

    def forward(self, observations, recurrent_state=None):
        features, recurrent_state = self.backbone(observations, recurrent_state)
        values = self.stop(features)
        return values.view(*values.shape[:-1], MAX_QUERIES, 2), recurrent_state

    def set_backbone_trainable(self, trainable):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)


class QAnchoredPolicyStopNet(nn.Module):
    """Budgeted critic conditioned on the most recent numeric lookup result.

    Context fields are: lookup-present, last verified q, normalized pushes and
    TCP travel since lookup, and overall step/travel progress relative to this
    scene's teacher-relative hard envelope. Auxiliary estimates of current q,
    q change, and near-term threshold-crossing hazard provide dense
    simulator-only representation learning; runtime remains an action-value
    argmax.
    """
    def __init__(self, action_state=None, hidden=128, embed=64, gru_hidden=256,
                 auxiliary_dim=3):
        super().__init__()
        self.backbone = StopBackbone(action_state, embed=embed, gru_hidden=gru_hidden)
        gru_hidden = self.backbone.gru_hidden
        self.context = nn.Sequential(nn.LayerNorm(LOOKUP_CONTEXT_DIM),
            nn.Linear(LOOKUP_CONTEXT_DIM, 32), nn.ELU())
        self.stop = nn.Sequential(nn.LayerNorm(gru_hidden + 32),
            nn.Linear(gru_hidden + 32, hidden), nn.ELU(),
            nn.Linear(hidden, MAX_QUERIES * 2))
        self.auxiliary = nn.Sequential(nn.LayerNorm(gru_hidden + 32),
            nn.Linear(gru_hidden + 32, hidden // 2), nn.ELU(),
            nn.Linear(hidden // 2, auxiliary_dim))
        for layer in (self.context[1], self.stop[1], self.stop[3],
                      self.auxiliary[1], self.auxiliary[3]):
            nn.init.orthogonal_(layer.weight, gain=.5)
            nn.init.zeros_(layer.bias)

    def forward(self, observations, lookup_context, recurrent_state=None):
        if lookup_context.shape != (*observations.shape[:2], LOOKUP_CONTEXT_DIM):
            raise ValueError('Invalid lookup context shape')
        features, recurrent_state = self.backbone(observations, recurrent_state)
        combined = torch.cat([features, self.context(lookup_context)], dim=-1)
        values = self.stop(combined)
        auxiliary = self.auxiliary(combined)
        return (values.view(*values.shape[:-1], MAX_QUERIES, 2),
                auxiliary, recurrent_state)

    def set_backbone_trainable(self, trainable):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = bool(trainable)


class HazardWindowPolicyStopNet(QAnchoredPolicyStopNet):
    """q-anchored critic with dense transient-window hazard predictions.

    Auxiliary outputs retain the v2 prefix (current q, q delta, crossing within
    three decisions) and add current graspability, crossings within 1/2/4/8
    pushes, and loss of the current window after the next push.
    """
    def __init__(self, action_state=None, hidden=128, embed=64, gru_hidden=256):
        super().__init__(action_state, hidden=hidden, embed=embed,
                         gru_hidden=gru_hidden,
                         auxiliary_dim=HAZARD_WINDOW_AUX_DIM)


def load_checkpoint(path, action_checkpoint_sha256=None, device='cpu'):
    payload = torch.load(path, map_location='cpu')
    protocol = payload.get('protocol')
    if (protocol not in (PROTOCOL, Q_ANCHORED_PROTOCOL, HAZARD_WINDOW_PROTOCOL)
            or payload.get('target') != 'continue_and_query_action_values'
            or payload.get('decision_rule') != 'argmax_continue_query'
            or payload.get('max_queries') != MAX_QUERIES):
        raise ValueError('Not a policy-conditioned stopping checkpoint')
    if (action_checkpoint_sha256 is not None
            and payload.get('action_checkpoint_sha256') != action_checkpoint_sha256):
        raise ValueError('Stopping model was trained for a different action policy')
    architecture = payload.get('architecture')
    expected_architecture = {
        PROTOCOL: 'target_preserving_recurrent_v2',
        Q_ANCHORED_PROTOCOL: 'q_anchored_recurrent_v1',
        HAZARD_WINDOW_PROTOCOL: 'hazard_window_recurrent_v1',
    }[protocol]
    if architecture != expected_architecture:
        raise ValueError('Unknown policy-stop architecture')
    state = payload['model']
    model_type = ({PROTOCOL: PolicyStopNet,
                   Q_ANCHORED_PROTOCOL: QAnchoredPolicyStopNet,
                   HAZARD_WINDOW_PROTOCOL: HazardWindowPolicyStopNet}[protocol])
    model = model_type(None, hidden=int(payload['stop_hidden']),
                       embed=int(payload['embed']),
                       gru_hidden=int(payload['gru_hidden']))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    # A two-action argmax is equivalent to a fixed 0.5 stop probability.  This
    # value is serialized only for trace compatibility; it is never calibrated.
    threshold = float(payload['threshold'])
    if threshold != .5 or payload.get('threshold_source') != 'fixed_argmax_equivalent':
        raise ValueError('Optimal stopping must use the fixed argmax rule')
    payload['teacher_relative_budget'] = validate_budget(payload['teacher_relative_budget'])
    payload['lookup_feedback'] = ('numeric_q' if protocol in
                                  (Q_ANCHORED_PROTOCOL, HAZARD_WINDOW_PROTOCOL)
                                  else payload.get('lookup_feedback','binary'))
    return model, payload
