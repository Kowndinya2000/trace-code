"""Stage-2 student: token-set encoder + GRU + policy/value heads.

Mirrors the teacher's torso deliberately (learning/token_set_builder.py): same
set encoder, same mean+max pooling, same LayerNorm placement. The teacher's
tokens are 8 geometry dims and the encoder appends a target flag; the student's
are 8 geometry + [visible, staleness] and the encoder appends the same target
flag, so with token_set.studentCompat=True on the teacher side the two embed
inputs are identical in width AND semantics and weights can transfer.

The GRU is what makes blackout survivable: when V_t = 0 the tokens carry no
geometry at all, so the only things left are proprioception, the plan prior, the
previous action and memory. That is the intended behaviour, not a degenerate
case -- the student is open-loop through a blackout and re-anchors when evidence
returns.
"""
import torch
import torch.nn as nn

from isaacgymenvs.open_loop import student_obs as so


class StudentNet(nn.Module):
    def __init__(self, n_tokens=so.N_TOKENS, token_dim=so.TOKEN_DIM,
                 n_extra=so.N_GLOBAL + so.N_ACTIONS + so.PLAN_DIM,
                 embed=64, layers=2, gru_hidden=256, n_actions=so.N_ACTIONS,
                 termination_head=False):
        super().__init__()
        self.n_tokens, self.token_dim, self.n_extra = n_tokens, token_dim, n_extra
        mods, d = [], token_dim + 1                      # +1 target flag
        for _ in range(layers):
            mods += [nn.Linear(d, embed), nn.LayerNorm(embed), nn.ELU()]
            d = embed
        self.embed = nn.Sequential(*mods)
        self.pool_norm = nn.LayerNorm(2 * embed)
        trunk_in = 2 * embed + n_extra
        self.trunk = nn.Sequential(nn.Linear(trunk_in, 256), nn.LayerNorm(256), nn.ELU(),
                                   nn.Linear(256, 256), nn.LayerNorm(256), nn.ELU())
        self.gru = nn.GRU(256, gru_hidden, batch_first=False)
        self.pi = nn.Linear(gru_hidden, n_actions)
        self.vf = nn.Linear(gru_hidden, 1)
        self.stop = nn.Linear(gru_hidden, 1) if termination_head else None
        self.gru_hidden = gru_hidden
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def encode(self, obs):
        """(B, OBS_DIM) -> (B, 2*embed + n_extra)."""
        B = obs.shape[0]
        n = self.n_tokens * self.token_dim
        toks = obs[:, :n].view(B, self.n_tokens, self.token_dim)
        flags = torch.zeros(B, self.n_tokens, 1, device=obs.device, dtype=obs.dtype)
        flags[:, 0, 0] = 1.0                             # target token
        e = self.embed(torch.cat([toks, flags], dim=-1))
        pooled = self.pool_norm(torch.cat([e.mean(1), e.amax(1)], dim=-1))
        return torch.cat([pooled, obs[:, n:]], dim=-1)

    def recurrent_features(self, obs_seq, h=None):
        T, B, _ = obs_seq.shape
        x = self.trunk(self.encode(obs_seq.reshape(T * B, -1))).view(T, B, -1)
        return self.gru(x, h)

    def forward(self, obs_seq, h=None):
        """Legacy action/value interface; does not evaluate stopping."""
        y, h = self.recurrent_features(obs_seq, h)
        return self.pi(y), self.vf(y).squeeze(-1), h

    def forward_with_stop(self, obs_seq, h=None):
        """One recurrent update; returns action, value, stop logit, memory.

        The stop head estimates current graspability from student inputs.
        The value head estimates RL return and is a separate quantity.
        """
        if self.stop is None: raise ValueError('Checkpoint has no trained termination head')
        y, h = self.recurrent_features(obs_seq, h)
        return self.pi(y), self.vf(y).squeeze(-1), self.stop(y).squeeze(-1), h


def distillation_loss(student_logits, teacher_logits, student_value, teacher_value,
                      value_coef=0.5, mask=None):
    """KL(teacher || student) on the 16-way policy + MSE on the value.

    KL, not cross-entropy on the argmax: the teacher's full distribution says
    which alternatives were nearly as good, and in clutter several pushes often
    are. Collapsing that to a hard label throws away most of the supervision.
    """
    logp_s = torch.log_softmax(student_logits, dim=-1)
    p_t = torch.softmax(teacher_logits, dim=-1)
    kl_t = (p_t * (torch.log_softmax(teacher_logits, dim=-1) - logp_s)).sum(-1)
    v_t = (student_value - teacher_value) ** 2
    if mask is None:
        kl, v = kl_t.mean(), v_t.mean()
    else:                       # ragged shards are zero-padded; ignore the pad
        d = mask.sum().clamp(min=1.0)
        kl, v = (kl_t * mask).sum() / d, (v_t * mask).sum() / d
    return kl + value_coef * v, kl.detach(), v.detach()


def demo():
    torch.manual_seed(0)
    T, B = 7, 5
    net = StudentNet()
    obs = torch.randn(T, B, so.OBS_DIM)
    logits, value, h = net(obs)
    assert logits.shape == (T, B, so.N_ACTIONS), logits.shape
    assert value.shape == (T, B) and h.shape == (1, B, net.gru_hidden)

    # memory must matter: identical blacked-out inputs after different histories
    # should NOT give identical outputs, or the GRU is doing nothing
    blind = torch.zeros(1, B, so.OBS_DIM)
    _, _, h1 = net(torch.randn(4, B, so.OBS_DIM))
    _, _, h2 = net(torch.randn(4, B, so.OBS_DIM))
    o1, _, _ = net(blind, h1)
    o2, _, _ = net(blind, h2)
    assert not torch.allclose(o1, o2, atol=1e-5), "GRU state is being ignored"

    tl = torch.randn(T, B, so.N_ACTIONS)
    loss, kl, v = distillation_loss(logits, tl, value, torch.randn(T, B))
    assert torch.isfinite(loss)
    # KL against itself must be ~0
    _, kl0, _ = distillation_loss(tl, tl, value, value)
    assert kl0.abs() < 1e-6, kl0
    n_par = sum(p.numel() for p in net.parameters())
    print(f"StudentNet OK: obs {so.OBS_DIM} -> logits {tuple(logits.shape)}, "
          f"{n_par/1e6:.2f}M params, KL(self)={float(kl0):.2e}")


if __name__ == "__main__":
    demo()
