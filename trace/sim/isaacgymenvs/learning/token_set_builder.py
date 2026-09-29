"""Permutation-invariant token-set network for rl_games (Stage 1b).

DeepSets-style torso injected into A2CBuilder's actor_cnn slot: the rest of
the proven pipeline (MLP -> GRU -> discrete head) is inherited untouched.
Expects the MoreTeacher 94-D obs: [11 x 8 egocentric corner tokens |
EEF abs (2) | wall distances (4)]; token 0 is the target (flagged).

Train config selects it via network.name: token_set (+ optional
network.token_set: {embed: 64, layers: 2}). Register happens in train.py.
"""
import torch
import torch.nn as nn
from rl_games.algos_torch import network_builder


class TokenSetEncoder(nn.Module):
    def __init__(self, n_tokens=11, token_dim=8, n_global=6, embed=64, layers=2,
                 student_compat=False):
        super().__init__()
        self.n_tokens, self.token_dim, self.n_global = n_tokens, token_dim, n_global
        # LayerNorm after every embedding layer and after pooling. Without it
        # the max-pool branch is unbounded, activations drift with the running
        # obs normalisation, and the categorical head has produced NaN logits
        # (t3-set-s1 died at ep 63). Normalisation is the standard remedy for
        # set encoders and costs almost nothing at this width.
        # student_compat: pad every token with the two slots the STUDENT needs
        # ([visible, staleness]) so teacher and student share an embed input
        # width and semantics, making torso weights transferable. The teacher
        # always feeds visible=1, staleness=0 -- it stays fully privileged.
        # OFF by default: turning it on changes embed.0.weight from (E, D+1) to
        # (E, D+3) and so cannot load any existing checkpoint.
        self.student_compat = bool(student_compat)
        self.n_flags = 3 if self.student_compat else 1
        mods, d = [], token_dim + self.n_flags           # target flag [+ vis, stale]
        for _ in range(layers):
            mods += [nn.Linear(d, embed), nn.LayerNorm(embed), nn.ELU()]
            d = embed
        self.embed = nn.Sequential(*mods)
        self.pool_norm = nn.LayerNorm(2 * embed)
        self.out_dim = 2 * embed + n_global              # mean+max pool + globals
        for m in self.modules():                         # small, stable init
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, obs):
        B = obs.shape[0]
        toks = obs[:, :self.n_tokens * self.token_dim].view(B, self.n_tokens,
                                                            self.token_dim)
        flags = torch.zeros(B, self.n_tokens, self.n_flags, device=obs.device,
                            dtype=obs.dtype)
        flags[:, 0, 0] = 1.0                             # target token
        if self.student_compat:
            flags[:, :, 1] = 1.0                         # visible: teacher sees all
            flags[:, :, 2] = 0.0                         # staleness: never stale
        e = self.embed(torch.cat([toks, flags], dim=-1))  # (B, T, E)
        pooled = self.pool_norm(torch.cat([e.mean(dim=1), e.amax(dim=1)], dim=-1))
        out = torch.cat([pooled, obs[:, self.n_tokens * self.token_dim:]], dim=-1)
        # last-resort guard: one blown-up env must not poison the whole batch's
        # logits (a NaN anywhere makes the categorical sampler raise)
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class TokenSetBuilder(network_builder.A2CBuilder):
    def build(self, name, **kwargs):
        ts = self.params.get("token_set", {})
        obs_dim = kwargs["input_shape"][0]
        n_tokens = int(ts.get("nTokens", 11))
        token_dim = int(ts.get("tokenDim", 8))
        enc = TokenSetEncoder(
            n_tokens=n_tokens, token_dim=token_dim,
            n_global=obs_dim - n_tokens * token_dim,
            embed=int(ts.get("embed", 64)), layers=int(ts.get("layers", 2)),
            student_compat=bool(ts.get("studentCompat", False)))
        kwargs["input_shape"] = (enc.out_dim,)           # mlp sees encoder output
        net = network_builder.A2CBuilder.Network(self.params, **kwargs)
        net.actor_cnn = enc                              # injected torso
        return net
