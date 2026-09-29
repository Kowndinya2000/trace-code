"""Exact simulator-teacher inference without constructing Isaac Gym.

The selected teacher is an rl_games discrete A2C model with the repository's
TokenSet encoder and a GRU.  Building it through rl_games keeps input
normalisation, token flags, MLP/GRU ordering, and checkpoint loading identical
to ``solve_in_twin.py`` while avoiding simulator startup in a hardware loop.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch


class TeacherPolicy:
    """Stateful deterministic wrapper around the selected recurrent teacher."""

    def __init__(self, checkpoint, network_json, device="cuda"):
        from rl_games.algos_torch import model_builder
        from isaacgymenvs.learning.token_set_builder import TokenSetBuilder

        self.checkpoint = Path(checkpoint).resolve()
        self.network_json = Path(network_json).resolve()
        self.device = torch.device(device)
        self.sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        params = json.loads(self.network_json.read_text())

        # Registration is process-global. Re-registering the same builder is
        # harmless and makes this class usable after other rl_games utilities.
        model_builder.register_network("token_set", lambda **_: TokenSetBuilder())
        with contextlib.redirect_stdout(io.StringIO()):
            factory = model_builder.ModelBuilder().load(params)
            model = factory.build(dict(
                actions_num=16,
                input_shape=(94,),
                num_seqs=1,
                value_size=1,
                normalize_input=True,
                normalize_value=True,
            ))
        saved = torch.load(str(self.checkpoint), map_location="cpu")
        model.load_state_dict(saved["model"], strict=True)
        self.model = model.to(self.device).eval()
        self.epoch = int(saved.get("epoch", -1))
        self.reset()

    def reset(self):
        states = self.model.a2c_network.get_default_rnn_state()
        self.states = tuple(s.to(self.device) for s in states)

    def step(self, observation):
        """Advance the teacher GRU and return ``(action, logits)``.

        ``observation`` is the raw 94-D MoreTeacher observation. The model's
        saved RunningMeanStd performs the same clipping and normalisation as
        the simulator player.
        """
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (94,) or not np.isfinite(obs).all():
            raise ValueError(f"teacher observation must be finite (94,), got {obs.shape}")
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        inputs = dict(is_train=False, prev_actions=None, obs=x,
                      rnn_states=self.states)
        with torch.no_grad():
            result = self.model(inputs)
        self.states = tuple(s.detach() for s in result["rnn_states"])
        logits = result["logits"]
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        action = int(logits.argmax(dim=-1).item())
        return action, logits[0].detach().cpu().numpy().copy()

