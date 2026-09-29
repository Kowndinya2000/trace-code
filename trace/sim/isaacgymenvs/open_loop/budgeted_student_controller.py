"""Hardware-side action/stop decisions without simulator or teacher queries."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaacgymenvs.learning.policy_stop import MAX_QUERIES, budget_progress, load_checkpoint
from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.open_loop.evaluation_core import primitive_vectors, sha256
from isaacgymenvs.open_loop.teacher_relative_budget import validate as validate_budget


@dataclass(frozen=True)
class BudgetedStudentDecision:
    kind: str
    primitive: object
    stop_score: float
    budget_progress: float
    completed_decisions: int
    travelled_m: float
    step_limit: int
    length_limit_m: float
    lookups_used: int
    lookups_remaining: int


class BudgetedStudentController:
    """Run a frozen action policy and its separate optimal-stopping critic."""
    def __init__(self, action_checkpoint, stop_checkpoint, device='cpu'):
        self.device = torch.device(device)
        action_path = Path(action_checkpoint).resolve()
        action = torch.load(action_path, map_location='cpu')
        if (action.get('termination_head') or action.get('head') != 'categorical_kl'
                or action.get('smoke')):
            raise ValueError('Expected a production action-only categorical checkpoint')
        self.settings = validate_budget(action.get('teacher_relative_budget'))
        state = action['model']
        self.action = StudentNet(n_actions=state['pi.weight'].shape[0],
            embed=state['embed.0.weight'].shape[0],
            gru_hidden=state['gru.weight_ih_l0'].shape[0] // 3,
            termination_head=False).to(self.device)
        self.action.load_state_dict(state, strict=True)
        self.action.eval()
        self.stop, self.stop_payload = load_checkpoint(
            stop_checkpoint, sha256(action_path), self.device)
        if self.stop_payload['teacher_relative_budget'] != self.settings:
            raise ValueError('Action and stop budget contracts differ')
        self.primitives = torch.tensor(primitive_vectors(
            self.settings['primitive_length_m']), device=self.device)
        self.reset(0)

    def reset(self, teacher_steps):
        if not isinstance(teacher_steps, int) or isinstance(teacher_steps, bool) or teacher_steps < 0:
            raise ValueError('Teacher step count must be a non-negative integer')
        self.teacher_steps = teacher_steps
        self.step_limit = teacher_steps + self.settings['extra_steps']
        self.length_limit_m = (teacher_steps * self.settings['primitive_length_m']
                               + self.settings['extra_length_m'])
        self.action_hidden = None
        self.stop_hidden = None
        self.completed = 0
        self.lookups_used = 0
        self.lookups_remaining = MAX_QUERIES
        self.last_lookup_q = None
        self.last_lookup_step = None
        self.last_lookup_travel_m = None
        self.awaiting_lookup = False
        self.pending = None
        self.terminal = None

    def _decision(self, kind, primitive, score, progress, travelled):
        return BudgetedStudentDecision(kind, primitive, score, progress,
            self.completed, travelled, self.step_limit, self.length_limit_m,
            self.lookups_used, self.lookups_remaining)

    @torch.no_grad()
    def step(self, student_observation, travelled_m):
        if self.terminal is not None:
            return self.terminal
        if self.awaiting_lookup:
            raise RuntimeError('Resolve the outstanding lookup before another policy step')
        observation = torch.as_tensor(student_observation, dtype=torch.float32,
                                      device=self.device)
        if observation.shape != (166,) or not torch.isfinite(observation).all():
            raise ValueError('Expected one finite 166-D student observation')
        travelled = float(travelled_m)
        if not np.isfinite(travelled) or travelled < 0:
            raise ValueError('Expected non-negative measured cumulative TCP travel')
        x = observation[None, None]
        logits, _, self.action_hidden = self.action(x, self.action_hidden)
        if self.stop_payload.get('lookup_feedback') == 'numeric_q':
            context = np.zeros(6, np.float32)
            context[4] = min(self.completed/max(self.step_limit,1),1.)
            context[5] = min(travelled/max(self.length_limit_m,1e-6),1.)
            if self.last_lookup_q is not None:
                context[:4] = [1., self.last_lookup_q,
                    min((self.completed-self.last_lookup_step)/max(self.step_limit,1),1.),
                    min((travelled-self.last_lookup_travel_m)/max(self.length_limit_m,1e-6),1.)]
            stop_values, _, self.stop_hidden = self.stop(
                x, torch.as_tensor(context,device=self.device)[None,None], self.stop_hidden)
        else:
            stop_values, self.stop_hidden = self.stop(x, self.stop_hidden)
        if self.lookups_remaining:
            values = stop_values[0, 0, self.lookups_remaining - 1]
            score = float(values.softmax(-1)[1])
            query = bool(values[1] > values[0])
        else:
            score, query = 0., False
        progress = float(budget_progress(self.completed, travelled,
            self.teacher_steps, self.settings))
        timed_out = (self.completed >= self.step_limit
                or travelled >= self.length_limit_m - 1e-9
                or travelled + self.settings['primitive_length_m'] > self.length_limit_m + 1e-9)
        primitive = int(logits[0, 0].argmax())
        if timed_out:
            decision = self._decision('timeout', None, score, progress, travelled)
            self.terminal = decision
            return decision
        if query:
            decision = self._decision('lookup', None, score, progress, travelled)
            self.awaiting_lookup = True
            self.pending = dict(primitive=primitive, score=score, progress=progress,
                                travelled=travelled)
            return decision
        kind = 'push'
        decision = self._decision(kind, primitive, score, progress, travelled)
        if kind == 'push':
            self.completed += 1
        else:
            self.terminal = decision
        return decision

    def resolve_lookup(self, lookup_result):
        """Consume one lookup; a negative verdict resumes the pending push."""
        if not self.awaiting_lookup or self.pending is None:
            raise RuntimeError('No lookup is awaiting a verdict')
        q_anchored = self.stop_payload.get('lookup_feedback') == 'numeric_q'
        if q_anchored:
            q = float(lookup_result)
            if not np.isfinite(q) or not 0. <= q <= 1.:
                raise ValueError('Numeric lookup q must lie in [0, 1]')
            graspable = q > float(self.stop_payload.get('graspability_threshold', .9))
        else:
            if not isinstance(lookup_result, (bool, np.bool_)):
                raise ValueError('Lookup verdict must be boolean')
            graspable = bool(lookup_result)
        pending = self.pending
        self.awaiting_lookup = False
        self.pending = None
        self.lookups_used += 1
        self.lookups_remaining -= 1
        if graspable:
            decision = self._decision('grasp', None, pending['score'],
                pending['progress'], pending['travelled'])
            self.terminal = decision
            return decision
        if q_anchored:
            self.last_lookup_q = q
            self.last_lookup_step = self.completed
            self.last_lookup_travel_m = pending['travelled']
        decision = self._decision('push', pending['primitive'], pending['score'],
            pending['progress'], pending['travelled'])
        self.completed += 1
        return decision
