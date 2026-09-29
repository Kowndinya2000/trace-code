"""Deployable student decisions: current observation in; push/stop/timeout out.

No simulator, true graspability, teacher query, or image of the hidden scene is
used. Call once initially, then once after each completed motion primitive.
The caller handles retraction/verification on hardware after stop or timeout.
"""
from dataclasses import dataclass
import numpy as np
import torch
from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.learning.student_ablate import obs_mask, strip_recurrence
from isaacgymenvs.open_loop.evaluation_core import primitive_vectors


@dataclass(frozen=True)
class StudentDecision:
    kind: str
    primitive: object
    predicted_graspability: float
    completed_decisions: int


class StudentController:
    def __init__(self, checkpoint, device='cpu', hard_limit=120):
        ck=torch.load(checkpoint,map_location='cpu')
        if not ck.get('termination_head') or ck.get('smoke'):
            raise ValueError('A production checkpoint with a trained graspability head is required')
        sd=ck['model'];hidden=sd['pi.weight'].shape[1]
        self.device=torch.device(device)
        self.net=StudentNet(n_actions=sd['pi.weight'].shape[0],embed=sd['embed.0.weight'].shape[0],
                            gru_hidden=hidden,termination_head=True).to(self.device)
        if ck['ablate']=='no_gru':self.net=strip_recurrence(self.net)
        self.net.load_state_dict(sd,strict=True);self.net.eval()
        self.mask=obs_mask(ck['ablate'],self.device)
        self.P=torch.tensor(primitive_vectors(.04),device=self.device)
        self.threshold=float(ck['stop_threshold'])
        if not 0<self.threshold<1 or not isinstance(hard_limit,int) or hard_limit<0:
            raise ValueError('Invalid stopping configuration')
        self.hard_limit=hard_limit
        self.reset()

    def reset(self):
        self.h=None;self.count=0;self.terminal=None

    @torch.no_grad()
    def step(self, student_observation):
        if self.terminal is not None:return self.terminal
        x=torch.as_tensor(student_observation,dtype=torch.float32,device=self.device)
        if x.shape!=(166,) or not torch.isfinite(x).all():raise ValueError('Expected finite 166-D student observation')
        if self.mask is not None:x=x*self.mask
        action,_,logit,self.h=self.net.forward_with_stop(x[None,None,:],self.h)
        score=float(logit.sigmoid()[0,0])
        if not np.isfinite(score):raise RuntimeError('Non-finite predicted graspability')
        if score>self.threshold or self.count>=self.hard_limit:
            self.terminal=StudentDecision('stop' if score>self.threshold else 'timeout',None,score,self.count)
            return self.terminal
        primitive=(int(torch.cdist(action[0],self.P).argmin()) if action.shape[-1]==2
                   else int(action[0,0].argmax()))
        decision=StudentDecision('push',primitive,score,self.count)
        self.count+=1
        return decision
