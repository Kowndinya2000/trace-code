"""Resolve and verify a teacher identity throughout collection and evaluation."""
from pathlib import Path
import os
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]
DATA=Path(os.environ.get('TRACE_DATA', ROOT.parents[2]/'data'))
DEFAULT=DATA/'checkpoints/teacher_ep210.pth'
ZERO_TERMS=['lambdaC','lambdaC3','lambdaArc','lambdaB','lambdaEef','lambdaDisturb']


def teacher_path(explicit=None):
    path=Path(explicit or os.environ.get('TRACE_TEACHER_CHECKPOINT') or DEFAULT).resolve()
    if not path.is_file(): raise FileNotFoundError(path)
    return path


def verify_minimal_teacher(path):
    path=teacher_path(path)
    done=json.loads((path.parent/'complete.json').read_text())
    provenance=json.loads((path.parent/'provenance.json').read_text())
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if not done['complete'] or done.get('smoke',True) or done.get('epochs')!=450:
        raise ValueError('Require a completed production teacher at epoch 450')
    if done['checkpoint_sha256']!=actual: raise ValueError('Teacher checkpoint hash changed')
    if provenance.get('protocol')!='graspability-only-teacher-v1': raise ValueError('Wrong teacher protocol')
    if any(float(provenance['reward'][k])!=0 for k in ZERO_TERMS):
        raise ValueError('Excluded shaping term found in teacher provenance')
    expected=dict(lambdaG=2.,lambdaStep=.1,rSuccess=10.,rOow=12.,rOutOfView=11.,gamma=.99)
    if any(float(provenance['reward'][k])!=v for k,v in expected.items()):
        raise ValueError('Wrong graspability, step, terminal, or discount coefficient')
    if provenance['spec']['seed']!=done['seed'] or provenance['spec']['num_envs']!=1024:
        raise ValueError('Teacher seed or environment budget differs from its specification')
    inference=json.loads((path.parent/'inference_validation.json').read_text())
    if (provenance.get('grasp_inference_batch_size')!=64 or not inference['valid'] or
        not inference['threshold_labels_match'] or inference['max_logit_error']>1e-5):
        raise ValueError('Frozen classifier inference adapter was not validated')
    return dict(checkpoint=str(path),sha256=actual,seed=done['seed'],reward_recipe='graspability_only')
