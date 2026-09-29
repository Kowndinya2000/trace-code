"""Hash-verified model loading and explicitly attributed dual-GN decisions.

No hardware or simulator calls. The release manifest, not historical metadata
embedded in a checkpoint, specifies the chosen operating threshold.
"""
import hashlib
import json
import math
from pathlib import Path


def read_lock(path):
    lock = json.loads(Path(path).read_text())
    if lock.get('schema') != 'locked-student-grasp-models-v1':
        raise ValueError('Unknown model-lock schema')
    if set(lock['models']) != {'bc', 'dagger_r3', 'occlusion_gn'}:
        raise ValueError('Incomplete model lock')
    return lock


def verified_path(lock, asset_dir, role):
    spec = lock['models'][role]
    filename = spec['file']
    if Path(filename).name != filename:
        raise ValueError('Model filename must be a basename')
    path = Path(asset_dir)/filename
    if path.stat().st_size != spec['bytes']:
        raise ValueError('Wrong model size: '+role)
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Wrong model SHA-256: '+role)
    return path


def load_locked_model(manifest, asset_dir, role, device='cpu'):
    import torch
    lock = read_lock(manifest)
    path = verified_path(lock, asset_dir, role)
    spec = lock['models'][role]
    if role == 'occlusion_gn':
        from isaacgymenvs.learning.recurrent_grasp import load_recurrent_grasp
        model, state = load_recurrent_grasp(path, device)
        if model.architecture != spec['architecture']:
            raise ValueError('Grasp architecture differs from lock')
        return model, dict(role=role, threshold=float(spec['threshold']),
            embedded_historical_threshold=state.get('threshold'),
            threshold_source='release_manifest', source_sha256=spec['sha256'])
    from isaacgymenvs.learning.student_net import StudentNet
    state = torch.load(path, map_location='cpu')
    if (state.get('protocol') != spec['protocol'] or state.get('termination_head')
            or state.get('smoke') or state.get('ablate') != 'none'
            or state.get('obs_dim') != spec['observation_dim']):
        raise ValueError('Wrong action model contract')
    weights = state['model']
    model = StudentNet(n_actions=spec['actions'], embed=weights['embed.0.weight'].shape[0],
                       gru_hidden=weights['pi.weight'].shape[1], termination_head=False)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval(), dict(role=role, threshold=None,
        observation_dim=spec['observation_dim'], source_sha256=spec['sha256'])


def dual_gn_decision(occlusion_score, helper_score=None, *, helper_available=False,
                     state_valid=True, helper_source=None, occlusion_threshold=.96,
                     helper_threshold=.9):
    if not 0 < occlusion_threshold < 1 or not 0 < helper_threshold < 1:
        raise ValueError('Invalid thresholds')
    if helper_available and helper_source not in ('oracle', 'real_gn', 'stock_image_gn'):
        raise ValueError('Available helper must have an explicit source')
    if not state_valid:
        return dict(graspable=False, source='none', helper_source=helper_source, state_valid=False)
    if not math.isfinite(occlusion_score) or not 0 <= occlusion_score <= 1:
        raise ValueError('Invalid occlusion-GN score')
    occ = occlusion_score > occlusion_threshold
    helper = False
    if helper_available:
        if helper_score is None or not math.isfinite(helper_score) or not 0 <= helper_score <= 1:
            raise ValueError('Invalid helper-GN score')
        helper = helper_score > helper_threshold
    source = 'both' if occ and helper else 'occlusion_only' if occ else 'helper_only' if helper else 'none'
    return dict(graspable=bool(occ or helper), source=source, helper_source=helper_source,
                state_valid=True, occlusion_score=occlusion_score, helper_score=helper_score)
