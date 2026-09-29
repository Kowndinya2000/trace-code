"""Representation/observation ablation refits on a fixed expert+DAgger aggregate.

Same data validation, grouped split, categorical-KL loss, recovery weighting,
optimizer, update count and validation-loss checkpoint selection as
train_student_repaired.py. Differences are explicit and bound into the checkpoint:

  --plan-mode/--plan-k        look-ahead block of the nominal plan (obs_variants)
  --visibility-schedule/...   re-derive token visibility from the shards' clean
                              observations (training blackout/dropout sweeps)
  --ablate no_gru_wide        capacity-matched memoryless control

Recovery weights are always computed from the released 166-D observations, whose
plan geometry does not depend on the variant, so every variant sees identical weights.
"""
from pathlib import Path
import argparse
import json
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from train_student_repaired import load_data, grouped_split, reduce_masked
from isaacgymenvs.learning.student_net import StudentNet
from isaacgymenvs.learning.student_ablate import obs_mask, strip_recurrence
from isaacgymenvs.open_loop.evaluation_core import PROTOCOL, sha256, read_manifest
from isaacgymenvs.open_loop import obs_variants as ov
from isaacgymenvs.open_loop.teacher_relative_budget import config as make_budget_config, recovery_weight_numpy

WIDE_WIDTH = 768


def shard_nominal(path, d):
    """The nominal plan file this shard was collected with.

    Nominal rollouts are regenerated per collection and are not identical across
    runs, so the plan must be the shard's own: its sibling nominal.json, or for
    cache-derived expert shards the cache rollout whose plans reproduce every
    stored look-ahead block exactly.
    """
    key = tuple(d['scene_hash'].tolist())
    candidates = [path.parent.parent / 'nominal.json']
    if 'derived_from_cache_sha256' in d:
        runtime = next((p for p in path.parents if (p / 'expert-trajectory-cache').is_dir()), None)
        if runtime is not None:
            cache = runtime / 'expert-trajectory-cache' / str(d['derived_from_cache_sha256']) / 'rollouts'
            candidates += sorted(cache.glob('*/nominal.json'))
    t = np.arange(d['obs'].shape[0])
    for candidate in candidates:
        if not candidate.exists():
            continue
        nominal = json.loads(candidate.read_text())
        if tuple(r['scene']['sha256'] for r in nominal['rows']) != key:
            continue
        plans = nominal['plans']
        if [len(p['actions']) for p in plans] != d['step_budget'].tolist():
            continue
        exact = all(np.array_equal(
            ov.lookahead(plans[i]['xy'], t, d['obs'][:, i, ov.TOKEN_END:ov.TOKEN_END + 2], ov.DEFAULT_LAYOUT)
            [d['decision_observed'][:, i]], d['obs'][:, i, ov.BASE_DIM:ov.BASE_DIM + 8][d['decision_observed'][:, i]])
            for i in range(len(plans)))
        if exact:
            return candidate, nominal
    raise ValueError(f'No nominal plan reproduces the stored plan block of {path}')


def build_variant(files, base, lay, visibility, initial_scene=False):
    """(T, N, D) variant observations aligned with load_data's aggregate order.

    initial_scene substitutes the plan block with obs_variants.initial_scene_context, using
    each shard's own nominal object trajectory (index 0), after the released-block checks.
    """
    T, N, _ = base.shape
    D = ov.obs_dim(lay)
    out = np.zeros((T, N, D), np.float32)
    offset = 0
    audit = dict(nominal_files=[], token_block_identical=True, plan_tail_identical=True)
    for path in files:
        with np.load(path) as archive:
            d = {k: archive[k] for k in archive.files}
        t, n = d['obs'].shape[:2]
        source = d['obs']
        if visibility is not None:
            source = source.copy()
            source[..., :ov.TOKEN_END] = ov.remask_tokens(d, visibility['p_drop'], visibility['blackout_len'],
                                                          visibility['schedule'])[..., :ov.TOKEN_END]
        nominal_path, nominal = shard_nominal(path, d)
        audit['nominal_files'].append(str(nominal_path))
        tt = np.arange(t)
        for i in range(n):
            plan = nominal['plans'][i]
            out[:t, offset + i] = ov.convert(source[:, i], plan['xy'], tt, lay, plan['objects'])
        block = base[:t, offset:offset + n]
        tail_start = ov.plan_components(lay)['drift'][0]
        if not np.array_equal(out[:t, offset:offset + n, tail_start:tail_start + ov.TAIL_DIM], block[..., 132 + 8:]):
            raise ValueError(f'Plan tail changed for {path}')
        if visibility is None and not np.array_equal(out[:t, offset:offset + n, :ov.BASE_DIM], block[..., :ov.BASE_DIM]):
            raise ValueError(f'Token/global block changed without a visibility override for {path}')
        audit['token_block_identical'] &= bool(np.array_equal(out[:t, offset:offset + n, :ov.TOKEN_END],
                                                              block[..., :ov.TOKEN_END]))
        if initial_scene:
            for i in range(n):
                out[:t, offset + i] = ov.initial_scene_context(out[:t, offset + i], nominal['plans'][i]['objects'])
        offset += n
    if offset != N:
        raise ValueError('Variant aggregate size mismatch')
    return out, audit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', nargs='+', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split-seed', type=int, default=20260907)
    ap.add_argument('--updates', type=int, default=10000)
    ap.add_argument('--batch-seqs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--ablate', default='none', choices=['none', 'no_gru', 'no_gru_wide', 'no_plan',
                                                         'initial_scene', 'no_plan_no_gru'])
    ap.add_argument('--plan-mode', default='window', choices=['window', 'full_static'])
    ap.add_argument('--plan-k', type=int, default=4)
    ap.add_argument('--plan-objects', default='current', choices=['current', 'window'])
    ap.add_argument('--visibility-schedule', default=None, choices=list(ov.SCHEDULES))
    ap.add_argument('--train-p-drop', type=float, default=None)
    ap.add_argument('--train-blackout-len', type=int, default=None)
    ap.add_argument('--drop-plan-components', nargs='*', default=[], choices=list(ov.PLAN_COMPONENTS))
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    lay = ov.layout(dict(plan_mode=a.plan_mode, plan_k=a.plan_k, plan_objects=a.plan_objects))
    plan_drop = ov.validate_drop(a.drop_plan_components)
    if plan_drop and a.ablate in ('no_plan', 'no_plan_no_gru'):
        ap.error('no_plan already removes every plan component')
    if a.ablate in ('no_plan', 'no_plan_no_gru', 'initial_scene') and not ov.is_default(lay):
        ap.error(f'{a.ablate} replaces the released 166-D plan block; use the default layout')
    if plan_drop and a.ablate == 'initial_scene':
        ap.error('initial_scene already defines the whole plan block')
    overrides = (a.visibility_schedule, a.train_p_drop, a.train_blackout_len)
    if any(v is not None for v in overrides) and not all(v is not None for v in overrides):
        ap.error('Visibility overrides require schedule, p_drop and blackout length together')
    visibility = (dict(schedule=a.visibility_schedule, p_drop=float(a.train_p_drop),
                       blackout_len=int(a.train_blackout_len)) if a.visibility_schedule else None)
    out = Path(a.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    if read_manifest(a.manifest).get('role') != 'training':
        raise ValueError('Production fitting requires a training-role manifest')
    budget_config = make_budget_config(20, .15, .04, .70, .10, 3.)
    started_load = time.monotonic()
    d, files = load_data(a.data, a.manifest, True, True, False, False, budget_config)
    base = d['obs']
    obs, audit = build_variant(files, base, lay, visibility, initial_scene=a.ablate == 'initial_scene')
    ti, vi = grouped_split(d['scene_hash'], a.split_seed)
    mk = d['valid'].copy()
    available = int(mk[:, ti].sum())
    ti = ti[mk[:, ti].sum(0) > 0]
    vi = vi[mk[:, vi].sum(0) > 0]
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(torch.device(a.device))
    net = StudentNet(n_actions=16, n_extra=obs.shape[-1] - ov.TOKEN_END).to(a.device)
    if a.ablate in ('no_gru', 'no_plan_no_gru'):
        net = strip_recurrence(net)
    elif a.ablate == 'no_gru_wide':
        net = strip_recurrence(net, WIDE_WIDTH)
    X = torch.tensor(obs, device=a.device)
    mask = obs_mask(a.ablate if a.ablate != 'no_gru_wide' else 'none', a.device)
    component = ov.component_mask(lay, plan_drop)
    if component is not None:
        component = torch.tensor(component, device=a.device)
        mask = component if mask is None else mask * component
    if mask is not None:
        X = X * mask
    L = torch.tensor(d['teacher_logits'], device=a.device)
    M = torch.tensor(mk, device=a.device)
    W = torch.tensor(recovery_weight_numpy(base, d['step_budget'], budget_config), device=a.device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    ti = torch.tensor(ti, device=a.device)
    vi = torch.tensor(vi, device=a.device)
    source = Path(__file__).resolve()
    n_params = sum(p.numel() for p in net.parameters())
    memory_params = sum(p.numel() for p in net.gru.parameters())
    provenance = dict(protocol=PROTOCOL, configuration=vars(a), obs_layout=lay, obs_dim=int(obs.shape[-1]),
                      plan_drop=plan_drop, plan_components=ov.plan_components(lay),
                      training_visibility_override=visibility, visibility_config=d['visibility_config'],
                      source_sha256=sha256(source), variant_audit=audit,
                      manifest_sha256=sha256(a.manifest), data_sha256={str(p): sha256(p) for p in files},
                      teacher_relative_budget_config=budget_config,
                      available_training_labels=available, used_training_labels=int(M[:, ti].sum().item()),
                      validation_labels=int(M[:, vi].sum().item()), trajectories=len(d['scene_hash']),
                      parameters=n_params, memory_module_parameters=memory_params,
                      action_weight_mean=float(W[M > 0].mean().item()),
                      load_seconds=time.monotonic() - started_load, gpu=torch.cuda.get_device_name())
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'FIT {a.ablate} {lay} vis={visibility}: obs {obs.shape[-1]}, {len(ti)} train / {len(vi)} val sequences, '
          f'{provenance["used_training_labels"]} labels, {n_params} params', flush=True)

    def loss(sel):
        pred, _, _ = net(X[:, sel])
        lp = torch.log_softmax(L[:, sel], -1)
        per = (lp.exp() * (lp - torch.log_softmax(pred, -1))).sum(-1)
        return reduce_masked(per, M[:, sel], False, W[:, sel])

    started = time.monotonic()
    cursor, order = len(ti), ti
    history, best_loss, best_state, best_update = [], float('inf'), None, None
    for update in range(a.updates):
        if cursor >= len(order):
            order = ti[torch.randperm(len(ti), device=a.device)]
            cursor = 0
        sel = order[cursor:cursor + a.batch_seqs]
        cursor += len(sel)
        net.train()
        act = loss(sel)
        if not torch.isfinite(act):
            raise RuntimeError('Non-finite fitting loss')
        opt.zero_grad()
        act.backward()
        opt.step()
        if (update + 1) % 500 == 0 or update + 1 == a.updates:
            net.eval()
            with torch.no_grad():
                validation = float(loss(vi))
            history.append(dict(update=update + 1, train_action=float(act), validation_action=validation,
                                seconds=time.monotonic() - started))
            print(json.dumps(history[-1]), flush=True)
            (out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
            if validation < best_loss:
                best_loss, best_update = validation, update + 1
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if sha256(source) != provenance['source_sha256']:
        raise RuntimeError('Trainer changed during fitting')
    net.load_state_dict(best_state)
    (out / 'selection.json').write_text(json.dumps(dict(
        rule='minimum grouped training-scene validation action loss', update=best_update,
        validation_loss=best_loss, development_used=False), indent=2) + '\n')
    torch.save(dict(model=net.state_dict(), obs_dim=int(obs.shape[-1]), obs_layout=lay, plan_drop=plan_drop,
                    head='categorical_kl',
                    protocol=PROTOCOL, ablate=a.ablate, seed=a.seed, smoke=False,
                    visibility_config=d['visibility_config'], training_visibility_override=visibility,
                    simulator_oracle_experiment=True, bounded_dagger_config=None,
                    teacher_relative_budget=budget_config, cartesian_policy=None,
                    termination_head=False, stop_threshold=None, stop_target=None), out / 'student.pt')
    (out / 'complete.json').write_text(json.dumps(dict(complete=True, updates=a.updates,
        checkpoint_sha256=sha256(out / 'student.pt'), selected_update=best_update,
        seconds=time.monotonic() - started), indent=2) + '\n')
    print('FIT COMPLETE', out, flush=True)


if __name__ == '__main__':
    main()
